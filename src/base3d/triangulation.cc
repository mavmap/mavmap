/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "triangulation.h"


Eigen::Vector3d
triangulate_point(const Eigen::Matrix<double, 3, 4>& proj_matrix1,
                  const Eigen::Matrix<double, 3, 4>& proj_matrix2,
                  const Eigen::Vector2d& point1,
                  const Eigen::Vector2d& point2) {

  Eigen::Matrix<double, 6, 4> A;

  const double x1 = point1(0);
  const double y1 = point1(1);
  const double x2 = point2(0);
  const double y2 = point2(1);

  // Set up Jacobian as point2D x (P x point3D) = 0
  for (size_t k=0; k<4; ++k) {
    // first set of points
    A(0, k) = x1 * proj_matrix1(2, k) - proj_matrix1(0, k);
    A(1, k) = y1 * proj_matrix1(2, k) - proj_matrix1(1, k);
    A(2, k) = x1 * proj_matrix1(1, k) - y1 * proj_matrix1(0, k);
    // second set of points
    A(3, k) = x2 * proj_matrix2(2, k) - proj_matrix2(0, k);
    A(4, k) = y2 * proj_matrix2(2, k) - proj_matrix2(1, k);
    A(5, k) = x2 * proj_matrix2(1, k) - y2 * proj_matrix2(0, k);
  }

  // Homogeneous 3D point is eigen vector corresponding to smallest singular
  // value. JacobiSVD is the most accurate method, though generally slow -
  // but fast for small matrices.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
  Eigen::Matrix<double, 4, 4> V = svd.matrixV();

  // Normalize point
  V(0, 3) /= V(3, 3);
  V(1, 3) /= V(3, 3);
  V(2, 3) /= V(3, 3);

  return V.block<3, 1>(0, 3);

}


std::vector<Eigen::Vector3d>
triangulate_points(const Eigen::Matrix<double, 3, 4>& proj_matrix1,
                   const Eigen::Matrix<double, 3, 4>& proj_matrix2,
                   const std::vector<Eigen::Vector2d>& points1,
                   const std::vector<Eigen::Vector2d>& points2) {

  std::vector<Eigen::Vector3d> points3D(points1.size());

  #pragma omp parallel shared(points3D, proj_matrix1, proj_matrix2, \
                              points1, points2)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<points3D.size(); ++i) {
      points3D[i] = triangulate_point(proj_matrix1, proj_matrix2,
                                      points1[i], points2[i]);
    }
  }

  return points3D;

}


Eigen::Matrix<double, Eigen::Dynamic, 3>
triangulate_points(const Eigen::Matrix<double, 3, 4>& proj_matrix1,
                   const Eigen::Matrix<double, 3, 4>& proj_matrix2,
                   const Eigen::Matrix<double, Eigen::Dynamic, 2>& points1,
                   const Eigen::Matrix<double, Eigen::Dynamic, 2>& points2) {

  Eigen::Matrix<double, Eigen::Dynamic, 3> points3D(points1.rows(), 3);

  #pragma omp parallel shared(points3D, proj_matrix1, proj_matrix2, \
                              points1, points2)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<points3D.size(); ++i) {
      points3D.row(i) = triangulate_point(proj_matrix1, proj_matrix2,
                                          points1.row(i), points2.row(i));
    }
  }

  return points3D;

}


std::vector<double>
calc_tri_angles(const Eigen::Matrix<double, 3, 4>& proj_matrix1,
                const Eigen::Matrix<double, 3, 4>& proj_matrix2,
                const std::vector<Eigen::Vector3d>& points3D) {

  const Eigen::Matrix<double, 3, 4> inv_proj_matrix1
    = invert_proj_matrix(proj_matrix1);
  const Eigen::Matrix<double, 3, 4> inv_proj_matrix2
    = invert_proj_matrix(proj_matrix2);

  // Camera positions in object coordinate system
  const Eigen::Vector3d& tvec1 = inv_proj_matrix1.block<3, 1>(0, 3);
  const Eigen::Vector3d& tvec2 = inv_proj_matrix2.block<3, 1>(0, 3);

  // Baseline length between cameras
  const double baseline = (tvec1 - tvec2).norm();
  double baseline2 = baseline * baseline;

  std::vector<double> angles(points3D.size());

  #pragma omp parallel shared(points3D, angles, baseline2)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<points3D.size(); ++i) {
      const Eigen::Vector3d& point3D = points3D[i];

      // Ray lengths from cameras to point
      const double ray1 = (point3D - tvec1).norm();
      const double ray2 = (point3D - tvec2).norm();

      // Angle between rays at point within the enclosing triangle,
      // see "law of cosines"
      const double angle = acos((ray1 * ray1 + ray2 * ray2 - baseline2)
                                / (2 * ray1 * ray2));
      if (std::isnan(angle)) {
        angles[i] = 0;
      } else {
        angles[i] = angle;
      }

    }
  }

  return angles;

}
