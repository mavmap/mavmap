/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <Eigen/Core>

#include "base3d/similarity_transform.h"
#include "base3d/triangulation.h"
#include "util/test.h"


void test_triangulate_point() {

  const std::vector<Eigen::Vector3d> points3D = {
    Eigen::Vector3d(0, 0.1, 0.1),
    Eigen::Vector3d(0, 1, 3),
    Eigen::Vector3d(0, 1, 2),
    Eigen::Vector3d(0.01, 0.2, 3),
    Eigen::Vector3d(-1, 0.1, 1),
    Eigen::Vector3d(0.1, 0.1, 0.2),
  };

  Eigen::Matrix<double, 3, 4> proj_matrix1 = Eigen::MatrixXd::Identity(3, 4);

  for (double rx=0; rx<1; rx+=0.2) {

    for (double tx=0; tx<10; tx+=2) {

      SimilarityTransform3D tform(1, rx, 0.2, 0.3, tx, 2, 3);

      Eigen::Matrix<double, 3, 4> proj_matrix2
        = tform.matrix().block<3, 4>(0, 0);

      for (size_t i=0; i<points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector4d point3D1(point3D(0), point3D(1), point3D(2), 1);
        Eigen::Vector3d point2D1 = proj_matrix1 * point3D1;
        Eigen::Vector3d point2D2 = proj_matrix2 * point3D1;
        point2D1 /= point2D1(2);
        point2D2 /= point2D2(2);

        const Eigen::Vector2d point2D1_N(point2D1(0), point2D1(1));
        const Eigen::Vector2d point2D2_N(point2D2(0), point2D2(1));

        const Eigen::Vector3d tri_point3D
          = triangulate_point(proj_matrix1, proj_matrix2,
                              point2D1_N, point2D2_N);

        ASSERT_ALMOST_EQUAL((point3D - tri_point3D).norm(), 0, 1e-10);
      }

    }

  }

}


int main(int argc, char* argv[]) {

  test_triangulate_point();

  return 0;

}
