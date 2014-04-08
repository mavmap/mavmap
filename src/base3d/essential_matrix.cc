/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "essential_matrix.h"


std::vector<Eigen::MatrixXd>
EssentialMatrixEstimator::estimate(const Eigen::MatrixXd& points1,
                                   const Eigen::MatrixXd& points2) {

  const size_t num_points = points1.rows();

  const Eigen::MatrixXd& P1 = points1;
  const Eigen::MatrixXd& P2 = points2;


  // Step 1: Extraction of the nullspace x, y, z, w

  Eigen::MatrixXd Q(num_points, 9);
  Q.col(0) = P1.col(0).cwiseProduct(P2.col(0));
  Q.col(1) = P1.col(1).cwiseProduct(P2.col(0));
  Q.col(2) = P2.col(0);
  Q.col(3) = P1.col(0).cwiseProduct(P2.col(1));
  Q.col(4) = P1.col(1).cwiseProduct(P2.col(1));
  Q.col(5) = P2.col(1);
  Q.col(6) = P1.col(0);
  Q.col(7) = P1.col(1);
  Q.col(8).fill(1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q, Eigen::ComputeFullV);
  // Eigen::Matrix<double, 9, 9> V = ;
  // Extract 4 eigen vectors corresponding to the smallest singular values
  Eigen::Matrix<double, 4, 9, Eigen::RowMajor> E
    = svd.matrixV().block<9, 4>(0, 5).transpose();


  // Step 3: Gauss-Jordan elimination with partial pivoting on the
  //         10x20 matrix A

  Eigen::Matrix<double, 10, 20, Eigen::RowMajor> A;
  #include "essential_matrix_poly.h"
  Eigen::Matrix<double, 10, 10> AA = A.block<10, 10>(0, 0).inverse()
                                     * A.block<10, 10>(0, 10);


  // Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
  //         matrix B to obtain the tenth degree polynomial

  Eigen::Matrix<double, 13, 3> B;
  for (int i=0; i<3; ++i) {
    Eigen::Matrix<double, 1, 13> row1;
    row1.fill(0);
    row1.block<1, 3>(0, 1) = AA.block<1, 3>(i * 2 + 4, 0);
    row1.block<1, 3>(0, 5) = AA.block<1, 3>(i * 2 + 4, 3);
    row1.block<1, 4>(0, 9) = AA.block<1, 4>(i * 2 + 4, 6);
    Eigen::Matrix<double, 1, 13> row2;
    row2.fill(0);
    row2.block<1, 3>(0, 0) = AA.block<1, 3>(i * 2 + 5, 0);
    row2.block<1, 3>(0, 4) = AA.block<1, 3>(i * 2 + 5, 3);
    row2.block<1, 4>(0, 8) = AA.block<1, 4>(i * 2 + 5, 6);
    B.col(i) = row1 - row2;
  }

  // Step 5: Extraction of roots from the tenth degree polynomial
  std::vector<double> coeffs(11);
  #include "essential_matrix_coeff.h"

  std::vector<std::complex<double>> roots = poly_solve(coeffs);

  std::vector<double> xs, ys, zs;

  std::vector<Eigen::MatrixXd> models;

  #pragma omp parallel shared(roots, models)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<roots.size(); ++i) {

      if (fabs(roots[i].imag()) > 1e-10) {
        continue;
      }

      const double z1 = roots[i].real();
      const double z2 = z1 * z1;
      const double z3 = z2 * z1;
      const double z4 = z3 * z1;

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Bz;
      for (size_t j=0; j<3; ++j) {
          const double* br = b + j * 13;
          Bz(j, 0) = br[0] * z3 + br[1] * z2 + br[2] * z1 + br[3];
          Bz(j, 1) = br[4] * z3 + br[5] * z2 + br[6] * z1 + br[7];
          Bz(j, 2) = br[8] * z4 + br[9] * z3 + br[10] * z2
                     + br[11] * z1 + br[12];
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(Bz, Eigen::ComputeFullV);
      Eigen::Vector3d X = svd.matrixV().block<3, 1>(0, 2);

      if (fabs(X(2)) < 1e-10) {
        continue;
      }

      Eigen::MatrixXd essential_vec
        = E.row(0) * (X(0) / X(2))
          + E.row(1) * (X(1) / X(2))
          + E.row(2) * z1
          + E.row(3);

      essential_vec /= essential_vec.norm();
      essential_vec.resize(3, 3);
      const Eigen::Matrix3d essential_matrix = essential_vec.transpose();

      #pragma omp critical
      models.push_back(essential_matrix);

    }
  }

  return models;

}


void
EssentialMatrixEstimator::residuals(const Eigen::MatrixXd& points1,
                                    const Eigen::MatrixXd& points2,
                                    const Eigen::MatrixXd& E,
                                    std::vector<double>& residuals) {

  // Compute Sampson distance for each point

  const Eigen::Matrix3d Et = E.transpose();

  residuals.resize(points1.rows());

  #pragma omp parallel shared(points1, points2, E, residuals)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<points1.rows(); ++i) {

      const Eigen::Vector3d x1(points1(i, 0), points1(i, 1), 1);
      const Eigen::Vector3d x2(points2(i, 0), points2(i, 1), 1);

      const Eigen::Vector3d Ex1 = E * x1;
      const Eigen::Vector3d Etx2 = Et * x2;
      const double x2tEx1 = x2.dot(Ex1);

      residuals[i] = x2tEx1 / sqrt(Ex1(0)*Ex1(0) + Ex1(1)*Ex1(1)
                                   + Etx2(0)*Etx2(0) + Etx2(1)*Etx2(1));

    }
  }

}


void decompose_essential_matrix(const Eigen::Matrix3d& E,
                                Eigen::Matrix3d& R1,
                                Eigen::Matrix3d& R2,
                                Eigen::Vector3d& t) {

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU
                                           | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV().transpose();

  if (U.determinant() < 0) {
    U *= -1;
  }
  if (V.determinant() < 0) {
    V *= -1;
  }

  Eigen::Matrix3d W;
  W << 0, 1, 0,
      -1, 0, 0,
       0, 0, 1;

  R1 = U * W * V;
  R2 = U * W.transpose() * V;
  t = U.col(2);

}


size_t pose_from_essential_matrix(const Eigen::Matrix3d& E,
                                  const Eigen::MatrixXd& points1,
                                  const Eigen::MatrixXd& points2,
                                  const std::vector<bool> inlier_mask,
                                  Eigen::Matrix3d& R,
                                  Eigen::Vector3d& t) {

  if (inlier_mask.size() != (size_t)points1.rows()
      || inlier_mask.size() != (size_t)points2.rows()) {
    throw std::invalid_argument("Size of inlier mask must match number "
                                "of points.");
  }

  std::vector<Eigen::Vector2d> inlier_points1, inlier_points2;
  inlier_points1.reserve(inlier_mask.size());
  inlier_points2.reserve(inlier_mask.size());
  for (size_t i=0; i<inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_points1.push_back(points1.row(i));
      inlier_points2.push_back(points2.row(i));
    }
  }

  Eigen::Matrix3d R1, R2;
  decompose_essential_matrix(E, R1, R2, t);

  // Projection matrix of first image
  const Eigen::Matrix<double, 3, 4> proj_matrix0
    = Eigen::MatrixXd::Identity(3, 4);

  // Generate all possible projection matrix combinations
  std::vector<Eigen::Matrix3d> Rs = {R1, R2, R1, R2};
  std::vector<Eigen::Vector3d> ts = {t, t, -t, -t};

  size_t best_num_inliers = 0;

  // Cheirality constraint: choose solution where the largest number of
  // 3D points lie in front of the camera for each
  // combination of {R1, R2, t, -t} of the essential matrix.

  for (size_t j=0; j<4; ++j) {

    Eigen::Matrix<double, 3, 4> proj_matrix
      = compose_proj_matrix(Rs[j], ts[j]);

    const std::vector<Eigen::Vector3d> points3D
      = triangulate_points(proj_matrix0, proj_matrix,
                           inlier_points1, inlier_points2);

    // Make sure that far away points do not break the solution
    const double max_depth = 100;

    // Calculate depth for each 3D point and check whether it lies in front
    // or behind the cameras
    size_t num_inliers = 0;
    for (size_t k=0; k<points3D.size(); ++k) {
      const double depth0 = calc_depth(proj_matrix0, points3D[k]);
      if (depth0 > 0 && depth0 < max_depth) {
        const double depthj = calc_depth(proj_matrix, points3D[k]);
        if (depthj > 0 && depthj < max_depth) {
          num_inliers += 1;
        }
      }
    }

    // New best essential_matrix, R, t
    if (num_inliers > best_num_inliers) {
      best_num_inliers = num_inliers;
      R = Rs[j];
      t = ts[j];
    }
  }

  return best_num_inliers;

}
