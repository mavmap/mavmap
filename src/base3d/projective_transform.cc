/**
 * Copyright (C) 2014
 *
 *   Johannes L. Schönberger <jsch (at) cs.unc.edu>
 *

 */

#include "projective_transform.h"


std::vector<Eigen::MatrixXd>
ProjectiveTransformEstimator::estimate(const Eigen::MatrixXd& src,
                                       const Eigen::MatrixXd& dst) {

  const size_t num_points = src.rows();

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * num_points, 9);

  A.block(0, 0, num_points, 1) = src.col(0);
  A.block(0, 1, num_points, 1) = src.col(1);
  A.block(0, 2, num_points, 1).fill(1);
  A.block(0, 6, num_points, 1) = - src.col(0).cwiseProduct(dst.col(0));
  A.block(0, 7, num_points, 1) = - src.col(1).cwiseProduct(dst.col(0));
  A.block(0, 8, num_points, 1) = dst.col(0);
  A.block(num_points, 3, num_points, 1) = src.col(0);
  A.block(num_points, 4, num_points, 1) = src.col(1);
  A.block(num_points, 5, num_points, 1).fill(1);
  A.block(num_points, 6, num_points, 1) = -src.col(0).cwiseProduct(dst.col(1));
  A.block(num_points, 7, num_points, 1) = -src.col(1).cwiseProduct(dst.col(1));
  A.block(num_points, 8, num_points, 1) = dst.col(1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

  Eigen::MatrixXd h = svd.matrixV().col(8);
  h /= - h(8);
  h.resize(3, 3);
  h(2, 2) = 1;

  std::vector<Eigen::MatrixXd> models(1);
  models[0] = h.transpose();

  return models;

}


void ProjectiveTransformEstimator::residuals(const Eigen::MatrixXd& src,
                                             const Eigen::MatrixXd& dst,
                                             const Eigen::MatrixXd& H,
                                             std::vector<double>& residuals) {

  residuals.resize(src.rows());

  #pragma omp parallel shared(src, dst, H, residuals)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<src.rows(); ++i) {

      const Eigen::Vector3d src_i(src(i, 0), src(i, 1), 1);

      const Eigen::Vector3d dst_t = H * src_i;

      const double dx = dst_t(0) / dst_t(2) - dst(i, 0);
      const double dy = dst_t(1) / dst_t(2) - dst(i, 1);

      residuals[i] = sqrt(dx * dx + dy * dy);

    }

  }

}
