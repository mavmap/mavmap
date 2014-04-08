/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "similarity_transform.h"


std::vector<Eigen::MatrixXd>
SimilarityTransform3DEstimator::estimate(const Eigen::MatrixXd& src,
                                         const Eigen::MatrixXd& dst) {
  std::vector<Eigen::MatrixXd> models(1);
  models[0] = Eigen::umeyama(src.transpose(), dst.transpose(), true);
  return models;
}

void
SimilarityTransform3DEstimator::residuals(const Eigen::MatrixXd& src,
                                          const Eigen::MatrixXd& dst,
                                          const Eigen::MatrixXd& matrix,
                                          std::vector<double>& residuals) {
  residuals.resize(src.rows());

  Eigen::Transform<double, 3, Eigen::Affine> transform;
  transform.matrix() = matrix;

  #pragma omp parallel shared(src, dst, matrix, residuals)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<src.rows(); ++i) {
      const Eigen::Vector3d& src_i = src.row(i);
      const Eigen::Vector3d& dst_i = dst.row(i);
      Eigen::Vector3d dst_predict = transform * src_i;
      residuals[i] = (dst_i - dst_predict).norm();
    }
  }

}


SimilarityTransform3D::SimilarityTransform3D() {
  SimilarityTransform3D(1, 0, 0, 0, 0, 0, 0);
}


SimilarityTransform3D::SimilarityTransform3D(
      const Eigen::Matrix<double, 3, 4> matrix) {
  transform_.matrix().block<3, 4>(0, 0) = matrix;
}


SimilarityTransform3D::SimilarityTransform3D(
      const Eigen::Transform<double, 3, Eigen::Affine>& transform)
    : transform_(transform) {}


SimilarityTransform3D::SimilarityTransform3D(
      const double scale, const double rx, const double ry, const double rz,
      const double tx, const double ty, const double tz) {

  const Eigen::Vector3d rvec(rx, ry, rz);
  const Eigen::Vector3d tvec(tx, ty, tz);

  Eigen::Matrix4d matrix = Eigen::MatrixXd::Identity(4, 4);
  matrix.block<3, 4>(0, 0) = compose_proj_matrix(rvec, tvec);
  matrix.block<3, 3>(0, 0) *= scale;

  transform_.matrix() = matrix;

}


void SimilarityTransform3D::estimate(const Eigen::MatrixXd& src,
                                     const Eigen::MatrixXd& dst) {

  transform_.matrix() = SimilarityTransform3DEstimator().estimate(src, dst)[0];

}


SimilarityTransform3D SimilarityTransform3D::inverse() {
  return SimilarityTransform3D(transform_.inverse());
}


void SimilarityTransform3D::transform_point(Eigen::Vector3d& xyz) {
  xyz = transform_ * xyz;
}


void SimilarityTransform3D::transform_pose(Eigen::Vector3d& rvec,
                                           Eigen::Vector3d& tvec) {

  // Projection matrix P1 projects 3D object points to image plane and thus to
  // 2D image points in the source coordinate system:
  //    x' = P1 * X1
  // 3D object points can be transformed to the destination system by applying
  // the similarity transformation S:
  //    X2 = S * X1
  // To obtain the projection matrix P2 that transforms the object point in the
  // destination system to the 2D image points, which do not change:
  //    x' = P2 * X2 = P2 * S * X1 = P1 * S^-1 * S * X1 = P1 * I * X1
  // and thus:
  //    P2' = P1 * S^-1
  // Finally, undo the inverse scaling of the rotation matrix:
  //    P2 = s * P2'

  Eigen::Matrix4d src_matrix = Eigen::MatrixXd::Identity(4, 4);
  src_matrix.block<3, 4>(0, 0) = compose_proj_matrix(rvec, tvec);
  Eigen::Matrix4d dst_matrix = src_matrix.matrix()
                               * transform_.inverse().matrix();
  dst_matrix *= scale();

  Eigen::AngleAxisd rot(dst_matrix.block<3, 3>(0, 0));
  rvec = rot.angle() * rot.axis();
  tvec = dst_matrix.block<3, 1>(0, 3);

}


Eigen::Matrix4d SimilarityTransform3D::matrix() {
  return transform_.matrix();
}


double SimilarityTransform3D::scale() {
  return matrix().block<1, 3>(0, 0).norm();
}


Eigen::Vector3d SimilarityTransform3D::rvec() {
  Eigen::AngleAxisd rot(matrix().block<3, 3>(0, 0) / scale());
  return rot.angle() * rot.axis();
}


Eigen::Vector3d SimilarityTransform3D::tvec() {
  return matrix().block<3, 1>(0, 3);
}
