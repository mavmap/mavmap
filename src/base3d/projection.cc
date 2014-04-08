/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "projection.h"


Eigen::AngleAxisd angle_axis_from_rvec(const Eigen::Vector3d& rvec) {
  double angle = rvec.norm();
  Eigen::Vector3d axis;
  // make sure, that approx. 0 angle does not scale the axis vector badly
  if (angle < std::numeric_limits<double>::epsilon()) {
    angle = 0;
    axis = Eigen::Vector3d(0, 0, 1);
  } else {
    axis = rvec.normalized();
  }
  return Eigen::AngleAxisd(angle, axis);
}


void euler_angles_from_rot_mat(const Eigen::Matrix3d& R,
                               double& rx, double& ry, double& rz) {

  rx = atan2f(R(2, 1), R(2, 2));
  ry = atan2f(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
  rz = atan2f(R(1, 0), R(0, 0));

  rx = std::isnan(rx) ? 0 : rx;
  ry = std::isnan(ry) ? 0 : ry;
  rz = std::isnan(rz) ? 0 : rz;

}


Eigen::Matrix3d rot_mat_from_euler_angles(const double& rx,
                                          const double& ry,
                                          const double& rz) {

  const Eigen::Matrix3d Rx
    = Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d Ry
    = Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d Rz
    = Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const Eigen::Matrix3d R = Rz * Ry * Rx;

  return R;

}


Eigen::Matrix<double, 3, 4>
compose_proj_matrix(const Eigen::Vector3d& rvec, const Eigen::Vector3d& tvec,
                    const Eigen::Matrix3d& calib_matrix) {

  Eigen::Matrix<double, 3, 4> Rt;
  Rt.block<3, 3>(0, 0) = angle_axis_from_rvec(rvec).toRotationMatrix();
  Rt.block<3, 1>(0, 3) = tvec;

  return calib_matrix * Rt;
}


Eigen::Matrix<double, 3, 4>
compose_proj_matrix(const Eigen::Matrix3d& R, const Eigen::Vector3d& tvec,
                    const Eigen::Matrix3d& calib_matrix) {

  Eigen::Matrix<double, 3, 4> Rt;
  Rt.block<3, 3>(0, 0) = R;
  Rt.block<3, 1>(0, 3) = tvec;

  return calib_matrix * Rt;
}


Eigen::Matrix<double, 3, 4>
invert_proj_matrix(const Eigen::Matrix<double, 3, 4>& matrix) {
  Eigen::Matrix4d matrix44 = Eigen::MatrixXd::Identity(4, 4);
  matrix44.block<3, 4>(0, 0) = matrix;
  return matrix44.inverse().eval().block<3, 4>(0, 0);
}


void extract_exterior_params(const Eigen::Vector3d& rvec,
                             const Eigen::Vector3d& tvec,
                             double& rx, double& ry, double& rz,
                             double& tx, double& ty, double& tz) {

  Eigen::Matrix<double, 3, 4> matrix
    = invert_proj_matrix(compose_proj_matrix(rvec, tvec));

  euler_angles_from_rot_mat(matrix.block<3, 3>(0, 0), rx, ry, rz);

  tx = matrix(0, 3);
  ty = matrix(1, 3);
  tz = matrix(2, 3);

}


std::vector<double>
calc_reproj_errors(const std::vector<Eigen::Vector2d>& points2D,
                   const std::vector<Eigen::Vector3d>& points3D,
                   const Eigen::Matrix<double, 3, 4>& proj_matrix) {

  std::vector<double> reproj_errors(points3D.size());

  const Eigen::Matrix3d R = proj_matrix.block<3, 3>(0, 0);
  const Eigen::Vector3d t = proj_matrix.block<3, 1>(0, 3);

  // Determine reprojection error for each correspondence
  for (size_t i=0; i<reproj_errors.size(); ++i) {
    const Eigen::Vector2d& point2D = points2D[i];
    const Eigen::Vector3d& point3D = points3D[i];
    Eigen::Vector3d point2Dp = R * point3D + t;
    point2Dp(0) /= point2Dp(2);
    point2Dp(1) /= point2Dp(2);
    const double dx = point2Dp(0) - point2D(0);
    const double dy = point2Dp(1) - point2D(1);
    reproj_errors[i] = sqrt(dx * dx + dy * dy);
  }

  return reproj_errors;
}


double calc_depth(const Eigen::Matrix<double, 3, 4>& proj_matrix,
                  const Eigen::Vector3d& point3D) {

  Eigen::Vector4d point3D1(point3D(0), point3D(1), point3D(2), 1);

  Eigen::Vector3d point2D = proj_matrix * point3D1;

  const double w = point2D(2);
  const double W = point3D1(3);

  const double mx = proj_matrix(0, 2);
  const double my = proj_matrix(1, 2);
  const double mz = proj_matrix(2, 2);

  return (w / W) * sqrt(mx*mx + my*my + mz*mz);

}
