/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE3D_PROJECTION_H_
#define MAVMAP_SRC_BASE3D_PROJECTION_H_

#include <vector>
#include <limits>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>


/**
 * Create Eigen::AngleAxis from rotation vector.
 *
 * @param rvec           3x1 rotation vector.
 *
 * @return               Eigen::AngleAxis object.
 */
Eigen::AngleAxisd angle_axis_from_rvec(const Eigen::Vector3d& rvec);


/**
 * Extract Euler angles from 3D rotation matrix.
 *
 * The convention `R = Rx * Ry * Rz` is used.
 *
 * @param R              3x3 rotation matrix.
 * @param rx, ry, rz     Euler angles in radians.
 */
void euler_angles_from_rot_mat(const Eigen::Matrix3d& R,
                               double& rx, double& ry, double& rz);


/**
 * Compose 3D rotation matrix from Euler angles.
 *
 * The convention `R = Rx * Ry * Rz` is used.
 *
 * @param rx, ry, rz     Euler angles in radians.
 *
 * @return               3x3 rotation matrix.
 */
Eigen::Matrix3d rot_mat_from_euler_angles(const double& rx,
                                          const double& ry,
                                          const double& rz);


/**
 * Compose projection matrix from rotation and translation vector (with
 * optionally integrated calibration).
 *
 * The projection matrix transforms 3D world to image points.
 *
 * @param rvec           3x1 rotation vector.
 * @param tvec           3x1 translation vector.
 * @param calib_matrix   3x3 calibration matrix of intrinsic parameters.
 *
 * @return               3x4 projection matrix.
 */
Eigen::Matrix<double, 3, 4>
compose_proj_matrix(const Eigen::Vector3d& rvec,
                    const Eigen::Vector3d& tvec,
                    const Eigen::Matrix3d& calib_matrix
                      =Eigen::MatrixXd::Identity(3, 3));


/**
 * Compose projection matrix from rotation matrix and translation vector (with
 * optionally integrated calibration).
 *
 * The projection matrix transforms 3D world to image points.
 *
 * @param R              3x3 rotation matrix.
 * @param tvec           3x1 translation vector.
 * @param calib_matrix   3x3 calibration matrix of intrinsic parameters.
 *
 * @return               3x4 projection matrix.
 */
Eigen::Matrix<double, 3, 4>
compose_proj_matrix(const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& tvec,
                    const Eigen::Matrix3d& calib_matrix
                      =Eigen::MatrixXd::Identity(3, 3));


/**
 * Invert projection matrix.
 *
 * The absolute pose of a camera can be extracted from the inverse projection
 * matrix.
 *
 * @param rvec           3x4 projection matrix.
 *
 * @return               3x4 inverse projection matrix.
 */
Eigen::Matrix<double, 3, 4>
invert_proj_matrix(const Eigen::Matrix<double, 3, 4>& matrix);


/**
 * Extract the exterior parameters from the projection matrix.
 *
 * This function first composes the projection matrix, then inverts it and
 * finally extracts the exterior parameters.
 *
 * @param rvec           3x1 rotation vector.
 * @param tvec           3x1 translation vector.
 * @param rx, ry, rz     Euler angles in radians with the convention
 *                       R = Rz * Ry * Rx.
 * @param tx, ty, tz     Translation components.
 */
void extract_exterior_params(const Eigen::Vector3d& rvec,
                             const Eigen::Vector3d& tvec,
                             double& rx, double& ry, double& rz,
                             double& tx, double& ty, double& tz);


/**
 * Calculate the reprojection error given a set of 2D-3D point
 * correspondences and a projection matrix.
 *
 * @param points2D     Vector of 2D image points as 2x1 vector.
 * @param points3D     Vector of 3D world points as 3x1 vector.
 * @param proj_matrix  3x4 projection matrix.
 *
 * @return             Output vector of reprojection errors for
 *                     each 2D-3D point pair.
 */
std::vector<double>
calc_reproj_errors(const std::vector<Eigen::Vector2d>& points2D,
                   const std::vector<Eigen::Vector3d>& points3D,
                   const Eigen::Matrix<double, 3, 4>& proj_matrix);


/**
 * Calculate depth of 3D point with respect to camera.
 *
 * The depth is defined as the Euclidean distance of a 3D point from the
 * camera and is positive if the 3D point is in front and negative if
 * behind the camera.
 *
 * @param proj_matrix  3x4 projection matrix.
 * @param point3D      3D point as 3x1 vector.
 *
 * @return             Depth of 3D point.
 */
double calc_depth(const Eigen::Matrix<double, 3, 4>& proj_matrix,
                  const Eigen::Vector3d& point3D);


#endif // MAVMAP_SRC_BASE3D_PROJECTION_H_
