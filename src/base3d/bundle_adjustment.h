/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE3D_BUNDLE_ADJUSTMENT_H_
#define MAVMAP_SRC_BASE3D_BUNDLE_ADJUSTMENT_H_

#include <iomanip>
#include <stdexcept>
#include <list>
#include <unordered_set>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "base3d/camera_models.h"
#include "base3d/projection.h"
#include "base3d/similarity_transform.h"
#include "fm/feature_management.h"


#define BA_POSE_FREE       0
#define BA_POSE_FIXED      1
#define BA_POSE_FIXED_X    2


struct BundleAdjustmentOptions {

  BundleAdjustmentOptions() : max_num_iterations(100),
                              function_tolerance(1e-4),
                              gradient_tolerance(1e-8),
                              update_point3D_errors(false),
                              min_track_len(2),
                              loss_scale_factor(1),
                              constrain_rotation(false),
                              constrain_rotation_weight(0),
                              refine_camera_params(false),
                              print_progress(false),
                              print_summary(true) {}

  /**
   * Maximum number of iterations.
   */
  size_t max_num_iterations;

  /**
   * Function tolerance, which controls when bundle adjustment stops:
   *
   *    (new_cost - old_cost) < function_tolerance * old_cost;
   */
  double function_tolerance;

  /**
   * Gradient tolerance, which controls when bundle adjustment stops:
   *
   *    max_i |gradient_i| < gradient_tolerance * max_i|initial_gradient_i|
   *
   * This value should typically be set to 1e-4 * `function_tolerance`.
   */
  double gradient_tolerance;

  /**
   * Whether to calculate the mean robust residuals of 3D points.
   */
  bool update_point3D_errors;

  /**
   * Minimum track length for a 3D point to be used in the bundle adjustment.
   */
  size_t min_track_len;

  /**
   * Scale factor of the robust Cauchy loss function.
   */
  double loss_scale_factor;

  /**
   * Whether to constrain the rotation with given rotation from e.g. IMUs.
   */
  bool constrain_rotation;

  /**
   * Weight for rotation constraint residuals. Each camera adds one residual.
   */
  double constrain_rotation_weight;

  /**
   * Whether to refine / calibrate the camera parameters by setting the
   * parameters as variable.
   */
  bool refine_camera_params;

  /**
   * Whether to print the results for each iteration of the bundle adjustment.
   */
  bool print_progress;

  /**
   * Whether to print a final summary of the bundle adjustment.
   */
  bool print_summary;

};


template <typename CameraModel>
class BACostFunction {

public:

  BACostFunction(const Eigen::Vector2d& point2D) : point2D_(point2D) {};

  static ceres::CostFunction* create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction
      <BACostFunction<CameraModel>,
       2, 3, 1, 1, 1, 3, CameraModel::num_params>
        (new BACostFunction(point2D)));
  }

  template <typename T> bool operator()(const T* const rvec,
                                        const T* const tx,
                                        const T* const ty,
                                        const T* const tz,
                                        const T* const point3D,
                                        const T* const camera_params,
                                        T* residuals) const {

    T point3D_cam[3];

    // Rotate
    ceres::AngleAxisRotatePoint(rvec, point3D, point3D_cam);

    // Translate
    point3D_cam[0] += tx[0];
    point3D_cam[1] += ty[0];
    point3D_cam[2] += tz[0];

    T u, v;
    CameraModel::world2image(point3D_cam[0], point3D_cam[1], point3D_cam[2],
                             u, v, camera_params);

    // Re-projection error
    residuals[0] = u - T(point2D_(0));
    residuals[1] = v - T(point2D_(1));

    return true;

  }

private:

  const Eigen::Vector2d& point2D_;

};


class BANoDistortionCostFunction {

public:

  BANoDistortionCostFunction(const Eigen::Vector3d& point2D_normalized);

  static ceres::CostFunction*
    create(const Eigen::Vector3d& point2D_normalized);

  template <typename T> bool operator()(const T* const rvec,
                                        const T* const tx,
                                        const T* const ty,
                                        const T* const tz,
                                        const T* const point3D,
                                        T* residuals) const;

private:

  const Eigen::Vector3d point2D_normalized_;

};


class BARotationConstraintCostFunction {

public:

  BARotationConstraintCostFunction(const double weight,
                                   const Eigen::Vector3d& rvec0);

  static ceres::CostFunction* create(const double weight,
                                     const Eigen::Vector3d& rvec0);

  template <typename T> bool operator()(const T* const rvec,
                                        T* residuals) const;

private:

  const double weight_;
  double rotmat0_[9];

};


double pose_refinement(Eigen::Vector3d& rvec,
                       Eigen::Vector3d& tvec,
                       std::vector<double>& camera_params,
                       const std::vector<Eigen::Vector2d>& points2D,
                       std::vector<Eigen::Vector3d>& points3D,
                       const std::vector<bool>& inlier_mask,
                       const BundleAdjustmentOptions& options);


double bundle_adjustment(
         FeatureManager& feature_manager,
         const std::vector<size_t>& free_image_ids,
         const std::vector<size_t>& fixed_image_ids,
         const std::vector<size_t>& fixed_x_image_ids,
         const BundleAdjustmentOptions& options,
         std::unordered_map<size_t, double>& point3D_errors,
         const std::unordered_map<size_t, Eigen::Vector3d>&
           rotation_constraints=std::unordered_map<size_t, Eigen::Vector3d>(),
         const std::set<size_t>& gcp_ids=std::set<size_t>());


#endif // MAVMAP_SRC_BASE3D_BUNDLE_ADJUSTMENT_H_
