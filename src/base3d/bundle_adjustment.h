/**
 * Copyright (C) 2013
 *
 *   Johannes Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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


template <typename CameraModel>
class BACostFunction {

public:

  BACostFunction(const Eigen::Vector2d& point2D) : point2D_(point2D) {};

  static ceres::CostFunction* create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction
      <BACostFunction<CameraModel>,
       2, 3, 1, 1, 1, 3, CameraModel::num_params>
        (new BACostFunction(point2D)));
  };

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

  };

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
                       const double loss_scale_factor,
                       const bool print_summary=true,
                       const bool print_progress=false);


double bundle_adjustment(
         FeatureManager& feature_manager,
         const std::vector<size_t>& free_image_ids,
         const std::vector<size_t>& fixed_image_ids,
         const std::vector<size_t>& fixed_x_image_ids,
         std::unordered_map<size_t, double>& point3D_errors,
         const size_t min_track_len=2,
         const double loss_scale_factor=1,
         const bool update_point3D_errors=false,
         const bool constrain_poses=false,
         const double constrain_rot_weight=100,
         const std::unordered_map<size_t, Eigen::Vector3d>& constrain_rvecs
          =std::unordered_map<size_t, Eigen::Vector3d>(),
         const bool print_summary=true,
         const bool print_progress=true);


#endif // MAVMAP_SRC_BASE3D_BUNDLE_ADJUSTMENT_H_
