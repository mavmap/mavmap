/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "bundle_adjustment.h"


BANoDistortionCostFunction::BANoDistortionCostFunction(
      const Eigen::Vector3d& point2D_normalized)
    : point2D_normalized_(point2D_normalized) {}


ceres::CostFunction*
BANoDistortionCostFunction::create(const Eigen::Vector3d& point2D_normalized) {
  return (new ceres::AutoDiffCostFunction
    <BANoDistortionCostFunction, 2, 3, 1, 1, 1, 3>
      (new BANoDistortionCostFunction(point2D_normalized)));
}


template <typename T> bool
BANoDistortionCostFunction::operator()(const T* const rvec,
                                       const T* const tx,
                                       const T* const ty,
                                       const T* const tz,
                                       const T* const point3D,
                                       T* residuals) const {
  T point3D_cam[3];

  // Rotate
  ceres::AngleAxisRotatePoint(rvec, point3D, point3D_cam);

  // Translate
  point3D_cam[0] += tx[0];
  point3D_cam[1] += ty[0];
  point3D_cam[2] += tz[0];

  // Normalize
  point3D_cam[0] /= point3D_cam[2];
  point3D_cam[1] /= point3D_cam[2];

  // No scaling, as point2D_normalized is not in image but in normalized camera
  // coordinate system (faster computation)

  // Re-projection error
  residuals[0] = point3D_cam[0] - T(point2D_normalized_(0));
  residuals[1] = point3D_cam[1] - T(point2D_normalized_(1));

  return true;
}


BARotationConstraintCostFunction::BARotationConstraintCostFunction(
    const double weight, const Eigen::Vector3d& rvec0) : weight_(weight) {
  ceres::AngleAxisToRotationMatrix(rvec0.data(), rotmat0_);
}


ceres::CostFunction*
BARotationConstraintCostFunction::create(const double weight,
                                         const Eigen::Vector3d& rvec0) {
  return (new ceres::AutoDiffCostFunction
    <BARotationConstraintCostFunction, 1, 3>
      (new BARotationConstraintCostFunction(weight, rvec0)));
}


template <typename T> bool
BARotationConstraintCostFunction::operator()(const T* const rvec,
                                             T* residuals) const {

  T rotmat[9];

  ceres::AngleAxisToRotationMatrix(rvec, rotmat);

  // norm(rotmat.T - rotmat0, 'fro')

  // Using the inverse (transpose) as rotmat0 is the extrinsic rotation of
  // the camera

  residuals[0] = T(0);

  T diff;

  diff = rotmat[0] - T(rotmat0_[0]);
  residuals[0] += diff * diff;
  diff = rotmat[3] - T(rotmat0_[1]);
  residuals[0] += diff * diff;
  diff = rotmat[6] - T(rotmat0_[2]);
  residuals[0] += diff * diff;
  diff = rotmat[1] - T(rotmat0_[3]);
  residuals[0] += diff * diff;
  diff = rotmat[4] - T(rotmat0_[4]);
  residuals[0] += diff * diff;
  diff = rotmat[7] - T(rotmat0_[5]);
  residuals[0] += diff * diff;
  diff = rotmat[2] - T(rotmat0_[6]);
  residuals[0] += diff * diff;
  diff = rotmat[6] - T(rotmat0_[7]);
  residuals[0] += diff * diff;
  diff = rotmat[8] - T(rotmat0_[8]);
  residuals[0] += diff * diff;

  residuals[0] = T(weight_) * sqrt(residuals[0]);

  return true;
}


void _print_report(const ceres::Solver::Summary& summary) {
  std::cout << std::right << std::setw(18) << "Residuals : ";
  std::cout << std::left
            << summary.num_residuals_reduced
            << std::endl;
  std::cout << std::right << std::setw(18) << "Parameters : ";
  std::cout << std::left
            << summary.num_parameters_reduced
            << std::endl;
  std::cout << std::right << std::setw(18) << "Iterations : ";
  std::cout << std::left
            << summary.num_successful_steps + summary.num_unsuccessful_steps
            << std::endl;
  std::cout << std::right << std::setw(18) << "Initial cost : ";
  std::cout << std::right << std::setprecision(6)
            << sqrt(summary.initial_cost / summary.num_residuals) << " [px]"
            << std::endl;
  std::cout << std::right << std::setw(18) << "Final cost : ";
  std::cout << std::right << std::setprecision(6)
            << sqrt(summary.final_cost / summary.num_residuals) << " [px]"
            << std::endl;
  std::cout << std::endl;
}


double pose_refinement(Eigen::Vector3d& rvec,
                       Eigen::Vector3d& tvec,
                       std::vector<double>& camera_params,
                       const std::vector<Eigen::Vector2d>& points2D,
                       std::vector<Eigen::Vector3d>& points3D,
                       const std::vector<bool>& inlier_mask,
                       const BundleAdjustmentOptions& options) {

  // Define robust lost function
  ceres::LossFunction *loss_function
    = new ceres::CauchyLoss(options.loss_scale_factor);

  double* rvec_data = rvec.data();
  double* tx = tvec.data();
  double* ty = tx + 1;
  double* tz = tx + 2;

  double* camera_params_data = camera_params.data();

  ceres::Problem problem;

  for (size_t i=0; i<points2D.size(); ++i) {

    if (!inlier_mask[i]) {
      continue;
    }

    const Eigen::Vector2d& point2D = points2D[i];
    Eigen::Vector3d& point3D = points3D[i];
    double* point3D_data = point3D.data();

    ceres::CostFunction* cost_function = NULL;

    switch ((int)camera_params.back()) {
      case PinholeCameraModel::code:
        cost_function = BACostFunction<PinholeCameraModel>::create(point2D);
        break;
      case OpenCVCameraModel::code:
        cost_function = BACostFunction<OpenCVCameraModel>::create(point2D);
        break;
      case CataCameraModel::code:
        cost_function = BACostFunction<CataCameraModel>::create(point2D);
        break;
    }

    problem.AddResidualBlock(cost_function, loss_function,
                             rvec_data, tx, ty, tz, point3D_data,
                             camera_params_data);
    problem.SetParameterBlockConstant(point3D_data);

  }

  // Always set camera parameters as constant, otherwise the parameters
  // will be changed "arbitrarily"
  problem.SetParameterBlockConstant(camera_params_data);

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.function_tolerance = options.function_tolerance;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_consecutive_invalid_steps = 10;
  solver_options.max_consecutive_nonmonotonic_steps = 10;
  solver_options.minimizer_progress_to_stdout = options.print_progress;

#ifdef OPENMP_FOUND
  solver_options.num_threads = omp_get_max_threads();
  solver_options.num_linear_solver_threads = omp_get_max_threads();
#endif

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (options.print_progress) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    std::cout << "Pose Refinement Report" << std::endl;
    std::cout << "----------------------" << std::endl;
    _print_report(summary);
  }

  const double final_cost = sqrt(summary.final_cost / summary.num_residuals);

  return final_cost;
}


void _bundle_adjustment_extract_data(
       FeatureManager& feature_manager,
       const std::vector<size_t>& image_ids,
       std::unordered_map<size_t, Eigen::Vector3d*>& rvecs,
       std::unordered_map<size_t, Eigen::Vector3d*>& tvecs,
       std::unordered_map<size_t, std::vector<double>*>& camera_params,
       std::unordered_map<size_t, std::vector<Eigen::Vector2d*> >& points2D,
       std::unordered_map<size_t, std::vector<Eigen::Vector3d*> >& points3D,
       std::unordered_map<size_t, std::vector<size_t> >& point3D_ids,
       std::unordered_map<size_t, size_t>& point3D_num_points2D) {

  for (auto it=image_ids.begin(); it!=image_ids.end(); ++it) {

    const size_t image_id = *it;

    rvecs[image_id] = &feature_manager.rvecs[image_id];
    tvecs[image_id] = &feature_manager.tvecs[image_id];

    const size_t camera_id = feature_manager.image_to_camera[image_id];
    camera_params[image_id] = &feature_manager.camera_params[camera_id];

    std::vector<Eigen::Vector2d*> curr_points2D;
    std::vector<Eigen::Vector3d*> curr_points3D;
    std::vector<size_t> curr_point3D_ids;

    // All image points of current image
    const std::vector<size_t>& point2D_ids
      = feature_manager.image_to_points2D[image_id];

    for (size_t i=0; i<point2D_ids.size(); ++i) {

      const size_t point2D_id = point2D_ids[i];

      // Does 3D point have a corresponding 3D point?
      if (feature_manager.point2D_to_point3D.count(point2D_id) == 0) {
        continue;
      }

      const size_t point3D_id
        = feature_manager.point2D_to_point3D[point2D_id];

      curr_points2D.push_back(&feature_manager.points2D[point2D_id]);
      curr_points3D.push_back(&feature_manager.points3D[point3D_id]);
      curr_point3D_ids.push_back(point3D_id);

      if (point3D_num_points2D.count(point3D_id)) {
        point3D_num_points2D[point3D_id] += 1;
      } else {
        point3D_num_points2D[point3D_id] = 1;
      }

    }

    points2D[image_id] = curr_points2D;
    points3D[image_id] = curr_points3D;
    point3D_ids[image_id] = curr_point3D_ids;

  }
}


void _bundle_adjustment_fill_problem(
       ceres::Problem& problem, const int pose_state,
       const std::vector<size_t>& image_ids,
       std::unordered_map<size_t, Eigen::Vector3d*>& rvecs,
       std::unordered_map<size_t, Eigen::Vector3d*>& tvecs,
       std::unordered_map<size_t, std::vector<double>*>& camera_params,
       std::unordered_map<size_t, std::vector<Eigen::Vector2d*> >& points2D,
       std::unordered_map<size_t, std::vector<Eigen::Vector3d*> >& points3D,
       std::unordered_map<size_t, std::vector<size_t> >& point3D_ids,
       std::unordered_map<size_t, size_t>& point3D_num_points2D,
       std::vector<size_t>& ordered_point3D_ids,
       const BundleAdjustmentOptions& options,
       ceres::LossFunctionWrapper* loss_function) {

  for (auto it=image_ids.begin(); it!=image_ids.end(); ++it) {

    size_t num_residuals = 0;

    const size_t image_id = *it;

    Eigen::Vector3d* rvec = rvecs[image_id];
    Eigen::Vector3d* tvec = tvecs[image_id];
    double* rvec_data = rvec->data();
    double* tx = tvec->data();
    double* ty = tx + 1;
    double* tz = tx + 2;

    std::vector<double>& curr_camera_params = *camera_params[image_id];
    double* camera_params_data = curr_camera_params.data();

    const std::vector<Eigen::Vector2d*>& curr_points2D = points2D[image_id];
    const std::vector<Eigen::Vector3d*>& curr_points3D = points3D[image_id];
    const std::vector<size_t>& curr_point3D_ids = point3D_ids[image_id];

    for (size_t i=0; i<curr_points2D.size(); ++i) {

      const size_t point3D_id = curr_point3D_ids[i];
      const size_t num_points2D = point3D_num_points2D[point3D_id];

      // Make sure only points with >= 2 observations in separate images
      // are added to the problem
      if (num_points2D < options.min_track_len) {
        continue;
      }

      Eigen::Vector2d* point2D = curr_points2D[i];
      Eigen::Vector3d* point3D = curr_points3D[i];

      ceres::CostFunction* cost_function = NULL;

      switch ((int)curr_camera_params.back()) {
        case PinholeCameraModel::code:
          cost_function = BACostFunction<PinholeCameraModel>::create(*point2D);
          break;
        case OpenCVCameraModel::code:
          cost_function = BACostFunction<OpenCVCameraModel>::create(*point2D);
          break;
        case CataCameraModel::code:
          cost_function = BACostFunction<CataCameraModel>::create(*point2D);
          break;
      }

      problem.AddResidualBlock(cost_function, loss_function,
                               rvec_data, tx, ty, tz, point3D->data(),
                               camera_params_data);

      ordered_point3D_ids.push_back(point3D_id);

      num_residuals += 1;

    }

    if (num_residuals > 1) {
      // Parameter blocks are variable by default, so set them as constant for
      // fixed and partially fixed poses
      switch (pose_state) {
        case BA_POSE_FIXED:
          problem.SetParameterBlockConstant(rvec_data);
          problem.SetParameterBlockConstant(tx);
          problem.SetParameterBlockConstant(ty);
          problem.SetParameterBlockConstant(tz);
          if (!options.refine_camera_params) { //FF
            problem.SetParameterBlockConstant(camera_params_data);
          }
          break;
        case BA_POSE_FIXED_X:
          problem.SetParameterBlockConstant(tx);
          if (!options.refine_camera_params) {
            problem.SetParameterBlockConstant(camera_params_data);
          }
          break;
        case BA_POSE_FREE:
          if (!options.refine_camera_params) {
            problem.SetParameterBlockConstant(camera_params_data);
          }
      }
    }
  }
}


void _bundle_adjustment_add_pose_constraints(
       ceres::Problem& problem,
       FeatureManager& feature_manager,
       const std::vector<size_t>& free_image_ids,
       const std::vector<size_t>& fixed_image_ids,
       const double constrain_rotation_weight,
       const std::unordered_map<size_t, Eigen::Vector3d>& rotation_constraints,
       std::unordered_map<size_t, Eigen::Vector3d*>& rvecs) {

  // Determine rotation between constraints and SfM rotations from first
  // fixed image
  const size_t image_id = fixed_image_ids[0];
  // const Eigen::Vector3d rvec_FM = ;
  // const Eigen::AngleAxisd rot_FM(rvec_FM);
  const Eigen::Matrix3d R_FM
    = angle_axis_from_rvec(*rvecs[image_id]).toRotationMatrix();
  const Eigen::Matrix3d R_C
    = angle_axis_from_rvec(
        rotation_constraints.at(image_id)).toRotationMatrix();

  // "Difference" between rotations, i.e. rotation transformation from SfM to
  // constrained coordinate system
  Eigen::Matrix<double, 3, 4> matrix_FM_C
    = Eigen::Matrix<double, 3, 4>::Zero();
  matrix_FM_C.block<3, 3>(0, 0) = R_FM.transpose() * R_C;
  SimilarityTransform3D rot_FM_C(matrix_FM_C);

  // Rotate all camera poses and 3D points in feature manager
  for (auto it=feature_manager.rvecs.begin();
       it!=feature_manager.rvecs.end(); ++it) {
    rot_FM_C.transform_pose(it->second, feature_manager.tvecs[it->first]);
  }
  for (auto it=feature_manager.points3D.begin();
       it!=feature_manager.points3D.end(); ++it) {
    rot_FM_C.transform_point(it->second);
  }

  // Finally, add rotation constraints
  for (auto it=free_image_ids.begin(); it!=free_image_ids.end(); ++it) {

    const size_t image_id = *it;

    // rotation to be estimated
    Eigen::Vector3d* rvec = rvecs[image_id];

    // rotation constraint, e.g. from IMU sensor
    const Eigen::Vector3d& rvec0 = rotation_constraints.at(image_id);

    ceres::CostFunction* cost_function
      = BARotationConstraintCostFunction::create(constrain_rotation_weight,
                                                 rvec0);

    problem.AddResidualBlock(cost_function, NULL, rvec->data());

  }

}


double bundle_adjustment(
         FeatureManager& feature_manager,
         const std::vector<size_t>& free_image_ids,
         const std::vector<size_t>& fixed_image_ids,
         const std::vector<size_t>& fixed_x_image_ids,
         const BundleAdjustmentOptions& options,
         std::unordered_map<size_t, double>& point3D_errors,
         const std::unordered_map<size_t, Eigen::Vector3d>& rotation_constraints,
         const std::set<size_t>& gcp_ids) {

  const size_t num_fixed_params = fixed_image_ids.size() * 6
                                  + fixed_x_image_ids.size()
                                  + gcp_ids.size() * 3;
  if (num_fixed_params < 7) {
    throw std::invalid_argument("At least 7 parameters should be set as fixed "
                                "to avoid datum defects resulting in a "
                                "singular Jacobian.");
  }

  if (options.min_track_len < 2) {
    throw std::invalid_argument("Minimum track length must be >= 2 in order "
                                "build valid bundle adjustment problem.");
  }

  ceres::Problem problem;
  std::vector<size_t> ordered_point3D_ids;
  std::unordered_map<size_t, size_t> point3D_num_points2D;

  ceres::LossFunctionWrapper* loss_function
    = new ceres::LossFunctionWrapper(new ceres::CauchyLoss(options.loss_scale_factor), ceres::TAKE_OWNERSHIP);

  // limited problem setup scope, so intermediate storage containers are
  // deallocated automatically to reduce memory footprint
  {

    std::unordered_map<size_t, Eigen::Vector3d*> rvecs, tvecs;
    std::unordered_map<size_t, std::vector<double>*> camera_params;
    std::unordered_map<size_t, std::vector<Eigen::Vector2d*> > points2D;
    std::unordered_map<size_t, std::vector<Eigen::Vector3d*> > points3D;
    std::unordered_map<size_t, std::vector<size_t> > point3D_ids;

    // Extract data for all poses

    // Free images
    _bundle_adjustment_extract_data(feature_manager, free_image_ids,
                                    rvecs, tvecs, camera_params,
                                    points2D, points3D, point3D_ids,
                                    point3D_num_points2D);
    // Fixed x images
    _bundle_adjustment_extract_data(feature_manager, fixed_x_image_ids,
                                    rvecs, tvecs, camera_params,
                                    points2D, points3D, point3D_ids,
                                    point3D_num_points2D);
    // Fixed images
    _bundle_adjustment_extract_data(feature_manager, fixed_image_ids,
                                    rvecs, tvecs, camera_params,
                                    points2D, points3D, point3D_ids,
                                    point3D_num_points2D);

    // Fill problem with data

    // Free images
    _bundle_adjustment_fill_problem(problem, BA_POSE_FREE,
                                    free_image_ids,
                                    rvecs, tvecs, camera_params,
                                    points2D, points3D, point3D_ids,
                                    point3D_num_points2D,
                                    ordered_point3D_ids, options,
                                    loss_function);
    // Fixed images
    _bundle_adjustment_fill_problem(problem, BA_POSE_FIXED,
                                    fixed_image_ids,
                                    rvecs, tvecs, camera_params,
                                    points2D, points3D, point3D_ids,
                                    point3D_num_points2D,
                                    ordered_point3D_ids, options,
                                    loss_function);
    // Fixed x images
    _bundle_adjustment_fill_problem(problem, BA_POSE_FIXED_X,
                                    fixed_x_image_ids,
                                    rvecs, tvecs, camera_params,
                                    points2D, points3D, point3D_ids,
                                    point3D_num_points2D,
                                    ordered_point3D_ids, options,
                                    loss_function);

    if (options.constrain_rotation) {
      // Pose constraints only for free images
      _bundle_adjustment_add_pose_constraints(
        problem, feature_manager, free_image_ids, fixed_image_ids,
        options.constrain_rotation_weight, rotation_constraints, rvecs);
    }

    // Set GCP coordinates as fixed
    // this is slow, but let's assume the number of GCPs is small
    // and the GCP feature is not used during incremental reconstruction
    for (const auto& gcp_id : gcp_ids) {
      if (std::count(ordered_point3D_ids.begin(), ordered_point3D_ids.end(), gcp_id)) {
        problem.SetParameterBlockConstant(feature_manager.points3D[gcp_id].data());
      }
    }

  } // end limited problem setup scope

  // Solve bundle adjustment problem
  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.function_tolerance = options.function_tolerance;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_consecutive_invalid_steps = 10;
  solver_options.max_consecutive_nonmonotonic_steps = 10;
  solver_options.minimizer_progress_to_stdout = options.print_progress;

#ifdef OPENMP_FOUND
  solver_options.num_threads = omp_get_max_threads();
  solver_options.num_linear_solver_threads = omp_get_max_threads();
#endif

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (ordered_point3D_ids.size() == 0) {
    std::cout << "No observations in bundle adjustment. Consider relaxing the constraints." << std::endl;
  }

  if (options.update_point3D_errors) {

    // Clean object point errors, so their mean can be recalculated below
    for (size_t i=0; i<ordered_point3D_ids.size(); ++i) {
      const size_t point3D_id = ordered_point3D_ids[i];
      point3D_errors[point3D_id] = 0;
    }

    loss_function->Reset(new ceres::TrivialLoss(), ceres::TAKE_OWNERSHIP);

    // Determine mean re-projection residuals (=error) for each 3D point
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(),
                     NULL, &residuals, NULL, NULL);

    for (size_t i=0; i<(size_t)problem.NumResidualBlocks(); ++i) {
      const size_t point3D_id = ordered_point3D_ids[i];
      const size_t num_points2D = point3D_num_points2D[point3D_id];
      const double rx = residuals[2*i];
      const double ry = residuals[2*i+1];
      point3D_errors[point3D_id] += sqrt(rx*rx + ry*ry) / num_points2D;
    }

  }

  if (options.print_progress) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    std::cout << "Bundle Adjustment Report" << std::endl;
    std::cout << "------------------------" << std::endl;
    _print_report(summary);
  }

  const double final_cost = sqrt(summary.final_cost / summary.num_residuals);

  return final_cost;
}
