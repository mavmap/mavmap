/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "sequential_mapper.h"


SequentialMapper::SequentialMapper(const std::vector<Image>& image_data_,
                                   const std::string& cache_path,
                                   const std::string& voc_tree_path,
                                   const SURFOptions& surf_options,
                                   const bool loop_detection,
                                   const bool debug_,
                                   const std::string& debug_path_)
    : debug(debug_),
      debug_path(debug_path_),
      image_data(image_data_),
      loop_detection_(loop_detection) {

  if (debug_path.length() > 0) {
    debug_path = ensure_trailing_slash(debug_path);
  }

  if (loop_detection_) {
    loop_detector_ = LoopDetector(voc_tree_path, image_data_.size(),
                                  MAX_NUM_VISUAL_WORDS);
    loop_detector_.init();
  }

  feature_cache_ = FeatureCache(cache_path, image_data, surf_options);

  num_proc_images_ = 0;
  first_image_idx_ = -1;
  second_image_idx_ = -1;
  prev_image_idx_ = -1;
  prev_prev_image_idx_ = -1;
  max_image_idx_ = 0;

}


bool SequentialMapper::process_initial(const size_t first_image_idx,
                                       const size_t second_image_idx,
                                       const SequentialMapperOptions&
                                         options) {

  if (num_proc_images_ > 0) {
    throw std::invalid_argument("Initial processing can only be called once.");
  }


  if (first_image_idx == second_image_idx) {
    throw std::invalid_argument("Initial image pair must not be the "
                                "same images.");
  }

  if (debug) {
    cv::Mat image_matches;
    cv::Mat prev_image = image_data[first_image_idx].read();
    cv::Mat curr_image = image_data[second_image_idx].read();
    cv::drawMatches(prev_image, std::vector<cv::KeyPoint>(0),
                    curr_image, std::vector<cv::KeyPoint>(0),
                    std::vector<cv::DMatch>(0), image_matches);
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(first_image_idx) + "-"
        + boost::lexical_cast<std::string>(second_image_idx)
        + "-images.jpg";
    cv::imwrite(debug_path_full, image_matches);
  }


  query_and_prepare_features_(first_image_idx);
  query_and_prepare_features_(second_image_idx);
  match_features_(options);


  if (debug) {
    std::cout << "DEBUG: Number of matches: "
              << matches_.size() << std::endl;
    cv::Mat image_matches;
    cv::Mat prev_image = image_data[first_image_idx].read();
    cv::Mat curr_image = image_data[second_image_idx].read();
    cv::drawMatches(prev_image, prev_keypoints_,
                    curr_image, curr_keypoints_,
                    matches_, image_matches);
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(first_image_idx) + "-"
        + boost::lexical_cast<std::string>(second_image_idx)
        + "-matches-all.jpg";
    cv::imwrite(debug_path_full, image_matches);
  }


  if (options.min_disparity > 0) {
    const size_t disparity = median_feature_disparity(prev_keypoints_, curr_keypoints_, matches_);
    float frame_diagonal;
    feature_cache_.query_dimensions(first_image_idx, NULL, NULL, NULL, &frame_diagonal);
    const double min_disparity = rel2abs_threshold(options.min_disparity, frame_diagonal);

    if (debug) {
      std::cout << "DEBUG: min-dispariy: " << disparity << " < " << min_disparity << std::endl;
    }

    if (disparity < min_disparity) {
      return false;
    }
  }


  if (options.max_homography_inliers < 1) {

    // Extract feature matches
    Eigen::MatrixXd prev_matches(matches_.size(), 2);
    Eigen::MatrixXd curr_matches(matches_.size(), 2);
    for (size_t i=0; i<matches_.size(); ++i) {
      cv::DMatch& match = matches_[i];
      prev_matches.row(i) = prev_points2D_[match.queryIdx];
      curr_matches.row(i) = curr_points2D_[match.trainIdx];
    }

    // Test whether there is enough view-point change between first and second
    // image by estimating the homography and checking for the number of
    // inliers

    const size_t stop_num_inliers
      = matches_.size() * options.max_homography_inliers;

    size_t num_inliers;
    std::vector<bool> inlier_mask;

    ProjectiveTransformEstimator projective_transform_estimator;
    RANSAC(projective_transform_estimator,
           prev_matches,
           curr_matches,
           4, // min_samples
           options.ransac_max_reproj_error, num_inliers, inlier_mask,
           100, // max_trials
           stop_num_inliers // stop_num_inliers
           );

    if (debug) {
      std::cout << "DEBUG: max-homography-inliers: "
                << (double)num_inliers / inlier_mask.size() << " > "
                << options.max_homography_inliers << std::endl;
    }

    if ((double)num_inliers / inlier_mask.size()
        > options.max_homography_inliers) {
      return false;
    }

  }


  // Get camera models
  const size_t first_camera_id = get_camera_id(first_image_idx);
  const size_t second_camera_id = get_camera_id(second_image_idx);

  const int first_camera_model_code = get_camera_model_code(first_image_idx);
  const int second_camera_model_code = get_camera_model_code(second_image_idx);

  const std::vector<double>& first_camera_params
    = feature_manager.camera_params[first_camera_id];
  const std::vector<double>& second_camera_params
    = feature_manager.camera_params[second_camera_id];


  // Extract normalized feature matches
  Eigen::MatrixXd prev_matches_normalized(matches_.size(), 2);
  Eigen::MatrixXd curr_matches_normalized(matches_.size(), 2);
  // Eigen::MatrixXd prev_matches_normalized_df(matches_.size(), 2);
  // Eigen::MatrixXd curr_matches_normalized_df(matches_.size(), 2);
  for (size_t i=0; i<matches_.size(); ++i) {
    cv::DMatch& match = matches_[i];
    prev_matches_normalized.row(i) = prev_points2D_N_[match.queryIdx];
    curr_matches_normalized.row(i) = curr_points2D_N_[match.trainIdx];
  }

  // Essential matrix estimation

  const double first_residual_threshold
    = camera_model_image2world_threshold(options.ransac_max_reproj_error,
                                         first_camera_model_code,
                                         first_camera_params.data());
  const double second_residual_threshold
    = camera_model_image2world_threshold(options.ransac_max_reproj_error,
                                         second_camera_model_code,
                                         second_camera_params.data());
  const double residual_threshold
    = (first_residual_threshold + second_residual_threshold) / 2;

  size_t num_inliers;
  std::vector<bool> inlier_mask;
  Eigen::Matrix3d essential_matrix;

  try {
    EssentialMatrixEstimator essential_matrix_estimator;
    essential_matrix
      = RANSAC(essential_matrix_estimator,
               prev_matches_normalized,
               curr_matches_normalized,
               5, // min_samples
               residual_threshold, num_inliers, inlier_mask,
               1000, // max_trials, many iterations for reliable solution
               matches_.size() + 1 // stop_num_inliers, run full max_trials
               );
  }
  catch (std::domain_error) {
    return false;
  }


  // Build inlier correspondence array
  std::vector<Eigen::Vector2i> correspondences;
  std::vector<Eigen::Vector2d> inlier_prev_points2D_N;
  std::vector<Eigen::Vector2d> inlier_curr_points2D_N;
  for (size_t i=0; i<matches_.size(); ++i) {
    if (inlier_mask[i]) {
      cv::DMatch& match = matches_[i];
      Eigen::Vector2i corr(match.queryIdx, match.trainIdx);
      correspondences.push_back(corr);
      inlier_prev_points2D_N.push_back(prev_points2D_N_[match.queryIdx]);
      inlier_curr_points2D_N.push_back(curr_points2D_N_[match.trainIdx]);
    }
  }


  if (debug) {
    std::cout << "DEBUG: Inliers: " << num_inliers << std::endl;
    std::cout << "DEBUG: Outliers: " << matches_.size() - num_inliers
              << std::endl;

    cv::Mat image_matches;
    cv::Mat prev_image = image_data[first_image_idx].read();
    cv::Mat curr_image = image_data[second_image_idx].read();
    std::vector<char> inlier_mask_char(inlier_mask.begin(), inlier_mask.end());
    cv::drawMatches(prev_image, prev_keypoints_,
                    curr_image, curr_keypoints_,
                    matches_, image_matches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    inlier_mask_char);
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(first_image_idx) + "-"
        + boost::lexical_cast<std::string>(second_image_idx)
        + "-matches-inlier.jpg";
    cv::imwrite(debug_path_full, image_matches);
  }

  // Make sure we have a sufficient number of inliers
  const size_t min_inlier_threshold
    = rel2abs_threshold(options.ransac_min_inlier_threshold, matches_.size());
  if (debug) {
    std::cout << "DEBUG: min-inlier-threshold: "
              << num_inliers << " < "
              << min_inlier_threshold << std::endl;
  }
  if (num_inliers < min_inlier_threshold) {
    return false;
  }


  // First camera pose
  const Eigen::Vector3d first_rvec = Eigen::Vector3d(0, 0, 0);
  const Eigen::Vector3d first_tvec = Eigen::Vector3d(0, 0, 0);


  // Extract rvec and tvec from second pose
  Eigen::Matrix3d R;
  Eigen::Vector3d second_tvec;
  pose_from_essential_matrix(essential_matrix,
                             prev_matches_normalized,
                             curr_matches_normalized,
                             inlier_mask, R, second_tvec);
  const Eigen::AngleAxisd second_rotation(R);
  const Eigen::Vector3d second_rvec = second_rotation.angle()
                                      * second_rotation.axis();

  const Eigen::Matrix<double, 3, 4> first_proj_matrix
    = compose_proj_matrix(first_rvec, first_tvec);
  const Eigen::Matrix<double, 3, 4> second_proj_matrix
    = compose_proj_matrix(second_rvec, second_tvec);


  // Make sure we do not have strong forward motion
  const Eigen::Matrix<double, 3, 4> inv_second_proj_matrix
    = invert_proj_matrix(second_proj_matrix);
  if (debug) {
    std::cout << "DEBUG: z-component: " << inv_second_proj_matrix(2, 3) << std::endl;
  }
  if (std::abs(inv_second_proj_matrix(2, 3)) > 0.99) {
    return false;
  }


  // Triangulate all inlier image point pairs
  std::vector<Eigen::Vector3d> points3D
    = triangulate_points(first_proj_matrix, second_proj_matrix,
                         inlier_prev_points2D_N, inlier_curr_points2D_N);


  // Calculate mean triangulation angles for all inlier correspondences
  const std::vector<double> tri_angles =
    calc_tri_angles(first_proj_matrix, second_proj_matrix, points3D);
  double mean_tri_angle = 0;
  for (size_t i=0; i<tri_angles.size(); ++i) {
    // Triangulation is unstable for acute angles (far away points)
      // and obtuse angles (close points)
      const double tri_angle = tri_angles[i];
      mean_tri_angle += std::min(tri_angle, 180 * DEG2RAD - tri_angle);
  }
  mean_tri_angle /= tri_angles.size();
  mean_tri_angle *= RAD2DEG;
  if (debug) {
    std::cout << "DEBUG: tri-min-angle: "
              << mean_tri_angle << " < "
              << options.tri_min_angle << std::endl;
  }
  if (mean_tri_angle < options.tri_min_angle) {
    return false;
  }

  // Add first image to feature manager
  const size_t first_image_id
    = feature_manager.add_image(first_camera_id, prev_points2D_);
  feature_manager.set_pose(first_image_id, first_rvec, first_tvec);

  // Add second image to feature manager
  const size_t second_image_id
    = feature_manager.add_image(second_camera_id, curr_points2D_);
  feature_manager.set_pose(second_image_id, second_rvec, second_tvec);

  size_t num_tris = 0;
  for (size_t i=0; i<correspondences.size(); ++i) {
    if (calc_depth(first_proj_matrix, points3D[i]) > 0) {
      size_t point3D_id
        = feature_manager.add_correspondence(second_image_id, first_image_id,
                                             correspondences[i](1),
                                             correspondences[i](0));
      feature_manager.set_point3D(point3D_id, points3D[i]);
      num_tris += 1;
    }
  }

  if (debug) {
    std::cout << "DEBUG: Number of triangulations: "
              << num_tris << std::endl;
  }


  image_idx_to_id_[first_image_idx] = first_image_id;
  image_id_to_idx_[first_image_id] = first_image_idx;

  image_idx_to_id_[second_image_idx] = second_image_id;
  image_id_to_idx_[second_image_id] = second_image_idx;

  num_proc_images_ += 2;

  if (first_image_idx > second_image_idx) {
    first_image_idx_ = second_image_idx;
    second_image_idx_ = first_image_idx;
    min_image_idx_ = second_image_idx;
    max_image_idx_ = first_image_idx;
  } else {
    first_image_idx_ = first_image_idx;
    second_image_idx_ = second_image_idx;
    min_image_idx_ = first_image_idx;
    max_image_idx_ = second_image_idx;
  }

  image_pair_idxs_[first_image_idx].insert(second_image_idx);
  image_pair_idxs_[second_image_idx].insert(first_image_idx);

  if (loop_detection_) {
    loop_detector_.add_image(first_image_idx, prev_descriptors_);
    loop_detector_.add_image(second_image_idx, curr_descriptors_);
  }

  return true;
}


bool SequentialMapper::process(size_t image_idx,
                               size_t prev_image_idx,
                               const SequentialMapperOptions& options,
                               const BundleAdjustmentOptions&
                                 refinement_options) {

  if (num_proc_images_ == 0) {
    throw std::invalid_argument("At least one image must be processed "
                                "prior to sequential processing.");
  }

  if (!is_image_processed(prev_image_idx)) {
    if (is_image_processed(image_idx)) {
      std::swap(image_idx, prev_image_idx);
    } else {
      return false;
    }
  }

  const size_t prev_image_id = get_image_id(prev_image_idx);

  if (debug) {
    cv::Mat image_matches;
    cv::Mat prev_image = image_data[prev_image_idx].read();
    cv::Mat curr_image = image_data[image_idx].read();
    cv::drawMatches(prev_image, std::vector<cv::KeyPoint>(0),
                    curr_image, std::vector<cv::KeyPoint>(0),
                    std::vector<cv::DMatch>(0), image_matches);
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(image_idx) + "-"
        + boost::lexical_cast<std::string>(prev_image_idx)
        + "-images.jpg";
    cv::imwrite(debug_path_full, image_matches);
  }

  query_and_prepare_features_(prev_image_idx);
  query_and_prepare_features_(image_idx);
  match_features_(options);

  if (debug) {
    std::cout << "DEBUG: Number of matches: "
              << matches_.size() << std::endl;
    cv::Mat image_matches;
    cv::Mat prev_image = image_data[prev_image_idx].read();
    cv::Mat curr_image = image_data[image_idx].read();
    cv::drawMatches(prev_image, prev_keypoints_,
                    curr_image, curr_keypoints_,
                    matches_, image_matches);
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(image_idx) + "-"
        + boost::lexical_cast<std::string>(prev_image_idx)
        + "-matches-all.jpg";
    cv::imwrite(debug_path_full, image_matches);
  }


  if (options.min_disparity > 0) {
    const size_t disparity = median_feature_disparity(prev_keypoints_, curr_keypoints_, matches_);
    float frame_diagonal;
    feature_cache_.query_dimensions(image_idx, NULL, NULL, NULL, &frame_diagonal);
    const double min_disparity = rel2abs_threshold(options.min_disparity, frame_diagonal);

    if (debug) {
      std::cout << "DEBUG: min-dispariy: " << disparity << " < " << min_disparity << std::endl;
    }

    if (disparity < min_disparity) {
      return false;
    }
  }


  // Test whether there is enough view-point change between first and second
  // image by estimating the homography and checking for the number of inliers
  if (options.max_homography_inliers < 1) {

    // Extract feature matches
    Eigen::MatrixXd prev_matches(matches_.size(), 2);
    Eigen::MatrixXd curr_matches(matches_.size(), 2);
    for (size_t i=0; i<matches_.size(); ++i) {
      cv::DMatch& match = matches_[i];
      prev_matches.row(i) = prev_points2D_[match.queryIdx];
      curr_matches.row(i) = curr_points2D_[match.trainIdx];
    }

    size_t num_inliers;
    std::vector<bool> inlier_mask;

    const size_t stop_num_inliers
      = matches_.size() * options.max_homography_inliers;

    ProjectiveTransformEstimator projective_transform_estimator;
    RANSAC(projective_transform_estimator,
           prev_matches,
           curr_matches,
           4, // min_samples
           options.ransac_max_reproj_error, num_inliers, inlier_mask,
           100, // max_trials
           stop_num_inliers // stop_num_inliers
           );

    if (debug) {
      std::cout << "DEBUG: max-homography-inliers: "
                << (double)num_inliers / inlier_mask.size() << " > "
                << options.max_homography_inliers << std::endl;
    }

    if ((double)num_inliers / inlier_mask.size()
        > options.max_homography_inliers) {
      return false;
    }

  }

  // All correspondences
  std::vector<Eigen::Vector2i> correspondences = cv2eigen_matches(matches_);

  // Find correspondences which have already been triangulated in previous
  // process chain
  std::vector<bool> tri_mask;
  std::vector<size_t> tri_point3D_ids;
  std::vector<size_t> prev_point2D_idxs(correspondences.size());
  for (size_t i=0; i<correspondences.size(); ++i) {
    prev_point2D_idxs[i] = correspondences[i](0);
  }
  feature_manager.find_tri_points(prev_image_id, prev_point2D_idxs,
                                  tri_point3D_ids, tri_mask);

  // Sort correspondences in two groups:
  //    1. Already triangulated in previous process chain
  //       => Use for 2D-3D pose estimation (if sufficiently long track length)
  //       => Continue track (if small re-projection error)
  //    2. New 2D-2D correspondence
  //       => Triangulate and add to feature manager
  //          (if small re-projection error)

  std::vector<Eigen::Vector2d> tri_curr_points2D_stable;
  std::vector<Eigen::Vector2d> tri_curr_points2D_N_stable;
  std::vector<Eigen::Vector2d> tri_curr_points2D_N;
  std::vector<Eigen::Vector2d> new_prev_points2D_N;
  std::vector<Eigen::Vector2d> new_curr_points2D_N;
  std::vector<Eigen::Vector3d> tri_points3D, tri_points3D_stable;
  std::vector<Eigen::Vector2i> tri_correspondences;
  std::vector<Eigen::Vector2i> new_correspondences;

  size_t tri_point3D_idx = 0;

  for (size_t i=0; i<correspondences.size(); ++i) {

    const Eigen::Vector2i& correspondence = correspondences[i];
    const size_t prev_point2D_idx = correspondence(0);
    const size_t curr_point2D_idx = correspondence(1);

    const Eigen::Vector2d& prev_point2D_N = prev_points2D_N_[prev_point2D_idx];
    const Eigen::Vector2d& curr_point2D_N = curr_points2D_N_[curr_point2D_idx];

    if (tri_mask[i]) {  // Already triangulated

      tri_correspondences.push_back(correspondence);
      tri_curr_points2D_N.push_back(curr_point2D_N);

      const size_t point3D_id = tri_point3D_ids[tri_point3D_idx];
      const Eigen::Vector3d& point3D = feature_manager.points3D[point3D_id];
      tri_points3D.push_back(point3D);

      // Stable points with minimum track length, used for pose estimation
      if (feature_manager.point3D_to_points2D[point3D_id].size()
          >= options.min_track_len) {
        tri_curr_points2D_stable.push_back(curr_points2D_[curr_point2D_idx]);
        tri_curr_points2D_N_stable.push_back(curr_point2D_N);
        tri_points3D_stable.push_back(point3D);
      }
      tri_point3D_idx++;

    } else {  // Add to triangulation todo-list

      new_correspondences.push_back(correspondence);
      new_prev_points2D_N.push_back(prev_point2D_N);
      new_curr_points2D_N.push_back(curr_point2D_N);

    }

  }

  const size_t min_inlier_threshold
    = rel2abs_threshold(options.ransac_min_inlier_threshold,
                        tri_points3D_stable.size());

  // We can never reach the minimum number of inlier threshold,
  // or number of stable points not sufficient for 2D-3D pose estimation
  if (debug) {
    std::cout << "DEBUG: min-inlier-threshold: "
              << tri_points3D_stable.size() << " < "
              << min_inlier_threshold << std::endl;
  }

  if (tri_points3D_stable.size() < min_inlier_threshold
      || tri_points3D_stable.size() < 4) {
    return false;
  }

  if (debug) {
    std::cout << "DEBUG: Triangulated matches: " << tri_curr_points2D_N.size()
              << std::endl;
    std::cout << "DEBUG: Triangulated stable matches: "
              << tri_curr_points2D_stable.size() << std::endl;
    std::cout << "DEBUG: New matches: " << new_curr_points2D_N.size()
              << std::endl;

    std::vector<cv::DMatch> tri_m;
    for (size_t i=0; i<tri_correspondences.size(); ++i) {
        Eigen::Vector2i c = tri_correspondences[i];
        tri_m.push_back(cv::DMatch(c(0), c(1), 0));
    }
    cv::Mat image_matches;
    cv::Mat prev_image = image_data[prev_image_idx].read();
    cv::Mat curr_image = image_data[image_idx].read();
    cv::drawMatches(prev_image, prev_keypoints_,
                    curr_image, curr_keypoints_,
                    tri_m, image_matches,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1));
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(image_idx) + "-"
        + boost::lexical_cast<std::string>(prev_image_idx)
        + "-matches-tri.jpg";
    cv::imwrite(debug_path_full, image_matches);
  }

  const size_t camera_id = get_camera_id(image_idx);
  const int camera_model_code = get_camera_model_code(image_idx);
  const std::vector<double>& camera_params
    = feature_manager.camera_params[camera_id];
  const double ransac_max_reproj_error_N
    = camera_model_image2world_threshold(options.ransac_max_reproj_error,
                                         camera_model_code,
                                         camera_params.data());

  Eigen::Matrix<double, 3, 4> proj_matrix;

  // Minimum number of inliers which must be found before RANSAC stops
  const size_t ransac_min_inliers_stop
    = rel2abs_threshold(options.ransac_min_inlier_stop,
                        tri_points3D_stable.size());

  size_t num_inliers;
  std::vector<bool> inlier_mask;

  try {

    Eigen::MatrixXd points2D(tri_curr_points2D_N_stable.size(), 2);
    Eigen::MatrixXd points3D(tri_points3D_stable.size(), 3);
    for (size_t i=0; i<(size_t)points2D.rows(); ++i) {
      points2D.row(i) = tri_curr_points2D_N_stable[i];
      points3D.row(i) = tri_points3D_stable[i];
    }

    P3PEstimator p3p_estimator;
    proj_matrix = RANSAC(p3p_estimator, points2D, points3D,
                         4, // min_samples
                         ransac_max_reproj_error_N, num_inliers, inlier_mask,
                         500, // max_trials
                         ransac_min_inliers_stop // stop_num_inliers
                         );
  }
  catch (std::domain_error) {  // No valid consensus set found
    return false;
  }

  if (debug) {
    std::cout << "DEBUG: 2D-3D inliers: " << num_inliers << std::endl;
    std::cout << "DEBUG: 2D-3D outliers: " << tri_points3D_stable.size()
                                              - num_inliers << std::endl;

    std::vector<cv::DMatch> matches_stable;
    std::vector<Eigen::Vector2d> tri_prev_points2D_stable;
    size_t tri_point3D_idx = 0;
    for (size_t i=0; i<correspondences.size(); ++i) {
      if (tri_mask[i]) {
        const size_t point3D_id = tri_point3D_ids[tri_point3D_idx];
        if (feature_manager.point3D_to_points2D[point3D_id].size()
            >= options.min_track_len) {
          matches_stable.push_back(matches_[i]);
        }
        tri_point3D_idx += 1;
      }
    }
    std::vector<cv::DMatch> matches_stable_inlier;
    for (size_t i=0; i<inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        matches_stable_inlier.push_back(matches_stable[i]);
      }
    }
    cv::Mat image_matches;
    cv::Mat prev_image = image_data[prev_image_idx].read();
    cv::Mat curr_image = image_data[image_idx].read();
    cv::drawMatches(prev_image, prev_keypoints_,
                    curr_image, curr_keypoints_,
                    matches_stable_inlier, image_matches);
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(image_idx) + "-"
        + boost::lexical_cast<std::string>(prev_image_idx)
        + "-matches-inlier.jpg";
    cv::imwrite(debug_path_full, image_matches);
  }

  if (debug) {
    std::cout << "DEBUG: min-inlier-threshold: "
              << num_inliers << " < "
              << min_inlier_threshold << std::endl;
  }

  if (num_inliers < min_inlier_threshold) {
    return false;
  }

  // Extract rotation and translation vector from projection matrix
  const Eigen::AngleAxisd rotation(proj_matrix.block<3, 3>(0, 0));
  Eigen::Vector3d curr_rvec = rotation.angle() * rotation.axis();
  Eigen::Vector3d curr_tvec = proj_matrix.block<3, 1>(0, 3);

  // Refine pose of current image using all inliers of RANSAC
  const double final_cost
    = pose_refinement(curr_rvec, curr_tvec,
                      feature_manager.camera_params[camera_id],
                      tri_curr_points2D_stable,
                      tri_points3D_stable, inlier_mask,
                      refinement_options);

  // Stop if worse than threshold

  if (debug) {
    std::cout << "DEBUG: final-cost-threshold: "
              << final_cost << " < "
              << options.final_cost_threshold << std::endl;
  }

  if (final_cost > options.final_cost_threshold) {
    return false;
  }

  // Use curr_rvec, curr_tvec rather than data from feature_manager if already
  // processed, so triangulation is reliable also if current pose is subject
  // to drift
  Eigen::Matrix<double, 3, 4> curr_proj_matrix
    = compose_proj_matrix(curr_rvec, curr_tvec);
  Eigen::Matrix<double, 3, 4> prev_proj_matrix
    = compose_proj_matrix(feature_manager.rvecs[prev_image_id],
                          feature_manager.tvecs[prev_image_id]);

  size_t curr_image_id;

  if (is_image_processed(image_idx)) {
    // Image already processed
    curr_image_id = get_image_id(image_idx);
  } else {
    // Add new image to feature manager
    curr_image_id = feature_manager.add_image(camera_id, curr_points2D_);
  }

  feature_manager.set_pose(curr_image_id, curr_rvec, curr_tvec);

  // Determine projection error of all correspondences with respect to
  // current image to classify as inlier or outlier and
  // accordingly append to feature manager

  const double tri_max_reproj_error_N
    = camera_model_image2world_threshold(options.tri_max_reproj_error,
                                         camera_model_code,
                                         camera_params.data());

  size_t num_tris = 0;

  // Triangulated 3D points
  std::vector<double> tri_proj_errors
    = calc_reproj_errors(tri_curr_points2D_N, tri_points3D, curr_proj_matrix);
  for (size_t i=0; i<tri_points3D.size(); ++i) {
    if (tri_proj_errors[i] < tri_max_reproj_error_N) {
      // Only add correspondence and keep 3D position from previous processing
      feature_manager.add_correspondence(curr_image_id, prev_image_id,
                                         tri_correspondences[i](1),
                                         tri_correspondences[i](0));
      num_tris += 1;
    }
  }

  if (debug) {
    std::cout << "DEBUG: Number of continued tracks: "
              << num_tris << std::endl;
  }

  // Triangulate all new image point pairs
  num_tris = 0;
  std::vector<Eigen::Vector3d> new_points3D
    = triangulate_points(prev_proj_matrix, curr_proj_matrix,
                         new_prev_points2D_N, new_curr_points2D_N);
  std::vector<double> tri_new_angles
    = calc_tri_angles(prev_proj_matrix, curr_proj_matrix, new_points3D);
  const double tri_min_angle_rad = options.tri_min_angle * DEG2RAD;
  // New 3D points
  std::vector<double> new_proj_errors
    = calc_reproj_errors(new_curr_points2D_N, new_points3D, curr_proj_matrix);
  for (size_t i=0; i<new_points3D.size(); ++i) {
    if (new_proj_errors[i] < tri_max_reproj_error_N) {
      // Triangulation is unstable for acute angles (far away points)
      // and obtuse angles (close points)
      const double tri_angle = tri_new_angles[i];
      if (tri_min_angle_rad < std::min(tri_angle, 180 * DEG2RAD - tri_angle)
          && calc_depth(curr_proj_matrix, new_points3D[i]) > 0) {
        const size_t new_point3D_id
          = feature_manager.add_correspondence(curr_image_id, prev_image_id,
                                               new_correspondences[i](1),
                                               new_correspondences[i](0));
        feature_manager.set_point3D(new_point3D_id, new_points3D[i]);
        num_tris += 1;
      }
    }
  }

  if (debug) {
    std::cout << "DEBUG: Number of new triangulations: "
              << num_tris << std::endl;
  }

  if (debug) {  // Write track lengths of current image

    std::ofstream debug_file;
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(image_idx) + "-"
        + boost::lexical_cast<std::string>(prev_image_idx)
        + "-track-length.log";
    debug_file.open(debug_path_full.c_str());

    const std::vector<size_t>& point2D_ids
      = feature_manager.image_to_points2D[curr_image_id];
    for (size_t i=0; i<point2D_ids.size(); ++i) {
      const size_t point2D_id = point2D_ids[i];
      if (feature_manager.point2D_to_point3D.count(point2D_id)) {
        const size_t point3D_id
          = feature_manager.point2D_to_point3D[point2D_id];
        debug_file << "Point 3D-ID: "
                   << point3D_id
                   << "\t\t, Track-length: "
                   << feature_manager.point3D_to_points2D[point3D_id].size()
                   << "\t\t, Z-coord: "
                   << feature_manager.points3D[point3D_id](2)
                   << std::endl;
      }
    }
    debug_file.close();
  }

  if (debug) {  // Write 3D points of current image

    std::ofstream debug_file;
    std::string debug_path_full = debug_path
        + boost::lexical_cast<std::string>(get_num_proc_images()) + "-"
        + boost::lexical_cast<std::string>(image_idx) + "-"
        + boost::lexical_cast<std::string>(prev_image_idx)
        + "-scene.wrl";
    debug_file.open(debug_path_full.c_str());

    debug_file << "#VRML V2.0 utf8\n";
    debug_file << "Background { skyColor [1.0 1.0 1.0] } \n";
    debug_file << "Shape{ appearance Appearance {\n";
    debug_file << " material Material {emissiveColor 1 1 1} }\n";
    debug_file << " geometry PointSet {\n";
    debug_file << " coord Coordinate {\n";
    debug_file << "  point [\n";

    std::vector<int> red_color;
    std::vector<int> green_color;
    std::vector<int> blue_color;

    const std::vector<size_t>& point2D_ids
      = feature_manager.image_to_points2D[curr_image_id];
    for (size_t i=0; i<point2D_ids.size(); ++i) {
      const size_t point2D_id = point2D_ids[i];
      if (feature_manager.point2D_to_point3D.count(point2D_id)) {
        const size_t point3D_id
          = feature_manager.point2D_to_point3D[point2D_id];
        const Eigen::Vector3d& point3D = feature_manager.points3D[point3D_id];
        const size_t track_len
          = feature_manager.point3D_to_points2D[point3D_id].size();
        if (track_len == 2) {
          // newly triangulated 3D points in red
          red_color.push_back(1);
          green_color.push_back(0);
          blue_color.push_back(0);
        } else if (track_len > options.min_track_len) {
          // points used for 2D-3D pose estimation in green
          red_color.push_back(0);
          green_color.push_back(1);
          blue_color.push_back(0);
        } else {
          // all other points with only small track length in blue
          red_color.push_back(0);
          green_color.push_back(0);
          blue_color.push_back(1);
        }
        debug_file << point3D(0) << " "
                   << point3D(1) << " "
                   << point3D(2) << std::endl;
      }
    }
    debug_file << " ] }\n";
    debug_file << " color Color { color [\n";

    // write color data for each point
    for (size_t i = 0; i <red_color.size(); ++i) {
      debug_file << red_color[i] << " "
                 << green_color[i] << " "
                 << blue_color[i] << std::endl;
    }

    debug_file << " ] } } }\n";
    debug_file.close();
  }

  if (!is_image_processed(image_idx)) {
    if (loop_detection_) {
      loop_detector_.add_image(image_idx, curr_descriptors_);
    }
    num_proc_images_ += 1;
  }

  image_idx_to_id_[image_idx] = curr_image_id;
  image_id_to_idx_[curr_image_id] = image_idx;

  if (image_idx < min_image_idx_) {
    min_image_idx_ = image_idx;
  } else if (image_idx > max_image_idx_) {
    max_image_idx_ = image_idx;
  }

  if (!is_pair_processed(image_idx, prev_image_idx)) {
    image_pair_idxs_[image_idx].insert(prev_image_idx);
    image_pair_idxs_[prev_image_idx].insert(image_idx);
  }

  return true;
}


bool SequentialMapper::is_image_processed(const size_t image_idx) {
  return image_idx_to_id_.count(image_idx) != 0;
}


bool SequentialMapper::is_pair_processed(const size_t image_idx1,
                                         const size_t image_idx2) {
  if (image_pair_idxs_.count(image_idx1) != 0) {
    if (image_pair_idxs_[image_idx1].count(image_idx2) != 0) {
      return true;
    }
  }
  return false;
}


size_t SequentialMapper::get_camera_id(const size_t image_idx) {
  const size_t camera_idx = image_data[image_idx].camera_idx;
  size_t camera_id;
  if (camera_idx_to_id_.count(camera_idx) == 0) {
    std::vector<double> camera_params = image_data[image_idx].camera_params;
    const int camera_model_code = get_camera_model_code(image_idx);
    // Append camera model code as last camera parameter, so
    // bundle_adjustment function is able to dynamically determine the
    // necessary model. The first N-1 parameters are then used by the camera
    // model functions, whereas the model code is not touched within the
    // camera model functions.
    camera_params.push_back((double)camera_model_code);
    camera_id
      = feature_manager.add_camera(camera_params);
    camera_idx_to_id_[camera_idx] = camera_id;
  } else {
    camera_id = camera_idx_to_id_[camera_idx];
  }
  return camera_id;
}


int SequentialMapper::get_camera_model_code(const size_t image_idx) {
  const std::string& camera_model = image_data[image_idx].camera_model;
  return camera_model_name_to_code(camera_model);
}


size_t SequentialMapper::get_image_id(const size_t image_idx) {
  if (!is_image_processed(image_idx)) {
    throw std::range_error("Image IDX does not exist.");
  }
  return image_idx_to_id_[image_idx];
}


size_t SequentialMapper::get_image_idx(const size_t image_id) {
  if (image_id_to_idx_.count(image_id) == 0) {
    throw std::range_error("Image ID does not exist.");
  }
  return image_id_to_idx_[image_id];
}


std::set<size_t> SequentialMapper::get_image_idxs() {
  std::set<size_t> image_idxs;
  for (auto it=image_idx_to_id_.begin(); it!=image_idx_to_id_.end(); ++it) {
    image_idxs.insert(it->first);
  }
  return image_idxs;
}


double SequentialMapper::get_point3D_error(const size_t point3D_id) {
  if (point3D_errors_.count(point3D_id) == 0) {
    throw std::range_error("3D point ID does not exist.");
  }
  return point3D_errors_[point3D_id];
}


size_t SequentialMapper::get_point3D_track_len(const size_t point3D_id) {
  if (feature_manager.points3D.count(point3D_id) == 0) {
    throw std::range_error("3D point ID does not exist.");
  }
  return feature_manager.point3D_to_points2D[point3D_id].size();
}


void SequentialMapper::get_features(const size_t image_idx,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    cv::Mat& descriptors) {
  feature_cache_.query(image_idx, keypoints, descriptors);
}


double SequentialMapper::adjust_bundle(
          const std::vector<size_t>& free_image_idxs,
          const std::vector<size_t>& fixed_image_idxs,
          const std::vector<size_t>& fixed_x_image_idxs,
          const BundleAdjustmentOptions& options) {

  // Get feature manager ID of image indices

  std::vector<size_t> free_image_ids;
  std::vector<size_t> fixed_image_ids;
  std::vector<size_t> fixed_x_image_ids;

  std::vector<size_t>::const_iterator it;

  std::unordered_map<size_t, Eigen::Vector3d> rotation_constraints;

  for (it=free_image_idxs.begin(); it!=free_image_idxs.end(); ++it) {
    const size_t image_id = get_image_id(*it);
    free_image_ids.push_back(image_id);
  }
  for (it=fixed_image_idxs.begin(); it!=fixed_image_idxs.end(); ++it) {
    const size_t image_id = get_image_id(*it);
    fixed_image_ids.push_back(image_id);
  }
  for (it=fixed_x_image_idxs.begin(); it!=fixed_x_image_idxs.end(); ++it) {
    const size_t image_id = get_image_id(*it);
    fixed_x_image_ids.push_back(image_id);
  }

  if (options.constrain_rotation) {
    for (it=free_image_idxs.begin(); it!=free_image_idxs.end(); ++it) {
      const size_t image_id = get_image_id(*it);
      rotation_constraints[image_id] = image_data[*it].rvec();
    }
    for (it=fixed_image_idxs.begin(); it!=fixed_image_idxs.end(); ++it) {
      const size_t image_id = get_image_id(*it);
      rotation_constraints[image_id] = image_data[*it].rvec();
    }
    for (it=fixed_x_image_idxs.begin(); it!=fixed_x_image_idxs.end(); ++it) {
      const size_t image_id = get_image_id(*it);
      rotation_constraints[image_id] = image_data[*it].rvec();
    }
  }

  return bundle_adjustment(feature_manager,
                           free_image_ids,
                           fixed_image_ids,
                           fixed_x_image_ids,
                           options,
                           point3D_errors_,
                           rotation_constraints);
}


double SequentialMapper::adjust_global_bundle(const BundleAdjustmentOptions& options) {
  if (num_proc_images_ == 0) {
    throw std::invalid_argument("At least the initial image pair must be "
                                "processed prior to global bundle "
                                "adjustment.");
  }

  std::vector<size_t> free_image_idxs;
  std::vector<size_t> fixed_image_idxs;
  std::vector<size_t> fixed_x_image_idxs;

  // adjust first two poses as fixed and with fixed x-coordinate
  fixed_image_idxs.push_back(first_image_idx_);
  fixed_x_image_idxs.push_back(second_image_idx_);

  // All other poses as free
  const size_t start_image_idx = min_image_idx_;
  const size_t end_image_idx = max_image_idx_;
  for (size_t image_idx=start_image_idx;
       image_idx<=end_image_idx; ++image_idx) {
    if (is_image_processed(image_idx)
        && image_idx != first_image_idx_
        && image_idx != second_image_idx_) {
      free_image_idxs.push_back(image_idx);
    }
  }

  return adjust_bundle(free_image_idxs, fixed_image_idxs, fixed_x_image_idxs,
                       options);

}


double SequentialMapper::adjust_global_bundle_gcp(const std::set<size_t>& gcp_ids,
                                                  const BundleAdjustmentOptions& options) {

  if (num_proc_images_ == 0) {
    throw std::invalid_argument("At least the initial image pair must be "
                                "processed prior to global bundle "
                                "adjustment.");
  }

  std::vector<size_t> free_image_ids;
  std::vector<size_t> fixed_image_ids;
  std::vector<size_t> fixed_x_image_ids;

  std::unordered_map<size_t, Eigen::Vector3d> rotation_constraints;

  const size_t start_image_idx = min_image_idx_;
  const size_t end_image_idx = max_image_idx_;
  for (size_t image_idx=start_image_idx;
       image_idx<=end_image_idx; ++image_idx) {
    if (is_image_processed(image_idx)
        && image_idx != first_image_idx_
        && image_idx != second_image_idx_) {
      free_image_ids.push_back(get_image_id(image_idx));
    }
  }

  if (options.constrain_rotation) {
    std::vector<size_t>::const_iterator it;
    for (it=free_image_ids.begin(); it!=free_image_ids.end(); ++it) {
      rotation_constraints[*it] = image_data[get_image_idx(*it)].rvec();
    }
  }

  return bundle_adjustment(feature_manager,
                           free_image_ids,
                           fixed_image_ids,
                           fixed_x_image_ids,
                           options,
                           point3D_errors_,
                           rotation_constraints,
                           gcp_ids);
}


size_t SequentialMapper::detect_loop(const size_t image_idx,
                                     const size_t num_images,
                                     const size_t num_nh_images,
                                     const size_t nh_distance,
                                     const SequentialMapperOptions& options,
                                     const BundleAdjustmentOptions& ba_options,
                                     const bool verbose) {

  if (!loop_detection_) {
    return 0;
  }

  std::vector<int> image_idxs;
  std::vector<float> scores;

  find_similar_images_(image_idx, num_images, image_idxs, scores);

  size_t num_successes = 0;
  size_t num_nh_successes = 0;

  // Try to process all similar images
  for (size_t i=0; i<image_idxs.size(); ++i) {

    const size_t other_image_idx = image_idxs[i];
    // "Distance" expressed in number of images between current and other image
    const size_t distance = std::abs<int>(other_image_idx - image_idx);

    if (image_idx != other_image_idx
        // Make sure the same image pairs are not processed twice
        && !is_pair_processed(image_idx, other_image_idx)
        // Either number of successfully processed pairs in direct neighborhood
        // of current image is not sufficient or "distance" to other image
        // is large enough
        && (num_nh_successes < num_nh_images || distance > nh_distance)) {

      // Try to process image pair
      if (process(image_idx, other_image_idx, options, ba_options)) {

        if (verbose) {
          std::cout << "Closed loop to image #" << other_image_idx
                    << std::endl << std::endl;
        }

        num_successes += 1;
        if (distance <= nh_distance) {
          num_nh_successes += 1;
        }

      }
    }
  }

  return num_successes;

}


bool SequentialMapper::merge(SequentialMapper& other,
                             const size_t num_similar_images,
                             const size_t num_skip_images,
                             const SequentialMapperOptions& options,
                             BundleAdjustmentOptions ba_options,
                             const bool verbose) {

  if (!loop_detection_) {
    return false;
  }

  if (&other == this) {
    throw std::invalid_argument("Cannot self-merge mapper.");
  }

  if (get_num_proc_images() == 0) {
    *this = other;
    return true;
  } else if (other.get_num_proc_images() == 0) {
    return true;
  }

  // Images that have been processed in this and other mapper
  std::set<size_t> corresponding_image_idxs;


  // Search for images which have already been processed in both mappers and
  // can thus be used for the similarity transform estimation.
  std::set<size_t> other_image_idxs = other.get_image_idxs();

  for (auto it=other_image_idxs.begin(); it!=other_image_idxs.end(); ++it) {
    const size_t other_image_idx = *it;
    if (is_image_processed(other_image_idx)) {
      corresponding_image_idxs.insert(other_image_idx);
    }
  }


  // First search for most similar images between two mapper sequences for all
  // images, this makes sure that we do not find similar images in one and the
  // same mapper but just between the mappers. Because successfully processed
  // images will be added to the vocabulary tree and will thus be also found
  // for subsequent similar image searches if we mixed the search and
  // processing.

  std::unordered_map<size_t, std::vector<int>> similar_images;
  size_t num_trials = 0;
  for (auto it=other_image_idxs.begin(); it!=other_image_idxs.end(); ++it) {
    const size_t other_image_idx = *it;

    // Only search for images every `num_skip_images`
    if (num_trials % num_skip_images == 0) {

      std::vector<int> this_image_idxs;
      std::vector<float> scores;

      find_similar_images_(other_image_idx, num_similar_images,
                           this_image_idxs, scores);

      similar_images[other_image_idx] = this_image_idxs;

    }

    num_trials += 1;

  }


  // Try to process similar image pairs, as found above

  std::unordered_map<size_t, size_t> proc_image_pair_idxs;

  for (auto it=similar_images.begin(); it!=similar_images.end(); ++it) {

    const size_t other_image_idx = it->first;
    const std::vector<int>& this_image_idxs = it->second;


    for (size_t i=0; i<this_image_idxs.size(); ++i) {
      const size_t this_image_idx = this_image_idxs[i];
      if (process(other_image_idx, this_image_idx, options, ba_options)) {
        corresponding_image_idxs.insert(other_image_idx);
        proc_image_pair_idxs[other_image_idx] = this_image_idx;
        if (verbose) {
          std::cout << "Detected loop between images #" << other_image_idx
                    << " and #" << this_image_idx << std::endl << std::endl;
        }
      }
    }

  }


  // Not enough image pairs to estimate similarity transform with 7 degrees
  // of freedom
  if (corresponding_image_idxs.size() < 3) {
    return false;
  }

  // Adjust global bundle to improve newly processed poses above, otherwise the
  // estimation of the similarity transform is inaccurate
  ba_options.print_progress = true;
  ba_options.print_summary = true;
  adjust_global_bundle(ba_options);

  // Extract corresponding image poses for similarity transform estimation

  Eigen::MatrixXd src(corresponding_image_idxs.size(), 3);
  Eigen::MatrixXd dst(corresponding_image_idxs.size(), 3);
  size_t i = 0;
  for (auto it=corresponding_image_idxs.begin();
       it!=corresponding_image_idxs.end(); ++it) {

    const size_t other_image_id = other.get_image_id(*it);
    const size_t this_image_id = get_image_id(*it);

    Eigen::Matrix<double, 3, 4> other_matrix = invert_proj_matrix(
      compose_proj_matrix(other.feature_manager.rvecs[other_image_id],
                          other.feature_manager.tvecs[other_image_id]));
    Eigen::Matrix<double, 3, 4> this_matrix = invert_proj_matrix(
      compose_proj_matrix(feature_manager.rvecs[this_image_id],
                          feature_manager.tvecs[this_image_id]));

    // Global positions of source and and destination cameras
    src.row(i) = other_matrix.block<3, 1>(0, 3);
    dst.row(i) = this_matrix.block<3, 1>(0, 3);
    i += 1;

  }

  // Estimate similarity transformation from other to this mapper
  SimilarityTransform3D other_to_this_transform;
  other_to_this_transform.estimate(src, dst);

  // Add all images from other to this mapper (must be done before adding
  // the correspondences, because feature manager requires existing images
  // for this)
  for (auto it=other_image_idxs.begin(); it!=other_image_idxs.end(); ++it) {
    const size_t other_image_idx = *it;

    // Only add image to this mapper if not already processed

    if (!is_image_processed(other_image_idx)) {

      // Add as new image

      const size_t other_image_id = other.get_image_id(other_image_idx);

      const std::vector<size_t>& other_point2D_ids
        = other.feature_manager.image_to_points2D[other_image_id];
      std::vector<Eigen::Vector2d> points2D(other_point2D_ids.size());
      for (size_t i=0; i<other_point2D_ids.size(); ++i) {
        const size_t point2D_id = other_point2D_ids[i];
        points2D[i] = other.feature_manager.points2D[point2D_id];
      }
      const size_t camera_id = get_camera_id(other_image_idx);
      const size_t image_id
        = feature_manager.add_image(camera_id, points2D);

      // Transform camera pose from other to this mapper:

      Eigen::Vector3d rvec = other.feature_manager.rvecs[other_image_id];
      Eigen::Vector3d tvec = other.feature_manager.tvecs[other_image_id];

      other_to_this_transform.transform_pose(rvec, tvec);

      feature_manager.set_pose(image_id, rvec, tvec);

      // Update this mapper info

      std::vector<cv::KeyPoint> _keypoints;
      cv::Mat descriptors;
      feature_cache_.query(other_image_idx, _keypoints, descriptors);
      loop_detector_.add_image(other_image_idx, descriptors);

      num_proc_images_ += 1;

      if (other_image_idx > max_image_idx_) {
        max_image_idx_ = other_image_idx;
      }

      image_idx_to_id_[other_image_idx] = image_id;
      image_id_to_idx_[image_id] = other_image_idx;
    }
  }

  // Add all correspondences from other to this mapper

  std::set<size_t> point3D_ids_merged;
  for (auto it=other.feature_manager.points3D.begin();
       it!=other.feature_manager.points3D.end(); ++it) {

    const size_t other_point3D_id = it->first;

    Eigen::Vector3d other_point3D = it->second;
    other_to_this_transform.transform_point(other_point3D);

    const std::vector<size_t>& other_track
      = other.feature_manager.point3D_to_points2D[other_point3D_id];

    const size_t prev_other_point2D_id = other_track[0];

    // Image index is equal for both mappers, assuming the same
    // feature detector was used
    size_t prev_point2D_idx
        = other.feature_manager.get_point2D_idx(prev_other_point2D_id);
    const size_t prev_other_image_id
      = other.feature_manager.point2D_to_image[prev_other_point2D_id];
    size_t prev_this_image_id
      = get_image_id(other.get_image_idx(prev_other_image_id));

    for (size_t j=1; j<other_track.size(); ++j) {
      const size_t other_point2D_id = other_track[j];
      // Image index is equal for both mappers, assuming the same
      // feature detector was used
      const size_t curr_point2D_idx
        = other.feature_manager.get_point2D_idx(other_point2D_id);
      const size_t curr_other_image_id
        = other.feature_manager.point2D_to_image[other_point2D_id];
      const size_t curr_this_image_id
        = get_image_id(other.get_image_idx(curr_other_image_id));

      const size_t this_point3D_id
        = feature_manager.add_correspondence(prev_this_image_id,
                                             curr_this_image_id,
                                             prev_point2D_idx,
                                             curr_point2D_idx);
      feature_manager.set_point3D(this_point3D_id, other_point3D);

      prev_this_image_id = curr_this_image_id;
      prev_point2D_idx = curr_point2D_idx;

    }
  }


  // Set correct image index range
  if (other.get_first_image_idx() < first_image_idx_) {
    first_image_idx_ = other.get_first_image_idx();
    second_image_idx_ = other.get_second_image_idx();
  }
  if (other.get_min_image_idx() < min_image_idx_) {
    min_image_idx_ = other.get_min_image_idx();
  }
  if (other.get_max_image_idx() > max_image_idx_) {
    max_image_idx_ = other.get_max_image_idx();
  }

  // Add successfully processed images in this function as processed pairs
  for (auto it=proc_image_pair_idxs.begin();
       it!=proc_image_pair_idxs.end(); ++it) {
    image_pair_idxs_[it->first].insert(it->second);
    image_pair_idxs_[it->second].insert(it->first);
  }

  // Add all processed image pairs from other to this mapper
  std::unordered_map<size_t, std::set<size_t> > other_image_pair_idxs
    = other.get_image_pair_idxs();
  for (auto it=other_image_pair_idxs.begin();
       it!=other_image_pair_idxs.end(); it++) {
    image_pair_idxs_[it->first].insert(it->second.begin(), it->second.end());
  }

  return true;
}


void SequentialMapper::write_image_data(const std::string& path) {

  std::ofstream file;
  file.open(path.c_str());

  file << "# BASENAME, ROLL, PITCH, YAW, LAT, LON, ALT, LOCAL_HEIGHT, ";
  file << "TX, TY, TZ, CAM_IDX, CAM_MODEL, CAM_PARAMS[]" << std::endl;

  for (size_t image_idx=0; image_idx<image_data.size(); ++image_idx) {

    size_t image_id;
    try {
      image_id = get_image_id(image_idx);
    }
    catch(std::exception& e) {
      continue;
    }

    double rx, ry, rz, tx, ty, tz;

    extract_exterior_params(feature_manager.rvecs[image_id],
                            feature_manager.tvecs[image_id],
                            rx, ry, rz, tx, ty, tz);

    const size_t camera_id = feature_manager.image_to_camera[image_id];
    const std::vector<double>& camera_params
      = feature_manager.camera_params[camera_id];

    file << image_data[image_idx].name << ", ";
    file << std::setprecision(12) << rx << ", ";
    file << std::setprecision(12) << ry << ", ";
    file << std::setprecision(12) << rz << ", ";
    // Use original data from input imagedata.txt
    file << std::setprecision(12) << image_data[image_idx].lat << ", ";
    file << std::setprecision(12) << image_data[image_idx].lon << ", ";
    file << std::setprecision(12) << image_data[image_idx].alt << ", ";
    file << std::setprecision(12) << image_data[image_idx].local_height << ", ";
    file << std::setprecision(12) << tx << ", ";
    file << std::setprecision(12) << ty << ", ";
    file << std::setprecision(12) << tz << ", ";
    file << image_data[image_idx].camera_idx << ", ";
    file << image_data[image_idx].camera_model << ", ";
    for (size_t i=0; i<camera_params.size()-2; ++i) {
      file << std::setprecision(12) << camera_params[i] << ", ";
    }
    file << std::setprecision(12) << camera_params[camera_params.size()-2];

    file << std::endl;

  }

  file << std::endl;

  file.close();

}


void SequentialMapper::write_point_cloud_data(const std::string& path) {

  std::unordered_map<size_t, Eigen::Vector3d>& points3D
    = feature_manager.points3D;
  std::unordered_map<size_t, Eigen::Vector2d>& points2D
    = feature_manager.points2D;
  std::unordered_map<size_t, size_t>& point2D_to_point3D
    = feature_manager.point2D_to_point3D;
  std::unordered_map<size_t, std::vector<size_t> >& point3D_to_points2D
    = feature_manager.point3D_to_points2D;
  std::unordered_map<size_t, std::vector<size_t> >&
    image_to_points2D = feature_manager.image_to_points2D;

  std::unordered_map<size_t, std::vector<Eigen::Vector3d>> point3D_to_colors;

  // Extract color for each point in all observed images
  for (size_t image_idx=0; image_idx<image_data.size(); image_idx++) {
    size_t image_id;
    try {
      image_id = get_image_id(image_idx);
    }
    catch(std::exception& e) {
      continue;
    }
    // Always read as color image, flag=1
    const cv::Mat image = image_data[image_idx].read(1);
    // Split image channels into separate color images in the order BGR
    cv::Mat image_rgb[3];
    cv::split(image, image_rgb);

    // Iterate through all 2D points in image
    const std::vector<size_t>& point2D_ids = image_to_points2D[image_id];
    for (size_t j=0; j<point2D_ids.size(); ++j) {
      const size_t point2D_id = point2D_ids[j];
      if (point2D_to_point3D.count(point2D_id)) {
        const size_t point3D_id = point2D_to_point3D[point2D_id];
        const Eigen::Vector2d& point2D = points2D[point2D_id];
        // Determine mean color around point in 3x3 window,
        // to account for image noise
        cv::Rect roi((size_t)point2D(0)-1, (size_t)point2D(1)-1, 3, 3);
        // OpenCV uses BGR color ordering
        const Eigen::Vector3d rgb(cv::mean(image_rgb[2](roi))[0] / 255.0,
                                  cv::mean(image_rgb[1](roi))[0] / 255.0,
                                  cv::mean(image_rgb[0](roi))[0] / 255.0);
        point3D_to_colors[point3D_id].push_back(rgb);
      }
    }
  }

  std::ofstream file;
  file.open(path.c_str());

  file << "# X, Y, Z, MEAN_R, MEAN_G, MEAN_B, "
       << "TRACK_LEN, MEAN_RESIDUAL" << std::endl;

  std::vector<Eigen::Vector3d> point3D_to_color;

  for (auto it=points3D.begin(); it!=points3D.end(); ++it) {

    const size_t point3D_id = it->first;

    const size_t track_len = point3D_to_points2D[point3D_id].size();

    const Eigen::Vector3d& point3D = it->second;

    double error;
    try {
      error = get_point3D_error(point3D_id);
    }
    catch(std::exception& e) {
      error = -1;
    }

    // Average colors of 3D point in all observed images
    std::vector<Eigen::Vector3d>& colors = point3D_to_colors[point3D_id];
    Eigen::Vector3d color(0, 0, 0);
    for (size_t i=0; i<colors.size(); ++i) {
      color += colors[i];
    }
    color /= colors.size();

    // Write point coordinates
    file << std::setprecision(12) << point3D(0) << ", "
         << std::setprecision(12) << point3D(1) << ", "
         << std::setprecision(12) << point3D(2) << ", "
         << std::setprecision(12) << color(0) << ", "
         << std::setprecision(12) << color(1) << ", "
         << std::setprecision(12) << color(2) << ", "
         << std::setprecision(12) << track_len << ", "
         << std::setprecision(12) << error << std::endl;


    point3D_to_color.push_back(color);

  }

  file << std::endl;

  file.close();

}


void SequentialMapper::write_point_cloud_vrml(const std::string& path,
                                              const size_t min_track_len,
                                              const double max_error,
                                              const double max_coord_norm) {

  std::unordered_map<size_t, Eigen::Vector3d>& points3D
    = feature_manager.points3D;
  std::unordered_map<size_t, Eigen::Vector2d>& points2D
    = feature_manager.points2D;
  std::unordered_map<size_t, size_t>& point2D_to_point3D
    = feature_manager.point2D_to_point3D;
  std::unordered_map<size_t, std::vector<size_t> >& point3D_to_points2D
    = feature_manager.point3D_to_points2D;
  std::unordered_map<size_t, std::vector<size_t> >&
    image_to_points2D = feature_manager.image_to_points2D;

  std::unordered_map<size_t, std::vector<Eigen::Vector3d>> point3D_to_colors;

  // Extract color for each point in all observed images
  for (size_t image_idx=0; image_idx<image_data.size(); image_idx++) {
    size_t image_id;
    try {
      image_id = get_image_id(image_idx);
    }
    catch(std::exception& e) {
      continue;
    }
    // Always read as color image, flag=1
    const cv::Mat image = image_data[image_idx].read(1);
    // Split image channels into separate color images in the order BGR
    cv::Mat image_rgb[3];
    cv::split(image, image_rgb);

    // Iterate through all 2D points in image
    const std::vector<size_t>& point2D_ids = image_to_points2D[image_id];
    for (size_t j=0; j<point2D_ids.size(); ++j) {
      const size_t point2D_id = point2D_ids[j];
      if (point2D_to_point3D.count(point2D_id)) {
        const size_t point3D_id = point2D_to_point3D[point2D_id];
        const Eigen::Vector2d& point2D = points2D[point2D_id];
        // Determine mean color around point in 3x3 window,
        // to account for image noise
        cv::Rect roi((size_t)point2D(0)-1, (size_t)point2D(1)-1, 3, 3);
        // OpenCV uses BGR color ordering
        const Eigen::Vector3d rgb(cv::mean(image_rgb[2](roi))[0] / 255.0,
                                  cv::mean(image_rgb[1](roi))[0] / 255.0,
                                  cv::mean(image_rgb[0](roi))[0] / 255.0);
        point3D_to_colors[point3D_id].push_back(rgb);
      }
    }
  }

  std::ofstream file;
  file.open(path.c_str());

  file << "#VRML V2.0 utf8\n";
  file << "Background { skyColor [1.0 1.0 1.0] } \n";
  file << "Shape{ appearance Appearance {\n";
  file << " material Material {emissiveColor 1 1 1} }\n";
  file << " geometry PointSet {\n";
  file << " coord Coordinate {\n";
  file << "  point [\n";

  std::vector<Eigen::Vector3d> point3D_to_color;

  for (auto it=points3D.begin(); it!=points3D.end(); ++it) {

    const size_t point3D_id = it->first;

    if (point3D_to_points2D[point3D_id].size() < min_track_len) {
      continue;
    }

    const Eigen::Vector3d& point3D = it->second;

    double error;
    try {
      error = get_point3D_error(point3D_id);
    }
    catch(std::exception& e) {
      continue;
    }

    if (error > max_error || point3D.norm() > max_coord_norm) {
      continue;
    }

    // Write point coordinates
    file << point3D(0) << ", "
         << point3D(1) << ", "
         << point3D(2) << std::endl;

    // Average colors of 3D point in all observed images
    std::vector<Eigen::Vector3d>& colors = point3D_to_colors[point3D_id];
    Eigen::Vector3d color(0, 0, 0);
    for (size_t i=0; i<colors.size(); ++i) {
      color += colors[i];
    }
    color /= colors.size();
    point3D_to_color.push_back(color);
  }

  file << " ] }\n";
  file << " color Color { color [\n";

  // Write color data for each point
  for (size_t i = 0; i <point3D_to_color.size(); ++i) {
    const Eigen::Vector3d color = point3D_to_color[i];
    file << color(0) << " " << color(1) << " " << color(2) << "\n";
  }

  file << " ] } } }\n";

  file.close();

}


void SequentialMapper::write_camera_models_vrml(const std::string& path,
                                                const double red,
                                                const double green,
                                                const double blue) {

  std::ofstream file;
  file.open(path.c_str());

  const auto& first_rvec = feature_manager.rvecs[get_image_id(first_image_idx_)];
  const auto& first_tvec = feature_manager.tvecs[get_image_id(first_image_idx_)];
  const auto& second_rvec = feature_manager.rvecs[get_image_id(second_image_idx_)];
  const auto& second_tvec = feature_manager.tvecs[get_image_id(second_image_idx_)];

  const Eigen::Vector3d baseline
    = invert_proj_matrix(compose_proj_matrix(first_rvec, first_tvec)).block<3, 1>(0, 3)
      - invert_proj_matrix(compose_proj_matrix(second_rvec, second_tvec)).block<3, 1>(0, 3);

  const double scale = 0.05 * baseline.norm();

  // Build camera base model at origin

  double six = scale * 1.5;
  double siy = scale;

  Eigen::Vector3d p1(-six, -siy, six*1.0*2.0);
  Eigen::Vector3d p2(+six, -siy, six*1.0*2.0);
  Eigen::Vector3d p3(+six, +siy, six*1.0*2.0);
  Eigen::Vector3d p4(-six, +siy, six*1.0*2.0);

  Eigen::Vector3d p5(0, 0, 0);
  Eigen::Vector3d p6(-six/3.0, -siy/3.0, six*1.0*2.0);
  Eigen::Vector3d p7(+six/3.0, -siy/3.0, six*1.0*2.0);
  Eigen::Vector3d p8(+six/3.0, +siy/3.0, six*1.0*2.0);
  Eigen::Vector3d p9(-six/3.0, +siy/3.0, six*1.0*2.0);

  std::vector<Eigen::Vector3d> points(9);
  points[0] = p1;
  points[1] = p2;
  points[2] = p3;
  points[3] = p4;
  points[4] = p5;
  points[5] = p6;
  points[6] = p7;
  points[7] = p8;
  points[8] = p9;

  file << "#VRML V2.0 utf8\n";

  for (const auto& rvec : feature_manager.rvecs) {

    file << "Shape{\n";
    file << " appearance Appearance {\n";
    file << "  material DEF Default-ffRffGffB Material {\n";
    file << "  ambientIntensity 0\n";
    file << "  diffuseColor " << " " << red << " "
                                     << green << " "
                                     << blue << "\n";
    file << "  emissiveColor 0.1 0.1 0.1 } }\n";
    file << " geometry IndexedFaceSet {\n";
    file << " solid FALSE \n";
    file << " colorPerVertex TRUE \n";
    file << " ccw TRUE \n";

    file << " coord Coordinate {\n";
    file << " point [\n";

    Eigen::Transform<double, 3, Eigen::Affine> transform;
    transform.matrix().block<3, 4>(0, 0)
      = invert_proj_matrix(compose_proj_matrix(rvec.second, feature_manager.tvecs[rvec.first]));

    // Move camera base model to camera pose
    for (size_t p=0; p<points.size(); p++) {
      Eigen::Vector3d pt = transform * points[p];
      file << std::setw(20) << pt(0)
           << std::setw(20) << pt(1)
           << std::setw(20) << pt(2)
           << "\n";
    }

    file << " ]\n }";

    file << "\n color Color {color [\n";
    for (size_t p=0; p<points.size(); p++) {
      file << " " << red << " " << green << " " << blue << "\n";
    }

    file << "\n] }\n";

    file << "\n coordIndex [\n";
    file << " 0, 1, 2, 3, -1\n";
    file << " 5, 6, 4, -1\n";
    file << " 6, 7, 4, -1\n";
    file << " 7, 8, 4, -1\n";
    file << " 8, 5, 4, -1\n";
    file << " \n] \n";

    file << " texCoord TextureCoordinate { point [\n";
    file << "  1 1,\n";
    file << "  0 1,\n";
    file << "  0 0,\n";
    file << "  1 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";

    file << " ] }\n";
    file << "} }\n";

  }

  file.close();

}


void SequentialMapper::write_camera_connections_vrml(const std::string& path) {

  std::ofstream file;
  file.open(path.c_str());

  const double eps = 1e-5;

  file << "#VRML V2.0 utf8\n";
  file << "Background { skyColor [1.0 1.0 1.0] } \n";

  for (auto it1=image_pair_idxs_.begin(); it1!=image_pair_idxs_.end(); ++it1) {

    const size_t image_id1 = get_image_id(it1->first);

    Eigen::Matrix<double, 3, 4> matrix1 = invert_proj_matrix(
      compose_proj_matrix(feature_manager.rvecs[image_id1],
                          feature_manager.tvecs[image_id1]));
    const Eigen::Vector3d tvec1 = matrix1.block<3, 1>(0, 3);

    for (auto it2=it1->second.begin(); it2!=it1->second.end(); ++it2) {

      const size_t image_id2 = get_image_id(*it2);

      Eigen::Matrix<double, 3, 4> matrix2 = invert_proj_matrix(
        compose_proj_matrix(feature_manager.rvecs[image_id2],
                            feature_manager.tvecs[image_id2]));
      const Eigen::Vector3d tvec2 = matrix2.block<3, 1>(0, 3);

      file << "Shape{\n";
      file << "appearance Appearance {\n";
      file << " material Material {emissiveColor 1 1 1} }\n";

      file << "geometry IndexedFaceSet {\n";
      file << " coord Coordinate {\n";
      file << "  point [\n";

      // Produce nearly degenerated polygon, as Meshlab is not able
      // to visualize IndexedLineSet

      file << "   " << tvec1(0) << " "
           << tvec1(1) << " "
           << tvec1(2) << std::endl;

      file << "   " << tvec1(0) << " "
           << tvec1(1) << " "
           << tvec1(2) + eps << std::endl;

      file << "   " << tvec2(0) << " "
           << tvec2(1) << " "
           << tvec2(2) << std::endl;

      file << "  ] }\n";

      file << " color Color {\n";
      file << "  color [\n";
      file << "   0 1.0 1.0\n";
      file << "  ] }\n";

      file << " coordIndex [\n";
      file << "  0, 1, 2, -1\n";
      file << " ]\n";

      file << " colorIndex [\n";
      file << "  0\n";
      file << " ]\n";

      file << "} }\n";

    }

  }

  file.close();

}


void SequentialMapper::write_tracks(std::string path,
                                    const size_t image_idx,
                                    const size_t max_num_points,
                                    const int radius, const int thickness) {

  path = ensure_trailing_slash(path);

  const size_t image_id = get_image_id(image_idx);

  std::vector<size_t> points2D = feature_manager.image_to_points2D[image_id];

  size_t num_points = 0;

  for (size_t i=0; i<points2D.size(); ++i) {

    if (num_points >= max_num_points) {
      return;
    }

    if (feature_manager.point2D_to_point3D[points2D[i]] == 0) {
      continue;
    }

   num_points += 1;

    const size_t point3D_id = feature_manager.point2D_to_point3D[points2D[i]];

    std::vector<size_t> point2D_ids
      = feature_manager.point3D_to_points2D[point3D_id];

    std::ostringstream point3D_id_str;
    point3D_id_str << point3D_id;

    std::ostringstream track_length_str;
    track_length_str << point2D_ids.size();

    for (size_t j=0; j<point2D_ids.size(); ++j) {

      size_t point2D_id = point2D_ids[j];

      for (size_t image_id=1; image_id<=feature_manager.get_num_images();
           image_id++) {

        const std::vector<size_t> image_points2D
          = feature_manager.image_to_points2D[image_id];

        if (std::find(image_points2D.begin(),
                      image_points2D.end(),
                      point2D_id)
            != image_points2D.end()) {

          const size_t image_idx = get_image_idx(image_id);

          // Read image
          cv::Mat image = image_data[image_idx].read();
          cv::Mat color_image;
          cv::cvtColor(image, color_image, cv::COLOR_GRAY2RGB);
          // Draw 2D point
          const Eigen::Vector2d& point2D
            = feature_manager.points2D[point2D_id];
          const cv::Point2f point2D_cv(point2D(0), point2D(1));
          cv::circle(color_image, point2D_cv, radius, cv::Scalar(0, 0, 255),
                     thickness);

          // Write to output path
          std::ostringstream image_id_str;
          image_id_str << image_id;
          cv::imwrite(path
                      + "LEN" + track_length_str.str()
                      + "-P3D#" + point3D_id_str.str()
                      + "-IMG#" + image_id_str.str() + ".jpg", color_image);
        }
      }
    }
  }
}


void SequentialMapper::query_and_prepare_features_(const size_t image_idx) {

  if (image_idx == prev_image_idx_) {

    prev_keypoints_ = curr_keypoints_;
    prev_points2D_ = curr_points2D_;
    prev_points2D_N_ = curr_points2D_N_;
    prev_descriptors_ = curr_descriptors_;

  } else if (image_idx == prev_prev_image_idx_) {

    std::swap(prev_keypoints_, curr_keypoints_);
    std::swap(prev_points2D_, curr_points2D_);
    std::swap(prev_points2D_N_, curr_points2D_N_);
    std::swap(prev_descriptors_, curr_descriptors_);

  } else {  // not in memory

    prev_keypoints_ = curr_keypoints_;
    prev_points2D_ = curr_points2D_;
    prev_points2D_N_ = curr_points2D_N_;
    prev_descriptors_ = curr_descriptors_;

    feature_cache_.query(image_idx, curr_keypoints_, curr_descriptors_);

    curr_points2D_ = cv2eigen_keypoints(curr_keypoints_);

    const size_t camera_id = get_camera_id(image_idx);
    const int camera_model_code = get_camera_model_code(image_idx);
    const std::vector<double>& camera_params
      = feature_manager.camera_params[camera_id];

    camera_model_image2world(curr_points2D_, curr_points2D_N_,
                             camera_model_code, camera_params.data());

  }

  prev_prev_image_idx_ = prev_image_idx_;
  prev_image_idx_ = image_idx;

}


void SequentialMapper::match_features_(const SequentialMapperOptions& options) {
  match_brute_force(prev_keypoints_, prev_descriptors_,
                    curr_keypoints_, curr_descriptors_, matches_,
                    true, options.match_max_ratio, options.match_max_distance, cv::NORM_L2);
}


void SequentialMapper::find_similar_images_(const size_t image_idx,
                                            const size_t num_images,
                                            std::vector<int>& image_idxs,
                                            std::vector<float>& scores) {

  // Make sure number of detected images does not exceed the number of
  // actually processed images in the mapper
  const size_t num_images_reduced = std::min(num_images, num_proc_images_);

  // Extract descriptors
  std::vector<cv::KeyPoint> _keypoints;
  cv::Mat descriptors;
  feature_cache_.query(image_idx, _keypoints, descriptors);

  // Detect similar images
  loop_detector_.query(descriptors, image_idxs, scores, num_images_reduced);

}
