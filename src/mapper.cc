/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

/**
 * This routine is intended for the reconstruction of sequential sets of
 * images, such as image sequences from UAVs.
 *
 * The mapper tries to sequentially reconstruct the camera poses and locally
 * adjusts the bundle over a local window of cameras. Loops are automatically
 * detected every N images. If the reconstruction of the image sequence breaks
 * in the middle, a new separate reconstruction is started for the remaining
 * images. After all images have been processed the routine tries to merge the
 * individual, separate reconstructed scenes. A final global bundle adjustment
 * is carried out at the end of the routine.
 *
 * Images with small feature disparity will be skipped in the initial
 * processing for more accurate triangulation and pose reconstruction. Skipped
 * images will be processed before the final global bundle adjustment.
 */


#include <vector>
#include <iomanip>
#include <iostream>
#include <exception>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <glog/logging.h>

#include "base3d/projection.h"
#include "sfm/sequential_mapper.h"
#include "util/io.h"
#include "util/timer.h"


namespace config = boost::program_options;


// State codes of camera poses in the local bundle adjustment
#define LOCAL_BA_POSE_FREE       0
#define LOCAL_BA_POSE_FIXED      1
#define LOCAL_BA_POSE_FIXED_X    2


void print_report_heading(const std::vector<Image>& image_data,
                          const size_t image_idx) {
  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Processing image #" << image_idx
            << " (" << image_data[image_idx].name << ")" << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;
}


void print_report_summary(SequentialMapper& mapper,
                          const size_t image_idx) {

  const size_t image_id = mapper.get_image_id(image_idx);

  Eigen::Matrix<double, 3, 4> matrix = invert_proj_matrix(
    compose_proj_matrix(mapper.feature_manager.rvecs[image_id],
                        mapper.feature_manager.tvecs[image_id]));
  const Eigen::Vector3d tvec = matrix.block<3, 1>(0, 3);

  std::cout << "Global position" << std::endl;
  std::cout << "---------------" << std::endl;
  std::cout << std::setw(15)
            << tvec(0) << std::endl
            << std::setw(15)
            << tvec(1) << std::endl
            << std::setw(15)
            << tvec(2) << std::endl
            << std::endl;

}


void write_mapper(SequentialMapper& mapper,
                  const SequentialMapperOptions& mapper_options,
                  const std::string& output_path,
                  const std::string& suffix) {

  mapper.write_image_data(output_path + "image-data-" + suffix + ".txt");

  mapper.write_point_cloud_data(output_path + "point-cloud-data-"
                                + suffix + ".txt");

  mapper.write_camera_models_vrml(output_path + "camera-models-" + suffix + ".wrl", 1, 0, 0);

  mapper.write_point_cloud_vrml(output_path + "point-cloud-min-track-len-2-" + suffix + ".wrl",
                                2, mapper_options.tri_max_reproj_error / 5.0);
  mapper.write_point_cloud_vrml(output_path + "point-cloud-min-track-len-3-" + suffix + ".wrl",
                                3, mapper_options.tri_max_reproj_error / 5.0);
  const size_t min_track_len
    = std::min<size_t>(3 * mapper_options.min_track_len,
                       mapper.get_num_proc_images() / 2);
  mapper.write_point_cloud_vrml(output_path + "point-cloud-" + suffix + ".wrl",
                                min_track_len, mapper_options.tri_max_reproj_error / 5.0);
  mapper.write_point_cloud_vrml(output_path + "point-cloud-all-" + suffix + ".wrl",
                                0, mapper_options.tri_max_reproj_error);

  mapper.write_camera_connections_vrml(output_path + "camera-connections-"
                                       + suffix + ".wrl");

}


void adjust_local_bundle(SequentialMapper& mapper,
                         const std::list<size_t>& image_idxs,
                         const std::vector<int>& local_ba_pose_state,
                         const BundleAdjustmentOptions& ba_options) {

  if (image_idxs.size() > local_ba_pose_state.size()) {
    throw std::range_error("Number of local images in `image_idxs` must not "
                           "be greater than number of poses defined in "
                           "`local_ba_pose_state`.");
  }

  std::vector<size_t> free_image_idxs;
  std::vector<size_t> fixed_image_idxs;
  std::vector<size_t> fixed_x_image_idxs;

  // Set parameters of image poses as constant or variable
  size_t i=0;
  for (auto it=image_idxs.begin(); it!=image_idxs.end(); ++it, ++i) {
    switch(local_ba_pose_state[i]) {
      case LOCAL_BA_POSE_FREE:
        free_image_idxs.push_back(*it);
        break;
      case LOCAL_BA_POSE_FIXED:
        fixed_image_idxs.push_back(*it);
        break;
      case LOCAL_BA_POSE_FIXED_X:
        fixed_x_image_idxs.push_back(*it);
        break;
    }
  }

  mapper.adjust_bundle(free_image_idxs, fixed_image_idxs, fixed_x_image_idxs,
                       ba_options);

}


void adjust_global_bundle(SequentialMapper& mapper,
                          BundleAdjustmentOptions ba_global_options) {

  // Mapper is empty, nothing to adjust
  if (mapper.get_num_proc_images() == 0) {
    return;
  }

  Timer timer;
  timer.start();

  // Globally adjust all poses with fixed initial two poses

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Global bundle adjustment" << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;

  ba_global_options.update_point3D_errors = true;
  ba_global_options.print_progress = true;
  ba_global_options.function_tolerance = 1e-6;
  ba_global_options.gradient_tolerance = 1e-10;
  ba_global_options.max_num_iterations = 200;

  mapper.adjust_global_bundle(ba_global_options);

  timer.print();

}


size_t detect_loop(SequentialMapper& mapper, const size_t image_idx,
                   const config::variables_map& vmap,
                   SequentialMapperOptions mapper_options,
                   const BundleAdjustmentOptions& ba_options) {

  // Do nothing if loop detection is disabled
  if (!vmap["loop-detection"].as<bool>()) {
    return 0;
  }

  // Adjust parameters
  mapper_options.max_homography_inliers = 1;
  mapper_options.min_track_len = 2;

  Timer timer;

  timer.start();

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Trying to detect loop for image #" << image_idx << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;

  const size_t num_successes
    = mapper.detect_loop(image_idx,
                         vmap["loop-detection-num-images"].as<size_t>(),
                         vmap["loop-detection-num-nh-images"].as<size_t>(),
                         vmap["loop-detection-nh-dist"].as<size_t>(),
                         mapper_options, ba_options);

  std::cout << "Successfully closed loop to " << num_successes
            << " images." << std::endl << std::endl;

  return num_successes;

}


void process_remaining_images(SequentialMapper& mapper,
                              const config::variables_map& vmap,
                              SequentialMapperOptions mapper_options,
                              const BundleAdjustmentOptions& ba_options) {

  const size_t start_image_idx = mapper.get_min_image_idx();
  const size_t end_image_idx = std::min(std::min(mapper.image_data.size() - 1,
                                                 vmap["end-image-idx"].as<size_t>() - 1),
                                        mapper.get_max_image_idx() + vmap["max-subsequent-trials"].as<size_t>());

  const size_t loop_detection_period = vmap["loop-detection-period"].as<size_t>();

  // Adjust parameters
  mapper_options.max_homography_inliers = 1;
  mapper_options.min_disparity = 0;
  mapper_options.min_track_len = 2;

  Timer timer;

  for (size_t image_idx=start_image_idx+1;
       image_idx<=end_image_idx; ++image_idx) {

    if (!mapper.is_image_processed(image_idx)) {

      // Find nearest processed images, whose pose is already estimated
      // (previous and next in image chain)
      size_t prev_proc_image_idx = image_idx - 1;
      size_t next_proc_image_idx;
      for (next_proc_image_idx=image_idx+1;
           next_proc_image_idx<=end_image_idx;
           ++next_proc_image_idx) {
        if (mapper.is_image_processed(next_proc_image_idx)) {
          break;
        }
      }
      if (next_proc_image_idx == end_image_idx) {
        next_proc_image_idx = UINT_MAX;
      }

      // Process skipped images and use previous or next nearest processed
      // images as matching "partner"
      for (; image_idx<next_proc_image_idx && image_idx<=end_image_idx;
           ++image_idx) {

        if (mapper.is_image_processed(image_idx)) {
          continue;
        }

        size_t prev_dist = image_idx - prev_proc_image_idx;
        size_t next_dist = next_proc_image_idx - image_idx;
        size_t partner_image_idx;
        if (prev_dist < next_dist) {
          partner_image_idx = prev_proc_image_idx;
        } else {
          partner_image_idx = next_proc_image_idx;
        }

        timer.restart();
        print_report_heading(mapper.image_data, image_idx);

        if (!mapper.process(image_idx, partner_image_idx, mapper_options, ba_options)) {
          mapper.detect_loop(image_idx, 5, 1, SIZE_MAX, mapper_options, ba_options);
        }

        // Run loop detection every 15 frames
        if (mapper.is_image_processed(image_idx)) {
          print_report_summary(mapper, image_idx);
          timer.print();

          if (mapper.get_num_proc_images() % loop_detection_period == 0) {
            detect_loop(mapper, image_idx, vmap, mapper_options, ba_options);
          }
        } else {
          timer.print();
        }
      }
    }
  }
}


void merge_mappers(std::vector<SequentialMapper*>& mappers,
                   const size_t num_skip_images,
                   const config::variables_map& vmap,
                   const SequentialMapperOptions& mapper_options,
                   const BundleAdjustmentOptions& ba_options,
                   const BundleAdjustmentOptions& ba_global_options,
                   const std::string& output_path) {

  if (mappers.size() < 2) {
    return;
  }

  bool merge_success = true;

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Starting merge routine for " << mappers.size()
            << " mappers" << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;

  size_t num_deleted = 0;

  while (merge_success) {

    merge_success = false;

    for (size_t i=0; i<mappers.size(); ++i) {
      for (size_t j=0; j<i; ++j) {

        // It is faster to merge the smaller mapper into the larger one
        if (mappers[i]->get_num_proc_images()
            < mappers[j]->get_num_proc_images()) {
          std::swap(mappers[i], mappers[j]);
        }

        std::cout << "Trying to merge mapper #" << i
                  << " and #" << j << std::endl << std::endl;

        if (mappers[i]->merge(*mappers[j],
                              15,  // num_similar_images
                              num_skip_images,
                              mapper_options,
                              ba_options)) {

          std::cout << "Successfully merged mapper #" << i
                    << " and #" << j << std::endl << std::endl;

          num_deleted += 1;

          std::string suffix = "merged-" + boost::lexical_cast<std::string>(num_deleted);
          write_mapper(*mappers[j], mapper_options, output_path, suffix);

          delete mappers[j];

          process_remaining_images(*mappers[i], vmap, mapper_options,
                                   ba_options);

          adjust_global_bundle(*mappers[i], ba_global_options);

          mappers.erase(mappers.begin() + j);
          merge_success = true;

          break;

        } else {

          std::cout << "Failed to merge mapper #" << i
                    << " and #" << j << std::endl << std::endl;

        }
      }
      if (merge_success) {
        break;
      }
    }
  }

}


void filter_point_cloud(SequentialMapper& mapper, BundleAdjustmentOptions& ba_global_options,
                        const double filter_max_error,
                        const std::set<size_t>& keep_point3D_ids=std::set<size_t>()) {
  std::list<size_t> filter_list;
  for (const auto point3D : mapper.feature_manager.points3D) {
    if (keep_point3D_ids.count(point3D.first) == 0
        && mapper.get_point3D_error(point3D.first) > filter_max_error) {
      filter_list.push_back(point3D.first);
    }
  }

  for (auto point3D_id : filter_list) {
    mapper.feature_manager.delete_point3D(point3D_id);
  }

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Filtered " << filter_list.size() << " 3D points" << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;

}


void apply_control_points(SequentialMapper& mapper, const std::string& output_file_path,
                          const std::vector<ControlPoint>& control_point_data,
                          BundleAdjustmentOptions ba_global_options, const double filter_max_error) {

  Timer timer;
  timer.start();

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Applying control points" << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;

  // Extract valid GCPs for mapper, i.e. a GCP is observed from at least 2 images
  size_t num_fixed_control_points = 0;
  std::vector<ControlPoint> valid_control_points;
  for (const auto& control_point : control_point_data) {
    std::vector<std::pair<size_t, Eigen::Vector2d>> valid_points2D;
    for (const auto& points2D : control_point.points2D) {
      if (mapper.is_image_processed(points2D.first)) {
        valid_points2D.push_back(points2D);
      }
    }
    if (valid_points2D.size() >= 2) {
      auto valid_control_point = control_point;
      valid_control_point.points2D = valid_points2D;
      valid_control_points.push_back(valid_control_point);
      if (valid_control_point.fixed) {
        num_fixed_control_points += 1;
      }
    }
  }

  std::vector<std::pair<size_t, ControlPoint>> control_points;

  if (valid_control_points.size() < 3) {
    std::cout << "Cannot apply control points to mapper - skipping. Need at least 3 valid GCPs "
                 "(fixed control points) with sufficient observations." << std::endl;

    if (filter_max_error > 0) {
      filter_point_cloud(mapper, ba_global_options, filter_max_error);
      adjust_global_bundle(mapper, ba_global_options);
    }
  } else {
    // Insert GCPs into feature manager
    for (const auto& control_point : valid_control_points) {

      auto& prev_observation = control_point.points2D[0];
      size_t prev_image_id = mapper.get_image_id(prev_observation.first);
      size_t prev_point2D_idx = mapper.feature_manager.image_to_points2D[prev_image_id].size();
      mapper.feature_manager.add_point2D(prev_image_id, prev_observation.second);

      for (size_t i=1; i<control_point.points2D.size(); ++i) {
        const auto& points2D = control_point.points2D[i];
        const size_t image_id = mapper.get_image_id(points2D.first);
        const size_t point2D_idx = mapper.feature_manager.image_to_points2D[image_id].size();
        mapper.feature_manager.add_point2D(image_id, points2D.second);
        const size_t point3D_id
          = mapper.feature_manager.add_correspondence(prev_image_id, image_id,
                                                      prev_point2D_idx, point2D_idx);

        // Triangulate (only once)
        if (mapper.feature_manager.point3D_to_points2D[point3D_id].size() == 2) {
          control_points.push_back(std::make_pair(point3D_id, control_point));

          // Extract projection matrices
          const Eigen::Matrix<double, 3, 4> prev_proj_matrix
            = compose_proj_matrix(mapper.feature_manager.rvecs[prev_image_id],
                                  mapper.feature_manager.tvecs[prev_image_id]);
          const Eigen::Matrix<double, 3, 4> proj_matrix
            = compose_proj_matrix(mapper.feature_manager.rvecs[image_id],
                                  mapper.feature_manager.tvecs[image_id]);

          // Normalize 2D observations
          const size_t prev_camera_id = mapper.get_camera_id(prev_observation.first);
          const int prev_camera_model_code = mapper.get_camera_model_code(prev_observation.first);
          const std::vector<double>& prev_camera_params
            = mapper.feature_manager.camera_params[prev_camera_id];

          std::vector<Eigen::Vector2d> prev_points(1), prev_points_N(1);
          prev_points[0] = prev_observation.second;
          camera_model_image2world(prev_points, prev_points_N,
                                   prev_camera_model_code, prev_camera_params.data());

          const size_t camera_id = mapper.get_camera_id(points2D.first);
          const int camera_model_code = mapper.get_camera_model_code(points2D.first);
          const std::vector<double>& camera_params
            = mapper.feature_manager.camera_params[camera_id];
          std::vector<Eigen::Vector2d> points(1), points_N(1);
          points[0] = points2D.second;
          camera_model_image2world(points, points_N,
                                   camera_model_code, camera_params.data());

          // Triangulate
          const Eigen::Vector3d point3D
            = triangulate_point(prev_proj_matrix, proj_matrix, prev_points_N[0], points_N[0]);
          mapper.feature_manager.set_point3D(point3D_id, point3D);
        }

        prev_image_id = image_id;
        prev_point2D_idx = point2D_idx;
      }
    }

    // Estimate similarity transform from mapper to control point system
    std::set<size_t> gcp_ids;
    Eigen::MatrixXd src(num_fixed_control_points, 3);
    Eigen::MatrixXd dst(num_fixed_control_points, 3);
    size_t i =0;
    for (const auto& control_point : control_points) {
      if (control_point.second.fixed) {
        src.row(i) = mapper.feature_manager.points3D[control_point.first];
        dst.row(i) = control_point.second.xyz;
        gcp_ids.insert(control_point.first);
        i += 1;
      }
    }

    SimilarityTransform3D mapper_to_gcp_transform;
    mapper_to_gcp_transform.estimate(src, dst);

    // Transform mapper to ground control point system
    for (auto& point3D : mapper.feature_manager.points3D) {
      mapper_to_gcp_transform.transform_point(point3D.second);
    }
    for (auto& el : mapper.feature_manager.rvecs) {
      mapper_to_gcp_transform.transform_pose(el.second, mapper.feature_manager.tvecs[el.first]);
    }

    // Set pre-defined control point coordinates
    // (they were previously set with the triangulated value and then transformed)
    for (const auto& control_point : control_points) {
      mapper.feature_manager.set_point3D(control_point.first, control_point.second.xyz);
    }

    // Globally adjust all poses all camera poses as free
    ba_global_options.update_point3D_errors = true;
    ba_global_options.print_progress = true;
    ba_global_options.function_tolerance = 1e-6;
    ba_global_options.gradient_tolerance = 1e-10;
    ba_global_options.max_num_iterations = 200;

    if (filter_max_error > 0) {
      std::set<size_t> keep_point3D_ids;
      for (const auto& control_point : control_points) {
        keep_point3D_ids.insert(control_point.first);
      }
      filter_point_cloud(mapper, ba_global_options, filter_max_error, keep_point3D_ids);
    }

    mapper.adjust_global_bundle_gcp(gcp_ids, ba_global_options);

    write_control_point_data(output_file_path, mapper, control_points);
  }

  timer.print();
}


int main(int argc, char* argv[]) {

  google::InitGoogleLogging(argv[0]);

  // Program options
  std::string input_path;
  std::string output_path;
  std::string cache_path;
  std::string voc_tree_path;
  std::string image_prefix;
  std::string image_suffix;
  std::string image_ext;
  size_t start_image_idx, end_image_idx;
  int first_image_idx, second_image_idx;
  bool debug;
  std::string debug_path;

  // Feature detection and extraction options
  SURFOptions surf_options;

  // Sequential mapper options
  SequentialMapperOptions init_mapper_options;
  SequentialMapperOptions mapper_options;

  // Bundle adjustment options
  BundleAdjustmentOptions ba_options;
  BundleAdjustmentOptions ba_local_options;
  BundleAdjustmentOptions ba_global_options;

  // Processing options
  bool process_curr_prev_prev;

  // Failure options
  size_t max_subsequent_trials;
  size_t failure_max_image_dist;
  size_t failure_skip_images;

  // Loop detection options
  bool loop_detection;
  size_t loop_detection_num_images;
  size_t loop_detection_num_nh_images;
  size_t loop_detection_nh_dist;
  size_t loop_detection_period;

  // Bundle adjustment options
  size_t local_ba_window_size;

  bool merge;
  size_t merge_num_skip_images;

  bool use_control_points;
  std::string control_point_data_path;

  double filter_max_error;

  config::variables_map vmap;

  try {
    config::options_description options_description("Options");
    options_description.add_options()
      ("help,h",
       "Print this help message.")

      // Path options
      ("input-path",
       config::value<std::string>(&input_path)
         ->required(),
       "Path to imagedata.txt and image files.")
      ("output-path",
       config::value<std::string>(&output_path)
         ->required(),
       "Path to output files.")
      ("cache-path",
       config::value<std::string>(&cache_path)
         ->required(),
       "Path to cache files.")
      ("voc-tree-path",
       config::value<std::string>(&voc_tree_path)
         ->required(),
       "Path to vocabulary tree.")

      // Image filename options
      ("image-prefix",
       config::value<std::string>(&image_prefix)
         ->default_value(""),
       "Prefix of image file names before name.")
      ("image-suffix",
       config::value<std::string>(&image_suffix)
         ->default_value(""),
       "Suffix of image file names after name.")
      ("image-ext",
       config::value<std::string>(&image_ext)
         ->default_value(".bmp"),
       "Image file name extension.")

      // Start and end image index options
      ("start-image-idx",
       config::value<size_t>(&start_image_idx)
         ->default_value(0),
       "Index of first image to be processed (position in image data file).")
      ("end-image-idx",
       config::value<size_t>(&end_image_idx)
         ->default_value(UINT_MAX),
       "Index of last image to be processed (position in image data file).")
      ("first-image-idx",
       config::value<int>(&first_image_idx)
         ->default_value(-1, "auto"),
       "Index of first image index of the initial pair. This is useful if the "
       "automatically chosen initial pair yields bad reconstruction results. "
       "Default is -1 and determines the image automatically.")
      ("second-image-idx",
       config::value<int>(&second_image_idx)
         ->default_value(-1, "auto"),
       "Index of second image index of the initial pair. This is useful if "
       "the automatically chosen initial pair yields bad reconstruction "
       "results. Default is -1 and determines the image automatically.")

      // Debug options
      ("debug",
       config::value<bool>(&debug)
         ->default_value(false, "false"),
       "Enable debug mode.")
      ("debug-path",
       config::value<std::string>(&debug_path)
         ->default_value(""),
       "Path to debug output files.")

      // Feature detection and extraction options
      ("surf-hessian-threshold",
       config::value<double>(&surf_options.hessian_threshold)
         ->default_value(1000),
       "Hessian threshold for SURF feature detection.")
      ("surf-num-octaves",
       config::value<size_t>(&surf_options.num_octaves)
         ->default_value(4),
       "The number of a gaussian pyramid octaves for the SURF detector.")
      ("surf-num-octave-layers",
       config::value<size_t>(&surf_options.num_octave_layers)
         ->default_value(3),
       "The number of images within each octave of a gaussian pyramid "
       "for the SURF detector.")
      ("surf-adaptive",
       config::value<bool>(&surf_options.adaptive)
         ->default_value(true, "true"),
       "Whether to use adaptive gridded SURF feature detection, which splits "
       "the image in grid of sub-images and detects features for each grid "
       "cell separately in order to ensure an evenly distributed number of "
       "features in each region of the image.")
      ("surf-adaptive-min-per-cell",
       config::value<size_t>(&surf_options.adaptive_min_per_cell)
         ->default_value(100),
       "Minimum number of features per grid cell.")
      ("surf-adaptive-max-per-cell",
       config::value<size_t>(&surf_options.adaptive_max_per_cell)
         ->default_value(300),
       "Maximum number of features per grid cell.")
      ("surf-adaptive-cell-rows",
       config::value<size_t>(&surf_options.adaptive_cell_rows)
         ->default_value(3),
       "Number of grid cells in the first dimension of the image. The total "
       "number of grid cells is defined as `surf-cell-rows` x "
       "`surf-cell-cols`.")
      ("surf-adaptive-cell-cols",
       config::value<size_t>(&surf_options.adaptive_cell_cols)
         ->default_value(3),
       "Number of grid cells in the first dimension of the image. The total "
       "number of grid cells is defined as `surf-cell-rows` x "
       "`surf-cell-cols`.")

      // Sequential mapper options
      //    General

      ("init-max-homography-inliers",
       config::value<double>(&init_mapper_options.max_homography_inliers)
         ->default_value(0.7, "0.7"),
       "Maximum allow relative number (0-1) of inliers in homography between "
       "two images in order to guarantee sufficient view-point change. "
       "Larger values result in requiring larger view-point changes.")
      ("init-min-disparity",
       config::value<double>(&init_mapper_options.min_disparity)
         ->default_value(0, "0"),
       "Minimum median feature disparity between "
       "two images in order to guarantee sufficient view-point change.")
      ("max-homography-inliers",
       config::value<double>(&mapper_options.max_homography_inliers)
         ->default_value(0.8, "0.8"),
       "Maximum allow relative number (0-1) of inliers in homography between "
       "two images in order to guarantee sufficient view-point change. "
       "Larger values result in requiring larger view-point changes.")
      ("min-disparity",
       config::value<double>(&mapper_options.min_disparity)
         ->default_value(0, "0"),
       "Minimum median feature disparity between "
       "two images in order to guarantee sufficient view-point change.")
      ("match-max-ratio",
       config::value<double>(&mapper_options.match_max_ratio)
         ->default_value(0.9, "0.9"),
       "Maximum distance ratio between first and second best matches.")
      ("match-max-distance",
       config::value<double>(&mapper_options.match_max_distance)
         ->default_value(-1, "-1"),
       "Maximum distance in pixels for valid correspondence.")
      ("min-track-len",
       config::value<size_t>(&mapper_options.min_track_len)
         ->default_value(3),
       "Minimum track length of a 3D point to be used for 2D-3D pose "
       "estimation. This threshold takes effect when the number of "
       "successfully processed images is > `2 * min-track-len`.")
      ("final-cost-threshold",
       config::value<double>(&mapper_options.final_cost_threshold)
         ->default_value(2),
       "Threshold for final cost of pose refinement.")
      ("loss-scale-factor",
       config::value<double>(&ba_options.loss_scale_factor)
         ->default_value(1),
       "Scale factor of robust Cauchy loss function for pose refinement and "
       "bundle adjustment.")
      ("tri-max-reproj-error",
       config::value<double>(&mapper_options.tri_max_reproj_error)
         ->default_value(4),
       "Maximum reprojection error for newly triangulated points to be saved.")
      ("init-tri-min-angle",
       config::value<double>(&init_mapper_options.tri_min_angle)
         ->default_value(10),
       "Minimum (or maximum as 180 - angle) angle between two rays of a newly "
       "triangulated point.")
      ("tri-min-angle",
       config::value<double>(&mapper_options.tri_min_angle)
         ->default_value(1),
       "Minimum (or maximum as 180 - angle) angle between two rays of a newly "
       "triangulated point.")
      ("ransac-min-inlier-stop",
       config::value<double>(&mapper_options.ransac_min_inlier_stop)
           ->default_value(0.6, "0.6"),
       "RANSAC algorithm for 2D-3D pose estimation stops when at least this "
       "number of inliers is found, as relative (<1) w.r.t. total number of "
       "features or absolute (>1) value.")
      ("ransac-max-reproj-error",
       config::value<double>(&mapper_options.ransac_max_reproj_error)
         ->default_value(4),
       "Maximum reprojection used for RANSAC.")
      ("ransac-min-inlier-threshold",
       config::value<double>(&mapper_options.ransac_min_inlier_threshold)
         ->default_value(30),
       "Processing of image pair fails if less than this number of inliers is "
       "found in the RANSAC 2D-3D pose estimation, as relative (<1) w.r.t. "
       "total number of features or absolute (>1) value.")

      // Processing options
      ("process-curr-prev-prev",
       config::value<bool>(&process_curr_prev_prev)
         ->default_value(true, "true"),
       "Whether to subsequently process current image not only against "
       "previous image but also against image before previous image.")

      // Failure options
      ("max-subsequent-trials",
       config::value<size_t>(&max_subsequent_trials)
         ->default_value(30),
       "Maximum number of times to skip subsequent images due to failed "
       "processing.")
      ("failure-max-image-dist",
       config::value<size_t>(&failure_max_image_dist)
         ->default_value(10),
       "If subsequent processing fails (after `max-subsequent-trials`) this "
       "routine tries to find a valid image pair by trying to process all "
       "possible combinations in the range "
       "[`last-image-idx - dist`; `last-image-idx + dist`].")
      ("failure-skip-images",
       config::value<size_t>(&failure_skip_images)
         ->default_value(1),
       "If all trials to find a valid image pair failed, a new mapper is "
       "created. This mapper starts to process images at image index "
       "`last-image-idx + failure-skip-images`.")

      // Loop detection options
      ("loop-detection",
       config::value<bool>(&loop_detection)
         ->default_value(true, "true"),
       "Whether to enable loop detection.")
      ("loop-detection-num-images",
       config::value<size_t>(&loop_detection_num_images)
         ->default_value(30),
       "Maximum number of most similar images to test for loop closure.")
      ("loop-detection-num-nh-images",
       config::value<size_t>(&loop_detection_num_nh_images)
         ->default_value(15),
       "Maximum number of most similar images in direct neighborhood of "
       "current image to test for loop closure.")
      ("loop-detection-nh-dist",
       config::value<size_t>(&loop_detection_nh_dist)
         ->default_value(30),
       "Distance which determines neighborhood of current image as "
       "[`curr-image-idx - dist`; `curr-image-idx + dist`]")
      ("loop-detection-period",
       config::value<size_t>(&loop_detection_period)
         ->default_value(20),
       "Loop detection is initiated every `loop-period` successfully "
       "processed images.")

      // Bundle adjustment options
      ("local-ba-window-size",
       config::value<size_t>(&local_ba_window_size)
         ->default_value(8),
       "Window size of the local bundle adjustment. The last N poses are "
       "adjusted after each successful pose reconstruction.")
      ("constrain-rotation",
       config::value<bool>(&ba_global_options.constrain_rotation)
         ->default_value(false, "false"),
       "Whether to constrain poses against given poses in `imagedata.txt` "
       "in the global bundle adjustment.")
      ("constrain-rotation-weight",
       config::value<double>(&ba_global_options.constrain_rotation_weight)
         ->default_value(0),
       "Weight for constraint residual of rotation.")
      ("refine-camera-params",
       config::value<bool>(&ba_global_options.refine_camera_params)
         ->default_value(true, "true"),
       "Whether to refine the camera parameters in global bundle adjustment.")
      ("local-ba-refine-camera-params",
       config::value<bool>(&ba_local_options.refine_camera_params)
         ->default_value(true, "true"),
       "Whether to refine the camera parameters in the local bundle "
       "adjustment.")


      // Merge routine options
      ("merge",
       config::value<bool>(&merge)
         ->default_value(true, "true"),
       "Whether to try to merge separate mappers.")
      ("merge-num-skip-images",
       config::value<size_t>(&merge_num_skip_images)->default_value(5),
       "Merge routine searches every `merge-num-skip-images` frames for "
       "a common image in the separate mappers.")

      // Control point options
      ("use-control-points",
       config::value<bool>(&use_control_points)
         ->default_value(false, "false"),
       "Whether to estimate GCP coordinates and to transform model to GCP system.")
      ("control-point-data-path",
       config::value<std::string>(&control_point_data_path)->default_value(""),
       "Path to GCP data.")

      // Filter options
      ("filter-max-error",
       config::value<double>(&filter_max_error)
         ->default_value(-1),
       "If >0 re-run bundle adjustment with 3D points filtered whose mean residual is larger "
       "than this threshold.");

    // Use the same loss-scale factor for global BA
    ba_local_options.loss_scale_factor = ba_options.loss_scale_factor;
    ba_global_options.loss_scale_factor = ba_options.loss_scale_factor;

    config::store(config::parse_command_line(argc, argv, options_description),
                  vmap);

    if (vmap.count("help")) {
      std::cout << options_description << std::endl;
      return 1;
    }

    vmap.notify();

  }
  catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << "Unknown error!" << std::endl;
    return 1;
  }

  if ((first_image_idx == -1) ^ (second_image_idx == -1)) {
    throw std::invalid_argument("You must specify both `first-image-idx` and "
                                "`second-image-idx`.");
  }

  input_path = ensure_trailing_slash(input_path);
  output_path = ensure_trailing_slash(output_path);


  // Check paths

  if (!boost::filesystem::exists(input_path + "imagedata.txt")) {
    std::cerr << "`imagedata.txt` does not exist." << std::endl;
    return 1;
  }
  if (!boost::filesystem::exists(output_path)) {
    std::cerr << "`output-path` does not exist." << std::endl;
    return 1;
  }
  if (!boost::filesystem::exists(cache_path)) {
    std::cerr << "`cache-path` does not exist." << std::endl;
    return 1;
  }
  if (!boost::filesystem::exists(debug_path)) {
    std::cerr << "`debug-path` does not exist." << std::endl;
    return 1;
  }
  if (use_control_points && !boost::filesystem::exists(control_point_data_path)) {
    std::cerr << "`control-point-data-path` does not exist." << std::endl;
    return 1;
  }


  // Read input data

  std::vector<Image> image_data
    = read_image_data(input_path + "imagedata.txt", input_path,
                      image_prefix, image_suffix, image_ext);

  std::vector<SequentialMapper*> mappers;
  SequentialMapper* mapper
    = new SequentialMapper(image_data,
                           cache_path,
                           voc_tree_path,
                           surf_options,
                           loop_detection,
                           debug,
                           debug_path);
  mappers.push_back(mapper);

  // Local bundle adjustment window
  std::vector<int> local_ba_pose_state(local_ba_window_size,
                                       LOCAL_BA_POSE_FREE);
  local_ba_pose_state[0] = LOCAL_BA_POSE_FIXED;
  local_ba_pose_state[1] = LOCAL_BA_POSE_FIXED;

  // Last image to be processed
  end_image_idx = std::min(end_image_idx, image_data.size() - 1);

  // Dynamically adjusted minimum track length for 2D-3D pose estimation
  // for the first poses
  const size_t custom_min_track_len = mapper_options.min_track_len;

  Timer timer;
  Timer timer_total;

  size_t image_idx = -1;
  size_t prev_image_idx = -1;
  size_t prev_prev_image_idx = -1;

  // Image list for local bundle adjustment
  std::list<size_t> ba_image_idxs;

  timer_total.start();

  for (image_idx=start_image_idx; image_idx<=end_image_idx; ++image_idx) {

    timer.restart();

    if (mapper->get_num_proc_images() == 0) {  // Initial processing

      if (first_image_idx == -1) {
        first_image_idx = image_idx;
      }

      print_report_heading(image_data, first_image_idx);

      // Search for good initial pair with sufficiently large disparity
      if (second_image_idx == -1) {
        for (image_idx=image_idx+1; image_idx<=end_image_idx; ++image_idx) {
          if (mapper->process_initial(first_image_idx, image_idx,
                                      init_mapper_options)) {
            break;
          }
        }
        if (mapper->get_num_proc_images() == 0) {
          throw std::runtime_error("Could not find good initial pair.");
        }
      } else {
        image_idx = second_image_idx;
        if (!mapper->process_initial(first_image_idx, image_idx,
                                     init_mapper_options)) {
          throw std::runtime_error("Manually specified initial pair could "
                                   "not be processed successfully.");
        }
      }

      print_report_summary(*mapper, first_image_idx);

      print_report_heading(image_data, image_idx);

      // Adjust first two poses
      std::vector<size_t> free_image_idxs;
      std::vector<size_t> fixed_image_idxs;
      std::vector<size_t> fixed_x_image_idxs;
      fixed_image_idxs.push_back(first_image_idx);
      fixed_x_image_idxs.push_back(image_idx);
      BundleAdjustmentOptions init_ba_options = ba_options;
      init_ba_options.update_point3D_errors = false;
      init_ba_options.min_track_len = 2;
      init_ba_options.refine_camera_params = false;
      mapper->adjust_bundle(free_image_idxs,
                            fixed_image_idxs,
                            fixed_x_image_idxs, init_ba_options);

      // Add to local bundle adjustment
      ba_image_idxs.push_back(mapper->get_first_image_idx());
      ba_image_idxs.push_back(mapper->get_second_image_idx());

      // Use automatic initial pair selection for next mapper, if the first
      // mapper cannot be continued
      first_image_idx = -1;
      second_image_idx = -1;

      print_report_summary(*mapper, image_idx);

    } else if (mapper->get_num_proc_images() >= 2) {  // Sequential processing

      print_report_heading(image_data, image_idx);

      // Increase minimum track length for 2D-3D pose estimation
      if (mapper->get_num_proc_images() < 2 * mapper_options.min_track_len) {
        mapper_options.min_track_len = 2;
      } else {
        mapper_options.min_track_len = custom_min_track_len;
      }

      // Try to process image
      bool success = false;
      for (size_t t=0; t<max_subsequent_trials; ++t) {
        if ((success = mapper->process(image_idx, prev_image_idx,
                                       mapper_options, ba_options))) {
          break;
        }
        image_idx += 1;
        if (image_idx > end_image_idx) {
          break;
        }
        timer.print();
        timer.restart();
        print_report_heading(image_data, image_idx);
      }
      if (image_idx <= end_image_idx) {

        if (!success) {
          // If not successful with previous image, try to process against other
          // arbitrary images
          // nh-distance=inf, num-nh-images=1: Stop after one successful image
          success = 0 < mapper->detect_loop(image_idx, 30, 1, SIZE_MAX,
                                            mapper_options, ba_options);
        }

        if (success) {

          if (process_curr_prev_prev && prev_prev_image_idx != -1) {
            SequentialMapperOptions mapper_options_copy = mapper_options;
            mapper_options_copy.max_homography_inliers = 1;
            mapper->process(image_idx, prev_prev_image_idx,
                            mapper_options_copy, ba_options);
          }

          // Adjust local bundle

          // Remove oldest image in local bundle adjustment
          if (ba_image_idxs.size() >= local_ba_pose_state.size()) {
            ba_image_idxs.pop_front();
          }

          // Add image to local bundle adjustment and make sure that it is not
          // added twice
          if (std::count(ba_image_idxs.begin(), ba_image_idxs.end(),
                         image_idx) == 0) {
            ba_image_idxs.push_back(image_idx);
          }

          adjust_local_bundle(*mapper, ba_image_idxs, local_ba_pose_state,
                              ba_local_options);

          print_report_summary(*mapper, image_idx);

          // Run loop detection every 15 frames
          if (mapper->get_num_proc_images() % loop_detection_period == 0) {
            detect_loop(*mapper, image_idx, vmap, mapper_options, ba_options);
          }

        }

      }

      // Start new, separate sequence and skip next 10 images
      // if ((!success && (image_idx < end_image_idx)) || image_idx == 42) {
      if (!success && image_idx < end_image_idx) {

        std::cout << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Starting mapper #" << mappers.size() + 1 << std::endl;
        std::cout << std::string(80, '=') << std::endl << std::endl;

        image_idx += failure_skip_images;

        mapper = new SequentialMapper(image_data,
                                      cache_path,
                                      voc_tree_path,
                                      surf_options, loop_detection,
                                      debug, debug_path);
        mappers.push_back(mapper);

        // Reset parameters
        ba_image_idxs.clear();
        prev_image_idx = -1;
        prev_prev_image_idx = -1;

        continue;

      }

    }

    // Save indexes of previously processed images
    prev_prev_image_idx = prev_image_idx;
    prev_image_idx = image_idx;

    timer.print();

  }


  // Process all skipped frames between first and last successfully processed
  // images in mapper and adjust the bundle
  for (auto it=mappers.begin(); it!=mappers.end(); ++it) {
    process_remaining_images(**it, vmap, mapper_options, ba_options);
    adjust_global_bundle(**it, ba_global_options);
  }


  // Try to merge individual mappers
  if (merge) {
    merge_mappers(mappers, merge_num_skip_images, vmap, mapper_options, ba_options,
                  ba_global_options, output_path);
  }

  // Try to process remaining frames again, if still any missing
  for (auto it=mappers.begin(); it!=mappers.end(); ++it) {
    if (image_data.size() > (*it)->get_num_proc_images()) {
      const size_t num_proc_images = (*it)->get_num_proc_images();
      process_remaining_images(**it, vmap, mapper_options, ba_options);
      if ((*it)->get_num_proc_images() > num_proc_images) {
        adjust_global_bundle(**it, ba_global_options);
      }
    }
  }

  if (use_control_points) {
    const std::vector<ControlPoint> control_point_data = read_control_point_data(control_point_data_path);
    size_t num_mappers = 0;
    for (auto it=mappers.begin(); it!=mappers.end(); ++it) {
      num_mappers += 1;
      const std::string output_file_path = output_path + "control-point-data-"
                                           + boost::lexical_cast<std::string>(num_mappers) + ".txt";
      apply_control_points(**it, output_file_path, control_point_data, ba_global_options, filter_max_error);
    }
  } else if (filter_max_error > 0) {
    for (auto it=mappers.begin(); it!=mappers.end(); ++it) {
      filter_point_cloud(**it, ba_global_options, filter_max_error);
      adjust_global_bundle(**it, ba_global_options);
    }
  }


  // Write output data for each SequentialMapper object

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Write output" << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;

  size_t num_mappers = 0;
  for (auto it=mappers.begin(); it!=mappers.end(); ++it) {

    num_mappers += 1;

    std::cout << "Writing output for mapper #" << num_mappers << std::endl;

    std::string suffix = boost::lexical_cast<std::string>(num_mappers);
    write_mapper(**it, mapper_options, output_path, suffix);

  }

  // Release memory of all mappers
  for (auto it=mappers.begin(); it!=mappers.end(); ++it) {
    delete *it;
  }

  std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "Total elapsed time: " << std::setprecision(2)
            << timer_total.elapsed_time() / 6e7
            << " [minutes]" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  return 0;
}
