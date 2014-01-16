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

#ifndef MAVMAP_SRC_SFM_SEQUENTIAL_MAPPER_H_
#define MAVMAP_SRC_SFM_SEQUENTIAL_MAPPER_H_

#include <list>
#include <vector>
#include <map>
#include <unordered_map>

#include <boost/lexical_cast.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "base2d/feature.h"
#include "base2d/feature_cache.h"
#include "base2d/image.h"
#include "base3d/bundle_adjustment.h"
#include "base3d/camera_models.h"
#include "base3d/essential_matrix.h"
#include "base3d/p3p.h"
#include "base3d/projection.h"
#include "base3d/projective_transform.h"
#include "base3d/similarity_transform.h"
#include "base3d/triangulation.h"
#include "fm/feature_management.h"
#include "loop/detection.h"
#include "util/estimation.h"
#include "util/opencv.h"
#include "util/path.h"


#include "util/timer.h"


// Maximum number of visual words (= number of features) per image which will
// be used in the vocabulary tree. This number greatly influences the amount
// of memory required for the vocabulary tree and should be adapted
// accordingly.
// TODO: Adaptively set this threshold depending on the SURFOptions when using
// adaptive detection mode.
#define MAX_NUM_VISUAL_WORDS 5000


struct SequentialMapperOptions {

  /**
   * Maximum allowed relative number of inliers for homography between two
   * consecutive images.
   *
   * This makes sure that two consecutive have sufficient view point change.
   */
  double max_inliers_homography;

  /**
   * Maximum final cost of pose refinement in `process` required successful
   * processing.
   */
  double final_cost_threshold;

  /**
   * Cauchy loss scale factor for bundle adjustment.
   */
  double loss_scale_factor;

  /**
   * RANSAC 2D-3D pose estimation stops when at least this number of inliers
   * has been found.
   *
   * Relative value [0; 1) as a fraction of the number of total points.
   * Absolute value [1; inf) as actual minimum number of inliers.
   */
  double ransac_min_inlier_stop;

  /**
   * Minimum required number of inliers required for successful processing.
   *
   * Relative value [0; 1) as a fraction of the number of total points.
   * Absolute value [1; inf) as actual minimum number of inliers.
   */
  double min_inlier_threshold;

  /**
   * Maximum allowed reprojection error in pixels.
   */
  double max_reproj_error;

  /**
   * Minimum (or maximum as 180 - angle) angle between two rays of a
   * triangulated point.
   */
  double min_triangulation_angle;

  /**
   * Minimum track length for a 3D point to be used in the estimation.
   */
  size_t min_track_len;

  SequentialMapperOptions() : max_inliers_homography(0.7),
                              final_cost_threshold(1),
                              loss_scale_factor(1),
                              ransac_min_inlier_stop(0.6),
                              min_inlier_threshold(30),
                              max_reproj_error(2),
                              min_triangulation_angle(0.5),
                              min_track_len(2) {}

};


struct SURFOptions {

  /**
   * Hessian threshold for SURF detection.
   */
  double hessian_threshold;

  /**
   * Number of octaves.
   */
  size_t num_octaves;

  /**
   * Number of layers per octave.
   */
  size_t num_octave_layers;

  /**
   * Whether to use gridded adaptive SURF feature detection.
   *
   * For each grid cell in the image the Hessian threshold is iteratively
   * adapted to match the feature limit defined by `adaptive_min_per_cell`
   * and `adaptive_max_per_cell`.
   */
  bool adaptive;

  /**
   * Minimum number of features per grid cell in adaptive detection mode.
   */
  size_t adaptive_min_per_cell;

  /**
   * Maximum number of features per grid cell in adaptive detection mode.
   */
  size_t adaptive_max_per_cell;

  /**
   * Number of grid cells in the first dimension of the image.
   *
   * The total number of grid cells is defined as
   * `adaptive_cell_rows` x `adaptive_cell_cols`.
   */
  size_t adaptive_cell_rows;

  /**
   * Number of grid cells in the second dimension of the image.
   *
   * The total number of grid cells is defined as
   * `adaptive_cell_rows` x `adaptive_cell_cols`.
   */
  size_t adaptive_cell_cols;

  SURFOptions() : hessian_threshold(100),
                  num_octaves(4),
                  num_octave_layers(3),
                  adaptive(true),
                  adaptive_min_per_cell(100),
                  adaptive_max_per_cell(300),
                  adaptive_cell_rows(3),
                  adaptive_cell_cols(3) {}

};


/**
 * Sequential Mapper for camera pose estimation and sparse 3D reconstruction.
 */
class SequentialMapper {

public:

  /**
   * Create sequential mapper.
   *
   * @param image_data                  Vector with data of all images.
   * @param cache_path                  Path to the base directory of the
   *                                    cache.
   *                                    The path must exist prior to
   *                                    instantiation.
   * @param voc_tree_path               Path to pre-computed vocabulary tree
   *                                    binary file.
   * @param surf_options                SURF detection and extraction options
   *                                    used for all images.
   * @param debug                       Enable debug mode, which saves debug
   *                                    output to `debug_path`.
   * @param debug_path                  Destination path to the base directory
   *                                    for debug output.
   */
  SequentialMapper(const std::vector<Image>& image_data,
                   const std::string& cache_path,
                   const std::string& voc_tree_path,
                   const SURFOptions& surf_options,
                   const bool debug=false, const std::string& debug_path="");


  /**
   * Process initial pair of images by essential matrix estimation.
   *
   * This function can only be called once successfully. It processes the
   * initial pose and structure configuration by computing the essential
   * matrix using the Five-Point algorithm.
   *
   * A sufficiently large disparity with images of good quality is recommended
   * to initialize a reliable and accurate structure configuration. It is the
   * foundation for an accurate reconstruction in the following process chain.
   *
   * Only for successful calls the reconstructed information (camera pose and
   * sparse 3D point cloud) is saved in the `feature_manager`.
   *
   * Uses the following options:
   *    - `max_inliers_homography`: Make sure image pair has large enough disparity
   *      for reliable initial pose estimation, otherwise return `false`.
   *    - `max_reproj_error`: Classify as inliers and outliers in RANSAC
   *      essential matrix estimation. Decide which 3D points are used for
   *      further processing and are added to the feature manager.
   *
   * @param first_image_idx   Index of first image in `image_data` array.
   * @param second_image_idx  Index of second image in `image_data` array.
   * @param options           Processing options.
   *
   * @return                  Whether image pair was processed successfully.
   */
  bool process_initial(const size_t first_image_idx,
                       const size_t second_image_idx,
                       const SequentialMapperOptions& options);


  /**
   * Process new image by 2D-3D pose estimation.
   *
   * This function must be used once `process_initial` has been called
   * successfully.
   *
   * The pose of the image is only estimated if the `image_idx` has not been
   * processed successfully before. In all cases new points are triangulated
   * and the tracks of already reconstructed 3D points will be continued.
   *
   * Only for successful calls the reconstructed information (camera pose and
   * sparse 3D point cloud) is saved in the `feature_manager`.
   *
   * Uses the following options:
   *    - `max_inliers_homography`: Make sure image pair has large enough disparity
   *      for reliable pose estimation, otherwise return `false`.
   *    - `max_reproj_error`: Classify as inliers and outliers in RANSAC
   *      2D-3D pose estimation in case the image has already been processed.
   *      Determine number of inliers of already triangulated points in case
   *      the image has been processed before. Decide which 3D points are used
   *      for further processing and are added to the feature manager.
   *    - `ransac_min_inlier_stop`: Number of inliers before RANSAC 2D-3D pose
   *      estimation stops.
   *    - `min_inlier_threshold`: Minimum number of inliers for successful
   *      processing, see `max_reproj_error`.
   *    - `final_cost_threshold`: Decide if pose refinement was successful or
   *      not and accordingly return `true` or `false`.
   *    - `loss_scale_factor`: Cauchy loss scale factor for pose refinement.
   *    - `min_track_len`: Minimum track length of a 3D point to be used
   *      for 2D-3D pose estimation. Longer track lengths typically mean more
   *      accurate 3D points and thus more reliable 2D-3D pose estimation.
   *      However, note that especially with only few successfully processed
   *      images at the beginning there are only few 3D points with long
   *      enough tracks.
   *
   * A successful call to this function must meet the following conditions:
   *    - Minimum median feature disparity between two images
   *    - Minimum number of inliers must be found:
   *        => in case the image has been processed before the number of
   *           inliers is defined as the number of already triangulated 3D
   *           points with a reprojection error smaller than
   *           `max_reproj_error`.
   *        => in case the image has not been processed before and the pose
   *           is estimated with 2D-3D correspondences, the number of inliers
   *           is defined as the number of inliers in the RANSAC 2D-3D pose
   *           estimation.
   *    - The final cost (pixel) in the pose refinement must not exceed
   *      the `final_cost_threshold`, in case the image has not been processed
   *      before.
   *    - The RANSAC 2D-3D pose estimation must find a valid consensus set, in
   *      case the image has not been processed before.
   *
   *
   * @param image_idx         Index of new image in `image_data` array
   *                          to be processed.
   * @param prev_image_idx    Index of corresponding image in `image_data`
   *                          array. This image must have been processed
   *                          successfully prior to this call.
   * @param options           Processing options.
   *
   * @return                  Whether image was processed successfully.
   */
  bool process(const size_t image_idx,
               const size_t prev_image_idx,
               const SequentialMapperOptions& options);


  /**
   * Bundle adjustment (local or global).
   *
   * Uses the following options:
   *    - `min_track_len`: Only use 3D points in bundle adjustment with
   *      this minimum track length.
   *
   * @param free_image_idxs        Indices of images in `image_data` array
   *                               whose pose parameters are free, i.e. they
   *                               are variable and thus re-estimated.
   * @param free_image_idxs        Indices of images in `image_data` array
   *                               whose pose parameters are fixed, i.e. they
   *                               are constant and thus not re-estimated.
   * @param fixed_image_idxs       Indices of images in `image_data` array
   *                               whose x-component of the translation is
   *                               fixed, i.e. they are all variable and thus
   *                               re-estimated except for the x-component of
   *                               the translation.
   * @param options                Processing options.
   * @param update_point3D_errors  Whether to update the mean reprojection
   *                               errors of all image observations for each
   *                               3D point in the bundle adjustment. The
   *                               values can be queried through
   *                               `get_point3D_error`.
   * @param constrain_poses        Whether to constrain the free poses using
   *                               the prior rotation information from the
   *                               `image_data` array.
   * @param constrain_rot_weight   Weight factor for the residual values
   *                               of the rotation constraint cost function.
   *                               Larger values result in stronger
   *                               constraints.
   * @param print_summary          Whether to print a short summary about the
   *                               bundle adjustment.
   * @param print_summary          Whether to print he current progress of
   *                               the bundle adjustment for each iteration.
   *
   * @return                       Final cost of the bundle adjustment
   *                               in pixels.
   */
  double adjust_bundle(const std::vector<size_t>& free_image_idxs,
                       const std::vector<size_t>& fixed_image_idxs,
                       const std::vector<size_t>& fixed_x_image_idxs,
                       const SequentialMapperOptions& options,
                       const bool update_point3D_errors=false,
                       const bool constrain_poses=false,
                       const double constrain_rot_weight=100,
                       const bool print_summary=true,
                       const bool print_progress=true);


  /**
   * Global bundle adjustment with initial poses as fixed.
   *
   * See `adjust_bundle`.
   */
  double adjust_global_bundle(const SequentialMapperOptions& options,
                              const bool update_point3D_errors=false,
                              const bool constrain_poses=false,
                              const double constrain_rot_weight=100,
                              const bool print_summary=true,
                              const bool print_progress=true);


  /**
   * Detect loop in all successfully processed images for given image.
   *
   * This function first searches for a specified number of most similar images
   * in all previously successfully processed images. Then the function tries
   * to process this image against those most similar images.
   *
   * Uses all the options used in `process`.
   *
   * @param image_idx        Reference image, for which similar images are
   *                         searched for.
   * @param num_images       Maximum number of similar images to search for.
   * @param num_nh_images    Maximum number of successful `process` calls
   *                         in direct neighborhood of `image_idx`. See
   *                         `nh_distance` for definition of neighborhood.
   * @param nh_distance      Defines the neighborhood image index range of
   *                         `image_idx` as [image_idx-nh_distance,
   *                                         image_idx+nh_distance].
   * @param options          Processing options.
   * @param verbose          Whether to print the status of the loop detection.
   *
   * @return                 Number of successfully detected loops.
   */
  size_t detect_loop(const size_t image_idx,
                     const size_t num_images,
                     const size_t num_nh_images,
                     const size_t nh_distance,
                     const SequentialMapperOptions& options,
                     const bool verbose=true);

  /**
   * Try to merge the other into this mapper.
   *
   * This function tries to detect loops of the most similar images between
   * the two mappers. If at at least 3 loops were detected, the positions of
   * the detected loop poses are used to estimate a similarity transformation
   * between the two coordinate systems. This similarity transformation is then
   * used to transform all data in the other mapper to the coordinate system of
   * this mapper. Finally, the transformed data of the other mapper is
   * completely cloned to this mapper. The other mapper can therefore safely
   * be deallocated.
   *
   * Furthermore, this function tries to detect loops between the two mapper
   * sequences throughout the complete sequence of the other mapper in order
   * to enable for a better bundle adjustment, which is able to resolve drift
   * effects, afterwards.
   *
   * Note, that it is typically much faster to merge the smaller mapper (other)
   * into the larger mapper (this) w.r.t. the size of `get_num_proc_images`.
   * Additionally, it is recommended to globally adjust both mappers
   * before calling the merge routine.
   *
   * If the mappers cannot be merged successfully, the state of this mapper
   * is not changed.
   *
   * Uses all the options used in `process`.
   *
   * @param other               `SequentialMapper` object.
   * @param num_similar_images  For each image in the other mapper this
   *                            function tries to find at least this number
   *                            of most similar images in this mapper.
   * @param num_skip_images     How many images to skip for loop detection, so
   *                            that every `num_skip_images` is tested for
   *                            possible loops.
   * @param options             Processing options.
   * @param verbose             Whether to print the status of the merge.
   *
   * @return                    Whether the two mappers have been merged
   *                            successfully.
   */
  bool merge(SequentialMapper& other,
             const size_t num_similar_images,
             const size_t num_skip_images,
             const SequentialMapperOptions& options,
             const bool verbose=true);


  /**
   * Check whether the image has been successfully processed by
   * `process_initial` or `process`.
   *
   * @param image_idx         Index of image in `image_data` array
   *                          to be checked.
   *
   * @return                  Whether the image has been processed
   *                          successfully.
   */
  bool is_image_processed(const size_t image_idx);


  /**
   * Check whether the image pair has been successfully processed by
   * `process_initial` or `process`.
   *
   * @param image_idx1        Index of first image in `image_data` array
   *                          to be checked.
   * @param image_idx2        Index of second image in `image_data` array
   *                          to be checked.
   *
   * @return                  Whether the two images have been processed
   *                          successfully.
   */
  bool is_pair_processed(const size_t image_idx1, const size_t image_idx2);


  /**
   * Get camera ID for image in `feature_manager`.
   *
   * This function either returns the ID of the camera if the camera has
   * already been added to the feature manager (this is the case if multiple
   * images were taken by the same camera). Otherwise it adds the camera of
   * of the image to the feature manager and returns the new camera ID.
   *
   * @param image_idx         Index of image in `image_data` array.
   *
   * @return                  The camera ID in the `feature_manager`.
   */
  size_t get_camera_id(const size_t image_idx);


  /**
   * Get the camera model code for image.
   *
   * The camera model codes are defined in `base3d/camera_models.h`.
   *
   * @param image_idx         Index of image in `image_data` array.
   *
   * @return                  The camera model code.
   */
  int get_camera_model_code(const size_t image_idx);


  /**
   * Get image ID for image in `feature_manager`.
   *
   * The image index in the `image_data` array is not equal to the image ID
   * in the `feature_manager`, as the latter depends on the order in which the
   * images were added to the `feature_manager`.
   *
   * @param image_idx         Index of image in `image_data` array.
   *
   * @return                  The image ID in the `feature_manager`.
   */
  size_t get_image_id(const size_t image_idx);


  /**
   * Get image index for image in `image_data`.
   *
   * The image index in the `image_data` array is not equal to the image ID
   * in the `feature_manager`, as the latter depends on the order in which the
   * images were added to the `feature_manager`.
   *
   * @param image_id          ID of image in `feature_manager`.
   *
   * @return                  The image index in the `image_data` array.
   */
  size_t get_image_idx(const size_t image_id);


  /**
   * Get the number of successfully processed images.
   *
   * @return                  The number of successfully processed images.
   */
  size_t get_num_proc_images() { return num_proc_images_; }


  /**
   * Get the first image index as passed to the successful call to
   * `process_initial`.
   *
   * @return                  The image index in the `image_data` array.
   */
  size_t get_first_image_idx() { return first_image_idx_; }


  /**
   * Get the second image index as passed to the successful call to
   * `process_initial`.
   *
   * @return                  The image index in the `image_data` array.
   */
  size_t get_second_image_idx() { return second_image_idx_; }


  /**
   * Get the minimum image index (smallest index value), which has been
   * processed successfully.
   *
   * @return                  The image index in the `image_data` array.
   */
  size_t get_min_image_idx() { return min_image_idx_; }


  /**
   * Get the maximum image index (largest index value), which has been
   * processed successfully.
   *
   * @return                  The image index in the `image_data` array.
   */
  size_t get_max_image_idx() { return max_image_idx_; }


  /**
   * Get the successfully processed image pair indices.
   *
   * Each key in the map was processed with its values successfully. Each
   * {key1: [..., value1, ...]} combination also exists as a
   * {value1: [..., key1, ...]}.
   *
   * @return                  The image pair indices.
   */
  std::unordered_map<size_t, std::set<size_t> >
    get_image_pair_idxs() { return image_pair_idxs_; }


  /**
   * Get all successfully processed image indices.
   *
   * @return                  A set of image indices in the `image_data` array.
   */
  std::set<size_t> get_image_idxs();


  /**
   * Get mean residual errors for 3D point in the `feature_manager`.
   *
   * Note that this information is only available if `adjust_bundle` has
   * been called with `update_point3D_errors=true`.
   *
   * @return                  Mean residual error.
   */
  double get_point3D_error(const size_t point3D_id);


  /**
   * Whether debug mode is enabled. This value can be safely changed from
   * outside without any restrictions as long as the `debug_path` is valid.
   */
  bool debug;


  /**
   * Destination path to the base directory for debug output.
   */
  std::string debug_path;

  FeatureManager feature_manager;

  std::vector<Image> image_data;

private:

  void extract_image_features_(const size_t image_idx);

  void match_image_features_();

  void find_similar_images_(const size_t image_idx,
                            const size_t num_images,
                            std::vector<int>& image_idxs,
                            std::vector<float>& scores);

  FeatureCache feature_cache_;

  LoopDetector loop_detector_;

  size_t num_proc_images_;

  size_t first_image_idx_;
  size_t second_image_idx_;

  size_t min_image_idx_;
  size_t max_image_idx_;

  size_t prev_image_idx_;
  size_t prev_prev_image_idx_;

  std::unordered_map<size_t, size_t> image_idx_to_id_;
  std::unordered_map<size_t, size_t> image_id_to_idx_;

  std::unordered_map<size_t, size_t> camera_idx_to_id_;

  std::unordered_map<size_t, std::set<size_t> > image_pair_idxs_;

  std::vector<cv::KeyPoint> prev_keypoints_, curr_keypoints_;
  std::vector<Eigen::Vector2d> prev_points2D_, curr_points2D_;
  std::vector<Eigen::Vector2d> prev_points2D_N_, curr_points2D_N_;

  cv::Mat prev_descriptors_, curr_descriptors_;

  std::vector<cv::DMatch> matches_;

  std::unordered_map<size_t, double> point3D_errors_;

};

#endif // MAVMAP_SRC_SFM_SEQUENTIAL_MAPPER_H_
