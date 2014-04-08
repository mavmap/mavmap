/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_FEATURE_FEATURE_CACHE_H_
#define MAVMAP_SRC_FEATURE_FEATURE_CACHE_H_

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "base2d/feature.h"
#include "base2d/image.h"
#include "util/path.h"


// File containing the general SURF parameters to automatically detect when
// the cache needs a reset.


// Actual cache files will be named Image::name + suffix
const static std::string params_suffix = "-params.ini";
const static std::string dimensions_suffix = "-metadata.ini";
const static std::string keypoints_suffix = "-keypoints.bin";
const static std::string descriptors_suffix = "-descriptors.bin";


class FeatureCache {

public:

  FeatureCache();

  /**
   * Create FeatureCache.
   *
   * This function caches extracted OpenCV SURF features (location and
   * descriptors). It automatically detects if a SURF parameter is changed and
   * clears the cache of the image accordingly.
   *
   * @param path                  Path to the base directory of the cache. The
   *                              path must exist prior to instantiation.
   * @param image_data            Vector with data of all images.
   * @param descriptor_extractor  Instance of the OpenCV SURF descriptor
   *                              extractor.
   * @param adaptive              Whether to use gridded adaptive feature
   *                              detection.
   * @param descriptor_extractor_adaptive
   *                              Adaptive SURF descriptor extractor, only used
   *                              if `adaptive=true`.
   */
  FeatureCache(const std::string& path,
               const std::vector<Image>& image_data,
               const SURFOptions& surf_options);

  /**
   * Query the cache for all features of an image.
   *
   * Either extracts the features using the `descriptor_extractor` instance if
   * the image has never been queried before. Otherwise retrieves the cached
   * features.
   *
   * @param image_idx      Image index in `image_data` array.
   * @param keypoints      Output keypoint data.
   * @param descriptors    Output descriptor data.
   */
  void query(const size_t image_idx,
             std::vector<cv::KeyPoint>& keypoints,
             cv::Mat& descriptors);

  /**
   * Query the dimensions of an image.
   *
   * Either reads the image from disk and determines its dimensions or
   * retrieves the dimensions from the cache without reading the
   * complete image from disk.
   *
   * @param image_idx      Image index in `image_data` array.
   * @param rows           Number of rows.
   * @param cols           Number of columns.
   * @param channels       Number of channels.
   * @param diagonal       Diagonal of image as ``sqrt(rows^2 + cols^2)``.
   */
  void query_dimensions(const size_t image_idx,
                        size_t* rows, size_t* cols, size_t* channels,
                        float* diagonal);

  /**
   * Clear the entire cache.
   */
  void clear();


  /**
   * Clear the cache of one image.
   *
   * @param image_idx     Image index in `image_data` array.
   */
  void clear(const size_t image_idx);


private:

  cv::Mat read_image_(const size_t image_idx);

  std::string path_;
  std::vector<Image> image_data_;
  SURFOptions surf_options_;
  cv::SURF descriptor_extractor_;
  AdaptiveSURF descriptor_extractor_adaptive_;

};

#endif // MAVMAP_SRC_FEATURE_FEATURE_CACHE_H_
