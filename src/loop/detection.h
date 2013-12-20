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

#ifndef MAVMAP_SRC_LOOP_DETECTION_H_
#define MAVMAP_SRC_LOOP_DETECTION_H_


#include <string>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>

#include "voc_tree.h"
#include "voc_tree_database.h"


/**
 * Loop detection using a scalable vocabulary tree based on SIFT or SURF
 * features.
 */
class LoopDetector {

public:

  LoopDetector();

  /**
   * Create loop detector.
   *
   * @param voc_tree_path         Path to pre-computed vocabulary tree binary.
   * @param max_num_images        Maximum number of images the vocabulary tree
   *                              is capable of storing.
   * @param max_num_visual_words  Maximum number of visual words / features
   *                              per image.
   */
  LoopDetector(const std::string& voc_tree_path,
               const size_t max_num_images,
               const size_t max_num_visual_words);

  /**
   * Initializes the vocabulary tree.
   */
  void init();

  /**
   * Add new image to the vocabulary tree.
   *
   * @param image_id     Image ID which serves as an unique identifier for the
   *                     image for the later querying.
   * @param descriptors  SURF or SIFT descriptors as Nx128 matrix. Only the
   *                     first `max_num_visual_words` features are used.
   */
  void add_image(const size_t image_id,
                 const cv::Mat& descriptors);

  /**
   * Query most similar images from the vocabulary tree.
   *
   * @param descriptors     SURF or SIFT descriptors of the sample image as
   *                        Nx128 matrix. Only the first `max_num_visual_words`
   *                        features are used.
   * @param image_ids       Unique image IDs as passed in `add_image` of most
   *                        similar images.
   * @param scores          Sorted scores (distance) of found images.
   * @param update_weights  Whether to re-weight and normalize the vocabulary
   *                        tree.
   */
  void query(const cv::Mat& descriptors,
             std::vector<int>& image_ids,
             std::vector<float>& scores,
             const size_t num_images=1,
             const bool update_weights=false);

private:

  uint32_t* descriptors_to_visual_words_(const cv::Mat& descriptors,
                                         size_t& num_visual_words);

  std::string voc_tree_path_;

  size_t max_num_images_;
  size_t max_num_visual_words_;
  size_t num_images_;

  VocTree voc_tree_;
  VocTreeDatabase database_;

};


#endif // MAVMAP_SRC_LOOP_DETECTION_H_
