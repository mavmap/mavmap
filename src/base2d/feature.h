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

#ifndef MAVMAP_SRC_BASE2D_FEATURE_H_
#define MAVMAP_SRC_BASE2D_FEATURE_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "base2d/image.h"
#include "util/math.h"


/**
 * Brute-force descriptor matcher.
 *
 * For each descriptor in the first set, this matcher finds the closest
 * descriptor in the second set by trying each one.
 *
 * @param keypoints1    First set of keypoints.
 * @param descriptors1  First set of descriptors.
 * @param keypoints2    Second set of keypoints.
 * @param descriptors2  Second set of descriptors.
 * @param matches       Output of brute-force matching.
 * @param max_distance  Maximum allowed distance in pixels for a match.
 * @param norm_type     Employed norm for calculating descriptor distance.
 */
void match_brute_force(const std::vector<cv::KeyPoint>& keypoints1,
                       const cv::Mat& descriptors1,
                       const std::vector<cv::KeyPoint>& keypoints2,
                       const cv::Mat& descriptors2,
                       std::vector<cv::DMatch>& matches,
                       const double max_distance=0,
                       const int norm_type=cv::NORM_L2);


/**
 * Calculate median feature disparity between two images.
 *
 * Median feature disparity is defined as median value of all distances
 * between coordinates of first and second set of corresponding keypoints.
 *
 * @param keypoints1    First set of keypoints.
 * @param keypoints2    Second set of keypoints.
 * @param matches       Corresponding matches between first and second set of
 *                      keypoints (queryIdx for first and trainIdx for second
 *                      set of keypoints).
 *
 * @return              Median feature disparity.
 */
double median_feature_disparity(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches);


/**
 * Adaptive SURF feature detector and extractor.
 *
 * The image is split into a grid of sub-images. Feature detection is carried
 * out for each sub-images in the grid to ensure evenly distributed features
 * for each region in the image. The Hessian threshold of SURF is automatically
 * increased / decreased to obtain the optimum number of features per grid
 * cell. It is assumed that the input images are similar, so that the
 * Hessian threshold is remembered for each grid cell and it generally not need
 * be adaptively re-determined (only if image content changes).
 */
class AdaptiveSURF {

public:

  AdaptiveSURF();

  /**
   * Create AdaptiveSURF.
   *
   * The total number of grid cells is defined as `cell_rows` x `cell_cols`.
   *
   * @param surf           Initial SURF detector used for new images.
   *                       Only `surf.hessianThreshold` is adaptively changed
   *                       for each grid cell.
   * @param min_per_cell   Minimum number of features per grid cell.
   * @param min_per_cell   Maximum number of features per grid cell.
   * @param cell_rows      Number of cells in the first dimension of image.
   * @param cell_cols      Number of cells in the second dimension of image.
   */
  AdaptiveSURF(const cv::SURF& surf,
               const size_t min_per_cell_,
               const size_t max_per_cell_,
               const size_t cell_rows_,
               const size_t cell_cols_);

  void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
              const size_t max_iter=10);

  void extract(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
               cv::Mat& descriptors);

  size_t min_per_cell;
  size_t max_per_cell;
  size_t cell_rows;
  size_t cell_cols;

private:

  double init_hessian_threshold_;
  cv::SURF surf_;
  size_t prev_rows_;
  size_t prev_cols_;
  std::vector<double> prev_cell_hessian_thresholds_;
  size_t max_filter_ext_;

};


#endif // MAVMAP_SRC_BASE2D_FEATURE_H_
