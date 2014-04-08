/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE2D_FEATURE_H_
#define MAVMAP_SRC_BASE2D_FEATURE_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "base2d/image.h"
#include "util/math.h"


struct SURFOptions {

  SURFOptions() : hessian_threshold(100),
                  num_octaves(4),
                  num_octave_layers(3),
                  adaptive(true),
                  adaptive_min_per_cell(100),
                  adaptive_max_per_cell(300),
                  adaptive_cell_rows(3),
                  adaptive_cell_cols(3) {}

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

};


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
 * @param ratio_test    Whether to use the ratio test by D. Lowe.
 * @param max_ratio         Maximum ratio used in the ratio test.
 * @param max_distance  Maximum allowed distance in pixels for a match.
 * @param norm_type     Employed norm for calculating descriptor distance.
 */
void match_brute_force(const std::vector<cv::KeyPoint>& keypoints1,
                       const cv::Mat& descriptors1,
                       const std::vector<cv::KeyPoint>& keypoints2,
                       const cv::Mat& descriptors2,
                       std::vector<cv::DMatch>& matches,
                       const bool ratio_test=true,
                       const double max_ratio=0.6,
                       const double max_distance=-1,
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
