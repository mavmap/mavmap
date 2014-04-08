/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "feature.h"


static void ratio_test_(std::vector<std::vector<cv::DMatch>>& matches, const double max_ratio) {
  for (auto it=matches.begin(); it!= matches.end(); ++it) {
    std::vector<cv::DMatch>& match = *it;
    if (match.size() > 1
        && match[0].distance / match[1].distance > max_ratio) {
      match.clear();
    }
  }
}


static void max_distance_mask_(const std::vector<cv::KeyPoint>& keypoints1,
                               const std::vector<cv::KeyPoint>& keypoints2,
                               const double max_distance, cv::Mat& mask12, cv::Mat& mask21) {
  mask12 = cv::Mat(keypoints1.size(), keypoints2.size(), CV_8UC1);
  mask21 = cv::Mat(keypoints2.size(), keypoints1.size(), CV_8UC1);

  const double max_distance2 = max_distance * max_distance;

  // Set masks depending on distance between keypoints
  for (size_t i=0; i<keypoints1.size(); ++i) {
    const cv::Point2f& left_point2D = keypoints1[i].pt;
    const double px = left_point2D.x;
    const double py = left_point2D.y;
    for (size_t j=0; j<keypoints2.size(); ++j) {
      const cv::Point2f& right_point2D = keypoints2[j].pt;
      const double dx = px - right_point2D.x;
      const double dy = py - right_point2D.y;
      if (dx*dx + dy*dy < max_distance2) {
        mask12.at<uint8_t>(i, j) = 1;
        mask21.at<uint8_t>(j, i) = 1;
      } else {
        mask12.at<uint8_t>(i, j) = 0;
        mask21.at<uint8_t>(j, i) = 0;
      }
    }
  }
}


void match_brute_force(const std::vector<cv::KeyPoint>& keypoints1,
                       const cv::Mat& descriptors1,
                       const std::vector<cv::KeyPoint>& keypoints2,
                       const cv::Mat& descriptors2,
                       std::vector<cv::DMatch>& matches,
                       const bool ratio_test,
                       const double max_ratio,
                       const double max_distance,
                       const int norm_type) {

  matches.clear();

  if (ratio_test) {

    cv::BFMatcher matcher(norm_type, false);


    std::vector<std::vector<cv::DMatch>> matches12, matches21;
    if (max_distance == -1) {
      matcher.knnMatch(descriptors1, descriptors2, matches12, 2);
      matcher.knnMatch(descriptors2, descriptors1, matches21, 2);
    } else {
      cv::Mat mask12, mask21;
      max_distance_mask_(keypoints1, keypoints2, max_distance, mask12, mask21);
      matcher.knnMatch(descriptors1, descriptors2, matches12, 2, mask12);
      matcher.knnMatch(descriptors2, descriptors1, matches21, 2, mask21);
    }

    // Ratio test by D. Lowe
    ratio_test_(matches12, max_ratio);
    ratio_test_(matches21, max_ratio);

    // Cross check / symmetry test
    for (auto it12=matches12.begin(); it12!=matches12.end(); ++it12) {
      const std::vector<cv::DMatch>& match12 = *it12;
      if (match12.size() < 2) {
        continue;
      }
      for (auto it21=matches21.begin(); it21!=matches21.end(); ++it21) {
        const std::vector<cv::DMatch>& match21 = *it21;
        if (it21->size() < 2) {
          continue;
        }
        if (match12[0].queryIdx == match21[0].trainIdx
            && match21[0].queryIdx == match12[0].trainIdx) {
          matches.push_back(match12[0]);
          break;
        }
      }
    }

  } else {

    if (max_distance == -1) {
      // Use built-in cross check
      cv::BFMatcher matcher(norm_type, true);
      matcher.match(descriptors1, descriptors2, matches);
    } else {
      cv::Mat mask12, mask21;
      max_distance_mask_(keypoints1, keypoints2, max_distance, mask12, mask21);

      // Manual cross check
      cv::BFMatcher matcher(norm_type, false);

      // One-directional matching in both directions
      std::vector<cv::DMatch> matches12, matches21;

      matcher.match(descriptors1, descriptors2,
                    matches12, mask12);
      matcher.match(descriptors2, descriptors1,
                    matches21, mask21);

      // Manual cross check
      for (size_t i=0; i<matches12.size(); ++i) {
        const cv::DMatch& match = matches12[i];
        if (match.queryIdx == matches21[match.trainIdx].trainIdx) {
          matches.push_back(match);
        }
      }
    }
  }
}


double median_feature_disparity(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches) {

  std::vector<double> disparities(matches.size());
  for (size_t i=0; i<matches.size(); ++i) {
    const cv::Point2f& left_point = keypoints1[matches[i].queryIdx].pt;
    const cv::Point2f& right_point = keypoints2[matches[i].trainIdx].pt;
    const double dx = left_point.x - right_point.x;
    const double dy = left_point.y - right_point.y;
    disparities[i] = sqrt(dx*dx + dy*dy);
  }

  return median(disparities);
}


AdaptiveSURF::AdaptiveSURF()
    : min_per_cell(100),
      max_per_cell(300),
      cell_rows(3),
      cell_cols(3),
      init_hessian_threshold_(1000),
      surf_(cv::SURF(1000)),
      prev_rows_(0),
      prev_cols_(0) {}


AdaptiveSURF::AdaptiveSURF(
      const cv::SURF& surf,
      const size_t min_per_cell_, const size_t max_per_cell_,
      const size_t cell_rows_, const size_t cell_cols_)
    : min_per_cell(min_per_cell_),
      max_per_cell(max_per_cell_),
      cell_rows(cell_rows_),
      cell_cols(cell_cols_),
      init_hessian_threshold_(surf.hessianThreshold),
      surf_(surf),
      prev_rows_(0),
      prev_cols_(0) {

  // Determine overlap for grid cells, otherwise no keypoints will be detected
  // at the borders of the grid cells
  const size_t num_total_layers = (surf_.nOctaveLayers + 2) * surf_.nOctaves;
  size_t max_filter_size;
  if (num_total_layers == 1) {
    max_filter_size = 9;
  } else {
    size_t prev_filter_size = 9;
    max_filter_size = 15;
    for (size_t i=1; i<num_total_layers; ++i) {
      const size_t curr_filter_size = max_filter_size;
      max_filter_size = 2 * max_filter_size - prev_filter_size;
      prev_filter_size = curr_filter_size;
    }
  }
  max_filter_ext_ = (max_filter_size - 1) / 2;

}


void AdaptiveSURF::detect(const cv::Mat& image,
                          std::vector<cv::KeyPoint>& keypoints,
                          const size_t max_iter) {

  keypoints.clear();

  const size_t num_cell_r = ceil(image.rows / (float)cell_rows);
  const size_t num_cell_c = ceil(image.cols / (float)cell_cols);

  // If image dimensions of previous image do not equal reset cell history
  if (prev_rows_ != (size_t)image.rows || prev_cols_ != (size_t)image.cols) {
    prev_rows_ = image.rows;
    prev_cols_ = image.cols;
    prev_cell_hessian_thresholds_
      = std::vector<double>(cell_rows * cell_cols,
                            init_hessian_threshold_);
  }

  size_t cell_idx = 0;

  uint8_t* mask_data = new uint8_t[(num_cell_r + 2 * max_filter_ext_)
                                   * (num_cell_c + 2 * max_filter_ext_)];

  for (size_t cell_r=0; cell_r<cell_rows; ++cell_r, ++cell_idx) {

    // Row range
    const size_t r0
      = std::max<int>((int)(cell_r * num_cell_r - max_filter_ext_), 0);
    const size_t r1
      = std::min<size_t>(r0 + num_cell_r + 2 * max_filter_ext_,
                         image.rows - 1);

    for (size_t cell_c=0; cell_c<cell_cols; ++cell_c) {

      // Col range
      const size_t c0
        = std::max<int>((int)(cell_c * num_cell_c - max_filter_ext_), 0);
      const size_t c1
        = std::min<size_t>(c0 + num_cell_c + 2 * max_filter_ext_,
                           image.cols - 1);

      // Crop cell
      cv::Rect image_roi(c0, r0, c1 - c0, r1 - r0);
      cv::Mat cell = image(image_roi);

      // Mask to ignore keypoints in overlapping regions in order to avoid
      // duplicate keypoints
      cv::Mat mask = cv::Mat(image_roi.height, image_roi.width,
                             CV_8U, mask_data);
      size_t mask_roi_r0 = std::min<size_t>(r0, max_filter_ext_ + 1);
      size_t mask_roi_c0 = std::min<size_t>(c0, max_filter_ext_ + 1);
      size_t mask_roi_height = image_roi.height - 2 * max_filter_ext_ - 1;
      size_t mask_roi_width = image_roi.width - 2 * max_filter_ext_ - 1;
      if (r1 + max_filter_ext_ > image.rows) {
        mask_roi_height = image_roi.height - max_filter_ext_ - 1;
      }
      if (c1 + max_filter_ext_ > image.cols) {
        mask_roi_width = image_roi.width - max_filter_ext_ - 1;
      }
      cv::Rect mask_roi(mask_roi_c0, mask_roi_r0,
                        mask_roi_width, mask_roi_height);
      mask.setTo(0);
      mask(mask_roi) = 1;

      std::vector<cv::KeyPoint> cell_keypoints;

      // Get threshold from previous call to this method
      surf_.hessianThreshold
        = prev_cell_hessian_thresholds_[cell_idx];

      // Adaptively determine optimum threshold
      for (size_t i=0; i<max_iter; ++i) {

        surf_.detect(cell, cell_keypoints, mask);

        if (cell_keypoints.size() < min_per_cell) {
          // Update threshold and re-detect
          surf_.hessianThreshold /= 1.5;
        } else if (cell_keypoints.size() > max_per_cell) {
          // Update threshold for next image, but do not re-detect,
          // simply keep best keypoints
          std::sort(cell_keypoints.begin(), cell_keypoints.end(),
                    [](cv::KeyPoint a, cv::KeyPoint b) {
                      return a.response > b.response;
                    });
          break;
        } else {
          break;
        }

      }

      // Save optimum threshold for next call to this method
      prev_cell_hessian_thresholds_[cell_idx]
        = surf_.hessianThreshold;

      // Save cell keypoints for output
      const size_t num_cell_keypoints = std::min(cell_keypoints.size(),
                                                 max_per_cell);
      for (size_t i=0; i<num_cell_keypoints; ++i) {
        cv::KeyPoint& keypoint = cell_keypoints[i];
        // Shift relative cell coordinates to absolute image coordinates
        keypoint.pt.x += c0;
        keypoint.pt.y += r0;
        keypoints.push_back(keypoint);
      }

    }
  }

  delete[] mask_data;
}


void AdaptiveSURF::extract(const cv::Mat& image,
                           std::vector<cv::KeyPoint>& keypoints,
                           cv::Mat& descriptors) {
  surf_(image, cv::noArray(), keypoints, descriptors, true);
}
