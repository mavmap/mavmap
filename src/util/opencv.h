/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_UTIL_OPENCV_H_
#define MAVMAP_SRC_UTIL_OPENCV_H_

#include <vector>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>


std::vector<Eigen::Vector2d>
cv2eigen_keypoints(const std::vector<cv::KeyPoint>& keypoints);

std::vector<Eigen::Vector2d>
cv2eigen_points(const std::vector<cv::Point2f>& points2D);

std::vector<Eigen::Vector3d>
cv2eigen_points(const std::vector<cv::Point3f>& points3D);

std::vector<Eigen::Vector2i>
cv2eigen_matches(const std::vector<cv::DMatch>& matches);

#endif // MAVMAP_SRC_UTIL_OPENCV_H_
