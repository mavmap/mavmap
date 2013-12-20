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
