/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "opencv.h"


std::vector<Eigen::Vector2d>
cv2eigen_keypoints(const std::vector<cv::KeyPoint>& keypoints) {
  std::vector<Eigen::Vector2d> points2D(keypoints.size());
  for (size_t i=0; i<keypoints.size(); ++i) {
    Eigen::Vector2d point2D(keypoints[i].pt.x, keypoints[i].pt.y);
    points2D[i] = point2D;
  }
  return points2D;
}


std::vector<Eigen::Vector2d>
cv2eigen_points(const std::vector<cv::Point2f>& points2D) {
  std::vector<Eigen::Vector2d> eigen_points2D(points2D.size());
  for (size_t i=0; i<points2D.size(); ++i) {
    Eigen::Vector2d point2D(points2D[i].x, points2D[i].y);
    eigen_points2D[i] = point2D;
  }
  return eigen_points2D;
}


std::vector<Eigen::Vector3d>
cv2eigen_points(const std::vector<cv::Point3f>& points3D) {
  std::vector<Eigen::Vector3d> eigen_points3D(points3D.size());
  for (size_t i=0; i<points3D.size(); ++i) {
    Eigen::Vector3d point3D(points3D[i].x, points3D[i].y, points3D[i].z);
    eigen_points3D[i] = point3D;
  }
  return eigen_points3D;
}


std::vector<Eigen::Vector2i>
cv2eigen_matches(const std::vector<cv::DMatch>& matches) {
  std::vector<Eigen::Vector2i> eigen_matches(matches.size());
  for (size_t i=0; i<matches.size(); ++i) {
    eigen_matches[i] = Eigen::Vector2i(matches[i].queryIdx,
                                       matches[i].trainIdx);
  }
  return eigen_matches;
}
