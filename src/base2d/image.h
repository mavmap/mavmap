/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE2D_IMAGE_H_
#define MAVMAP_SRC_BASE2D_IMAGE_H_


#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "base3d/projection.h"


class Image {

public:

  Image();

  cv::Mat read(const int flags=0);

  Eigen::Vector3d rvec();
  Eigen::Vector3d tvec();
  Eigen::Matrix<double, 3, 4> proj_matrix();

  // Image file data
  std::string path;
  std::string name;

  // Position and orientation data
  double roll, pitch, yaw;
  double lat, lon, alt;
  double local_height;
  double tx, ty, tz;

  // Image meta data
  size_t rows, cols, channels;
  double diagonal;

  // Camera data
  int camera_idx;
  std::string camera_model;
  std::vector<double> camera_params;

};


#endif // MAVMAP_SRC_BASE2D_IMAGE_H_
