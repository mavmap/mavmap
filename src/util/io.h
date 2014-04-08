/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_UTIL_IO_H_
#define MAVMAP_SRC_UTIL_IO_H_

#include <fstream>
#include <list>
#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "base2d/image.h"
#include "base3d/projection.h"
#include "fm/feature_management.h"
#include "sfm/sequential_mapper.h"
#include "util/path.h"


struct ControlPoint {

  std::string name;
  Eigen::Vector3d xyz;
  std::vector<std::pair<size_t, Eigen::Vector2d>> points2D;
  bool fixed;

};


std::vector<Image> read_image_data(const std::string& path,
                                   const std::string& root_path,
                                   const std::string& prefix="",
                                   const std::string& suffix="",
                                   const std::string& ext=".bmp");


Eigen::Matrix3d read_calib_matrix(const std::string& path);


std::vector<ControlPoint> read_control_point_data(const std::string& path);

void write_control_point_data(const std::string& path, SequentialMapper& mapper,
                              const std::vector<std::pair<size_t, ControlPoint>> control_points);


#endif // MAVMAP_SRC_UTIL_IO_H_
