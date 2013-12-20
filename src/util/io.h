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


std::vector<Image> read_image_data(const std::string path,
                                   const std::string root_path,
                                   const std::string prefix="",
                                   const std::string suffix="",
                                   const std::string ext=".bmp");


Eigen::Matrix3d read_calib_matrix(const std::string path);


void write_image_data(const std::string path,
                      SequentialMapper& mapper);


void write_point_cloud(const std::string path,
                       SequentialMapper& mapper,
                       const size_t min_track_len,
                       const double error_threshold,
                       const double coordinate_norm_threshold);


void write_camera_poses(const std::string path,
                        FeatureManager& feature_manager,
                        const double scale, const double red,
                        const double green, const double blue);


void write_track(std::string path,
                 SequentialMapper& mapper,
                 const size_t point3D_id,
                 const int radius=5, const int thickness=2);


void write_camera_connections(const std::string& path,
                              SequentialMapper& mapper);


#endif // MAVMAP_SRC_UTIL_IO_H_
