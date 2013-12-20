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

#include "image.h"


Image::Image(): path(""),
                timestamp(""),
                roll(0), pitch(0), yaw(0),
                lat(0), lon(0), alt(0),
                local_height(0),
                tx(0), ty(0), tz(0),
                rows(0), cols(0), channels(0), diagonal(0),
                camera_idx(0), camera_model(""),
                camera_params(std::vector<double>()) {}


cv::Mat Image::read(const int flags) {
  cv::Mat mat = cv::imread(path, flags);
  rows = mat.rows;
  cols = mat.cols;
  channels = mat.channels();
  diagonal = sqrt(rows*rows + cols*cols);
  return mat;
}


Eigen::Vector3d Image::rvec() {
  const Eigen::Matrix3d R = rot_mat_from_euler_angles(roll, pitch, yaw);
  const Eigen::AngleAxisd rot(R);
  return rot.angle() * rot.axis();
}


Eigen::Vector3d Image::tvec() {
  return Eigen::Vector3d(tx, ty, tz);
}


Eigen::Matrix<double, 3, 4> Image::proj_matrix() {
  return invert_proj_matrix(compose_proj_matrix(rvec(), tvec()));
}
