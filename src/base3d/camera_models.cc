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

#include "camera_models.h"


int camera_model_name_to_code(const std::string& name) {
  if (name == "PINHOLE") {
    return PinholeCameraModel::code;
  } else if (name == "OPENCV") {
    return OpenCVCameraModel::code;
  } else if (name == "CATA") {
    return CataCameraModel::code;
  }
  return -1;
}


void
camera_model_world2image(const double& x, const double& y, const double& z,
                         double& u, double& v,
                         const int model_code,
                         const double params[]) {
  switch (model_code) {
    case PinholeCameraModel::code:
      PinholeCameraModel::world2image(x, y, z, u, v, params);
      break;
    case OpenCVCameraModel::code:
      OpenCVCameraModel::world2image(x, y, z, u, v, params);
      break;
    case CataCameraModel::code:
      CataCameraModel::world2image(x, y, z, u, v, params);
      break;
  }
}


void
camera_model_image2world(const double& u, const double& v,
                         double& x, double& y, double& z,
                         const int model_code,
                         const double params[]) {
  switch (model_code) {
    case PinholeCameraModel::code:
      PinholeCameraModel::image2world(u, v, x, y, z, params);
      break;
    case OpenCVCameraModel::code:
      OpenCVCameraModel::image2world(u, v, x, y, z, params);
      break;
    case CataCameraModel::code:
      CataCameraModel::image2world(u, v, x, y, z, params);
      break;
  }
}


double camera_model_image2world_threshold(const double threshold,
                                          const int model_code,
                                          const double params[]) {
  const double focal = (params[0] + params[1]) / 2;
  return threshold / focal;
}
