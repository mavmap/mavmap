/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
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
camera_model_image2world(const std::vector<Eigen::Vector2d>& image_points,
                         std::vector<Eigen::Vector2d>& world_points,
                         const int model_code,
                         const double params[]) {
  world_points.resize(image_points.size());
  for (size_t i=0; i<world_points.size(); ++i) {
    const Eigen::Vector2d& image_point = image_points[i];
    Eigen::Vector2d& world_point = world_points[i];
    double world_point_z;
    camera_model_image2world(image_point(0), image_point(1),
                             world_point(0),
                             world_point(1),
                             world_point_z,
                             model_code,
                             params);
    world_point(0) /= world_point_z;
    world_point(1) /= world_point_z;
  }

}


double camera_model_image2world_threshold(const double threshold,
                                          const int model_code,
                                          const double params[]) {
  const double focal = (params[0] + params[1]) / 2;
  return threshold / focal;
}
