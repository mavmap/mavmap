/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "io.h"


std::vector<Image> read_image_data(const std::string& path,
                                   const std::string& root_path,
                                   const std::string& prefix,
                                   const std::string& suffix,
                                   const std::string& ext) {

  std::ifstream file(path.c_str());

  std::string line;
  std::string item;

  std::set<int> camera_idxs;

  std::vector<Image> image_data;

  while (std::getline(file, line)) {

    if (line.size() == 0 || line.at(0) == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    Image image;

    // NAME (, PATH)
    std::getline(line_stream, item, ',');
    image.name = item;
    image.path = root_path + prefix + image.name + suffix + ext;

    // ROLL
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.roll = boost::lexical_cast<double>(item);
    // PITCH
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.pitch = boost::lexical_cast<double>(item);
    // YAW
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.yaw = boost::lexical_cast<double>(item);

    // LAT
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.lat = boost::lexical_cast<double>(item);
    // LON
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.lon = boost::lexical_cast<double>(item);
    // ALT
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.alt = boost::lexical_cast<double>(item);

    // LOCAL_HEIGHT
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.local_height = boost::lexical_cast<double>(item);

    // TX
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.tx = boost::lexical_cast<double>(item);
    // TY
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.ty = boost::lexical_cast<double>(item);
    // TZ
    std::getline(line_stream, item, ',');
    boost::trim(item);
    image.tz = boost::lexical_cast<double>(item);

    // CAM
    if (line_stream.eof()) {
      if (image_data.size() == 0) {
        throw std::domain_error("You must specify a camera model for the "
                                "first image.");
      }
      // No separate camera data specified for this line, so reuse previous
      // camera model
      image.camera_idx = image_data.back().camera_idx;
      image.camera_model = image_data.back().camera_model;
      image.camera_params = image_data.back().camera_params;
    } else {

      // CAM_IDX
      std::getline(line_stream, item, ',');
      boost::trim(item);
      image.camera_idx = boost::lexical_cast<int>(item);

      if (camera_idxs.count(image.camera_idx) != 0) {
        throw std::domain_error("Two cameras with the same index have been "
                                "defined.");
      }

      camera_idxs.insert(image.camera_idx);

      // CAM_MODEL
      std::getline(line_stream, item, ',');
      boost::trim(item);
      image.camera_model = boost::to_upper_copy(item);

      // CAM_PARAMS (variable number of parameters)
      while (true) {
        if (line_stream.eof()) {
          break;
        }
        std::getline(line_stream, item, ',');
        boost::trim(item);
        image.camera_params.push_back(boost::lexical_cast<double>(item));
      }
    }

    if (image.camera_model == "") {
      throw std::domain_error("No camera model specified.");
    }
    if (image.camera_params.size() < 4) {
      throw std::domain_error("You must at least specify 4 parameters for "
                              "a camera model: focal_length (fx, fy) and "
                              "principal point (cx, cy)");
    }

    image_data.push_back(image);

  }

  file.close();

  return image_data;
}


Eigen::Matrix3d read_calib_matrix(const std::string& path) {

  std::ifstream file(path.c_str());

  Eigen::Matrix3d calib_matrix = Eigen::Matrix3d::Zero();

  std::string line;
  std::string item;

  size_t matrix_row = 0;
  while (std::getline(file, line) && matrix_row < 3) {

    if (line.size() == 0 || line.at(0) == '#') {
      continue;
    }

    std::istringstream line_stream(line);
    std::istringstream item_stream;

    std::getline(line_stream, item, ',');
    item_stream.clear();
    item_stream.str(item);
    item_stream >> calib_matrix(matrix_row, 0);

    std::getline(line_stream, item, ',');
    item_stream.clear();
    item_stream.str(item);
    item_stream >> calib_matrix(matrix_row, 1);

    std::getline(line_stream, item, ';');
    item_stream.clear();
    item_stream.str(item);
    item_stream >> calib_matrix(matrix_row, 2);

    matrix_row += 1;

  }

  file.close();

  return calib_matrix;
}


void init_gcp_(std::string line, ControlPoint& control_point) {
  if (line.at(1) == '#') {
    line = line.substr(2);
    control_point.fixed = true;
  } else {
    line = line.substr(1);
    control_point.fixed = false;
  }

  std::istringstream line_stream(line);

  std::string item;

  control_point.points2D.clear();

  std::getline(line_stream, item, ',');
  boost::trim(item);
  control_point.name = item;

  std::getline(line_stream, item, ',');
  boost::trim(item);
  control_point.xyz(0) = boost::lexical_cast<double>(item);

  std::getline(line_stream, item, ',');
  boost::trim(item);
  control_point.xyz(1) = boost::lexical_cast<double>(item);

  std::getline(line_stream, item, ',');
  boost::trim(item);
  control_point.xyz(2) = boost::lexical_cast<double>(item);
}


void read_control_point_observation_(std::string line, ControlPoint& control_point) {
  std::pair<size_t, Eigen::Vector2d> point2D;

  std::istringstream line_stream(line);

  std::string item;

  // Image index
  std::getline(line_stream, item, ',');
  boost::trim(item);
  point2D.first = boost::lexical_cast<size_t>(item);

  // Pixel X-coordinate
  std::getline(line_stream, item, ',');
  boost::trim(item);
  point2D.second(0) = boost::lexical_cast<double>(item);

  // Pixel Y-coordinate
  std::getline(line_stream, item, ',');
  boost::trim(item);
  point2D.second(1) = boost::lexical_cast<double>(item);

  control_point.points2D.push_back(point2D);
}


std::vector<ControlPoint> read_control_point_data(const std::string& path) {
  std::ifstream file(path.c_str());

  std::string line;

  std::vector<ControlPoint> control_points;
  ControlPoint control_point;

  while (std::getline(file, line)) {
    if (line.size() == 0) {
      continue;
    }

    if (line.at(0) == '#') {
      init_gcp_(line, control_point);
    } else {
      read_control_point_observation_(line, control_point);
    }

    while (std::getline(file, line)) {

      // New 3D point
      if (line.size() == 0 || line.at(0) == '#') {
        if (control_point.points2D.size() == 0) {
          throw std::domain_error("control_point must have at least two points2D.");
        }
        control_points.push_back(control_point);
        if (line.size() > 0 && line.at(0) == '#') {
          init_gcp_(line, control_point);
        }
        break;
      }
      if (line.at(0) == '#') {
        init_gcp_(line, control_point);
      }

      read_control_point_observation_(line, control_point);
    }
  }

  if (control_points.size() > 0 && control_points[control_points.size()-1].name != control_point.name) {
    control_points.push_back(control_point);
  }

  file.close();

  return control_points;
}


void write_control_point_data(const std::string& path, SequentialMapper& mapper,
                              const std::vector<std::pair<size_t, ControlPoint>> control_points) {
  std::ofstream file;
  file.open(path.c_str());

  file << "# NAME, X, Y, Z, TRACK_LEN, MEAN_RESIDUAL" << std::endl;



  std::vector<Eigen::Vector3d> point3D_to_color;

  for (const auto& control_point : control_points) {
    const auto& point3D = mapper.feature_manager.points3D[control_point.first];

    file << control_point.second.name << ", "
         << std::setprecision(12) << point3D(0) << ", "
         << std::setprecision(12) << point3D(1) << ", "
         << std::setprecision(12) << point3D(2) << ", "
         << std::setprecision(12) << mapper.feature_manager.point3D_to_points2D[control_point.first].size() << ", "
         << std::setprecision(12) << mapper.get_point3D_error(control_point.first) << std::endl;
  }

  file << std::endl;

  file.close();
}
