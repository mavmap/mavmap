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

#include "io.h"


std::vector<Image> read_image_data(const std::string path,
                                   const std::string root_path,
                                   const std::string prefix,
                                   const std::string suffix,
                                   const std::string ext) {

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
    std::stringstream item_stream;

    Image image;

    // TIMESTAMP (, PATH)
    std::getline(line_stream, item, ',');
    image.timestamp = item;
    image.path = root_path + prefix + image.timestamp + suffix + ext;

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


Eigen::Matrix3d read_calib_matrix(const std::string path) {

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

  return calib_matrix;

}


void write_image_data(const std::string path,
                      SequentialMapper& mapper) {

  std::vector<Image> image_data = mapper.image_data;

  std::ofstream file;
  file.open(path.c_str());

  file << "# BASENAME, ROLL, PITCH, YAW, LAT, LON, ALT, LOCAL_HEIGHT, ";
  file << "TX, TY, TZ" << std::endl;

  for (size_t i=0; i<image_data.size(); ++i) {

    size_t image_id;
    try {
      image_id = mapper.get_image_id(i);
    }
    catch(std::exception& e) {
      continue;
    }

    double rx, ry, rz, tx, ty, tz;

    extract_exterior_params(mapper.feature_manager.rvecs[image_id],
                            mapper.feature_manager.tvecs[image_id],
                            rx, ry, rz, tx, ty, tz);

    file << image_data[i].timestamp << ", ";
    file << rx << ", ";
    file << ry << ", ";
    file << rz << ", ";
    // Use original data from input imagedata.txt
    file << image_data[i].lat << ", ";
    file << image_data[i].lon << ", ";
    file << image_data[i].alt << ", ";
    file << image_data[i].local_height << ", ";
    file << tx << ", ";
    file << ty << ", ";
    file << tz;

    file << std::endl;

  }

  file << std::endl;

  file.close();

}


void write_point_cloud(const std::string path,
                       SequentialMapper& mapper,
                       const size_t min_track_len,
                       const double error_threshold,
                       const double coordinate_norm_threshold) {

  std::unordered_map<size_t, Eigen::Vector3d>& points3D
    = mapper.feature_manager.points3D;
  std::unordered_map<size_t, Eigen::Vector2d>& points2D
    = mapper.feature_manager.points2D;
  std::unordered_map<size_t, size_t>& point2D_to_point3D
    = mapper.feature_manager.point2D_to_point3D;
  std::unordered_map<size_t, std::vector<size_t> >& point3D_to_points2D
    = mapper.feature_manager.point3D_to_points2D;
  std::unordered_map<size_t, std::vector<size_t> >&
    image_to_points2D = mapper.feature_manager.image_to_points2D;

  std::unordered_map<size_t, std::vector<Eigen::Vector3d>> point3D_to_colors;

  std::vector<Image>& image_data = mapper.image_data;

  // Extract color for each point in all observed images
  for (size_t image_idx=0; image_idx<image_data.size(); image_idx++) {
    size_t image_id;
    try {
      image_id = mapper.get_image_id(image_idx);
    }
    catch(std::exception& e) {
      continue;
    }
    // Always read as color image, flag=1
    const cv::Mat image = image_data[image_idx].read(1);
    // Split image channels into separate color images in the order BGR
    cv::Mat image_rgb[3];
    cv::split(image, image_rgb);

    // Iterate through all 2D points in image
    const std::vector<size_t>& point2D_ids = image_to_points2D[image_id];
    for (size_t j=0; j<point2D_ids.size(); ++j) {
      const size_t point2D_id = point2D_ids[j];
      if (point2D_to_point3D.count(point2D_id)) {
        const size_t point3D_id = point2D_to_point3D[point2D_id];
        const Eigen::Vector2d& point2D = points2D[point2D_id];
        // Determine mean color around point in 3x3 window,
        // to account for image noise
        cv::Rect roi((size_t)point2D(0)-1, (size_t)point2D(1)-1, 3, 3);
        // OpenCV uses BGR color ordering
        const Eigen::Vector3d rgb(cv::mean(image_rgb[2](roi))[0] / 255.0,
                                  cv::mean(image_rgb[1](roi))[0] / 255.0,
                                  cv::mean(image_rgb[0](roi))[0] / 255.0);
        point3D_to_colors[point3D_id].push_back(rgb);
      }
    }
  }

  std::ofstream file;
  file.open(path.c_str());

  file << "#VRML V2.0 utf8\n";
  file << "Background { skyColor [1.0 1.0 1.0] } \n";
  file << "Shape{ appearance Appearance {\n";
  file << " material Material {emissiveColor 1 1 1} }\n";
  file << " geometry PointSet {\n";
  file << " coord Coordinate {\n";
  file << "  point [\n";

  std::vector<Eigen::Vector3d> point3D_to_color;

  for (auto it=points3D.begin(); it!=points3D.end(); ++it) {

    const size_t point3D_id = it->first;

    if (point3D_to_points2D[point3D_id].size() < min_track_len) {
      continue;
    }

    const Eigen::Vector3d& point3D = it->second;

    double error;
    try {
      error = mapper.get_point3D_error(point3D_id);
    }
    catch(std::exception& e) {
      continue;
    }

    if (error > error_threshold
        || point3D.norm() > coordinate_norm_threshold) {
      continue;
    }

    // Write point coordinates
    file << point3D(0) << ", "
         << point3D(1) << ", "
         << point3D(2) << std::endl;

    // Average colors of 3D point in all observed images
    std::vector<Eigen::Vector3d>& colors = point3D_to_colors[point3D_id];
    Eigen::Vector3d color(0, 0, 0);
    for (size_t i=0; i<colors.size(); ++i) {
      color += colors[i];
    }
    color /= colors.size();
    point3D_to_color.push_back(color);
  }

  file << " ] }\n";
  file << " color Color { color [\n";

  // Write color data for each point
  for (size_t i = 0; i <point3D_to_color.size(); ++i) {
    const Eigen::Vector3d color = point3D_to_color[i];
    file << color(0) << " " << color(1) << " " << color(2) << "\n";
  }

  file << " ] } } }\n";

  file.close();

}


void write_camera_poses(const std::string path,
                        FeatureManager& feature_manager,
                        const double scale, const double red,
                        const double green, const double blue) {

  std::unordered_map<size_t, Eigen::Vector3d> rvecs
    = feature_manager.rvecs;
  std::unordered_map<size_t, Eigen::Vector3d> tvecs
    = feature_manager.tvecs;

  std::ofstream file;
  file.open(path.c_str());

  // Build camera base model at origin

  double six = scale * 1.5;
  double siy = scale;

  Eigen::Vector3d p1(-six, -siy, six*1.0*2.0);
  Eigen::Vector3d p2(+six, -siy, six*1.0*2.0);
  Eigen::Vector3d p3(+six, +siy, six*1.0*2.0);
  Eigen::Vector3d p4(-six, +siy, six*1.0*2.0);

  Eigen::Vector3d p5(0, 0, 0);
  Eigen::Vector3d p6(-six/3.0, -siy/3.0, six*1.0*2.0);
  Eigen::Vector3d p7(+six/3.0, -siy/3.0, six*1.0*2.0);
  Eigen::Vector3d p8(+six/3.0, +siy/3.0, six*1.0*2.0);
  Eigen::Vector3d p9(-six/3.0, +siy/3.0, six*1.0*2.0);

  std::vector<Eigen::Vector3d> points = {p1, p2, p3, p4, p5, p6, p7, p8, p9};

  file << "#VRML V2.0 utf8\n";

  for (size_t i=1; i<=rvecs.size(); ++i) {

    file << "Shape{\n";
    file << " appearance Appearance {\n";
    file << "  material DEF Default-ffRffGffB Material {\n";
    file << "  ambientIntensity 0\n";
    file << "  diffuseColor " << " " << red << " "
                                     << green << " "
                                     << blue << "\n";
    file << "  emissiveColor 0.1 0.1 0.1 } }\n";
    file << " geometry IndexedFaceSet {\n";
    file << " solid FALSE \n";
    file << " colorPerVertex TRUE \n";
    file << " ccw TRUE \n";

    file << " coord Coordinate {\n";
    file << " point [\n";

    Eigen::Transform<double, 3, Eigen::Affine> transform;
    transform.matrix().block<3, 4>(0, 0)
      = invert_proj_matrix(compose_proj_matrix(rvecs[i], tvecs[i]));

    // Move camera base model to camera pose
    for (size_t p=0; p<points.size(); p++) {
      Eigen::Vector3d pt = transform * points[p];
      file << std::setw(20) << pt(0)
           << std::setw(20) << pt(1)
           << std::setw(20) << pt(2)
           << "\n";
    }

    file << " ]\n }";

    file << "\n color Color {color [\n";
    for (size_t p=0; p<points.size(); p++) {
      file << " " << red << " " << green << " " << blue << "\n";
    }

    file << "\n] }\n";

    file << "\n coordIndex [\n";
    file << " 0, 1, 2, 3, -1\n";
    file << " 5, 6, 4, -1\n";
    file << " 6, 7, 4, -1\n";
    file << " 7, 8, 4, -1\n";
    file << " 8, 5, 4, -1\n";
    file << " \n] \n";

    file << " texCoord TextureCoordinate { point [\n";
    file << "  1 1,\n";
    file << "  0 1,\n";
    file << "  0 0,\n";
    file << "  1 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";
    file << "  0 0,\n";

    file << " ] }\n";
    file << "} }\n";

  }

  file.close();

}


void write_track(std::string path,
                 SequentialMapper& mapper,
                 const size_t point3D_id,
                 const int radius, const int thickness) {

  path = ensure_trailing_slash(path);

  std::vector<size_t> point2D_ids
    = mapper.feature_manager.point3D_to_points2D[point3D_id];

  if (point2D_ids.size() == 0) {
    return;
  }

  std::ostringstream point3D_id_str;
  point3D_id_str << point3D_id;

  std::ostringstream track_length_str;
  track_length_str << point2D_ids.size();

  for (size_t i=0; i<point2D_ids.size(); ++i) {

    size_t point2D_id = point2D_ids[i];

    for (size_t image_id=1; image_id<=mapper.feature_manager.get_num_images();
         image_id++) {

      const std::vector<size_t> image_points2D
        = mapper.feature_manager.image_to_points2D[image_id];
      if (std::find(image_points2D.begin(),
                    image_points2D.end(),
                    point2D_id)
          != image_points2D.end()) {
        const size_t image_idx = mapper.get_image_idx(image_id);
        // read image
        cv::Mat image = mapper.image_data[image_idx].read(1);
        // draw 2D point
        const Eigen::Vector2d& point2D
          = mapper.feature_manager.points2D[point2D_id];
        const cv::Point2f point2D_cv(point2D(0), point2D(1));
        cv::circle(image, point2D_cv, radius, cv::Scalar(0, 0, 255),
                   thickness);
        // write to output path
        std::ostringstream image_id_str;
        image_id_str << image_id;
        cv::imwrite(path
                    + "LEN" + track_length_str.str()
                    + "-P3D#" + point3D_id_str.str()
                    + "-IMG#" + image_id_str.str() + ".jpg", image);
      }
    }

  }
}


void write_camera_connections(const std::string& path,
                              SequentialMapper& mapper) {

  std::unordered_map<size_t, std::set<size_t> > image_pair_idxs
    = mapper.get_image_pair_idxs();

  std::ofstream file;
  file.open(path.c_str());

  const double eps = 1e-5;

  file << "#VRML V2.0 utf8\n";
  file << "Background { skyColor [1.0 1.0 1.0] } \n";

  for (auto it1=image_pair_idxs.begin(); it1!=image_pair_idxs.end(); ++it1) {

    const size_t image_id1 = mapper.get_image_id(it1->first);

    Eigen::Matrix<double, 3, 4> matrix1 = invert_proj_matrix(
      compose_proj_matrix(mapper.feature_manager.rvecs[image_id1],
                          mapper.feature_manager.tvecs[image_id1]));
    const Eigen::Vector3d tvec1 = matrix1.block<3, 1>(0, 3);

    for (auto it2=it1->second.begin(); it2!=it1->second.end(); ++it2) {

      const size_t image_id2 = mapper.get_image_id(*it2);

      Eigen::Matrix<double, 3, 4> matrix2 = invert_proj_matrix(
        compose_proj_matrix(mapper.feature_manager.rvecs[image_id2],
                            mapper.feature_manager.tvecs[image_id2]));
      const Eigen::Vector3d tvec2 = matrix2.block<3, 1>(0, 3);

      file << "Shape{\n";
      file << "appearance Appearance {\n";
      file << " material Material {emissiveColor 1 1 1} }\n";

      file << "geometry IndexedFaceSet {\n";
      file << " coord Coordinate {\n";
      file << "  point [\n";

      // Produce nearly degenerated polygon, as Meshlab is not able
      // to visualize IndexedLineSet

      file << "   " << tvec1(0) << " "
           << tvec1(1) << " "
           << tvec1(2) << std::endl;

      file << "   " << tvec1(0) << " "
           << tvec1(1) << " "
           << tvec1(2) + eps << std::endl;

      file << "   " << tvec2(0) << " "
           << tvec2(1) << " "
           << tvec2(2) << std::endl;

      file << "  ] }\n";

      file << " color Color {\n";
      file << "  color [\n";
      file << "   0 1.0 1.0\n";
      file << "  ] }\n";

      file << " coordIndex [\n";
      file << "  0, 1, 2, -1\n";
      file << " ]\n";

      file << " colorIndex [\n";
      file << "  0\n";
      file << " ]\n";

      file << "} }\n";

    }

  }

  file.close();

}
