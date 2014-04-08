/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "feature_cache.h"


FeatureCache::FeatureCache() {}


FeatureCache::FeatureCache(const std::string& path,
                           const std::vector<Image>& image_data,
                           const SURFOptions& surf_options)
    : path_(ensure_trailing_slash(path)),
      image_data_(image_data),
      surf_options_(surf_options) {

  descriptor_extractor_ = cv::SURF(surf_options_.hessian_threshold,
                                   surf_options_.num_octaves,
                                   surf_options_.num_octave_layers,
                                   true, false);
  descriptor_extractor_adaptive_
    = AdaptiveSURF(descriptor_extractor_,
                   surf_options_.adaptive_min_per_cell,
                   surf_options_.adaptive_max_per_cell,
                   surf_options_.adaptive_cell_rows,
                   surf_options_.adaptive_cell_cols);
}


void FeatureCache::query(const size_t image_idx,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors) {

  if (image_idx >= image_data_.size()) {
    throw std::range_error("Image index does not exist.");
  }

  keypoints.clear();
  descriptors.release();

  const std::string params_path
    = path_ + image_data_[image_idx].name + params_suffix;
  const std::string kp_path
    = path_ + image_data_[image_idx].name + keypoints_suffix;
  const std::string desc_path
    = path_ + image_data_[image_idx].name + descriptors_suffix;

  bool params_changed = false;
  if (!boost::filesystem::exists(params_path)) {
    params_changed = true;
  } else {
    // Check whether one of the parameters has changed
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(params_path, pt);

    try {
      if (surf_options_.adaptive != pt.get<bool>("adaptive")
          || surf_options_.hessian_threshold
            != pt.get<double>("hessian_threshold")
          || surf_options_.num_octaves
            != pt.get<int>("num_octaves")
          || surf_options_.num_octave_layers
            != pt.get<int>("num_octave_layers")) {
        params_changed = true;
      }
    }
    catch(...) {
      params_changed = true;
    }

    if (surf_options_.adaptive) {  // adaptive=true
      try {
        if (surf_options_.adaptive_min_per_cell
              != pt.get<size_t>("adaptive_min_per_cell")
            || surf_options_.adaptive_max_per_cell
              != pt.get<size_t>("adaptive_max_per_cell")
            || surf_options_.adaptive_cell_rows
              != pt.get<size_t>("adaptive_cell_rows")
            || surf_options_.adaptive_cell_cols
              != pt.get<size_t>("adaptive_cell_cols")) {
          params_changed = true;
        }
      }
      catch(...) {
        params_changed = true;
      }
    }
  }

  if (params_changed) {
    clear(image_idx);
    // Save new parameters
    boost::property_tree::ptree pt;
    pt.put("hessian_threshold", surf_options_.hessian_threshold);
    pt.put("num_octaves", surf_options_.num_octaves);
    pt.put("num_octave_layers", surf_options_.num_octave_layers);
    pt.put("adaptive", surf_options_.adaptive);
    if (surf_options_.adaptive) {
      pt.put("adaptive_min_per_cell", surf_options_.adaptive_min_per_cell);
      pt.put("adaptive_max_per_cell", surf_options_.adaptive_max_per_cell);
      pt.put("adaptive_cell_rows", surf_options_.adaptive_cell_rows);
      pt.put("adaptive_cell_cols", surf_options_.adaptive_cell_cols);
    }
    boost::property_tree::write_ini(params_path, pt);
  }

  if (!boost::filesystem::exists(kp_path)
      || !boost::filesystem::exists(desc_path)) {

    const cv::Mat image = read_image_(image_idx);
    if (surf_options_.adaptive) {

      descriptor_extractor_adaptive_.detect(image, keypoints);
      descriptor_extractor_adaptive_.extract(image, keypoints, descriptors);
    } else {
      descriptor_extractor_(image, cv::noArray(),
                            keypoints, descriptors, false);
    }

    // write keypoints
    std::ofstream kp_file(kp_path.c_str(), std::ios::binary);
    const size_t kp_num_bytes = keypoints.size() * sizeof(cv::KeyPoint);
    kp_file.write((char*)&kp_num_bytes, sizeof(size_t));
    kp_file.write((char*)&keypoints[0], kp_num_bytes);
    kp_file.close();

    // write descriptors
    std::ofstream desc_file(desc_path.c_str(), std::ios::binary);
    const size_t desc_num_bytes = descriptors.total() * descriptors.elemSize();
    const int desc_type = descriptors.type();
    desc_file.write((char*)&desc_num_bytes, sizeof(size_t));
    desc_file.write((char*)&descriptors.rows, sizeof(size_t));
    desc_file.write((char*)&descriptors.cols, sizeof(size_t));
    desc_file.write((char*)&desc_type, sizeof(int));
    desc_file.write((char*)descriptors.data, desc_num_bytes);
    desc_file.close();

  } else {

    // Read keypoints
    std::ifstream kp_file(kp_path.c_str(), std::ios::binary);
    size_t kp_num_bytes;
    kp_file.read((char*)&kp_num_bytes, sizeof(size_t));
    keypoints.resize(kp_num_bytes / sizeof(cv::KeyPoint));
    kp_file.read((char*)&keypoints[0], kp_num_bytes);

    // Read descriptors
    std::ifstream desc_file(desc_path.c_str(), std::ios::binary);
    size_t desc_num_bytes, desc_rows, desc_cols;
    int desc_type;
    desc_file.read((char*)&desc_num_bytes, sizeof(size_t));
    desc_file.read((char*)&desc_rows, sizeof(size_t));
    desc_file.read((char*)&desc_cols, sizeof(size_t));
    desc_file.read((char*)&desc_type, sizeof(int));
    descriptors = cv::Mat(desc_rows, desc_cols, desc_type);
    desc_file.read((char*)descriptors.data, desc_num_bytes);

  }

}


void FeatureCache::query_dimensions(const size_t image_idx,
                                    size_t* rows, size_t* cols,
                                    size_t* channels, float* diagonal) {

  // Load image once to retrieve dimensions
  const std::string metadata_path
    = path_ + image_data_[image_idx].name + dimensions_suffix;
  if (!boost::filesystem::exists(metadata_path)) {
    read_image_(image_idx);
  }

  boost::property_tree::ptree pt;
  boost::property_tree::ini_parser::read_ini(metadata_path, pt);

  if (rows != NULL) {
    *rows = pt.get<size_t>("rows");
  }
  if (cols != NULL) {
    *cols = pt.get<size_t>("cols");
  }
  if (channels != NULL){
    *channels = pt.get<size_t>("channels");
  }
  if (diagonal != NULL) {
    *diagonal = pt.get<double>("diagonal");
  }

}


void FeatureCache::clear() {
  // Clear entire cache
  boost::filesystem::remove(path_);
  boost::filesystem::create_directory(path_);
}


void FeatureCache::clear(const size_t image_idx) {
  const Image& image = image_data_[image_idx];
  const std::string params_path
    = path_ + image.name + params_suffix;
  const std::string metadata_path
    = path_ + image.name + dimensions_suffix;
  const std::string kp_path
    = path_ + image.name + keypoints_suffix;
  const std::string desc_path
    = path_ + image.name + descriptors_suffix;
  boost::filesystem::remove(params_path);
  boost::filesystem::remove(metadata_path);
  boost::filesystem::remove(kp_path);
  boost::filesystem::remove(desc_path);
}


cv::Mat FeatureCache::read_image_(const size_t image_idx) {

  Image& image = image_data_[image_idx];

  cv::Mat mat = image.read(0);

  // Write frame dimensions, so they can be retrieved without loading the
  // complete image
  const std::string metadata_path
    = path_ + image.name + dimensions_suffix;
  if (!boost::filesystem::exists(metadata_path)) {
    boost::property_tree::ptree pt;
    pt.put("rows", image.rows);
    pt.put("cols", image.cols);
    pt.put("channels", image.channels);
    pt.put("diagonal", image.diagonal);
    boost::property_tree::write_ini(metadata_path, pt);
  }

  return mat;

}
