/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "detection.h"


LoopDetector::LoopDetector() {
  LoopDetector(boost::filesystem::unique_path().native(), 0, 0);
}


LoopDetector::LoopDetector(const std::string& voc_tree_path,
                           const size_t max_num_images,
                           const size_t max_num_visual_words)
  : voc_tree_path_(voc_tree_path),
    max_num_images_(max_num_images),
    max_num_visual_words_(max_num_visual_words),
    num_images_(0) {}


void LoopDetector::init() {
  voc_tree_ = VocTree();
  voc_tree_.init(voc_tree_path_.c_str());

  database_ = VocTreeDatabase();
  database_.init(max_num_images_, voc_tree_.nrvisualwords(),
                 max_num_visual_words_);
}


void LoopDetector::add_image(const size_t image_id,
                             const cv::Mat& descriptors) {

  if (num_images_ >= max_num_images_) {
    throw std::range_error("Maximum number of images reached.");
  }
  if (descriptors.cols != 128) {
    throw std::range_error("Descriptor length must be 128.");
  }

  size_t num_visual_words;
  uint32_t* visual_words = descriptors_to_visual_words_(descriptors,
                                                        num_visual_words);

  // Note: for simple similarity test not necessary
  // HeapSort sorter;
  // sorter.sortarray(visual_words, size);
  // unique(visual_words, (unsigned int*)&size);

  database_.insertdoc(visual_words, num_visual_words, image_id);

  num_images_ += 1;

  delete[] visual_words;

}


void LoopDetector::query(const cv::Mat& descriptors,
                         std::vector<int>& image_ids,
                         std::vector<float>& scores,
                         const size_t num_images,
                         const bool update_weights) {

  if (descriptors.cols != 128) {
    throw std::range_error("Descriptor length must be 128.");
  }

  if (update_weights) {
    database_.computeidf();
    database_.normalize();
  }

  size_t num_visual_words;
  uint32_t* visual_words = descriptors_to_visual_words_(descriptors,
                                                        num_visual_words);

  image_ids.clear();
  scores.clear();
  image_ids.resize(num_images);
  scores.resize(num_images);

  database_.querytopn(visual_words, num_visual_words, num_images,
                      &image_ids[0], &scores[0]);

  delete[] visual_words;

}


uint32_t* LoopDetector::descriptors_to_visual_words_(
    const cv::Mat& descriptors, size_t& num_visual_words) {

  num_visual_words = std::min<size_t>(descriptors.rows, max_num_visual_words_);
  uint32_t* visual_words = new uint32_t[num_visual_words];
  uint8_t descriptor_uint8[128];

  for (size_t i=0; i<num_visual_words; ++i) {
    // Pointer to descriptor of i-th feature, OpenCV arrays are stored as
    // C-contiguous (row-major order)
    const float* descriptor = (float*)descriptors.ptr(i);
    // Transform values between -1...+1 to -127...+127 and shift it to 0...255
    for (size_t j=0; j<128; ++j) {
      descriptor_uint8[j] = floor(descriptor[j] * 127 + 127);
    }
    voc_tree_.quantize(&visual_words[i], descriptor_uint8);
  }

  return visual_words;
}
