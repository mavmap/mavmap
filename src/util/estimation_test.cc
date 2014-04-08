/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base3d/similarity_transform.h"
#include "util/estimation.h"
#include "util/test.h"


void test_ransac() {

  const size_t num_samples = 1000;
  const size_t num_outliers = 400;

  // Create some arbitrary transformation
  SimilarityTransform3D orig_tform(2, 2, 1, 2, 100, 10, 10);

  // Generate exact data
  Eigen::MatrixXd src(num_samples, 3);
  Eigen::MatrixXd dst(num_samples, 3);
  for (size_t i=0; i<num_samples; ++i) {
    Eigen::Vector3d s(i, sqrt(i)+2, sqrt(2*i+2));
    Eigen::Vector3d d = s;
    orig_tform.transform_point(d);
    src.row(i) = s;
    dst.row(i) = d;
  }

  // Add some faulty data
  for (size_t i=0; i<num_outliers; ++i) {
    dst.row(i) = Eigen::Vector3d((rand() % 1000) / 1000.0 - 2000,
                                 (rand() % 1000) / 1000.0 - 4000,
                                 (rand() % 1000) / 1000.0 - 6000);
  }

  // Robustly estimate transformation using RANSAC
  std::vector<bool> inlier_mask;
  size_t num_inliers;
  SimilarityTransform3DEstimator est_tform;
  Eigen::Matrix4d robust_matrix
    = RANSAC(est_tform, src, dst, 3, 10, num_inliers, inlier_mask);

  // Make sure outliers were detected correctly
  assert(num_inliers == num_samples - num_outliers);
  for (size_t i=0; i<num_samples; ++i) {
    if (i < num_outliers) {
      assert(inlier_mask[i] == false);
    } else {
      assert(inlier_mask[i] == true);
    }
  }

  // Make sure original transformation is estimated correctly
  const double matrix_diff = (orig_tform.matrix() - robust_matrix).norm();
  ASSERT_ALMOST_EQUAL(matrix_diff, 0, 1e-6);

}


int main(int argc, char* argv[]) {

  test_ransac();

  return 0;

}
