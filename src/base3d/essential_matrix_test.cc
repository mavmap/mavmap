/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <Eigen/Core>

#include "base3d/camera_models.h"
#include "base3d/essential_matrix.h"
#include "util/test.h"


void test_ransac() {

  double points1_raw[] = {0.4964 ,1.0577,
                          0.3650, -0.0919,
                          -0.5412, 0.0159,
                          -0.5239, 0.9467,
                          0.3467, 0.5301,
                          0.2797, 0.0012,
                          -0.1986, 0.0460,
                          -0.1622, 0.5347,
                          0.0796, 0.2379,
                          -0.3946, 0.7969,
                          0.2, 0.7,
                          0.6, 0.3};

    double points2_raw[] = {0.7570, 2.7340,
                            0.3961, 0.6981,
                            -0.6014, 0.7110,
                            -0.7385, 2.2712,
                            0.4177, 1.2132,
                            0.3052, 0.4835,
                            -0.2171, 0.5057,
                            -0.2059, 1.1583,
                            0.0946, 0.7013,
                            -0.6236, 3.0253,
                            0.5, 0.9,
                            0.9, 0.2};

  const size_t num_points = 12;

  Eigen::MatrixXd points1(num_points, 2), points2(num_points, 2);
  for (size_t i=0; i<num_points; ++i) {
    points1.row(i) = Eigen::Vector2d(points1_raw[2*i], points1_raw[2*i+1]);
    points2.row(i) = Eigen::Vector2d(points2_raw[2*i], points2_raw[2*i+1]);
  }

  size_t num_inliers;
  std::vector<bool> inlier_mask;
  EssentialMatrixEstimator estimator;
  const Eigen::Matrix3d essential_matrix
    = RANSAC(estimator,
             points1,
             points2,
             5, // min_samples
             0.01, num_inliers, inlier_mask,
             50, // max_trials
             num_points + 1 // stop_num_inliers
             );

  std::vector<double> residuals;
  estimator.residuals(points1, points2, essential_matrix, residuals);

  for (size_t i=0; i<10; ++i) {
    assert(residuals[i] < 0.1);
  }
  assert(inlier_mask[10] == false);
  assert(inlier_mask[11] == false);

}


int main(int argc, char* argv[]) {

  test_ransac();

  return 0;

}
