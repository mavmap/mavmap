/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <Eigen/Core>

#include "base3d/essential_matrix.h"
#include "base3d/p3p.h"
#include "base3d/similarity_transform.h"
#include "util/test.h"


void test_estimation() {

  for (double rx=0; rx<1; rx+=0.2) {

    for (double tx=0; tx<50; tx+=2) {

      SimilarityTransform3D orig_tform(1, rx, 0.2, 0.3, tx, 2, 3);

      Eigen::Matrix<double, 4, 3> points3D_world;
      points3D_world << 1, 1, 1,
                        0, 1, 1,
                        3, 1, 4,
                        2, 1, 7;

      // Project points to camera coordinate system
      Eigen::Matrix<double, 4, 3> points3D_camera;
      for (size_t i=0; i<(size_t)points3D_world.rows(); ++i) {
        Eigen::Vector3d point3D_camera = points3D_world.row(i);
        orig_tform.transform_point(point3D_camera);
        points3D_camera.row(i) = point3D_camera;
      }

      // Generate image points
      Eigen::Matrix<double, 4, 2> points2D = points3D_camera.block<4, 2>(0, 0);
      points2D.col(0) = points2D.col(0).cwiseQuotient(points3D_camera.col(2));
      points2D.col(1) = points2D.col(1).cwiseQuotient(points3D_camera.col(2));

      P3PEstimator estimator;

      const Eigen::Matrix<double, 3, 4> model
        = estimator.estimate(points2D, points3D_world)[0];

      // Test if correct transformation has been determined
      const double matrix_diff
        = (orig_tform.matrix().block<3, 4>(0, 0) - model).norm();
      ASSERT_ALMOST_EQUAL(matrix_diff, 0, 1e-5);

      // Test residuals of exact points
      std::vector<double> residuals;
      estimator.residuals(points2D, points3D_world, model, residuals);
      for (size_t i=0; i<residuals.size(); ++i) {
        ASSERT_ALMOST_EQUAL(residuals[i], 0, 1e-6);
      }

      // Test residuals of faulty points
      points3D_world.col(0).fill(20);
      estimator.residuals(points2D, points3D_world, model, residuals);
      for (size_t i=0; i<residuals.size(); ++i) {
        ASSERT_GREATER(residuals[i], 2);
      }

    }

  }

}


int main(int argc, char* argv[]) {

  test_estimation();

  return 0;

}
