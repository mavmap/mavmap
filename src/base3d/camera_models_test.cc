/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <vector>
#include <iostream>

#include "camera_models.h"
#include "util/test.h"


template <typename CameraModel>
void test_model(const std::vector<double>& camera_params) {

  double u, v, x, y, z;

  // world 2 image 2 world
  double x0 = 0.5;
  double y0 = 0.23;
  double z0 = 1;
  CameraModel::world2image(x0, y0, z0, u, v, camera_params.data());
  CameraModel::image2world(u, v, x, y, z, camera_params.data());
  x /= z;
  y /= z;
  z /= z;
  ASSERT_ALMOST_EQUAL(x, x0, 1e-5);
  ASSERT_ALMOST_EQUAL(y, y0, 1e-5);
  ASSERT_ALMOST_EQUAL(z, z0, 1e-5);

  // image 2 world 2 image
  double u0 = 200;
  double v0 = 100;
  CameraModel::image2world(u0, v0, x, y, z, camera_params.data());
  CameraModel::world2image(x, y, z, u, v, camera_params.data());
  ASSERT_ALMOST_EQUAL(u, u0, 1e-1);
  ASSERT_ALMOST_EQUAL(v, v0, 1e-1);

  // principal point
  CameraModel::world2image(0.0, 0.0, 1.0, u, v, camera_params.data());
  ASSERT_ALMOST_EQUAL(u, camera_params[2], 1e-6);
  ASSERT_ALMOST_EQUAL(v, camera_params[3], 1e-6);
  CameraModel::image2world(camera_params[2], camera_params[3], x, y, z,
                           camera_params.data());
  x /= z;
  y /= z;
  z /= z;
  ASSERT_ALMOST_EQUAL(x, 0, 1e-4);
  ASSERT_ALMOST_EQUAL(y, 0, 1e-4);
  ASSERT_ALMOST_EQUAL(z, 1, 1e-4);

}


int main(int argc, char* argv[]) {

  test_model<PinholeCameraModel>({651.123, 655.123,
                                  386.123, 511.123});

  test_model<CataCameraModel>({651.123, 655.123,
                               386.123, 511.123,
                               -0.471, 0.223,
                               -0.001, 0.001,
                               0});
  test_model<CataCameraModel>({651.123, 655.123,
                               386.123, 511.123,
                               -0.471, 0.223,
                               -0.001, 0.001,
                               1});
  test_model<CataCameraModel>({651.123, 655.123,
                               386.123, 511.123,
                               -0.471, 0.223,
                               -0.001, 0.001,
                               0.5});

  test_model<OpenCVCameraModel>({651.123, 655.123,
                                 386.123, 511.123,
                                 -0.471, 0.223,
                                 -0.001, 0.001});

  return 0;

}
