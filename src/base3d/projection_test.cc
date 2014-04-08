/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <Eigen/Core>

#include "base3d/projection.h"
#include "util/test.h"


void test_euler_angles() {

  const double rx = 0.1;
  const double ry = 0.2;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  euler_angles_from_rot_mat(rot_mat_from_euler_angles(rx, ry, rz), rxx, ryy, rzz);

  ASSERT_ALMOST_EQUAL(rx, rxx, 1e-6);
  ASSERT_ALMOST_EQUAL(ry, ryy, 1e-6);
  ASSERT_ALMOST_EQUAL(rz, rzz, 1e-6);

}


int main(int argc, char* argv[]) {

  test_euler_angles();

  return 0;

}
