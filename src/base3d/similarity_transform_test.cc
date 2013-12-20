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

#include <Eigen/Core>

#include "base3d/similarity_transform.h"
#include "util/test.h"


void test_init() {

  SimilarityTransform3D tform(2, 1, 3.11, 0.66, 100, 10, 0.5);

  ASSERT_ALMOST_EQUAL(tform.scale(), 2, 1e-10);

  ASSERT_ALMOST_EQUAL(tform.rvec()[0], 1, 1e-10);
  ASSERT_ALMOST_EQUAL(tform.rvec()[1], 3.11, 1e-10);
  ASSERT_ALMOST_EQUAL(tform.rvec()[2], 0.66, 1e-10);

  ASSERT_ALMOST_EQUAL(tform.tvec()[0], 100, 1e-10);
  ASSERT_ALMOST_EQUAL(tform.tvec()[1], 10, 1e-10);
  ASSERT_ALMOST_EQUAL(tform.tvec()[2], 0.5, 1e-10);

}


void test_estimate(const size_t num_coords) {

  SimilarityTransform3D orig_tform(2, 1, 3.11, 0.66, 100, 10, 0.5);

  Eigen::MatrixXd src(num_coords, 3), dst(num_coords, 3);

  for (size_t i=0; i<num_coords; ++i) {
    Eigen::Vector3d s(i, i+2, i*i);
    Eigen::Vector3d d = s;
    orig_tform.transform_point(d);
    src.row(i) = s;
    dst.row(i) = d;
  }

  SimilarityTransform3D est_tform;
  est_tform.estimate(src, dst);

  const double matrix_diff = (orig_tform.matrix() - est_tform.matrix()).norm();
  ASSERT_ALMOST_EQUAL(matrix_diff, 0, 1e-6);

}


int main(int argc, char* argv[]) {

  test_init();
  test_estimate(3);
  test_estimate(100);

  return 0;

}
