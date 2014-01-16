/**
 * Copyright (C) 2014
 *
 *   Johannes Schönberger <jsch (at) cs.unc.edu>
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

#include "base3d/projective_transform.h"
#include "util/test.h"


void test_estimate() {

  for (double x=0; x<10; ++x) {

    Eigen::Matrix3d H0;
    H0 << x, 0.2, 0.3,
          30, 0.2, 0.1,
          0.3, 20, 1;

    Eigen::Matrix<double, 4, 2> src;
    src << x, 0,
           1, 0,
           2, 1,
           10, 30;

    Eigen::Matrix<double, 4, 2> dst;

    for (size_t i=0; i<4; ++i) {
      const Eigen::Vector3d src_i(src(i, 0), src(i, 1), 1);
      const Eigen::Vector3d dst_i = H0 * src_i;
      dst(i, 0) = dst_i(0) / dst_i(2);
      dst(i, 1) = dst_i(1) / dst_i(2);
    }

    ProjectiveTransformEstimator est_tform;
    std::vector<Eigen::MatrixXd> models = est_tform.estimate(src, dst);

    std::vector<double> residuals;
    est_tform.residuals(src, dst, models[0], residuals);

    for (size_t i=0; i<4; ++i) {
      // std::cout << residuals[i] << std::endl;
      ASSERT_ALMOST_EQUAL(residuals[i], 0, 1e-3);
    }

  }

}


int main(int argc, char* argv[]) {

  test_estimate();

  return 0;

}
