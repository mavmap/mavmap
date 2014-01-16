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

#include <vector>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "base3d/projection.h"
#include "util/estimation.h"


class ProjectiveTransformEstimator : public BaseEstimator {

public:

  /**
   * Estimate the projective transformation (homography).
   *
   * @param src      Set of corresponding 2D points in the source coordinate
   *                 system as a Nx2 matrix.
   * @param dst      Set of corresponding 2D points in the destination
   *                 coordinate system as a Nx2 matrix.
   *
   * @return         3x3 homogeneous transformation matrix.
   */
  std::vector<Eigen::MatrixXd> estimate(const Eigen::MatrixXd& src,
                                        const Eigen::MatrixXd& dst);

  /**
   * Calculate the residuals of a set of corresponding points.
   *
   * @param points1    First set of corresponding normalized points as Nx2
   *                   matrix.
   * @param points2    Second set of corresponding normalized points as Nx2
   *                   matrix.
   * @param E          3x3 projective matrix.
   * @param residuals  Output vector of residuals (Sampson-distance) for each
   *                   point pair.
   */
  void residuals(const Eigen::MatrixXd& src,
                 const Eigen::MatrixXd& dst,
                 const Eigen::MatrixXd& H,
                 std::vector<double>& residuals);

};
