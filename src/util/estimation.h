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

#ifndef MAVMAP_SRC_UTIL_RANSAC_H_
#define MAVMAP_SRC_UTIL_RANSAC_H_

#include <vector>
#include <set>
#include <stdexcept>
#include <cfloat>
#include <random>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <Eigen/Core>


#ifndef SIZE_T_MAX
#define SIZE_T_MAX std::numeric_limits<size_t>::max()
#endif


/**
 * Base class for estimators.
 */
class BaseEstimator {

public:

  virtual std::vector<Eigen::MatrixXd>
    estimate(const Eigen::MatrixXd& x,
             const Eigen::MatrixXd& y) = 0;

  virtual void
    residuals(const Eigen::MatrixXd& x,
              const Eigen::MatrixXd& y,
              const Eigen::MatrixXd& model,
              std::vector<double>& residuals) = 0;

};


/**
 * Robustly estimate a model using the RANSAC ("RANdom SAmple Consensus")
 * method.
 *
 * @param estimator           Sub-class of `BaseEstimator`.
 * @param x                   Independent variables as NxM matrix.
 * @param y                   Dependent variables as NxD matrix.
 * @param min_samples         Minimum number of samples necessary to estimate
 *                            the model defined by `estimator`.
 * @param residual_threshold  Residual threshold for a sample to be
 *                            classified as inlier.
 * @param num_inliers         Output value for number of inliers of final
 *                            estimated model.
 * @param inlier_mask         Output vector of length N indicating if a sample
 *                            is classified as in- or outlier in the final
 *                            estimated model.
 * @param max_trials          Maximum number of random trials to estimate
 *                            model from random subset of samples.
 * @param stop_num_inliers    Abort the iteration if at least this number of
 *                            inliers has been found for the currently best
 *                            model even if `max_trials` has not been reached
 *                            yet.
 *
 * @return                  The image index in the `image_data` array.
 */
Eigen::MatrixXd RANSAC(BaseEstimator& estimator,
                       const Eigen::MatrixXd& x,
                       const Eigen::MatrixXd& y,
                       const size_t min_samples,
                       const double residual_threshold,
                       size_t& num_inliers,
                       std::vector<bool>& inlier_mask,
                       const size_t max_trials=100,
                       const size_t stop_num_inliers=SIZE_T_MAX);


#endif // MAVMAP_SRC_UTIL_RANSAC_H_
