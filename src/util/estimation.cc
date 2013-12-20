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

#include "estimation.h"


Eigen::MatrixXd RANSAC(BaseEstimator& estimator,
                       const Eigen::MatrixXd& x,
                       const Eigen::MatrixXd& y,
                       const size_t min_samples,
                       const double residual_threshold,
                       size_t& num_inliers,
                       std::vector<bool>& inlier_mask,
                       const size_t max_trials,
                       const size_t stop_num_inliers) {

  if (x.rows() != y.rows()) {
    throw std::invalid_argument("`x` and `y` must have the same size.");
  }

  if ((size_t)x.rows() <= min_samples) {
    throw std::invalid_argument("`min_samples` must be smaller than number "
                                "samples in `x` and `y`.");
  }

  const size_t num_samples = x.rows();

  size_t best_num_inliers = 0;
  double best_residual_sum = DBL_MAX;
  std::vector<bool> best_inlier_mask;
  Eigen::MatrixXd best_model;

  std::default_random_engine random_generator;
  std::uniform_int_distribution<size_t> uniform_distribution(0, num_samples-1);

  {

    #pragma omp parallel shared(best_num_inliers, best_residual_sum, \
                                best_inlier_mask, best_model, \
                                random_generator, uniform_distribution)
    {

      std::vector<bool> subset_inlier_mask(num_samples);
      Eigen::MatrixXd subset_x(min_samples, x.cols());
      Eigen::MatrixXd subset_y(min_samples, y.cols());
      std::vector<double> residuals(num_samples);

      bool abort = false;

      int trial;
      #pragma omp for schedule(static, 1)
      for (trial=0; trial<max_trials; ++trial) {

        if (abort) {
          continue;
        }

        // Generate random subset indices (rand without replacement)
        std::set<size_t> subset_idxs;
        while (subset_idxs.size() < min_samples) {
          const size_t rand_idx = uniform_distribution(random_generator);
          if (subset_idxs.count(rand_idx) == 0) {
            subset_idxs.insert(rand_idx);
          }
        }

        // Extract random subset from all samples
        size_t i = 0;
        for (auto it=subset_idxs.begin(); it!=subset_idxs.end(); ++it, ++i) {
          const size_t subset_idx = *it;
          subset_x.row(i) = x.row(subset_idx);
          subset_y.row(i) = y.row(subset_idx);
        }

        // Estimate model for current subset
        const std::vector<Eigen::MatrixXd> subset_models
          = estimator.estimate(subset_x, subset_y);

        for (size_t i=0; i<subset_models.size(); ++i) {

          const Eigen::MatrixXd& subset_model = subset_models[i];

          // Determine residuals of all samples for previously estimated model
          size_t num_inliers = 0;
          double residual_sum = 0;

          estimator.residuals(x, y, subset_model, residuals);
          for (size_t j=0; j<num_samples; ++j) {
            const double abs_residual = fabs(residuals[j]);
            const bool inlier_test = abs_residual <= residual_threshold;
            subset_inlier_mask[j] = inlier_test;
            if (inlier_test) {
              residual_sum += abs_residual;
              num_inliers += 1;
            }
          }

          // Save as best subset if better than all previous subsets
          #pragma omp critical
          {
            if (num_inliers > best_num_inliers
                || (num_inliers == best_num_inliers
                    && residual_sum < best_residual_sum)) {
              best_num_inliers = num_inliers;
              best_residual_sum = residual_sum;
              best_inlier_mask = subset_inlier_mask;
              best_model = subset_model;
            }
          }

          // Stop iteration if a certain number of inliers has been found
          if (best_num_inliers >= stop_num_inliers) {
            abort = true;
          }
        }

      }

    }

  }

  if (best_num_inliers < min_samples) {
    throw std::domain_error("RANSAC could not find valid consensus set.");
  }

  num_inliers = best_num_inliers;
  inlier_mask = best_inlier_mask;

  return best_model;

}
