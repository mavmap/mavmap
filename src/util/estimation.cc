/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "util/estimation.h"


static size_t RANSAC_SEED_COUNTER = 0;


static size_t dynamic_max_trials_(const size_t num_inliers, const size_t num_samples,
                                  const size_t min_samples, const double probability) {
  const double e = 1 - num_inliers / (double)num_samples;
  const double nom = std::max(DBL_MIN, 1 - probability);
  const double denom = std::max(DBL_MIN, 1 - std::pow(1 - e, min_samples));
  return (size_t)std::ceil(std::log(nom) / std::log(denom));
}


Eigen::MatrixXd RANSAC(BaseEstimator& estimator, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
                       const size_t min_samples, const double residual_threshold,
                       size_t& num_inliers, std::vector<bool>& inlier_mask,
                       const size_t max_trials, const size_t stop_num_inliers, const double probability) {
  if (x.rows() != y.rows()) {
    throw std::domain_error("`x` and `y` must have the same size.");
  }

  if ((size_t)x.rows() <= min_samples) {
    throw std::domain_error("`min_samples` must be smaller than number samples in `x` and `y`.");
  }

  const size_t num_samples = x.rows();

  size_t best_num_inliers = 0;
  double best_residual_sum = DBL_MAX;
  std::vector<bool> best_inlier_mask;
  Eigen::MatrixXd best_model;

#ifdef OPENMP_FOUND
  // Make sure we do not create new threads within each RANSAC evaluator thread
  omp_set_nested(0);
#endif

  #pragma omp parallel shared(best_num_inliers, best_residual_sum, \
                              best_inlier_mask, best_model, RANSAC_SEED_COUNTER)
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

      // Mersenne twister engine is extremely fast, high quality, but never ever use rand() % N
      // Always use the generator locally per thread, so we do not generate the same samples across multiple threads
      // The SEED_COUNTER is used to ensure that different threads use different seeds and consecutive calls to
      // RANSAC result in different random samplings (as long as size_t does not overflow after two consecutive calls)
      std::mt19937_64 random_generator;
      #pragma omp critical
      {
        random_generator.seed(++RANSAC_SEED_COUNTER);
      }
      std::uniform_int_distribution<size_t> uniform_distribution(0, num_samples-1);

      // Generate random subset indices (rand without replacement)
      std::set<size_t> subset_idxs;
      while (subset_idxs.size() < min_samples) {
        size_t rand_idx;
        rand_idx = uniform_distribution(random_generator);
        if (subset_idxs.count(rand_idx) == 0) {
          subset_idxs.insert(rand_idx);
        }
      }

      // Extract random subset from all samples
      size_t i = 0;
      for (auto it=subset_idxs.begin(); it!=subset_idxs.end(); ++it, ++i) {
        subset_x.row(i) = x.row(*it);
        subset_y.row(i) = y.row(*it);
      }

      // Estimate model for current subset
      const std::vector<Eigen::MatrixXd> subset_models = estimator.estimate(subset_x, subset_y);

      for (const auto& subset_model : subset_models) {

        if (abort) {
          continue;
        }

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
              || (num_inliers == best_num_inliers && residual_sum < best_residual_sum)) {
            best_num_inliers = num_inliers;
            best_residual_sum = residual_sum;
            best_inlier_mask = subset_inlier_mask;
            best_model = subset_model;
          }
          if (best_num_inliers >= stop_num_inliers
              || trial >= dynamic_max_trials_(num_inliers, num_samples, min_samples, probability)) {
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
