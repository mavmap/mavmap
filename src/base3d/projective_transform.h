/**
 * Copyright (C) 2014
 *
 *   Johannes L. Schönberger <jsch (at) cs.unc.edu>
 *

 */

#ifndef MAVMAP_SRC_BASE3D_PROJECTIVE_TRANSFORM_H_
#define MAVMAP_SRC_BASE3D_PROJECTIVE_TRANSFORM_H_


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

#endif // MAVMAP_SRC_BASE3D_PROJECTIVE_TRANSFORM_H_
