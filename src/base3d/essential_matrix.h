/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE3D_ESSENTIAL_MATRIX_H_
#define MAVMAP_SRC_BASE3D_ESSENTIAL_MATRIX_H_

#include <vector>
#include <complex>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "base3d/projection.h"
#include "base3d/triangulation.h"
#include "util/estimation.h"
#include "util/math.h"


/**
 * Essential matrix estimator from corresponding point pairs.
 *
 * This algorithm solves the Five-Point problem and is based on the following
 * paper:
 *
 *    D.Nister, An efﬁcient solution to the ﬁve-point relative pose problem,
 *    IEEE-T-PAMI, 26(6), 2004.
 *    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.86.8769
 */
class EssentialMatrixEstimator : public BaseEstimator {

public:

  /**
   * Estimate up to 10 possible essential matrix solutions from a set of
   * corresponding points.
   *
   * @param points1    First set of corresponding normalized points as Nx2
   *                   matrix.
   * @param points2    Second set of corresponding normalized points as Nx2
   *                   matrix.
   *
   * @return           Up to 10 possible solutions as a vector of 3x3 essential
   *                   matrices.
   */
  std::vector<Eigen::MatrixXd> estimate(const Eigen::MatrixXd& points1,
                                        const Eigen::MatrixXd& points2);

  /**
   * Calculate the residuals of a set of corresponding points and a given
   * essential matrix.
   *
   * Residuals are defined as the Sampson-distance.
   *
   * @param points1    First set of corresponding normalized points as Nx2
   *                   matrix.
   * @param points2    Second set of corresponding normalized points as Nx2
   *                   matrix.
   * @param E          3x3 essential matrix.
   * @param residuals  Output vector of residuals (Sampson-distance) for each
   *                   point pair.
   */
  void residuals(const Eigen::MatrixXd& points1,
                 const Eigen::MatrixXd& points2,
                 const Eigen::MatrixXd& E,
                 std::vector<double>& residuals);

};


/**
 * Decompose an essential matrix into the possible rotation and translation.
 *
 * The first pose is assumed to be P = [I | 0] and the set of four other
 * possible second poses are defined as: {[R1 | t], [R2 | t],
 *                                        [R1 | -t], [R2 | -t]}
 *
 * @param E          3x3 essential matrix.
 * @param R1         First possible 3x3 rotation matrix.
 * @param R2         Second possible 3x3 rotation matrix.
 * @param t          3x1 possible translation vector (also -t possible).
 */
void decompose_essential_matrix(const Eigen::Matrix3d& E,
                                Eigen::Matrix3d& R1,
                                Eigen::Matrix3d& R2,
                                Eigen::Vector3d& t);


/**
 * Recover the most probable pose from the set of four possible solutions of
 * an essential matrix.
 *
 * First the set of four possible solutions is obtained from
 * `decompose_essential_matrix`. In the second step a cheirality test is
 * conducted which chooses the solution with the largest number of 3D points
 * in front of the camera by calculating the depth of the triangulated 3D
 * points.
 *
 * The pose of the first image is assumed to be P = [I | 0]
 *
 * @param E            3x3 essential matrix.
 * @param points1      First set of corresponding points as Nx2 matrix.
 * @param points2      Second set of corresponding points as Nx2 matrix.
 * @param inlier_mask  Only points with `true` in the inlier mask are
 *                     considered in the cheirality test. Size of the
 *                     inlier mask must match the number of points N.
 * @param R            Most probable 3x3 rotation matrix.
 * @param t            Most probable 3x1 translation vector.
 */
size_t pose_from_essential_matrix(const Eigen::Matrix3d& E,
                                  const Eigen::MatrixXd& points1,
                                  const Eigen::MatrixXd& points2,
                                  const std::vector<bool> inlier_mask,
                                  Eigen::Matrix3d& R,
                                  Eigen::Vector3d& t);


#endif // MAVMAP_SRC_BASE3D_ESSENTIAL_MATRIX_H_
