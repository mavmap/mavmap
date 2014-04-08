/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE3D_P3P_H_
#define MAVMAP_SRC_BASE3D_P3P_H_

#include <vector>
#include <iostream>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/estimation.h"
#include "util/math.h"


/**
 * Analytic solver for the P3P (Prespective-Three-Point) problem.
 *
 * The algorithm is based on the following paper:
 *
 *    X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang. Complete Solution
 *    Classification for the Perspective-Three-Point Problem.
 *    http://www.mmrc.iss.ac.cn/~xgao/paper/ieee.pdf
 */
class P3PEstimator : public BaseEstimator {

public:

  /**
   * Estimate the most probable solution of the P3P problem from a set of
   * four 2D-3D point correspondences. The first three points are used to
   * derive multiple solutions from the analytical problem, whereas the fourth
   * point is used to solve the ambiguity and choose the most probable
   * solution.
   *
   * @param points2D   Normalized 2D image points as 4x2 matrix.
   * @param points3D   3D world points as 4x3 matrix.
   *
   * @return           Most probable pose as length-1 vector of a 3x4 matrix.
   */
  std::vector<Eigen::MatrixXd> estimate(const Eigen::MatrixXd& points2D,
                                        const Eigen::MatrixXd& points3D);

  /**
   * Calculate the reprojection error given a set of 2D-3D point
   * correspondences and a projection matrix.
   *
   * @param points2D     Normalized 2D image points as 4x2 matrix.
   * @param points3D     3D world points as 4x3 matrix.
   * @param proj_matrix  3x4 projection matrix.
   * @param residuals    Output vector of residuals (reprojection errors) for
   *                     each 2D-3D point pair.
   */
  void residuals(const Eigen::MatrixXd& points2D,
                 const Eigen::MatrixXd& points3D,
                 const Eigen::MatrixXd& proj_matrix,
                 std::vector<double>& residuals);

};


#endif // MAVMAP_SRC_BASE3D_P3P_H_
