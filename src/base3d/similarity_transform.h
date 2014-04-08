/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE3D_SIMILARITY_TRANSFORM_H_
#define MAVMAP_SRC_BASE3D_SIMILARITY_TRANSFORM_H_

#include <vector>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base3d/projection.h"
#include "util/estimation.h"


/**
 * 3D similarity transform estimator from corresponding point pairs in the
 * source and destination coordinate systems.
 *
 * This algorithm is based on the following paper:
 *
 *      S. Umeyama. Least-Squares Estimation of Transformation Parameters
 *      Between Two Point Patterns. IEEE Transactions on Pattern Analysis and
 *      Machine Intelligence, Volume 13 Issue 4, Page 376-380, 1991.
 *      http://www.stanford.edu/class/cs273/refs/umeyama.pdf
 *
 * and uses the Eigen3 implementation.
 */
class SimilarityTransform3DEstimator : public BaseEstimator {

public:

  /**
   * Estimate the 3D similarity transform.
   *
   * @param src      Set of corresponding 3D points in the source coordinate
   *                 system as a Nx3 matrix.
   * @param dst      Set of corresponding 3D points in the destination
   *                 coordinate system as a Nx3 matrix.
   *
   * @return         4x4 homogeneous transformation matrix.
   */
  std::vector<Eigen::MatrixXd> estimate(const Eigen::MatrixXd& src,
                                        const Eigen::MatrixXd& dst);

  /**
   * Calculate the transformation error for each point pair.
   *
   * The transformation error is defined as: norm[2](dst_i - transform*src_i)
   *
   * @param src        Set of corresponding 3D points in the source coordinate
   *                   system as a Nx3 matrix.
   * @param dst        Set of corresponding 3D points in the destination
   *                   coordinate system as a Nx3 matrix.
   * @param matrix     4x4 homogeneous transformation matrix.
   * @param residuals  Output vector of residuals for each point pair.
   */
  void residuals(const Eigen::MatrixXd& src,
                 const Eigen::MatrixXd& dst,
                 const Eigen::MatrixXd& matrix,
                 std::vector<double>& residuals);

};


class SimilarityTransform3D {

public:

  SimilarityTransform3D();

  SimilarityTransform3D(const Eigen::Matrix<double, 3, 4> matrix);

  SimilarityTransform3D(const Eigen::Transform<double, 3, Eigen::Affine>&
                          transform);

  SimilarityTransform3D(const double scale,
                        const double rx, const double ry, const double rz,
                        const double tx, const double ty, const double tz);

  void estimate(const Eigen::MatrixXd& src,
                const Eigen::MatrixXd& dst);

  SimilarityTransform3D inverse();

  void transform_point(Eigen::Vector3d& xyz);
  void transform_pose(Eigen::Vector3d& rvec, Eigen::Vector3d& tvec);

  Eigen::Matrix4d matrix();
  double scale();
  Eigen::Vector3d rvec();
  Eigen::Vector3d tvec();


private:

  void compose_transform_();

  Eigen::Transform<double, 3, Eigen::Affine> transform_;

};


#endif // MAVMAP_SRC_BASE3D_SIMILARITY_TRANSFORM_H_
