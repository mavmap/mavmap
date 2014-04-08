/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "p3p.h"


double inline calc_cos_(const Eigen::Vector3d& point1,
                        const Eigen::Vector3d& point2) {
  return point1(0) * point2(0)
         + point1(1) * point2(1)
         + point1(2) * point2(2);
}


double inline calc_dist_(const Eigen::Vector3d& point1,
                         const Eigen::Vector3d& point2) {
  return (point1 - point2).norm();
}


std::vector<Eigen::MatrixXd>
P3PEstimator::estimate(const Eigen::MatrixXd& points2D,
                       const Eigen::MatrixXd& points3D) {

  // Implementation of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang,
  // "Complete Solution Classification for the Perspective-Three-Point
  // Problem"

  assert(points2D.rows() == 4 && points3D.rows() == 4);

  // Extract first 3 points A, B, C for computation of {R, t}
  const Eigen::Matrix<double, 3, 3> points3D_world
    = points3D.block<3, 3>(0, 0).transpose();

  const Eigen::Vector3d& A = points3D.row(0);
  const Eigen::Vector3d& B = points3D.row(1);
  const Eigen::Vector3d& C = points3D.row(2);
  // Use last point as D to obtain best solution of {R, t}
  const Eigen::Vector4d D(points3D(3, 0), points3D(3, 1), points3D(3, 2), 1);

  Eigen::Matrix<double, 4, 3> points2D_N;
  for (size_t i=0; i<(size_t)points2D.rows(); ++i) {
    const Eigen::Vector2d& point2D = points2D.row(i);
    const double N = sqrt(point2D(0)*point2D(0) + point2D(1)*point2D(1) + 1);
    points2D_N(i, 0) = point2D(0) / N;
    points2D_N(i, 1) = point2D(1) / N;
    points2D_N(i, 2) = 1 / N;
  }

  const Eigen::Vector3d& u = points2D_N.row(0);
  const Eigen::Vector3d& v = points2D_N.row(1);
  const Eigen::Vector3d& w = points2D_N.row(2);
  // Use last point as z to obtain best solution of {R, t}
  const Eigen::Vector2d& z = points2D.row(3);

  // Angles between 2D points
  const double cos_uv = calc_cos_(u, v);
  const double cos_uw = calc_cos_(u, w);
  const double cos_vw = calc_cos_(v, w);

  // Distances between 2D points
  const double dist_AB = calc_dist_(A, B);
  const double dist_AC = calc_dist_(A, C);
  const double dist_BC = calc_dist_(B, C);

  const double dist_AB_2 = dist_AB * dist_AB;
  const double a = (dist_BC * dist_BC) / dist_AB_2;
  const double b = (dist_AC * dist_AC) / dist_AB_2;

  // Helper variables for calculation of coefficients
  const double a2 = a * a;
  const double b2 = b * b;
  const double p = 2 * cos_vw;
  const double q = 2 * cos_uw;
  const double r = 2 * cos_uv;
  const double p2 = p * p;
  const double p3 = p2 * p;
  const double q2 = q * q;
  const double r2 = r * r;
  const double r3 = r2 * r;
  const double r4 = r3 * r;
  const double r5 = r4 * r;

  // Build polynomial coefficients: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0
  std::vector<double> coeffs_a(5);
  coeffs_a[4] = -2*b+b2+a2+1+a*b*(2-r2)-2*a; // a4
  coeffs_a[3] = -2*q*a2-r*p*b2+4*q*a+(2*q+p*r)*b+(r2*q-2*q+r*p)*a*b-2*q; // a3
  coeffs_a[2] = (2+q2)*a2+(p2+r2-2)*b2-(4+2*q2)*a-(p*q*r+p2)*b-(p*q*r+r2)*a*b+q2+2; // a2
  coeffs_a[1] = -2*q*a2-r*p*b2+4*q*a+(p*r+q*p2-2*q)*b+(r*p+2*q)*a*b-2*q; // a1
  coeffs_a[0] = a2+b2-2*a+(2-p2)*b-2*a*b+1; // a0

  std::vector<std::complex<double>> roots_a = poly_solve(coeffs_a);

  double best_error = DBL_MAX;
  Eigen::Matrix<double, 3, 4> best_model;

  #pragma omp parallel shared(roots_a, best_model, best_error)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<roots_a.size(); ++i) {

      const double x = roots_a[i].real();

      // Neglect all complex results as degenerate cases
      if (roots_a[i].imag() > 1e-10 || x < 0) {
        continue;
      }

      const double x2 = x * x;
      const double x3 = x2 * x;

      // Build polynomial coefficients: b1*y + b0 = 0
      const double _b1 = (p2-p*q*r+r2)*a+(p2-r2)*b-p2+p*q*r-r2;
      const double b1 = b*_b1*_b1;
      const double b0 = ((1-a-b)*x2+(a-1)*q*x-a+b+1)*(r3*(a2+b2-2*a-2*b+(2-r2)*a*b+1)*x3+r2*(p+p*a2-2*r*q*a*b+2*r*q*b-2*r*q-2*p*a-2*p*b+p*r2*b+4*r*q*a+q*r3*a*b-2*r*q*a2+2*p*a*b+p*b2-r2*p*b2)*x2+(r5*(b2-a*b)-r4*p*q*b+r3*(q2-4*a-2*q2*a+q2*a2+2*a2-2*b2+2)+r2*(4*p*q*a-2*p*q*a*b+2*p*q*b-2*p*q-2*p*q*a2)+r*(p2*b2-2*p2*b+2*p2*a*b-2*p2*a+p2+p2*a2))*x+(2*p*r2-2*r3*q+p3-2*p2*q*r+p*q2*r2)*a2+(p3-2*p*r2)*b2+(4*q*r3-4*p*r2-2*p3+4*p2*q*r-2*p*q2*r2)*a+(-2*q*r3+p*r4+2*p2*q*r-2*p3)*b+(2*p3+2*q*r3-2*p2*q*r)*a*b+p*q2*r2-2*p2*q*r+2*p*r2+p3-2*r3*q);

      // Solve for y
      const double y = b0 / b1;
      const double y2 = y * y;

      const double nu = x2 + y2 - 2 * x * y * cos_uv;

      const double dist_PC = dist_AB / sqrt(nu);
      const double dist_PB = y * dist_PC;
      const double dist_PA = x * dist_PC;

      Eigen::Matrix<double, 3, 3> points3D_camera;
      points3D_camera.col(0) = u * dist_PA; // A'
      points3D_camera.col(1) = v * dist_PB; // B'
      points3D_camera.col(2) = w * dist_PC; // C'

      // Find transformation from world to camera system (similarity transform
      // without scaling - Euclidean transform)
      const Eigen::Matrix4d matrix
        = Eigen::umeyama(points3D_world, points3D_camera, false);
      const Eigen::Matrix<double, 3, 4> model = matrix.block<3, 4>(0, 0);

      // Project point D into image and determine reprojection error
      Eigen::Vector3d zp = model * D;
      zp(0) /= zp(2);
      zp(1) /= zp(2);
      const double z_dx = zp(0) - z(0);
      const double z_dy = zp(1) - z(1);
      const double error = sqrt(z_dx * z_dx + z_dy * z_dy);

      // Use solution with minimum reprojection error
      #pragma omp critical
      {
        if (error < best_error) {
          best_error = error;
          best_model = model;
        }
      }

    }
  }

  std::vector<Eigen::MatrixXd> models(1);
  models[0] = best_model;

  return models;

}


void P3PEstimator::residuals(const Eigen::MatrixXd& points2D,
                             const Eigen::MatrixXd& points3D,
                             const Eigen::MatrixXd& model,
                             std::vector<double>& residuals) {

  residuals.resize(points2D.rows());

  const Eigen::Matrix3d R = model.block<3, 3>(0, 0);
  const Eigen::Vector3d t = model.block<3, 1>(0, 3);

  // Determine reprojection error for each correspondence
  #pragma omp parallel shared(points2D, points3D, model, residuals)
  {
    int i;
    #pragma omp for schedule(static, 1)
    for (i=0; i<residuals.size(); ++i) {
      const Eigen::Vector2d& point2D = points2D.row(i);
      const Eigen::Vector3d& point3D = points3D.row(i);
      Eigen::Vector3d point2Dp = R * point3D + t;
      point2Dp(0) /= point2Dp(2);
      point2Dp(1) /= point2Dp(2);
      const double dx = point2Dp(0) - point2D(0);
      const double dy = point2Dp(1) - point2D(1);
      residuals[i] = sqrt(dx * dx + dy * dy);
    }
  }

}
