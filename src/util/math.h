/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_UTIL_MATH_H_
#define MAVMAP_SRC_UTIL_MATH_H_

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <complex>
#include <cmath>
#include <limits>


#define DEG2RAD M_PI / 180.0
#define RAD2DEG 180.0 / M_PI


/**
 * Determine median value in vector.
 *
 * Returns NaN for empty vectors.
 *
 * @param elems   Elements in sequence.
 *
 * @return        Median value.
 */
double median(std::vector<double> elems);


/**
 * Transform relative (<1) or absolute (>1) to absolute threshold (>1).
 *
 * In case the threshold is relative it is converted to an absolute value
 * as a fraction of the `total` elements. Otherwise the absolute threshold
 * is not changed and returned.
 *
 * @param threshold   Relative or absolute threshold.
 * @param total       Total number of elements.
 *
 * @return            Absolute threshold.
 */
double rel2abs_threshold(const double threshold, const double total);


/**
 * Evaluate polynomial at given position.
 *
 * The polynomial coefficients must be given in increasing order:
 *
 *    y(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3 + ...
 *
 * where ``coeffs[0] = a0``, ``coeffs[1] = a1``, ``coeffs[2] = a2`` etc.
 *
 * @param coeffs      Vector of real coefficients.
 * @param x           Complex position at which to evaluate the polynomial.
 *
 * @return            Complex return value.
 */
std::complex<double> poly_eval(const std::vector<double>& coeffs,
                               const std::complex<double> x);


/**
 * Solve for the real or complex roots of a polynomial.
 *
 * The polynomial coefficients must be given in increasing order:
 *
 *    y(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3 + ...
 *
 * where ``coeffs[0] = a0``, ``coeffs[1] = a1``, ``coeffs[2] = a2`` etc.
 *
 * Implementation of the Durand-Kerner method, see:
 *
 *    https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
 *
 * @param coeffs      Vector of real coefficients.
 * @param max_iter    Maximum number of iterations.
 *
 * @return            Complex roots of the polynomial of length N-1,
 *                    where N is the number of coefficients.
 */
std::vector<std::complex<double>> poly_solve(const std::vector<double>& coeffs, size_t max_iter=100,
                                             const double eps=1e-10);


#endif // MAVMAP_SRC_UTIL_MATH_H_
