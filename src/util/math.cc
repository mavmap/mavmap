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

#include "math.h"


double median(std::vector<double> elems) {
  size_t size = elems.size();
  size_t mid = size / 2;
  if (size == 0) {
    return NAN;
  }

  std::sort(elems.begin(), elems.end());

  if (size % 2 == 0) {
    return (elems[mid] + elems[mid-1]) / 2;
  } else {
    return elems[mid];
  }
}


double rel2abs_threshold(const double threshold, const double total) {
  if (threshold >= 1) {
    return threshold;
  } else {
    return threshold * total;
  }
}


std::complex<double> poly_eval(const std::vector<double>& coeffs,
                               const std::complex<double> x) {
    const size_t n = coeffs.size();
    std::complex<double> xn = x;
    std::complex<double> y = 0;
    for (int i=n-2; i>=0; --i) {
      y += coeffs[i] * xn;
      xn *= x;
    }
    y += coeffs.back();
    return y;
}


std::vector<std::complex<double>> poly_solve(const std::vector<double>& coeffs,
                                             size_t max_iter) {

  const size_t cn = coeffs.size();
  const size_t n = cn - 1;

  std::vector<std::complex<double>> roots(n);

  std::complex<double> p(1, 0), r(1, 1);

  for(size_t i=0; i<n; ++i) {
    roots[i] = p;
    p = p * r;
  }

  for(size_t iter=0; iter<max_iter; ++iter) {
    double max_diff = 0;
    for(size_t i=0; i<n; ++i) {
      p = roots[i];
      std::complex<double> num = coeffs[n];
      std::complex<double> d = coeffs[n];
      for(size_t j=0; j<n; ++j) {
        num = num * p + coeffs[n-j-1];
        if (j != i) {
          d = d * (p - roots[j]);
        }
      }
      num /= d;
      roots[i] = p - num;
      max_diff = std::max(max_diff, std::abs(num));
    }
    if (max_diff <= 0) {
      break;
    }
  }

  return roots;
}
