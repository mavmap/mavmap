/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
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


std::vector<std::complex<double>> poly_solve(const std::vector<double>& coeffs, size_t max_iter, const double eps) {
  const size_t cn = coeffs.size();
  const size_t n = cn - 1;

  std::vector<std::complex<double>> roots(n);
  std::complex<double> p(1, 0), r(1, 1);

  // Initial roots
  for (size_t i=0; i<n; ++i) {
    roots[i] = p;
    p = p * r;
  }

  for (size_t iter=0; iter<max_iter; ++iter) {
    double max_diff = 0;
    for (size_t i=0; i<n; ++i) {
      p = roots[i];
      std::complex<double> num = coeffs[n];
      std::complex<double> d = coeffs[n];
      for (size_t j=0; j<n; ++j) {
        num = num * p + coeffs[n-j-1];
        if (j != i) {
          d = d * (p - roots[j]);
        }
      }
      num /= d;
      roots[i] = p - num;
      max_diff = std::max(max_diff, std::abs(num));
    }
    if (max_diff <= eps) {
      break;
    }
  }

  return roots;
}
