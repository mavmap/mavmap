/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <vector>

#include "util/math.h"
#include "util/test.h"


void test_median() {
  ASSERT_EQUAL(median({1, 2, 3, 4}), 2.5);
  ASSERT_EQUAL(median({1, 2, 3, 100}), 2.5);
  ASSERT_EQUAL(median({1, 2, 3, 4, 100}), 3);
  ASSERT_EQUAL(median({-100, 1, 2, 3, 4}), 2);
  ASSERT_EQUAL(median({-1, -2, -3, -4}), -2.5);
  ASSERT_EQUAL(median({-1, -2, 3, 4}), 1);
}


void test_rel2abs_threshold() {
  ASSERT_EQUAL(rel2abs_threshold(0.5, 100), 50);
  ASSERT_EQUAL(rel2abs_threshold(50, 100), 50);
  ASSERT_EQUAL(rel2abs_threshold(10, 100), 10);
  ASSERT_EQUAL(rel2abs_threshold(1, 100), 1);
  ASSERT_EQUAL(rel2abs_threshold(0, 100), 0);
}


void test_poly_eval() {
  ASSERT_EQUAL(poly_eval({1, -3, 3, -5, 10}, 1),
               std::complex<double>(1-3+3-5+10, 0));
  ASSERT_ALMOST_EQUAL(poly_eval({1, -3, 3, -5}, 2),
                      std::complex<double>(1*2*2*2-3*2*2+3*2-5, 0), 1e-6);
}


void test_poly_solve() {
  std::vector<std::complex<double>> roots = poly_solve({1, -3, 3, -5, 10});

  // Generated with OpenCV
  std::vector<std::complex<double>> ref = {{0.451826,0.160867},
                                           {0.451826,-0.160867},
                                           {-0.201826,0.627696},
                                           {-0.201826,-0.627696}};

  for (size_t i=0; i<roots.size(); ++i) {
    ASSERT_ALMOST_EQUAL(roots[i], ref[i], 1e-6);
  }
}


int main(int argc, char* argv[]) {

  test_median();
  test_rel2abs_threshold();
  test_poly_eval();
  test_poly_solve();

  return 0;

}
