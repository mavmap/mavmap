/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_UTIL_TEST_H_
#define MAVMAP_SRC_UTIL_TEST_H_

#include <iostream>
#include <assert.h>


#define ASSERT_EQUAL(x, y) assert((x) == (y))
#define ASSERT_ALMOST_EQUAL(x, y, eps) assert(std::abs((x) - (y)) < (eps))

#define ASSERT_LESS(x, y) assert((x) < (y))
#define ASSERT_GREATER(x, y) assert((x) > (y))

#define ASSERT_LESS_EQUAL(x, y) assert((x) <= (y))
#define ASSERT_GREATER_EQUAL(x, y) assert((x) >= (y))

#endif // MAVMAP_SRC_UTIL_TEST_H_
