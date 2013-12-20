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
