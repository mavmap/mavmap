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

#ifndef MAVMAP_SRC_UTIL_TIMER_H_
#define MAVMAP_SRC_UTIL_TIMER_H_

#include <boost/chrono.hpp>
#include <iostream>
#include <iomanip>


class Timer {

public:

  Timer() : running_(false) {}

  void start();
  void restart();
  void print();
  double elapsed_time();

private:

  bool running_;
  boost::chrono::high_resolution_clock::time_point start_time_;

};

#endif // MAVMAP_SRC_UTIL_TIMER_H_
