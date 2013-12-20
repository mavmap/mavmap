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

#include "timer.h"


void Timer::start() {
  if (running_) {
    return;
  }
  running_ = true;
  start_time_ = boost::chrono::high_resolution_clock::now();
}


void Timer::restart() {
  running_ = false;
  start();
}


double Timer::elapsed_time() {
  return boost::chrono::duration_cast<boost::chrono::microseconds>(
           boost::chrono::high_resolution_clock::now() - start_time_).count();
}


void Timer::print() {
  std::cout << "Elapsed time: "
            << std::setiosflags(std::ios::fixed)
            << std::setprecision(4)
            << elapsed_time() / 1e6
            << " [seconds]"
            << std::endl;
}
