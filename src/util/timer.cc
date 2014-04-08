/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
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
