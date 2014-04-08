/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
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
