/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "path.h"


std::string ensure_trailing_slash(std::string path) {
  if (path.at(path.length()-1) != '/') {
    path += "/";
  }
  return path;
}
