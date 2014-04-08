/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "feature_management.h"


/*

Notes
-----

* ``*_idx`` variables denote the position in the original input array
* ``*_id`` variables denote the internal ID of an object in the feature manager
* ``num_*`` keeps track of the number of added data and so determines the
  internal ID of objects
* Internal IDs are always returned from constructing methods as ``size_t``.

*/


FeatureManager::FeatureManager() {
  num_cameras_ = 0;
  num_images_ = 0;
  num_points2D_ = 0;
  num_points3D_ = 0;
}


size_t FeatureManager::get_point2D_idx(const size_t point2D_id) {
  // Assume that 2D points have been added one after each other
  return point2D_id - image_to_points2D[point2D_to_image[point2D_id]][0];
}


size_t FeatureManager::add_point2D(const size_t image_id,
                                   const Eigen::Vector2d& xy) {
  const size_t point2D_id = ++num_points2D_;

  points2D[point2D_id] = xy;
  image_to_points2D[image_id].push_back(point2D_id);
  point2D_to_image[point2D_id] = image_id;

  return point2D_id;
}


size_t FeatureManager::add_point3D() {
  const size_t point3D_id = ++num_points3D_;

  points3D[point3D_id] = Eigen::Vector3d::Zero();
  points3D_tri[point3D_id] = false;
  point3D_to_points2D[point3D_id] = std::vector<size_t>();

  return point3D_id;
}


size_t FeatureManager::add_camera(const std::vector<double>& params) {
  const size_t camera_id = ++num_cameras_;

  // Explicitly set size of vector, so vector does not take up more memory
  // than necessary
  camera_params[camera_id].reserve(params.size());
  for (size_t i=0; i<params.size(); ++i) {
    camera_params[camera_id].push_back(params[i]);
  }

  return camera_id;
}


size_t FeatureManager::add_image(const size_t camera_id) {

  const size_t image_id = ++num_images_;

  image_to_camera[image_id] = camera_id;
  rvecs[image_id] = Eigen::Vector3d::Zero();
  tvecs[image_id] = Eigen::Vector3d::Zero();
  image_to_points2D[image_id] = std::vector<size_t>();

  return image_id;
}


size_t FeatureManager::add_image(const size_t camera_id,
                                 const std::vector<Eigen::Vector2d>&
                                   points2D) {

  const size_t image_id = add_image(camera_id);

  image_to_points2D[image_id].reserve(points2D.size());

  // 2D points must be added one after each other, see `get_point2D_idx`
  for (size_t i=0; i<points2D.size(); ++i) {
    add_point2D(image_id, points2D[i]);
  }

  return image_id;
}


size_t FeatureManager::add_correspondence(const size_t image_id1,
                                          const size_t image_id2,
                                          const size_t point2D_idx1,
                                          const size_t point2D_idx2) {

  // corresponding internal FeatureManager image point ids
  const size_t point2D_id1
    = image_to_points2D[image_id1][point2D_idx1];
  const size_t point2D_id2
    = image_to_points2D[image_id2][point2D_idx2];

  const bool point3D_exists1 = point2D_to_point3D.count(point2D_id1);
  const bool point3D_exists2 = point2D_to_point3D.count(point2D_id2);

  size_t point3D_id;

  if (!point3D_exists1 && !point3D_exists2) {

    // add new 3D point and add both observations to track
    point3D_id = add_point3D();
    point3D_to_points2D[point3D_id].push_back(point2D_id1);
    point3D_to_points2D[point3D_id].push_back(point2D_id2);
    point2D_to_point3D[point2D_id1] = point3D_id;
    point2D_to_point3D[point2D_id2] = point3D_id;

  } else if (point3D_exists1 && !point3D_exists2) {

    // ID of existing 3D point
    point3D_id = point2D_to_point3D[point2D_id1];

    // make sure 3D point is observed only once in the same image
    if (!point3D_in_image(point3D_id, image_id2)) {
      // add new observation of 3D point
      point3D_to_points2D[point3D_id].push_back(point2D_id2);
      point2D_to_point3D[point2D_id2] = point3D_id;
    }

  } else if (!point3D_exists1 && point3D_exists2) {

    // ID of existing 3D point
    point3D_id = point2D_to_point3D[point2D_id2];

    // make sure 3D point is observed only once in the same image
    if (!point3D_in_image(point3D_id, image_id1)) {
      // add new observation of 3D point
      point3D_to_points2D[point3D_id].push_back(point2D_id1);
      point2D_to_point3D[point2D_id1] = point3D_id;
    }

  } else {  // both exist

    const size_t point3D_id1 = point2D_to_point3D[point2D_id1];
    const size_t point3D_id2 = point2D_to_point3D[point2D_id2];

    if (point3D_id1 != point3D_id2) {

      // merge two tracks and keep 3D point of longer track

      size_t point3D_id_keep, point3D_id_copy;
      if (point3D_to_points2D[point3D_id1].size()
          >= point3D_to_points2D[point3D_id2].size()) {
        point3D_id_keep = point3D_id1;
        point3D_id_copy = point3D_id2;
      } else {
        point3D_id_keep = point3D_id2;
        point3D_id_copy = point3D_id1;
      }

      // intentionally do not use references as:
      //   1) points2D_keep is filtered for duplicates later
      //   2) delete_point3D removes elements from points2D_copy which are
      //      used thereafter
      std::vector<size_t> points2D_keep
        = point3D_to_points2D[point3D_id_keep];
      const std::vector<size_t> points2D_copy
        = point3D_to_points2D[point3D_id_copy];

      // delete 3D point prior to copying track as `delete_point3D` deletes
      // all references to 3D point
      delete_point3D(point3D_id_copy);

      // copy track
      for (size_t i=0; i<points2D_copy.size(); ++i) {
        const size_t point2D_id = points2D_copy[i];
        points2D_keep.push_back(point2D_id);
        point2D_to_point3D[point2D_id] = point3D_id_keep;
      }

      // erase duplicate observations of 3D point in the same image and
      // keep the first observation (remove all thereafter from track)

      // store already existing image IDs
      std::set<size_t> image_ids;

      // clear complete track and only add unique 2D points per image
      std::vector<size_t>& point2D_ids = point3D_to_points2D[point3D_id_keep];
      point2D_ids.clear();

      for (size_t i=0; i<points2D_keep.size(); ++i) {
        const size_t point2D_id = points2D_keep[i];
        const size_t image_id = point2D_to_image[point2D_id];
        if (image_ids.count(image_id) != 0) {
          point2D_to_point3D.erase(point2D_id);
        } else {
          image_ids.insert(image_id);
          point2D_ids.push_back(point2D_id);
        }
      }

      point3D_id = point3D_id_keep;

    } else {
      // if both points are already part of the same track: no action required
      point3D_id = point3D_id1;  // == point3D_id2
    }

  }

  return point3D_id;
}


void FeatureManager::set_point3D(const size_t point3D_id,
                                 const Eigen::Vector3d& xyz) {
  points3D[point3D_id] = xyz;
  points3D_tri[point3D_id] = true;
}


void FeatureManager::set_pose(const size_t image_id,
                              const Eigen::Vector3d& rvec,
                              const Eigen::Vector3d& tvec) {
  if (!rvecs.count(image_id)) {
    throw std::range_error("Image ID does not exist.");
  }
  rvecs[image_id] = rvec;
  tvecs[image_id] = tvec;
}


void FeatureManager::delete_point3D(const size_t point3D_id) {
  points3D.erase(point3D_id);
  points3D_tri.erase(point3D_id);
  const std::vector<size_t>& point2D_ids = point3D_to_points2D[point3D_id];
  for (size_t i=0; i<point2D_ids.size(); ++i) {
    point2D_to_point3D.erase(point2D_ids[i]);
  }
  point3D_to_points2D.erase(point3D_id);
}


void FeatureManager::find_tri_points(const size_t image_id,
                                     const std::vector<size_t> point2D_idxs,
                                     std::vector<size_t>& point3D_ids,
                                     std::vector<bool>& mask) {

  if (!image_to_points2D.count(image_id)) {
    throw std::range_error("Image ID does not exist.");
  }

  point3D_ids.clear();
  mask.resize(point2D_idxs.size());

  const std::vector<size_t>& point2D_ids = image_to_points2D[image_id];

  for (size_t i=0; i<point2D_idxs.size(); ++i) {
    // find 3D point id corresponding to image point
    const size_t point2D_id = point2D_ids[point2D_idxs[i]];
    if (point2D_to_point3D.count(point2D_id)) {
      const size_t point3D_id = point2D_to_point3D[point2D_id];
      // check if already triangulated
      if (points3D.count(point3D_id) && points3D_tri[point3D_id]) {
        point3D_ids.push_back(point3D_id);
        mask[i] = true;
      } else {
        mask[i] = false;
      }
    } else {
      mask[i] = false;
    }
  }
}


bool FeatureManager::point3D_in_image(const size_t point3D_id,
                                      const size_t image_id) {

  std::vector<size_t>& point2D_ids = point3D_to_points2D[point3D_id];
  for (size_t i=0; i<point2D_ids.size(); ++i) {
    if (point2D_to_image[point2D_ids[i]] == image_id) {
      return true;
    }
  }
  return false;

}
