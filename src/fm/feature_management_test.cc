/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include <vector>

#include <Eigen/Core>

#include "feature_management.h"


const std::vector<double> camera_params;


void test_add_camera() {

  FeatureManager fm;

  size_t camera_id1 = fm.add_camera(camera_params);
  size_t image_id1 = fm.add_image(camera_id1);
  size_t image_id2 = fm.add_image(camera_id1);
  size_t image_id3 = fm.add_image(camera_id1);
  assert(fm.camera_params.size() == 1);
  assert(fm.image_to_camera[image_id1] == camera_id1);
  assert(fm.image_to_camera[image_id2] == camera_id1);
  assert(fm.image_to_camera[image_id3] == camera_id1);

  size_t camera_id2 = fm.add_camera(camera_params);
  size_t image_id4 = fm.add_image(camera_id2);
  size_t image_id5 = fm.add_image(camera_id2);
  size_t image_id6 = fm.add_image(camera_id1);
  assert(fm.camera_params.size() == 2);
  assert(fm.image_to_camera[image_id4] == camera_id2);
  assert(fm.image_to_camera[image_id5] == camera_id2);
  assert(fm.image_to_camera[image_id6] == camera_id1);

}


void test_add_image() {

  FeatureManager fm;

  size_t camera_id1 = fm.add_camera(camera_params);
  size_t image_id1 = fm.add_image(camera_id1);
  size_t image_id2 = fm.add_image(camera_id1);
  size_t image_id3 = fm.add_image(camera_id1);

  assert(fm.image_to_points2D.size() == 3);
  assert(fm.rvecs.size() == 3);
  assert(fm.tvecs.size() == 3);
  assert(fm.camera_params.size() == 1);

  assert(fm.image_to_points2D[image_id1].size() == 0);
  assert(fm.image_to_points2D[image_id2].size() == 0);
  assert(fm.image_to_points2D[image_id3].size() == 0);

}


void test_add_image_with_points2D() {

  FeatureManager fm;

  std::vector<Eigen::Vector2d> points2D;
  points2D.push_back(Eigen::Vector2d(0, 0));
  points2D.push_back(Eigen::Vector2d(0, 1));
  points2D.push_back(Eigen::Vector2d(1, 0));
  points2D.push_back(Eigen::Vector2d(1, 1));

  size_t camera_id1 = fm.add_camera(camera_params);
  size_t image_id1 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(2, 2));
  size_t image_id2 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(3, 3));
  size_t image_id3 = fm.add_image(camera_id1, points2D);

  assert(fm.image_to_points2D.size() == 3);
  assert(fm.rvecs.size() == 3);
  assert(fm.tvecs.size() == 3);
  assert(fm.camera_params.size() == 1);

  assert(fm.image_to_points2D[image_id1].size() == 4);
  assert(fm.image_to_points2D[image_id2].size() == 5);
  assert(fm.image_to_points2D[image_id3].size() == 6);

  assert(fm.points2D.size() == 15);

}


void test_set_pose() {

  FeatureManager fm;

  size_t camera_id1 = fm.add_camera(camera_params);
  size_t image_id1 = fm.add_image(camera_id1);
  size_t image_id2 = fm.add_image(camera_id1);
  size_t image_id3 = fm.add_image(camera_id1);

  fm.set_pose(image_id1, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(3, 2, 1.1));
  fm.set_pose(image_id2, Eigen::Vector3d(0, 1, 1), Eigen::Vector3d(3, 2, 1.2));
  fm.set_pose(image_id3, Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(3, 2, 1.3));

  assert(fm.rvecs[image_id1] == Eigen::Vector3d(0, 0, 1));
  assert(fm.rvecs[image_id2] == Eigen::Vector3d(0, 1, 1));
  assert(fm.rvecs[image_id3] == Eigen::Vector3d(1, 1, 1));

  assert(fm.tvecs[image_id1] == Eigen::Vector3d(3, 2, 1.1));
  assert(fm.tvecs[image_id2] == Eigen::Vector3d(3, 2, 1.2));
  assert(fm.tvecs[image_id3] == Eigen::Vector3d(3, 2, 1.3));

}


void test_add_correspondence() {

  FeatureManager fm;

  std::vector<Eigen::Vector2d> points2D;
  points2D.push_back(Eigen::Vector2d(0, 0));
  points2D.push_back(Eigen::Vector2d(0, 1));
  points2D.push_back(Eigen::Vector2d(1, 0));
  points2D.push_back(Eigen::Vector2d(1, 1));

  size_t camera_id1 = fm.add_camera(camera_params);
  size_t image_id1 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(2, 2));
  size_t image_id2 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(3, 3));
  size_t image_id3 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(4, 4));
  size_t image_id4 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(5, 5));
  size_t image_id5 = fm.add_image(camera_id1, points2D);

  // add first correspondence
  size_t point3D_id1 = fm.add_correspondence(image_id2, image_id1, 0, 0);
  assert(fm.points3D.size() == 1);
  assert(fm.points3D_tri.size() == 1);
  assert(fm.points3D_tri[point3D_id1] == false);
  assert(fm.point3D_to_points2D[point3D_id1].size() == 2);
  assert(fm.point3D_to_points2D.size() == 1);
  assert(fm.point2D_to_point3D.size() == 2);
  assert(fm.points2D[fm.point3D_to_points2D[point3D_id1][0]] == points2D[0]);
  assert(fm.points2D[fm.point3D_to_points2D[point3D_id1][1]] == points2D[0]);

  // add another correspondence to existing 3D point
  size_t point3D_id2 = fm.add_correspondence(image_id3, image_id2, 2, 0);
  assert(point3D_id1 == point3D_id2);
  assert(fm.points3D.size() == 1);
  assert(fm.points3D_tri.size() == 1);
  assert(fm.points3D_tri[point3D_id1] == false);
  assert(fm.point3D_to_points2D[point3D_id2].size() == 3);
  assert(fm.point3D_to_points2D.size() == 1);
  assert(fm.point2D_to_point3D.size() == 3);
  assert(fm.points2D[fm.point3D_to_points2D[point3D_id2][0]] == points2D[0]);
  assert(fm.points2D[fm.point3D_to_points2D[point3D_id2][1]] == points2D[0]);
  assert(fm.points2D[fm.point3D_to_points2D[point3D_id2][2]] == points2D[2]);

  // add another entirely new correspondence
  size_t point3D_id3 = fm.add_correspondence(image_id5, image_id4, 5, 1);
  assert(point3D_id1 != point3D_id3);
  assert(fm.points3D.size() == 2);
  assert(fm.points3D_tri.size() == 2);
  assert(fm.points3D_tri[point3D_id3] == false);
  assert(fm.point3D_to_points2D[point3D_id3].size() == 2);
  assert(fm.point3D_to_points2D.size() == 2);
  assert(fm.point2D_to_point3D.size() == 5);
  assert(fm.points2D[fm.point3D_to_points2D[point3D_id3][0]] == points2D[5]);
  assert(fm.points2D[fm.point3D_to_points2D[point3D_id3][1]] == points2D[1]);

  // merge two tracks
  size_t point3D_id4 = fm.add_correspondence(image_id4, image_id3, 1, 2);
  assert(point3D_id4 == point3D_id1);  // make sure longer track is kept
  assert(fm.points3D.size() == 1);
  assert(fm.points3D_tri.size() == 1);
  assert(fm.points3D_tri[point3D_id4] == false);
  assert(fm.point3D_to_points2D[point3D_id4].size() == 5);
  assert(fm.point3D_to_points2D.size() == 1);
  assert(fm.point2D_to_point3D.size() == 5);

  // check if duplicate observations of the same 3D point in the same image
  // are detected
  size_t point3D_id5 = fm.add_correspondence(image_id4, image_id3, 2, 2);
  assert(point3D_id5 == point3D_id1);  // make sure longer track is kept
  assert(fm.points3D.size() == 1);
  assert(fm.points3D_tri.size() == 1);
  assert(fm.points3D_tri[point3D_id5] == false);
  assert(fm.point3D_to_points2D[point3D_id5].size() == 5);
  assert(fm.point3D_to_points2D.size() == 1);
  assert(fm.point2D_to_point3D.size() == 5);
  assert(fm.point3D_to_points2D[point3D_id5][4]
         == fm.image_to_points2D[image_id4][1]);

  fm.set_point3D(point3D_id1, Eigen::Vector3d(0, 1, 2));
  assert(fm.points3D_tri[point3D_id1] == true);
  assert(fm.points3D_tri[point3D_id3] == false);
  assert(fm.points3D[point3D_id1] == Eigen::Vector3d(0, 1, 2));

  fm.set_point3D(point3D_id3, Eigen::Vector3d(2, 2, 2));
  assert(fm.points3D_tri[point3D_id1] == true);
  assert(fm.points3D_tri[point3D_id3] == true);
  assert(fm.points3D[point3D_id1] == Eigen::Vector3d(0, 1, 2));
  assert(fm.points3D[point3D_id3] == Eigen::Vector3d(2, 2, 2));

}


void test_find_tri_points() {

  FeatureManager fm;

  std::vector<Eigen::Vector2d> points2D;
  points2D.push_back(Eigen::Vector2d(0, 0));
  points2D.push_back(Eigen::Vector2d(0, 1));
  points2D.push_back(Eigen::Vector2d(1, 0));
  points2D.push_back(Eigen::Vector2d(1, 1));

  size_t camera_id1 = fm.add_camera(camera_params);
  size_t image_id1 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(2, 2));
  size_t image_id2 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(3, 3));
  size_t image_id3 = fm.add_image(camera_id1, points2D);
  points2D.push_back(Eigen::Vector2d(4, 4));
  size_t image_id4 = fm.add_image(camera_id1, points2D);

  // add first correspondence
  size_t point3D_id1 = fm.add_correspondence(image_id2, image_id1, 0, 0);
  // add another correspondence to existing 3D point
  fm.add_correspondence(image_id3, image_id2, 2, 0);
  // add another entirely new correspondence
  size_t point3D_id3 = fm.add_correspondence(image_id3, image_id2, 5, 1);
  // add another entirely new correspondence
  fm.add_correspondence(image_id4, image_id3, 6, 4);

  // set 1st 3D point
  fm.set_point3D(point3D_id1, Eigen::Vector3d(0, 1, 2));

  std::vector<size_t> point2D_idxs;
  point2D_idxs.push_back(0);
  point2D_idxs.push_back(1);
  point2D_idxs.push_back(2);
  point2D_idxs.push_back(3);
  std::vector<size_t> point3D_ids;
  std::vector<bool> mask;

  fm.find_tri_points(image_id1, point2D_idxs, point3D_ids, mask);
  assert(mask.size() == 4);
  assert(mask[0] == true);
  assert(mask[1] == false);
  assert(mask[2] == false);
  assert(mask[3] == false);
  assert(point3D_ids.size() == 1);
  assert(point3D_ids[0] == point3D_id1);

  fm.find_tri_points(image_id2, point2D_idxs, point3D_ids, mask);
  assert(mask.size() == 4);
  assert(mask[0] == true);
  assert(mask[1] == false);
  assert(mask[2] == false);
  assert(mask[3] == false);
  assert(point3D_ids.size() == 1);
  assert(point3D_ids[0] == point3D_id1);

  fm.find_tri_points(image_id4, point2D_idxs, point3D_ids, mask);
  assert(mask.size() == 4);
  assert(mask[0] == false);
  assert(mask[1] == false);
  assert(mask[2] == false);
  assert(mask[3] == false);
  assert(point3D_ids.size() == 0);

  // set 2nd 3D point
  fm.set_point3D(point3D_id3, Eigen::Vector3d(0, 1, 2));

  fm.find_tri_points(image_id2, point2D_idxs, point3D_ids, mask);
  assert(mask.size() == 4);
  assert(mask[0] == true);
  assert(mask[1] == true);
  assert(mask[2] == false);
  assert(mask[3] == false);
  assert(point3D_ids.size() == 2);
  assert(point3D_ids[0] == point3D_id1);
  assert(point3D_ids[1] == point3D_id3);

}


int main(int argc, char* argv[]) {

  test_add_camera();
  test_add_image();
  test_add_image_with_points2D();
  test_set_pose();
  test_add_correspondence();
  test_find_tri_points();

  return 0;

}
