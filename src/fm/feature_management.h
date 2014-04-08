/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_FM_FEATURE_MANAGEMENT_H_
#define MAVMAP_SRC_FM_FEATURE_MANAGEMENT_H_

#include <stdexcept>
#include <vector>
#include <set>
#include <unordered_map>

#include <Eigen/Core>


class FeatureManager {

public:

  FeatureManager();

  size_t get_num_cameras() { return num_cameras_; }
  size_t get_num_images() { return num_images_; }
  size_t get_num_points2D() { return num_points2D_; }
  size_t get_num_points3D() { return num_points3D_; }

  /**
   * Return index of 2D image point as added in the order when calling
   * the `add_image` function.
   *
   * @param point2D_id    Internal image ID of 2D point, returned by
   *                      `add_image`.
   *
   * @return              Index of 2D point.
   */
  size_t get_point2D_idx(const size_t point2D_id);

  /**
   * Add new 2D image point and all necessary references.
   *
   * @param image_id      Internal image ID of 2D point, returned by
   *                      `add_image`.
   * @param xy            Coordinate of 2D point.
   *
   * @return              Unique ID of new 2D point.
   */
  size_t add_point2D(const size_t image_id, const Eigen::Vector2d& xy);

  /**
   * Add new 3D object point and all necessary references.
   *
   * @return              Unique ID of new 3D point.
   */
  size_t add_point3D();

  /**
   * Add new camera. There is only one camera per image, while multiple images
   * might be taken by the same camera with the same camera parameters.
   *
   * @param params        Arbitrary camera parameters, e.g. focal length,
   *                      principal point, distortion parameters etc.
   *
   * @return              Unique ID of new camera.
   */
  size_t add_camera(const std::vector<double>& params);

  /**
   * Add new image.
   *
   * @param camera_id     Unique ID of camera. There is only one camera per
   *                      image, while multiple images might be taken by
   *                      the same camera with the same camera parameters.
   *
   * @return              Unique ID of new image.
   */
  size_t add_image(const size_t camera_id);

  /**
   * Add new image.
   *
   * @param camera_id     Unique ID of camera. There is only one camera per
   *                      image, while multiple images might be taken by
   *                      the same camera with the same camera parameters.
   * @param points2D      2D points of image.
   *
   * @return              Unique ID of new image.
   */
  size_t add_image(const size_t camera_id,
                   const std::vector<Eigen::Vector2d>& points2D);

  /**
   * Add new correspondence between two 2D image points.
   *
   * A new 3D point and its track list is automatically created if the
   * correspondence has not yet been defined. If one of the 2D points already
   * has a corresponding 3D point, the new point is automatically added to the
   * existing 3D point track list. In case both 2D points are already part of
   * different 3D point track lists, both tracks and 3D points are merged such
   * that the 3D point coordinate with the longer track list is kept.
   *
   * If the one and the same 3D point is observed in the corresponding image
   * already (i.e. `image_id1 == image_id2`), the new observation is not added
   * to the track list. Two merged tracks are also checked for duplicate
   * observations in the same image. Short: each 3D point track list is
   * guaranteed to only have one observation (2D point) per image.
   *
   * @param image_id1     ID of 1st image, returned by `add_image`.
   * @param image_id2     ID of 2nd image, returned by `add_image`.
   * @param point2D_idx1  Index of 2D point corresponding to 1st image. The
   *                      index is defined by the order in which the 2D points
   *                      have been added to the feature manager.
   * @param point2D_idx2  Index of 2D point corresponding to 2nd image. The
   *                      index is defined by the order in which the 2D points
   *                      have been added to the feature manager.
   *
   * @return              Unique ID of 3D point, which either already exists or
   *                      or is created as a new 3D point.
   */
  size_t add_correspondence(const size_t image_id1, const size_t image_id2,
                            const size_t point2D_idx1,
                            const size_t point2D_idx2);

  /**
   * Set coordinate of 3D point and set point as triangulated.
   *
   * @param point3D_id    ID of 3D point, returned by `new_point3D` or
   *                      `add_correspondence`.
   * @param xyz           Coordinate of 3D point.
   */
  void set_point3D(const size_t point3D_id, const Eigen::Vector3d& xyz);

  /**
   * Set pose of image.
   *
   * @param image_id      ID of image, returned by `add_image`.
   * @param rvec          Rotation vector.
   * @param tvec          Translation vector.
   */
  void set_pose(const size_t image_id, const Eigen::Vector3d& rvec,
                const Eigen::Vector3d& tvec);

  /**
   * Delete 3D point, its track and the triangulation flag.
   *
   * @param point3D_id    ID of 3D point, returned by `new_point3D` or
   *                      `add_correspondence`.
   */
  void delete_point3D(const size_t point3D_id);

  /**
   * Find points in image which have already been triangulated.
   *
   * @param image_id      ID of image, returned by `add_image`.
   * @param point2D_idx1  Indices of 2D points in image. The index is defined
   *                      by the order in which the 2D points have been added
   *                      to the feature manager.
   * @param point3D_ids   Output list of triangulated 3D point IDs. See
   *                      the `mask` parameter for relating this list to the
   *                      input 2D points: each 3D point is added
   *                      subsequently to this list depending on its
   *                      triangulation flag.
   * @param mask          Output mask indicating if a 2D point has a
   *                      corresponding 3D point or not (true if yes).
   */
  void find_tri_points(const size_t image_id,
                       const std::vector<size_t> point2D_idxs,
                       std::vector<size_t>& point3D_ids,
                       std::vector<bool>& mask);

  /**
   * Test if 3D point is observed in image (i.e. there exists a corresponding
   * 2D point in the track list of the 3D point).
   *
   * Note, that only the first occurrence is removed from the track.
   *
   * @param point3D_id    ID of 3D point, returned by `new_point3D` or
   *                      `add_correspondence`.
   * @param image_id      ID of image, returned by `add_image`.
   *
   * @return              `true` or `false`.
   */
  bool point3D_in_image(const size_t point3D_id, const size_t image_id);


  // list of all object points as
  //      {P3D_ID : (x, y, z), ...}
  std::unordered_map<size_t, Eigen::Vector3d> points3D;

  // boolean flag for triangulated object points
  std::unordered_map<size_t, bool> points3D_tri;

  // list of image point positions of all images as
  //      {P2D_ID : (x, y), ...}
  std::unordered_map<size_t, Eigen::Vector2d> points2D;

  // list of corresponding 3D point for all image points as
  //      {P2D_ID : P3D_ID, ...}
  std::unordered_map<size_t, size_t> point2D_to_point3D;

  // list of corresponding 3D point for all image points as
  //      {P2D_ID : IMG_ID, ...}
  std::unordered_map<size_t, size_t> point2D_to_image;

  // list of all image points for each image as
  //      {IMG_ID : (P2D_ID_1, P2D_ID_2, ...), ...}
  std::unordered_map<size_t, std::vector<size_t> > image_to_points2D;

  // list of image points for each 3D point as
  //      {P3D_ID : (P2D_ID_1, P2D_ID_2, ...), ...}
  std::unordered_map<size_t, std::vector<size_t> > point3D_to_points2D;

  // list of angle axis rotation vectors for each images as
  //      {IMG_ID : [rx, ry, rz], ...}
  std::unordered_map<size_t, Eigen::Vector3d> rvecs;

  // list of translation vectors for each images as
  //      {IMG_ID : [tx, ty, tz], ...}
  std::unordered_map<size_t, Eigen::Vector3d> tvecs;

  // list of camera indices for each image as
  //      {IMG_ID : CAM_ID, ...}
  std::unordered_map<size_t, size_t> image_to_camera;

  // list of parameters (focal, principal point etc.) for each camera as
  //      {CAM_ID : (fx, fy, cx, cy, k1, k2, ...), ...}
  std::unordered_map<size_t, std::vector<double> > camera_params;


private:

  // current total number of added object points, image points, images etc.:
  //      used for creating the internally unique IDs
  size_t num_cameras_;
  size_t num_images_;
  size_t num_points2D_;
  size_t num_points3D_;

};

#endif // MAVMAP_SRC_FM_FEATURE_MANAGEMENT_H_
