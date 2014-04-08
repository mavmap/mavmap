/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_BASE3D_CAMERA_MODELS_H_
#define MAVMAP_SRC_BASE3D_CAMERA_MODELS_H_

#include <vector>
#include <string>
#include <unordered_map>

#include <Eigen/Core>


/**
 * This file defines several different camera models and arbitrary new camera
 * models can be added by the following steps:
 *
 * 1. Add a new class in this file which implements the `world2image` and
 *    `image2world` methods.
 * 2. Define an unique name and code for the camera model and add it to
 *    the enum of the class and `CAMERA_MODEL_NAME_TO_CODE`.
 * 3. Add new specialization in `camera_model_image2world`,
 *    `camera_model_world2image`, `pose_refinement` and
 *    `bundle_adjustment` for the new model.
 * 4. Make sure the new camera model is still valid for the simplifications
 *    made in `camera_model_image2world_threshold`.
 * 5. Add new template specialization of test case for camera model to
 *    `camera_models_test.cc`.
 *
 * You can now use the camera model by specifying the model and the parameters
 * in the `imagedata.txt`. The ordering of the parameters in the
 * `imagedata.txt` must exactly match the ordering in which the camera model
 * indexes the `camera_params` array.
 */


/**
 * Get camera model code for a give camera model name.
 *
 * @param name        Camera model name, e.g. "PINHOLE", "OPENCV" etc.
 *
 * @returns           Camera model code as defined in CameraModel::code.
 */
int camera_model_name_to_code(const std::string& name);


/**
 * Transform image coordinates to world coordinates in camera coordinate system
 * using a specific camera model.
 *
 * This is the inverse of `camera_model_world2image`.
 *
 * @param u, v         Image coordinates in pixels.
 * @param x, y, z      Output world coordinates in camera coordinate system.
 * @param model_code   Unique code of camera model as defined in
 *                     `CAMERA_MODEL_NAME_TO_CODE`.
 * @param params       Array of arbitrary camera parameters in a specific
 *                     ordering, first 4 parameters must be {fx, fy, cx, cy}.
 */
void
camera_model_image2world(const std::vector<Eigen::Vector2d>& image_points,
                         std::vector<Eigen::Vector2d>& world_points,
                         const int model_code,
                         const double params[]);


/**
 * Transform threshold from pixel unit to normalized world coordinate system.
 *
 * Threshold is assumed to be at the principal point and assumes a
 * pinhole camera model.
 *
 * This is the inverse of `camera_model_world2image`.
 *
 * @param threshold    Threshold in pixels.
 * @param model_code   Unique code of camera model as defined in
 *                     `CAMERA_MODEL_NAME_TO_CODE`.
 * @param params       Array of arbitrary camera parameters in a specific
 *                     ordering, first 4 parameters must be {fx, fy, cx, cy}.
 *
 * @return             Threshold in normalized world coordinate system.
 */
double camera_model_image2world_threshold(const double threshold,
                                          const int model_code,
                                          const double params[]);


/**
 * Pinhole camera model.
 *
 * No distortion is assumed. Only focal length and principal point is modeled.
 *
 * Parameter array is expected in the following ordering:
 *
 *    fx, fy, cx, cy
 *
 * @see https://en.wikipedia.org/wiki/Pinhole_camera_model
 */
struct PinholeCameraModel {

  enum {
    code = 1,
    num_params = 4
  };

  template <typename T> static inline void
    world2image(const T& x, const T& y, const T& z, T& u, T& v,
                const T camera_params[]) {

    const T& f1 = camera_params[0];
    const T& f2 = camera_params[1];
    const T& c1 = camera_params[2];
    const T& c2 = camera_params[3];

    // Projection to normalized image plane
    u = x / z;
    v = y / z;

    // No distortion

    // Transform to image coordinates
    u = f1 * u + c1;
    v = f2 * v + c2;

  }

  template <typename T> static inline void
    image2world(const T& u, const T& v, T& x, T& y, T& z,
                const T camera_params[]) {

    const T& f1 = camera_params[0];
    const T& f2 = camera_params[1];
    const T& c1 = camera_params[2];
    const T& c2 = camera_params[3];

    x = (u - c1) / f1;
    y = (v - c2) / f2;
    z = 1;

  }

};


/**
 * OpenCV camera model.
 *
 * Based on the pinhole camera model. Additionally models radial and
 * tangential distortion (up to 2nd degree of coefficients). Not suitable for
 * large radial distortions of fish-eye cameras.
 *
 * Parameter array is expected in the following ordering:
 *
 *    fx, fy, cx, cy, k1, k2, p1, p2
 *
 * @see http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 */
struct OpenCVCameraModel {

  enum {
    code = 2,
    num_params = 8
  };

  template <typename T> static inline void
    world2image(const T& x, const T& y, const T& z, T& u, T& v,
                const T camera_params[]) {

    const T& f1 = camera_params[0];
    const T& f2 = camera_params[1];
    const T& c1 = camera_params[2];
    const T& c2 = camera_params[3];

    // Projection to normalized image plane
    u = x / z;
    v = y / z;

    // Distortion
    T du, dv;
    distortion(u, v, du, dv, camera_params);
    u += du;
    v += dv;

    // Transform to image coordinates
    u = f1 * u + c1;
    v = f2 * v + c2;

  }

  template <typename T> static inline void
    image2world(const T& u, const T& v, T& x, T& y, T& z,
                const T camera_params[]) {

    const T& f1 = camera_params[0];
    const T& f2 = camera_params[1];
    const T& c1 = camera_params[2];
    const T& c2 = camera_params[3];

    // Lift points to normalized plane
    x = (u - c1) / f1;
    y = (v - c2) / f2;

    // Recursive inverse distortion model
    size_t num_iterations = 10;
    T xx = x;
    T yy = y;
    T dx, dy;
    for (size_t i=0; i<num_iterations; ++i) {
      distortion(xx, yy, dx, dy, camera_params);
      xx = x - dx;
      yy = y - dy;
    }

    x = xx;
    y = yy;
    z =  1;

  }

  template <typename T> static inline void
    distortion(const T&u, const T&v, T& du, T& dv,
               const T camera_params[]) {

    const T& k1 = camera_params[4];
    const T& k2 = camera_params[5];
    const T& p1 = camera_params[6];
    const T& p2 = camera_params[7];

    T u2 = u * u;
    T uy = u * v;
    T y2 = v * v;
    T r2 = u2 + y2;
    T radial = k1 * r2 + k2 * r2 * r2;
    du = u * radial + T(2) * p1 * uy + p2 * (r2 + T(2) * u2);
    dv = v * radial + T(2) * p2 * uy + p1 * (r2 + T(2) * y2);

  }

};


/**
 * Cata camera model (by Christopher Mei).
 *
 * Based on the OpenCV camera model. It also models the distortion
 * introduced by the use of telecentric lenses with paracatadioptric sensors.
 * It is better suited for fisheye cameras while still being simple.
 *
 * This model is an extension to the models presented in:
 *
 *    Joao P. Barreto and Helder Araujo. Issues on the geometry of central
 *    catadioptric image formation. In CVPR, volume 2, pages 422-427, 2001.
 *    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.95.7958
 *
 *    Christopher Geyer and Kostas Daniilidis. A Unifying Theory for Central
 *    Panoramic Systems and Practical Implications. In ECCV, pages 445-461,
 *    2000. http://www.frc.ri.cmu.edu/users/cgeyer/papers/geyer_eccv00.pdf
 *
 * Parameter array is expected in the following ordering:
 *
 *    fx, fy, cx, cy, k1, k2, p1, p2, xi
 *
 * @see http://homepages.laas.fr/~cmei/uploads/Main/projection_model.pdf
 */
struct CataCameraModel {

  enum {
    code = 3,
    num_params = 9
  };

  template <typename T> static inline void
    world2image(const T& x, const T& y, const T& z, T& u, T& v,
                const T camera_params[]) {

    const T& f1 = camera_params[0];
    const T& f2 = camera_params[1];
    const T& c1 = camera_params[2];
    const T& c2 = camera_params[3];
    const T& xi = camera_params[8];

    // Projection to normalized image plane
    T zz = z + xi * sqrt(x * x + y * y + z * z);
    u = x / zz;
    v = y / zz;

    // Distortion
    T du, dv;
    distortion(u, v, du, dv, camera_params);
    u += du;
    v += dv;

    // Transform to image coordinates
    u = f1 * u + c1;
    v = f2 * v + c2;

  }

  template <typename T> static inline void
    image2world(const T& u, const T& v, T& x, T& y, T& z,
                const T camera_params[]) {

    const T& f1 = camera_params[0];
    const T& f2 = camera_params[1];
    const T& c1 = camera_params[2];
    const T& c2 = camera_params[3];
    const T& xi = camera_params[8];

    // Lift points to normalized plane
    x = (u - c1) / f1;
    y = (v - c2) / f2;

    // Recursive inverse distortion model
    size_t num_iterations = 10;
    T xx = x;
    T yy = y;
    T dx, dy;
    for (size_t i=0; i<num_iterations; ++i) {
      distortion(xx, yy, dx, dy, camera_params);
      xx = x - dx;
      yy = y - dy;
    }

    // Lift normalized points to the sphere
    x = xx;
    y = yy;
    if (xi == 1) {
      z = (1 - xx * xx - yy * yy) / 2;
    } else {
      T r2 = xx * xx+yy * yy;
      z = 1 - xi * (r2 + T(1)) / (xi + sqrt(T(1) + (T(1) - xi * xi) * r2));
    }
  }

  template <typename T> static inline void
    distortion(const T&u, const T&v, T& du, T& dv,
               const T camera_params[]) {

    const T& k1 = camera_params[4];
    const T& k2 = camera_params[5];
    const T& p1 = camera_params[6];
    const T& p2 = camera_params[7];

    T u2 = u * u;
    T uy = u * v;
    T y2 = v * v;
    T r2 = u2 + y2;
    T radial = k1 * r2 + k2 * r2 * r2;
    du = u * radial + T(2) * p1 * uy + p2 * (r2 + T(2) * u2);
    dv = v * radial + T(2) * p2 * uy + p1 * (r2 + T(2) * y2);

  }

};


/**
 * Transform world coordinates in camera coordinate system to image
 * coordinates using a specific camera model.
 *
 * This is the inverse of `camera_model_image2world`.
 *
 * @param x, y, z      World coordinates in camera coordinate system.
 * @param u, v         Output image coordinates in pixels.
 * @param model_code   Unique code of camera model as defined in
 *                     `CAMERA_MODEL_NAME_TO_CODE`.
 * @param params       Array of arbitrary camera parameters in a specific
 *                     ordering, first 4 parameters must be {fx, fy, cx, cy}.
 */
inline void
camera_model_world2image(const double& x, const double& y, const double& z,
                         double& u, double& v,
                         const int model_code,
                         const double params[]) {
  switch (model_code) {
    case PinholeCameraModel::code:
      PinholeCameraModel::world2image(x, y, z, u, v, params);
      break;
    case OpenCVCameraModel::code:
      OpenCVCameraModel::world2image(x, y, z, u, v, params);
      break;
    case CataCameraModel::code:
      CataCameraModel::world2image(x, y, z, u, v, params);
      break;
  }
}


/**
 * Transform image coordinates to world coordinates in camera coordinate system
 * using a specific camera model.
 *
 * This is the inverse of `camera_model_world2image`.
 *
 * @param u, v         Image coordinates in pixels.
 * @param x, y, z      Output world coordinates in camera coordinate system.
 * @param model_code   Unique code of camera model as defined in
 *                     `CAMERA_MODEL_NAME_TO_CODE`.
 * @param params       Array of arbitrary camera parameters in a specific
 *                     ordering, first 4 parameters must be {fx, fy, cx, cy}.
 */
inline void
camera_model_image2world(const double& u, const double& v,
                         double& x, double& y, double& z,
                         const int model_code,
                         const double params[]) {
  switch (model_code) {
    case PinholeCameraModel::code:
      PinholeCameraModel::image2world(u, v, x, y, z, params);
      break;
    case OpenCVCameraModel::code:
      OpenCVCameraModel::image2world(u, v, x, y, z, params);
      break;
    case CataCameraModel::code:
      CataCameraModel::image2world(u, v, x, y, z, params);
      break;
  }
}


#endif // MAVMAP_SRC_BASE3D_CAMERA_MODELS_H_
