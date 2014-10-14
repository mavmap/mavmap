======
MAVMAP
======


About
=====

MAVMAP is a structure-from-motion system.

The system is intended to take a sequence of images (taken from an arbitrary
number of cameras and different camera models) as input and produce a 3-D
reconstruction of the camera poses and the sparse scene geometry as output.

More information about the internals of the reconstruction process can be found
below.

If you find this project useful, please consider citing

    Schönberger, J. L., Fraundorfer, F., and Frahm, J.-M.:
    Structure-from-motion for MAV image sequence analysis with photogrammetric applications,
    Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XL-3, 305-312,
    doi:10.5194/isprsarchives-XL-3-305-2014, 2014.


Requirements
============

* Boost
* Eigen3
* Ceres-Solver
* OpenCV

Confirmed to work with the following versions (2013-12-09):

* Boost==1.55.0
* Eigen==3.2.0
* Ceres-Solver=={1.7.0, 1.8.0}
* OpenCV==2.4.7.0
* Clang==LLVM version 5.0 (clang-500.2.79), GCC==4.6.3

Additionally, it is fully functional on Ubuntu 12.04 LTS with system packages
and OpenCV>=2.4.X.

Currently, the system only works on 64bit platforms, but it also runs on 32bit
platforms, however with reduced performance, by adding the following compile
flag `EIGEN_DONT_ALIGN_STATICALLY`.

Installation
============

* Install Ceres-Solver:
```
    tar zxf ceres-solver-X.X.X.tar.gz
    cd ceres-solver-X.X.X
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
```

* Compile MAVMAP:
```
    cd mavmap
    mkdir build
    cd build
    cmake ..
    make
```


License
=======

Free for academic use. For information on commercial licensing, please contact
the authors.

    Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
    Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>


Usage
=====

A fully-functional mapping routine is included, which is aimed at
reconstructing sequential sets of images, such as image sequences from UAVs.

Image data format
-----------------

This routine is implemented in `src/mapper` and can be configured via its
command line interface. The input data consists of a set of images whose
properties are defined in the `imagedata.txt` file. All the files must reside
in the same directory, e.g.
```
imagedata.txt
image1.bmp
image2.bmp
image3.bmp
image4.bmp
image5.bmp
[...]
```

The `imagedata.txt` defines the ordering of the acquisition of the images.
Consecutive images should optimally have overlaps. Every image is defined in a
separate line with some meta-data and the camera model, e.g.
```
# COMMENT
# BASENAME, ROLL, PITCH, YAW, LAT, LON, ALT, LOCAL_HEIGHT, TX, TY, TZ, CAM_IDX, CAM_MODEL, CAM_PARAMS[N]
image1, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5, 1, PINHOLE, 100.000, 100.000, 368.000, 256.000
image2, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5
image3, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5
image4, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5, 2, OPENCV, 100.000, 100.000, 368.000, 256.000, 0.1, 0.2, 0.01, 0.02
image5, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5, 1
image6, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5
image7, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5
image8, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5, 2
image9, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5, 2
image10, 1.1, -0.3, -1.0, 47.4, 9.2, 485.4, 2.8, 20.3, -0.4, -4.5, 2, OPENCV, 100.000, 100.000, 368.000, 256.000, 0.1, 0.2, 0.01, 0.02
[...]
```

* `{BASENAME}`: Filename of image without file extension.
* `{ROLL, PITCH, YAW}`: Camera orientation in radians (using the convention
  `R = R_x(roll) * R_y(pitch) * R_z(yaw)`) in the world coordinate system.
* `{LAT, LON, ALT}`: Global camera position from e.g. a GPS-sensor.
* `{LOCAL_HEIGHT}`: Local altitude above ground from e.g. a proximity sensor.
* `{TX, TY, TZ}`: Camera position in the world coordinate system.
* `{CAM_IDX, CAM_MODEL, CAM_PARAMS[N]}`: Camera index, camera model and camera
  parameters. Each unique camera must get an unique camera index. The camera
  model is assumed to be equal for all the following images until a new camera
  model with a new index is defined. In the example above this means that
  image{1,2,3,5,6,7} were taken from the first and image{4,8,9,10} from the
  second camera with different camera models, respectively. Available camera
  models and their respective parameter ordering can be found in
  `src/base3d/camera_models.h`.

Apart from the image `BASENAME` only `{ROLL, PITCH, YAW}` are currently used as
input for the optional constraining of the camera poses from e.g. IMU-sensors.
The values `{ROLL, PITCH, YAW, TX, TY, TZ}` will be set in the output data
file `imagedataout.txt`.

Mappers for arbitrary image configurations can be easily built on top of the
core components of the library. See internals section below.

Control points and geo-registration
-----------------------------------

Definition of control points provides the following two features:

* Geo-registration w.r.t. coordinate system of fixed control points
  (ground control points)
* Estimation of specific points by defining variable control points

You must enable `--use-control-points` and set `--control-point-data-path` to
point at a file in the following format:

```
## FIXED_CONTROL_POINT_NAME, X, Y, Z
IMAGE_IDX1, IX, IY
IMAGE_IDX2, IX, IY
...
# VARIABLE_CONTROL_POINT_NAME, X, Y, Z
IMAGE_IDX1, IX, IY
IMAGE_IDX2, IX, IY
...
# VARIABLE_CONTROL_POINT_NAME, X, Y, Z
IMAGE_IDX1, IX, IY
IMAGE_IDX2, IX, IY
...
## FIXED_CONTROL_POINT_NAME, X, Y, Z
IMAGE_IDX1, IX, IY
IMAGE_IDX2, IX, IY
```

* `{FIXED_CONTROL_POINT_NAME,VARIABLE_CONTROL_POINT_NAME}`: Unique control
  point identifier. One `#` indicates a variable and two `#` a fixed control
  point.
* `{X, Y, Z}`: Position of control point, use dummy values for variable
  control points.
* `{IMAGE_IDX}`: 0-based index of image in `imagedata.txt` of observation.
* `{IX, IY}`: Pixel position of observation in image.


Internals
=========

MAVMAP is built on top of the following core components:

* FeatureManager:
  Manage 2-D image feature points, 3-D scene points, camera poses,
  camera models, 2-D feature correspondences and 3-D point tracks.

* FeatureCache:
  Extract feature (SURF) locations and descriptors and cache them on disk for
  fast later retrieval. Automatic detection of changed feature detection and
  extraction parameters.

* SequentialMapper
  Provide tools for the sequential processing of images, bundle adjustment of
  the scene and cameras, automatic loop-detection and merging of separate,
  overlapping scenes.

The `SequentialMapper` makes use of the `FeatureCache` for feature retrieval
and of `FeatureManager` for storing the complete reconstructed information
(camera poses, scene, feature locations etc.) for efficient memory usage and
fast data access. The `FeatureManager` is a self-contained class without any
external dependencies.

The `src/mapper` routine is intended for the reconstruction of sequential image
sets, where each image was sequentially taken one after another. The methods of
the `SequentialMapper` allow for the creation of mapping routines of arbitrary
image collections.

FeatureManager
--------------

The data structures in the class were specifically chosen for fast access,
convenient usage and low memory footprint. The data is generally guaranteed to
have constant access and insertion time.

The entire structure of the scene and the camera poses is stored in the
`FeatureManager` and does not require any additional information for bundle
adjustment and data output.

The core functionality of the manager is to keep track of feature
correspondences in that it automatically creates new 3-D points, merges
existing tracks, eliminates duplicate observations of the same 3-D point in the
same image.

Data access to the `std::unordered_map` containers should be careful. While
`std::unordered_map[]` is the fastest access, it automatically creates a new
key-value pair in the map if the key does not exist. It can destroy the
automatic track list functionality. It is thus recommended to use
`std::unordered_map::at` for data access from outside.

SequentialMapper
----------------

This class is intended for the following exemplary process chain:

1. Find good initial image pair and reconstruct structure by estimating the
   essential matrix. (See `process_initial`)
2. Sequentially reconstruct new camera poses and extend the existing scene
   geometry by processing new images against already reconstructed images or
   by processing already reconstructed images (but a new combination) against
   each other to extend and improve the scene geometry. This routine is based
   on 2D-3D pose estimation by using already reconstructed 3-D world points
   and their corresponding image features in the new image. (See `process`)
3. Adjust local or global bundle. (See `adjust_bundle`)
4. Detect loops in already reconstructed scene. (See `detect_loop`)
5. Try to merge different sequential mappers with overlapping scene parts.
   (See `merge`).

Exemplary usage:
```
SequentialMapper mapper1(...);
mapper1.process_initial(image1, image2);
mapper1.process(image3, image2);
mapper1.process(image3, image1);
mapper1.process(image4, image3);
mapper1.process(image5, image4);
[...]
mapper1.adjust_bundle(...);
mapper1.process(image100, image4);
mapper1.detect_loop(image100);
mapper1.adjust_bundle(...);

SequentialMapper mapper2(...);
mapper2.process_initial(image1, image2);
mapper2.process(image3, image2);
[...]

mapper1.merge(mapper2);
```

This class always keeps the previously extracted features in memory and it is
thus recommended to sequentially process images for optimum performance, so
that the next new image is processed against one of the last images, i.e.
```
SequentialMapper mapper(...);
mapper.process_initial(image1, image2);
mapper.process(image3, image2);
mapper.process(image4, image3);
mapper.process(image5, image4);
mapper.process(image6, image4);
mapper.process(image6, image5);
[...]
```

More details about the usage of the class can be found in the header source
file.

Bundle Adjustment
-----------------

The bundle adjustment is based on the Ceres-Solver library.

It is possible to combine arbitrary camera models in the same adjustment
problem. The solver uses the robust Cauchy loss function and the cost function
is based on the reprojection error in pixels. The following convention for the
projection of 3-D world (`X`) to 2-D image (`x`) points is employed:
```
T = [R | t]
    [0 | 1]
X' = T * X
x = C(X')
```

where `T` is a 4x4 matrix and transforms 3-D points from the world to the
camera coordinate system. `R` is a 3x3 transformation matrix and `t` is a 3x1
translation vector. `R` (in angle-axis representation) and `t` are stored in
the `FeatureManager`. The camera pose in the world coordinate system can be
calculated by extracting `R'` and `t'` from `T^-1`.


`C(X)` represents the camera model and projects the 3-D point in the camera
coordinate system to the image plane. For the standard pinhole camera model it
is defined as:
```
C(X) = f * X + c
```

where `f` is the focal length and and `c` the principal point.

The bundle adjuster optimizes the poses and 3-D points of the feature manager
in-place, i.e. it uses only very few additional memory for building the bundle
adjustment problem.

The solver uses the sparse Schur algorithm for solving the normal equations
and the parameter and residual ordering uses Ceres' automatic ordering
routine.

Each 3-D point in the bundle adjustment is guaranteed to have at least two
observations in separate images to avoid singularities in the Jacobian.
At least two poses should be set as fixed to avoid datum defects which result
in rank deficient Jacobians.
