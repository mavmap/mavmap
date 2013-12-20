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

#ifndef MAVMAP_SRC_BASE3D_TRIANGULATION_H_
#define MAVMAP_SRC_BASE3D_TRIANGULATION_H_

#include <vector>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/SVD>

#include "base3d/projection.h"


/**
 * Triangulate 3D point from corresponding image point observations.
 *
 * Implementation of the linear triangulation method described in
 *   R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision,
 *   Cambridge Univ. Press, 2003.
 *
 * @param proj_matrix1   Projection matrix of the first image as 3x4 matrix.
 * @param proj_matrix2   Projection matrix of the second image as 3x4 matrix.
 * @param point1         Corresponding 2D point in first image.
 * @param point2         Corresponding 2D point in second image.
 *
 * @return               Triangulated 3D point.
 */
Eigen::Vector3d
triangulate_point(const Eigen::Matrix<double, 3, 4>& proj_matrix1,
                  const Eigen::Matrix<double, 3, 4>& proj_matrix2,
                  const Eigen::Vector2d& point1,
                  const Eigen::Vector2d& point2);


/**
 * Triangulate 3D points from corresponding image point observations.
 *
 * Implementation of the linear triangulation method described in
 *   R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision,
 *   Cambridge Univ. Press, 2003.
 *
 * @param proj_matrix1   Projection matrix of the first image as 3x4 matrix.
 * @param proj_matrix2   Projection matrix of the second image as 3x4 matrix.
 * @param points1        Corresponding 2D points in first image.
 * @param points2        Corresponding 2D points in second image.
 *
 * @return               Triangulated 3D points.
 */
std::vector<Eigen::Vector3d>
triangulate_points(const Eigen::Matrix<double, 3, 4>& proj_matrix1,
                   const Eigen::Matrix<double, 3, 4>& proj_matrix2,
                   const std::vector<Eigen::Vector2d>& points1,
                   const std::vector<Eigen::Vector2d>& points2);


/**
 * Triangulate 3D points from corresponding image point observations.
 *
 * Implementation of the linear triangulation method described in
 *   R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision,
 *   Cambridge Univ. Press, 2003.
 *
 * @param proj_matrix1   Projection matrix of the first image as 3x4 matrix.
 * @param proj_matrix2   Projection matrix of the second image as 3x4 matrix.
 * @param points1        Corresponding 2D points in first image as Nx2 matrix.
 * @param points2        Corresponding 2D points in second image as Nx2 matrix.
 *
 * @return               Triangulated 3D points as Nx3 matrix.
 */
Eigen::Matrix<double, Eigen::Dynamic, 3>
triangulate_points(const Eigen::Matrix<double, 3, 4>& proj_matrix1,
                   const Eigen::Matrix<double, 3, 4>& proj_matrix2,
                   const Eigen::Matrix<double, Eigen::Dynamic, 2>& points1,
                   const Eigen::Matrix<double, Eigen::Dynamic, 2>& points2);


/**
 * Calculate angle between the two rays of a triangulated point.
 *
 * @param proj_matrix1   Projection matrix of the first image as 3x4 matrix.
 * @param proj_matrix2   Projection matrix of the second image as 3x4 matrix.
 * @param points         Triangulated 3D points.
 *
 * @return               Angle in radians.
 */
std::vector<double>
calc_triangulation_angles(const Eigen::Matrix<double, 3, 4> proj_matrix1,
                          const Eigen::Matrix<double, 3, 4> proj_matrix2,
                          const std::vector<Eigen::Vector3d> points);


#endif // MAVMAP_SRC_BASE3D_TRIANGULATION_H_
