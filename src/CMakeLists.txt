INCLUDE_DIRECTORIES(
    .
    ${OpenCV_INCLUDE_DIR}
    ${Eigen3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CERES_INCLUDES}
)

ADD_SUBDIRECTORY(base2d)
ADD_SUBDIRECTORY(base3d)
ADD_SUBDIRECTORY(fm)
ADD_SUBDIRECTORY(loop)
ADD_SUBDIRECTORY(sfm)
ADD_SUBDIRECTORY(util)

SET(MAVMAP_LIBRARIES
    sequential_mapper
    opencv
    detection
    voc_tree
    voc_tree_database
    voc_tree_inv_file
    bundle_adjustment
    camera_models
    essential_matrix
    p3p
    projection
    projective_transform
    triangulation
    similarity_transform
    feature_management
    feature_cache
    estimation
    io
    path
    feature
    image
    timer
    math)

ADD_EXECUTABLE(mapper mapper.cc)
TARGET_LINK_LIBRARIES(mapper
    ${MAVMAP_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
)
