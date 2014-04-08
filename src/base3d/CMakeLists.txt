INCLUDE_DIRECTORIES(
    ../
)

ADD_LIBRARY(bundle_adjustment bundle_adjustment.cc)
ADD_LIBRARY(camera_models camera_models.cc)
ADD_LIBRARY(essential_matrix essential_matrix.cc)
ADD_LIBRARY(p3p p3p.cc)
ADD_LIBRARY(projection projection.cc)
ADD_LIBRARY(projective_transform projective_transform.cc)
ADD_LIBRARY(similarity_transform similarity_transform.cc)
ADD_LIBRARY(triangulation triangulation.cc)

ADD_EXECUTABLE(camera_models_test camera_models_test.cc)
ADD_EXECUTABLE(essential_matrix_test essential_matrix_test.cc)
TARGET_LINK_LIBRARIES(essential_matrix_test
    essential_matrix
    projection
    triangulation
    estimation
    math
)

ADD_EXECUTABLE(p3p_test p3p_test.cc)
TARGET_LINK_LIBRARIES(p3p_test
    estimation
    p3p
    similarity_transform
    projection
    math
)

ADD_EXECUTABLE(projection_test projection_test.cc)
TARGET_LINK_LIBRARIES(projection_test
    projection
)

ADD_EXECUTABLE(projective_transform_test projective_transform_test.cc)
TARGET_LINK_LIBRARIES(projective_transform_test
    projective_transform
)

ADD_EXECUTABLE(similarity_transform_test similarity_transform_test.cc)
TARGET_LINK_LIBRARIES(similarity_transform_test
    similarity_transform
    projection
)

ADD_EXECUTABLE(triangulation_test triangulation_test.cc)
TARGET_LINK_LIBRARIES(triangulation_test
    similarity_transform
    triangulation
    projection
)
