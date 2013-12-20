INCLUDE_DIRECTORIES(
    ../
)

ADD_LIBRARY(io io.cc)
ADD_LIBRARY(math math.cc)
ADD_LIBRARY(opencv opencv.cc)
ADD_LIBRARY(path path.cc)
ADD_LIBRARY(estimation estimation.cc)
ADD_LIBRARY(timer timer.cc)

ADD_EXECUTABLE(estimation_test estimation_test.cc)
TARGET_LINK_LIBRARIES(estimation_test
    estimation
    similarity_transform
    projection
)
ADD_EXECUTABLE(math_test math_test.cc)
TARGET_LINK_LIBRARIES(math_test
    math
)
