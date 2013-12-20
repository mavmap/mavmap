ADD_LIBRARY(feature_management feature_management.cc)

ADD_EXECUTABLE(feature_management_test feature_management_test.cc)
TARGET_LINK_LIBRARIES(feature_management_test
    feature_management
)
