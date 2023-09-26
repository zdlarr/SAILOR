cmake_minimum_required(VERSION 3.0.0)
project(DepthToColor VERSION 1.0.0)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# eigen.
include_directories( "/usr/include/eigen3")
# libs.
pybind11_add_module(${PROJECT_NAME} depth2color.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC pybind11::module)
target_link_libraries(${PROJECT_NAME} PUBLIC opencv_core opencv_highgui opencv_imgcodecs)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("LIB Path: ${LIBRARY_OUTPUT_PATH}")