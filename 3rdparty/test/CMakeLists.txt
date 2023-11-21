
cmake_minimum_required(VERSION 3.8)

project(test)

find_package(CUDA 11.8 REQUIRED)

set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CUDA_PATH "/usr/local/cuda")

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
  ${CMAKE_SOURCE_DIR}/../DIOPI/proto/include
)
message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")
include_directories(
  ${COMMON_HEADER_DIRS}
)

add_subdirectory(runtime)
add_subdirectory(utils)
# enable_testing()
add_executable(demo test/main.cpp)

target_link_libraries(demo runtime logger tensor)
