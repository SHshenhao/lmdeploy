# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

set(UsedVendor cuda)

# dipu include path
set(DIPU_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${DIPU_PATH})

message(STATUS "DIPU_PATH:  ${DIPU_PATH}")

# src
# file(GLOB RT_SRC_FILES
#     runtime/core/guardimpl/*.cpp
#     runtime/core/allocator/*.cpp
#     runtime/core/DIPU*.cpp
#     runtime/core/MemChecker.cpp
#     runtime/distributed/*.cpp
#     runtime/devproxy/*.cpp
# )

file(GLOB RT_SRC_FILES core/*.cpp)
file(GLOB DIOPI_RT_FILES diopirt/*.cpp)

# vendor src
add_subdirectory(vendor/${UsedVendor})
include_directories(SYSTEM ${VENDOR_INCLUDE_DIRS})
link_directories(${VENDOR_LIB_DIRS})


# if (${UsedVendor} STREQUAL ${VendorMLU})
#     message(STATUS "---- vendor specific (but shouldn't exist)----")
# endif()

# set(SOURCE_FILES
#     ${RT_SRC_FILES}
#     ${OP_SRC_FILES}
#     ${DIOPI_RT_FILES}
#     ${BASE_FILES}
#     ${UTILS_FILES}
#     ${VENDOR_FILES}
#     ${PROFILER_FILES}
# )

set(SOURCE_FILES
    ${RT_SRC_FILES}
    ${DIOPI_RT_FILES}
    ${VENDOR_FILES}
)

# add_library(runtime SHARED ${SOURCE_FILES})
add_library(runtime SHARED ${SOURCE_FILES})
set_property(TARGET runtime PROPERTY POSITION_INDEPENDENT_CODE  ON)
if (${UsedVendor} EQUAL cuda)
    set_property(TARGET runtime PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
endif()

# link
target_link_libraries(runtime ${DIPU_VENDOR_LIB})

# diopi
if (EXISTS $ENV{DIOPI_PATH})
    message(STATUS "DIOPI_PATH: $ENV{DIOPI_PATH}")
    set(DIOPI_PROTO_PATH $ENV{DIOPI_PATH})
    include_directories($ENV{DIOPI_PATH}/include)
else()
    set(DIOPI_PROTO_PATH "${PROJECT_SOURCE_DIR}/3rdparty/DIOPI/proto")
    include_directories(${DIOPI_PROTO_PATH}/include)
endif()
add_definitions(-DDIOPI_ATTR_WEAK)
message(STATUS "DIOPI_PROTO_PATH:  ${DIOPI_PROTO_PATH}")
message(STATUS "PROJECT_SOURCE_DIR:  ${PROJECT_SOURCE_DIR}")
# include_directories(${PROJECT_SOURCE_DIR}/3rdparty/DIOPI/impl/libs)

# if(DEFINED ENV{DIOPI_IMPL_LIB})
#   find_library(DIOPI_IMPL_LIB NAMES diopi_impl HINTS ENV DIOPI_IMPL_LIB REQUIRED)
# else()
#   set(DIOPI_IMPL_LIB diopi_impl)
# endif()
set(DIOPI_IMPL_LIB ${PROJECT_SOURCE_DIR}/3rdparty/DIOPI/impl/lib/libdiopi_impl.so)

# need export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
target_link_libraries(runtime  -Wl,--no-as-needed ${DIOPI_IMPL_LIB} -Wl,--as-needed)

## copy vendor header file
# set(VENDOR_DIST "${DIPU_PATH}/vendor/")
# set(VENDOR_HEADER_DIR "${DIPU_PATH}/vendor/${UsedVendor}")
# message(STATUS "VENDOR_DIST:  ${VENDOR_DIST}")
# message(STATUS "VENDOR_HEADER_DIR:  ${VENDOR_HEADER_DIR}")
# add_custom_command(OUTPUT vendor_include
#     COMMAND mkdir -p ${VENDOR_DIST}
#     # COMMAND cmake -E create_symlink ${VENDOR_HEADER_DIR}/vendorapi.h ${VENDOR_DIST}/vendorapi.h)
#     COMMAND cmake -E create_symlink /home/shenhao/work/lmdeploy/src/turbomind/runtime/vendor/cuda/vendorapi.h /home/shenhao/work/lmdeploy/src/turbomind/runtime/vendor/vendorapi.h)
# if (EXISTS "${VENDOR_HEADER_DIR}/vendor_autocast.h" AND
#         NOT IS_DIRECTORY "${VENDOR_HEADER_DIR}/vendor_autocast.h")
#     add_custom_command(OUTPUT vendor_include APPEND
#         COMMAND cmake -E create_symlink ${VENDOR_HEADER_DIR}/vendor_autocast.h ${VENDOR_DIST}/vendor_autocast.h)
# else()
#     target_compile_definitions(runtime PRIVATE DIPU_NO_VENDOR_AUTOCAST)
# endif()
# add_custom_target(copy_include DEPENDS vendor_include)
# add_dependencies(runtime copy_include)
