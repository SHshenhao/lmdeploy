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
if ("${IMPL_OPT}" STREQUAL "cuda")
  include_directories("/usr/local/cuda/include")
endif()
if ("${IMPL_OPT}" STREQUAL "ascend")
  if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
  else()
    set(ASCEND_DIR /usr/local/Ascend)
  endif()

  if(EXISTS ${ASCEND_DIR}/ascend-toolkit/latest/)
    message(STATUS "ascend-toolkit exists:" ${ASCEND_DIR}/ascend-toolkit/latest/)
    message(STATUS "ASCEND_DIR:" ${ASCEND_DIR})
    include_directories(${ASCEND_DIR}/ascend-toolkit/latest/include/)
  #   link_directories(${ASCEND_DIR}/ascend-toolkit/latest/lib64)
  else()
      message(FATAL_ERROR "No ascend-toolkit found.")
  endif()
endif()

add_subdirectory(runtime)
add_subdirectory(utils)
# add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(models)
# if(BUILD_PYT)
#     add_subdirectory(th_op)
# endif()
if(BUILD_PY_FFI)
    add_subdirectory(python)
endif()
add_subdirectory(triton_backend)
