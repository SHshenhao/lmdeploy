#!/bin/sh
# export DIOPI_ROOT=$PWD/3rdparty/DIOPI/impl/lib/
# export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;

cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_PY_FFI=ON \
    -DBUILD_MULTI_GPU=OFF \
    -DUSE_NVTX=OFF \
    -DIMPL_OPT=cuda
