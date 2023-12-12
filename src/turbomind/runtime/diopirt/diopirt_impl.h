// Copyright (c) 2023, DeepLink.
#pragma once
#include <list>

#include "3rdparty/DIOPI/proto/include/diopi/diopirt.h"
#include "3rdparty/DIOPI/proto/include/diopi/functions.h"
#include "3rdparty/DIOPI/proto/include/diopi/functions_lmdeploy.h"
// #include <diopi/diopirt.h>
// #include <diopi/functions.h>

#include "src/turbomind/runtime/rthelper.h"
#include "src/turbomind/runtime/core/DIPURawGeneratorImpl.h"

#include "src/turbomind/utils/Tensor.h"

using deviceStream_t = dipu::deviceStream_t;

extern "C" {
struct diopiContext {
    deviceStream_t stream;
    // 1. use arrays to hold tensor that avoid tensor deleting when leaving scope
    // 2. The address of each array must be fixed, so use list instead of vector
    std::list<turbomind::Tensor> arrays;

    explicit diopiContext(const deviceStream_t& s) : stream(s) {}
};

}  // extern "C"

namespace dipu {

namespace diopi_helper {

::diopiTensorHandle_t toDiopiTensorHandle(turbomind::Tensor& tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const turbomind::Tensor& tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const turbomind::Tensor* tensor);

::diopiGeneratorHandle_t toDiopiGeneratorHandle(DIPURawGeneratorImpl& generator);

static std::map<turbomind::DataType, diopiDtype_t> toDiopiDataType = {
    {turbomind::DataType::TYPE_INVALID, diopiDtype_t::diopi_dtype_unsupported},
    {turbomind::DataType::TYPE_BOOL, diopiDtype_t::diopi_dtype_bool},
    {turbomind::DataType::TYPE_INT8, diopiDtype_t::diopi_dtype_int8},
    {turbomind::DataType::TYPE_INT32, diopiDtype_t::diopi_dtype_int32},
    {turbomind::DataType::TYPE_INT64, diopiDtype_t::diopi_dtype_int64},
    {turbomind::DataType::TYPE_UINT8, diopiDtype_t::diopi_dtype_uint8},
    {turbomind::DataType::TYPE_UINT32, diopiDtype_t::diopi_dtype_uint32},
    {turbomind::DataType::TYPE_UINT64, diopiDtype_t::diopi_dtype_uint64},
    {turbomind::DataType::TYPE_FP16, diopiDtype_t::diopi_dtype_float16},
    {turbomind::DataType::TYPE_FP32, diopiDtype_t::diopi_dtype_float32},
    {turbomind::DataType::TYPE_FP64, diopiDtype_t::diopi_dtype_float64}
};

static std::map<diopiDtype_t, turbomind::DataType> toTmDataType = {
    {diopiDtype_t::diopi_dtype_unsupported, turbomind::DataType::TYPE_INVALID},
    {diopiDtype_t::diopi_dtype_bool, turbomind::DataType::TYPE_BOOL},
    {diopiDtype_t::diopi_dtype_int8, turbomind::DataType::TYPE_INT8},
    {diopiDtype_t::diopi_dtype_int32, turbomind::DataType::TYPE_INT32},
    {diopiDtype_t::diopi_dtype_int64, turbomind::DataType::TYPE_INT64},
    {diopiDtype_t::diopi_dtype_uint8, turbomind::DataType::TYPE_UINT8},
    {diopiDtype_t::diopi_dtype_uint32, turbomind::DataType::TYPE_UINT32},
    {diopiDtype_t::diopi_dtype_uint64, turbomind::DataType::TYPE_UINT64},
    {diopiDtype_t::diopi_dtype_float16, turbomind::DataType::TYPE_FP16},
    {diopiDtype_t::diopi_dtype_float32, turbomind::DataType::TYPE_FP32},
    {diopiDtype_t::diopi_dtype_float64, turbomind::DataType::TYPE_FP64}
};

turbomind::MemoryType toTmDevice(::diopiDevice_t device);

}  // namespace diopi_helper

}  // namespace dipu