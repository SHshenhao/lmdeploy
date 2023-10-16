#include <stdio.h>

#include "diopirt_impl.h"

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

namespace dipu {

namespace diopi_helper {

::diopiTensorHandle_t toDiopiTensorHandle(turbomind::Tensor& tensor) {
    return tensor.data == nullptr ? nullptr : reinterpret_cast<::diopiTensorHandle_t>(&tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const turbomind::Tensor& tensor) {
    return tensor.data == nullptr ? nullptr : reinterpret_cast<::diopiConstTensorHandle_t>(&tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const turbomind::Tensor* tensor) {
    return tensor == nullptr ? nullptr : toDiopiTensorHandle(*tensor);
}

// ::diopiGeneratorHandle_t toDiopiGeneratorHandle(::Generator& generator) {
//     if (!generator.has_value()) return nullptr;
//         return toDiopiGeneratorHandle(generator.value());
// }

}  // namespace diopi_helper

}  // namespace dipu


bool diopiTensor::resetShape(const diopiSize_t* size) {
    int64_t numel = 1;
    for (int64_t i = 0; i < size->len; ++i) {
        numel *= size->data[i];
    }
    if (numel != numel_) return false;

    shape_.resize(size->len);
    for (int64_t i = size->len - 1; i >= 0; --i) {
        shape_[i] = size->data[i];
    }
    return true;
}

DIOPI_RT_API diopiError_t diopiGetTensorData(diopiTensorHandle_t th, void** pptr) {
    *pptr = th->getPtr<void*>();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDataConst(diopiConstTensorHandle_t th, const void** pptr) {
    *pptr = th->getPtr<void*>();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorShape(diopiConstTensorHandle_t th, diopiSize_t* size) {
    diopiSize_t thSize = {th->getPtr<int64_t>(), std::static_cast<int64_t>(size())};
    *size = thSize;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t th, diopiSize_t* stride) {
    diopiSize_t thStride = {th->getPtr<int64_t>(), std::static_cast<int64_t>(size())};
    *size = thStride;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDtype(diopiConstTensorHandle_t th, diopiDtype_t* dtype) {
    *dtype = toDiopiDataType[th->dtype()];
    return diopiSuccess;
}

// DIOPI_RT_API diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t th, diopiDevice_t* device) {
//     *device = th->device();
//     return diopiSuccess;
// }

DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t th, int64_t* numel) {
    diopiSize_t thNumel = {th->getPtr<int64_t>(), std::static_cast<int64_t>(size())};
    *numel = thNumel;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t th, int64_t* elemSize) {
    *elemSize = std::static_cast<int64_t>(th->sizeBytes());
    return diopiSuccess;
}