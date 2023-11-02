// Copyright (c) 2023, DeepLink.
#include <stdio.h>
#include <mutex>

#include "diopirt_impl.h"

namespace diopihelper = dipu::diopi_helper;

extern "C" {

static char diopiVersion[256] = {0};

DIOPI_RT_API const char* diopiGetVersion() {
    static bool inited = false;
    if (!inited) {
        inited = true;
        snprintf(diopiVersion, sizeof(diopiVersion), "DIOPI Version: %d.%d.%d", DIOPI_VER_MAJOR, DIOPI_VER_MINOR, DIOPI_VER_PATCH);
    }
    return diopiVersion;
}

// bool diopiTensor::resetShape(const diopiSize_t* size) {
//     int64_t numel = 1;
//     for (int64_t i = 0; i < size->len; ++i) {
//         numel *= size->data[i];
//     }
//     if (numel != numel_) return false;

//     shape_.resize(size->len);
//     for (int64_t i = size->len - 1; i >= 0; --i) {
//         shape_[i] = size->data[i];
//     }
//     return true;
// }

DIOPI_RT_API diopiError_t diopiGetTensorData(diopiTensorHandle_t th, void** pptr) {
    *pptr = (reinterpret_cast<turbomind::Tensor*>(th))->getPtr<void*>();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDataConst(diopiConstTensorHandle_t th, const void** pptr) {
    *pptr = (reinterpret_cast<const turbomind::Tensor*>(th))->getPtr<void*>();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorShape(diopiConstTensorHandle_t th, diopiSize_t* size) {
    const turbomind::Tensor* ptr = reinterpret_cast<const turbomind::Tensor*>(th);
    std::vector<int64_t> shape;
    for (auto& length: ptr->shape) {
        shape.emplace_back(length);
    }
    diopiSize_t tsize{shape.data(), int64_t(shape.size())};
    *size = tsize;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t th, diopiSize_t* stride) {
    const turbomind::Tensor* ptr = reinterpret_cast<const turbomind::Tensor*>(th);
    std::vector<int64_t> shape{1};
    for (size_t i = ptr->shape.size()-1; i >= 1; i--) {
        shape.insert(shape.begin(), shape.front() * ptr->shape[i]);
    }
    diopiSize_t tstride{shape.data(), int64_t(shape.size())};
    *stride = tstride;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDtype(diopiConstTensorHandle_t th, diopiDtype_t* dtype) {
    *dtype = diopihelper::toDiopiDataType[(reinterpret_cast<const turbomind::Tensor*>(th))->type];
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t th, diopiDevice_t* device) {
    *device = ((reinterpret_cast<const turbomind::Tensor*>(th))->where == turbomind::MemoryType::MEMORY_GPU ? diopi_device : diopi_host);
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t th, int64_t* numel) {
    if (th == nullptr) {
        *numel = 0;
        return diopiSuccess;
    }

    *numel = int64_t((reinterpret_cast<const turbomind::Tensor*>(th))->size());
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t th, int64_t* elemSize) {
    *elemSize = int64_t(turbomind::Tensor::getTypeSize(reinterpret_cast<const turbomind::Tensor*>(th)->type));
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream) {
    *stream = ctx->stream;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireTensor(
    diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
    const diopiSize_t* size, const diopiSize_t* stride,
    const diopiDtype_t dtype, const diopiDevice_t device) {
    turbomind::DataType data_type = diopihelper::toTmDataType[dtype];
    turbomind::MemoryType mem_type = diopihelper::toTmDevice(device);
    std::vector<size_t> _shape;
    int64_t numel = 1;
    for (int64_t i = 0; i < size->len; i++) {
        int64_t dimlength = *(size->data + i);
        _shape.emplace_back(dimlength);
        numel *= dimlength;
    }
    void* _data;
    if (mem_type == turbomind::MemoryType::MEMORY_GPU) {
        dipu::devapis::mallocDevice(&_data, size_t(numel) * turbomind::Tensor::getTypeSize(data_type));
    } else {
        dipu::devapis::mallocHost(&_data, size_t(numel) * turbomind::Tensor::getTypeSize(data_type));
    }
    turbomind::Tensor t{mem_type, data_type, _shape, _data};
    ctx->arrays.emplace_back(std::move(t));
    *tensor = reinterpret_cast<diopiTensorHandle_t>(&(ctx->arrays.back()));
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireBuffer(
    diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
    int64_t num_bytes, diopiDevice_t device) {
    diopiSize_t size{&num_bytes, 1};
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, device);
}

DIOPI_RT_API diopiError_t diopiGeneratorGetState(diopiContextHandle_t ctx, diopiConstGeneratorHandle_t th, diopiTensorHandle_t *state) {
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorSetState(diopiGeneratorHandle_t th, diopiConstTensorHandle_t new_state) {
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRecordStart(const char* record_name, void** record) {
    *record = nullptr;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRecordEnd(void** record) {
    *record = nullptr;
    return diopiSuccess;
}

}  // extern "C"