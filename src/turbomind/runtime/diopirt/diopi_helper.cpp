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
#include <unordered_set>

namespace dipu {

namespace diopi_helper {

diopiError_t clearDiopiContextAll(diopiContext& ctx) {
    int64_t arraysize{ctx.arrays.size()};
    if (arraysize <= 0) return diopiSuccess;
    std::unordered_set<void*> ptr_map;
    dipu::devapis::syncStream(ctx.stream);
    for (auto &tensor: ctx.arrays) {
        if (!tensor.preallocated && tensor.data != nullptr && ptr_map.find(tensor.data) == ptr_map.end()) {
            if (tensor.where == turbomind::MEMORY_GPU) {
                dipu::devapis::freeDevice(tensor.data);
                ptr_map.emplace(tensor.data);
                tensor.data = nullptr;
            }
            if (tensor.where == turbomind::MEMORY_CPU_PINNED) {
                dipu::devapis::freeHost(tensor.data);
                ptr_map.emplace(tensor.data);
                tensor.data = nullptr;
            }
        }
    }
    ptr_map.clear();
    ctx.arrays.clear();
    return diopiSuccess;
}

diopiError_t clearDiopiContextAfterN(diopiContext& ctx, int64_t num) {
    int64_t arraysize{ctx.arrays.size()};
    if (arraysize <= num) return diopiSuccess;
    std::unordered_set<void*> ptr_map;
    int64_t i = 0;
    dipu::devapis::syncStream(ctx.stream);
    for (auto &tensor: ctx.arrays) {
        ++i;
        if (i <= num) continue;
        if (!tensor.preallocated && tensor.data != nullptr && ptr_map.find(tensor.data) == ptr_map.end()) {
            if (tensor.where == turbomind::MEMORY_GPU) {
                ptr_map.emplace(tensor.data);
                dipu::devapis::freeDevice(tensor.data);
                tensor.data = nullptr;
            }
            if (tensor.where == turbomind::MEMORY_CPU_PINNED) {
                ptr_map.emplace(tensor.data);
                dipu::devapis::freeHost(tensor.data);
                tensor.data = nullptr;
            }
        }
    }
    ptr_map.clear();
    ctx.arrays.resize(std::min(num, arraysize));
    return diopiSuccess;
}

::diopiTensorHandle_t toDiopiTensorHandle(turbomind::Tensor& tensor) {
    return tensor.data == nullptr ? nullptr : reinterpret_cast<::diopiTensorHandle_t>(&tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const turbomind::Tensor& tensor) {
    return tensor.data == nullptr ? nullptr : reinterpret_cast<::diopiConstTensorHandle_t>(&tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const turbomind::Tensor* tensor) {
    return tensor == nullptr ? nullptr : toDiopiTensorHandle(*tensor);
}

::diopiGeneratorHandle_t toDiopiGeneratorHandle(dipu::DIPURawGeneratorImpl& generator) {
    return reinterpret_cast<::diopiGeneratorHandle_t>(&generator);
}

turbomind::MemoryType toTmDevice(::diopiDevice_t device) {
    switch (device) {
    case diopi_host:
        return turbomind::MemoryType::MEMORY_CPU_PINNED;
    case diopi_device:
        return turbomind::MemoryType::MEMORY_GPU;
    default:
        VENDOR_CHECK(false, "invalid diopi device, diopi device is ", device);
    }
}


}  // namespace diopi_helper

}  // namespace dipu