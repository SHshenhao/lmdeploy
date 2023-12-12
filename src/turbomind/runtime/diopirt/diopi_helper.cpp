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