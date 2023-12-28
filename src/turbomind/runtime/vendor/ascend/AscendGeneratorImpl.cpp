// Copyright (c) 2023, DeepLink.
#include "../../core/DIPURawGeneratorImpl.h"

namespace dipu {

static const size_t seed_size = sizeof(uint64_t);
static const size_t offset_size = sizeof(int64_t);
static const size_t total_size = seed_size + offset_size;

void DIPURawGeneratorImpl::set_state(const turbomind::Tensor& state) {
    auto state_size = state.sizeBytes();
    // assert(
    // state_size == total_size || state_size == total_size - offset_size,
    // "RNG state is wrong size");

    void* state_tmp_data = std::malloc(total_size);
    // assert(state.type == turbomind::MemoryType::MEMORY_CPU || state.type == turbomind::MemoryType::MEMORY_CPU_PINNED);
    turbomind::Tensor state_tmp{state.where, state.type, state.shape, reinterpret_cast<void*>(state_tmp_data)};
    memcpy(state_tmp_data, state.data, total_size);
    if (state_.data != nullptr) {
        // std::free(state_.data);
    }
    state_ = state_tmp;
    state_need_reset_ = false;
}

void DIPURawGeneratorImpl::update_state() {
    if (state_need_reset_) {
        void* state_tmp_data = std::malloc(total_size);
        turbomind::Tensor state_tmp{turbomind::MemoryType::MEMORY_CPU, turbomind::DataType::TYPE_UINT8, {total_size}, reinterpret_cast<void*>(state_tmp_data)};
        if (state_.data != nullptr) {
            // std::free(state_.data);
        }
        state_ = state_tmp;
        auto rng_state = state_.getPtr<uint8_t>();
        uint64_t seed = this->current_seed();
        int64_t offset = 0;
        memcpy(rng_state, &seed, seed_size);
        memcpy(rng_state + seed_size, &offset, offset_size);
        state_need_reset_ = false;
    }
}

}  // namespace dipu
