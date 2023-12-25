// Copyright (c) 2023, DeepLink.
#include "DIPURawGeneratorImpl.h"

namespace dipu {

DIPURawGeneratorImpl::~DIPURawGeneratorImpl() {
  if (state_.data != nullptr) {
    std::free(state_.data);
  }
}

/**
 * DIPUGeneratorImpl class implementation
 */
DIPURawGeneratorImpl::DIPURawGeneratorImpl()
  : state_need_reset_(true) {}

/**
 * Sets the seed to be used by MTGP
 *
 * See Note [Acquire lock when using random generators]
 */
void DIPURawGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  state_need_reset_ = true;
}

/**
 * Gets the current seed of DIPUGeneratorImpl.
 */
uint64_t DIPURawGeneratorImpl::current_seed() const {
  return seed_;
}

/**
 * get state
 *
 * See Note [Acquire lock when using random generators]
  */
turbomind::Tensor& DIPURawGeneratorImpl::get_state() {
  if (state_need_reset_) {
      update_state();
  }
  return state_;
}

/**
 * set state flag
 * See Note [Acquire lock when using random generators]
  */
void DIPURawGeneratorImpl::set_state_flag(bool flag) {
  state_need_reset_ = flag;
}

}  // namespace dipu
