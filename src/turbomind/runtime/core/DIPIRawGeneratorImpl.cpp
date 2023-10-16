// Copyright (c) 2023, DeepLink.
#include "DIPURawGeneratorImpl.h"
#include "../device/rawdeviceapis.h>

namespace dipu {
/**
 * DIPURawGeneratorImpl class implementation
 */
DIPURawGeneratorImpl::DIPURawGeneratorImpl() :state_need_reset_(true) {
  seed_ = 0;
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<DIPURawGeneratorImpl> DIPURawGeneratorImpl::clone() const {
  auto gen = new DIPURawGeneratorImpl();
  gen->set_current_seed(this->seed_);
  auto state = this->state_;
  const auto& state_clone = this->clone_state();
  gen->set_state(state_clone);
  gen->set_state_flag(this->state_need_reset_);
  return std::shared_ptr<DIPURawGeneratorImpl>(&gen);
}

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
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 */
uint64_t DIPURawGeneratorImpl::seed() {
  //TODO:随机生成seed
  uint64_t random = 42;
  this->set_current_seed(random);
  return random;
}

/**
 * get state
 *
 * See Note [Acquire lock when using random generators]
  */
RandState DIPURawGeneratorImpl::get_state() const {
  if (state_need_reset_) {
    update_state();
  }
  auto state_clone = this->clone_state();
  return state_clone;
}

RandState DIPURawGeneratorImpl::clone_state(const RandState& state) const {
  uint8_t newnum = *(state.state_);
  RandState newState{&newnum, state.size_};
  return newState;
}

/**
* set state
*
* See Note [Acquire lock when using random generators]
*/
void DIPURawGeneratorImpl::set_state(const RandState& state) {
  this->state_ = this->clone_state(state);
}

/**
 * set state flag
 * See Note [Acquire lock when using random generators]
  */
void DIPURawGeneratorImpl::set_state_flag(bool flag) {
  state_need_reset_ = flag;
}

}  // namespace dipu
