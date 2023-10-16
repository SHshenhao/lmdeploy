// Copyright (c) 2023, DeepLink.
#pragma once

#include <memory>
#include <cstdint>

#include "basedef.h"

namespace dipu {
struct RandState {
    std::share_ptr<uint8_t*> state_ = nullptr;
    size_t size_ = 0;
};

class DIPURawGeneratorImpl {
public:
  // Constructors
  explicit DIPURawGeneratorImpl();
  ~DIPURawGeneratorImpl() = default;

  std::shared_ptr<DIPURawGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed);
  uint64_t current_seed() const;
  uint64_t seed();
  RandState get_state() const;
  RandState clone_state(const RandState& state) const;
  virtual void set_state(const RandState& state) {};

  void set_state_flag(bool flag);
  virtual void update_state() const {};

  uint64_t seed_;
  mutable RandState state_;
  mutable bool state_need_reset_;
}

}  // namespace dipu
