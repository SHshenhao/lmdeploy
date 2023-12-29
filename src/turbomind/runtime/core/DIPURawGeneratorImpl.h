// Copyright (c) 2023, DeepLink.
#pragma once

// #include "src/turbomind/runtime/rthelper.h"
#include "src/turbomind/utils/Tensor.h"

namespace dipu {
class DIPURawGeneratorImpl {
public:
  // Constructors
  explicit DIPURawGeneratorImpl();
  ~DIPURawGeneratorImpl();

  void set_current_seed(uint64_t seed);
  uint64_t current_seed() const;
  turbomind::Tensor& get_state();
  void set_state(const turbomind::Tensor& state);
  // virtual void set_offset(uint64_t offset) {};
  // virtual uint64_t get_offset() const {return 0;};

protected:
  void set_state_flag(bool flag);
  void update_state();

  uint64_t seed_ = 0;
  turbomind::Tensor state_{};
  bool state_need_reset_ = true;
};

}  // namespace dipu
