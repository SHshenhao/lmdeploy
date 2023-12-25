#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cuda_fp16.h>
// #include "3rdparty/DIOPI/impl/third_party/half/include/half.hpp"
// using half=half_float::half;

namespace dipu {

#define VENDOR_CHECK(cond, ...) \
{ \
  if (!(cond)) { \
    throw std::runtime_error(std::string(" FILE:") + std::string(__FILE__) + "LINE:" + std::to_string(__LINE__) \
                              + " " + #__VA_ARGS__ + " \n"); \
  } \
}

#define DIPU_CALLCUDA(Expr)                                                     \
{                                                                               \
    cudaError_t ret = Expr;                                                     \
    if (!(ret == ::cudaSuccess)) { \
      auto msg = std::string(" FILE:") + std::string(__FILE__) + "LINE:" + std::to_string(__LINE__) + \
                  " call cuda " + #Expr + " error, ret = " + cudaGetErrorString(ret);  + " \n"; \
      throw std::runtime_error(msg); \
  } \
}

using deviceStream_t = cudaStream_t;
#define deviceDefaultStreamLiteral cudaStreamLegacy
using deviceEvent_t = cudaEvent_t;

using diclComm_t = ncclComm_t;
using commUniqueId = ncclUniqueId;

}