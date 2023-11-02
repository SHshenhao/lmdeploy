#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

namespace dipu {

#define VENDOR_CHECK(cond, ...) \
{ \
  if (!(cond)) { \
    auto msg = std::string(" FILE:") + std::string(__FILE__) + "LINE:" + std::to_string(__LINE__) + " \n"; \
    auto msg1 = std::to_string('##__VA_ARGS__');  \
    throw std::runtime_error(msg + msg1); \
  } \
}

// VENDOR_CHECK(ret == ::cudaSuccess, "call cuda error, expr = ", #Expr, ", ret = ", ret, mes);
#define DIPU_CALLCUDA(Expr)                                                     \
{                                                                               \
    cudaError_t ret = Expr;                                                     \
    if (!(ret == ::cudaSuccess)) { \
      auto msg = std::string(" FILE:") + std::string(__FILE__) + "LINE:" + std::to_string(__LINE__) + "call cuda error, ret = " + cudaGetErrorString(ret);  + " \n"; \
      throw std::runtime_error(msg); \
  } \
}

using deviceStream_t = cudaStream_t;
#define deviceDefaultStreamLiteral cudaStreamLegacy
using deviceEvent_t = cudaEvent_t;

using diclComm_t = ncclComm_t;
using commUniqueId = ncclUniqueId;

}