// Copyright (c) 2023, DeepLink.
#pragma once

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include "3rdparty/DIOPI/impl/third_party/half/include/half.hpp"
using half=half_float::half;

namespace dipu
{

#define VENDOR_CHECK(cond, ...) \
{ \
  if (!(cond)) { \
    throw std::runtime_error(std::string(" FILE:") + std::string(__FILE__) + "LINE:" + std::to_string(__LINE__) \
                              + " " + #__VA_ARGS__ + " \n"); \
  } \
}

#define TRACK_ACL(x)                                                   \
    {                                                                  \
        static bool enable = std::getenv("DIPU_TRACK_ACL") != nullptr; \
        if (enable)                                                    \
        {                                                              \
            printf("[%s: %d]:%s\n", __FILE__, __LINE__, x);            \
        }                                                              \
    }

#define DIPU_CALLACLRT(Expr)                               \
    {                                                      \
        TRACK_ACL(#Expr);                                  \
        ::aclError ret = Expr;                             \
        if (ret != ::ACL_SUCCESS)                          \
        {                                                  \
          throw std::runtime_error(std::string("ascend device error:") + aclGetRecentErrMsg()); \
        }                                                  \
    }

using deviceStream_t = aclrtStream;
#define deviceDefaultStreamLiteral nullptr;
using deviceEvent_t = aclrtEvent;
using deviceHandle_t = aclrtContext*;

using diclComm_t = HcclComm;
using commUniqueId = HcclRootInfo;
}
