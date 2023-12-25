#include <cstring>
#include "../../device/rawdiclapis.h"

namespace dipu {

namespace devapis {
  // NCCL op mapping
  static std::map<DiclReduceOp, ncclRedOp_t> ncclOp = {
    {DiclReduceOp::MIN, ncclMin},
    {DiclReduceOp::MAX, ncclMax},
    {DiclReduceOp::SUM, ncclSum},
    {DiclReduceOp::PRODUCT, ncclProd},
  #ifdef NCCL_HAS_AVG
    {DiclReduceOp::AVG, ncclAvg},
  #endif
  };

  static std::map<DiclDataType, ncclDataType_t> ncclDataType = {
    // {DiclDataType::TYPE_INVALID, },
    {DiclDataType::TYPE_BOOL, ncclUint8},
    {DiclDataType::TYPE_UINT8, ncclUint8},
    // {DiclDataType::TYPE_UINT16, },
    // {DiclDataType::TYPE_UINT32, },
    // {DiclDataType::TYPE_UINT64, },
    {DiclDataType::TYPE_INT8, ncclInt8},
    // {DiclDataType::TYPE_INT16, },
    {DiclDataType::TYPE_INT32, ncclInt32},
    {DiclDataType::TYPE_INT64, ncclInt64},
    {DiclDataType::TYPE_FP16, ncclHalf},
    {DiclDataType::TYPE_FP32, ncclFloat},
    {DiclDataType::TYPE_FP64, ncclDouble},
    // {DiclDataType::TYPE_BYTES, },
  #if HAS_NCCL_BF16_DATATYPE
    {DiclDataType::TYPE_BF16, ncclBfloat16},
  #endif
  };

// Macro to print and abort on a non-successful NCCL return value.
#define NCCL_THROW(cmd)                            \
  do {                                                   \
    ncclResult_t result = cmd;                           \
    if (result != ncclSuccess) {                         \
      std::string err = ncclGetErrorString(result); \
      fprintf(                                           \
          stderr,                                        \
          "NCCL error in: %s:%d, %s\n",                  \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
    }                                                    \
  } while (0)


  const int DICL_UNIQUE_ID_BYTES_SIZE = NCCL_UNIQUE_ID_BYTES;

  DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
    ncclResult_t ncclAsyncErr_;
    NCCL_THROW(ncclCommGetAsyncError(comm, &ncclAsyncErr_));
    if (ncclAsyncErr_ != ncclSuccess) {
      return DICL_SUCCESS;
    } else {
      return DICL_ERR_UNDEF;
    }
  }

  DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
    NCCL_THROW(ncclGetUniqueId(uniqueId));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId,
                                          int rank, int localDeviceId) {
    NCCL_THROW(ncclCommInitRank(comm, nranks, uniqueId, rank));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclCommDestroy(ncclComm_t comm) {
    NCCL_THROW(ncclCommDestroy(comm));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawAllReduce(const void *sendbuff, void *recvbuff, size_t count, DiclDataType datatype,
                              const DiclReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
    NCCL_THROW(ncclAllReduce(sendbuff, recvbuff, count, ncclDataType[datatype], ncclOp[reduceOp], comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawBroadcast(const void *sendbuff, void* recvbuff, size_t count, DiclDataType datatype,
                              int root, diclComm_t comm, deviceStream_t stream) {
    NCCL_THROW(ncclBroadcast(sendbuff, recvbuff, count, ncclDataType[datatype], root, comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawAllGather(const void *sendBuf, void *recvBuf, size_t count, DiclDataType datatype,
                              diclComm_t comm, deviceStream_t stream) {
    NCCL_THROW(ncclAllGather(sendBuf, recvBuf, count, ncclDataType[datatype], comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawReduce(const void* sendbuff, void* recvbuff, size_t count, DiclDataType datatype,
                            const DiclReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream) {
    NCCL_THROW(ncclReduce(sendbuff, recvbuff, count, ncclDataType[datatype], ncclOp[reduceOp], root, comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, DiclDataType datatype, 
                                  const DiclReduceOp& op, diclComm_t comm, deviceStream_t stream) {
    throw std::runtime_error("mlu Not implement diclReduceScatter");
  }

  DIPU_API diclResult_t diclRawSend(void* sendbuff, size_t count, DiclDataType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream){
    NCCL_THROW(ncclSend(sendbuff, count, ncclDataType[datatype], peer, comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawRecv(void* recvbuff, size_t count, DiclDataType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream) {
    NCCL_THROW(ncclRecv(recvbuff, count, ncclDataType[datatype], peer, comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawGroupStart() {
    NCCL_THROW(ncclGroupStart());
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRawGroupEnd() {
    NCCL_THROW(ncclGroupEnd());
    return DICL_SUCCESS;
  }

} // end namespace devapis
} // end namespace dipu
