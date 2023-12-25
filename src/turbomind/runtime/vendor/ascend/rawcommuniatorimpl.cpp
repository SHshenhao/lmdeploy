#include <cstring>
#include "../../device/rawdiclapis.h"

namespace dipu {

namespace devapis {
// HCCL ReduceOp mapping
  std::map<DiclReduceOp, HcclReduceOp> hcclOp = {
    {DiclReduceOp::MIN, HCCL_REDUCE_MIN},
    {DiclReduceOp::MAX, HCCL_REDUCE_MAX},
    {DiclReduceOp::SUM, HCCL_REDUCE_SUM},
    {DiclReduceOp::PRODUCT, HCCL_REDUCE_PROD}
  };

  static std::map<DiclDataType, HcclDataType> hcclDataType = {
  // {DiclDataType::TYPE_INVALID, },
  {DiclDataType::TYPE_BOOL, HCCL_DATA_TYPE_UINT8},
  {DiclDataType::TYPE_UINT8, HCCL_DATA_TYPE_UINT8},
  // {DiclDataType::TYPE_UINT16, },
  // {DiclDataType::TYPE_UINT32, },
  // {DiclDataType::TYPE_UINT64, },
  {DiclDataType::TYPE_INT8, HCCL_DATA_TYPE_INT8},
  // {DiclDataType::TYPE_INT16, },
  {DiclDataType::TYPE_INT32, HCCL_DATA_TYPE_INT32},
  {DiclDataType::TYPE_INT64, HCCL_DATA_TYPE_INT64},
  {DiclDataType::TYPE_FP16, HCCL_DATA_TYPE_FP16},
  {DiclDataType::TYPE_FP32, HCCL_DATA_TYPE_FP32},
  {DiclDataType::TYPE_FP64, HCCL_DATA_TYPE_FP64},
  // {DiclDataType::TYPE_BYTES, },
  #if HAS_HCCL_BF16_DATATYPE
  {DiclDataType::TYPE_BF16, ncclBfloat16},
  #endif
  };

#define HCCL_THROW(cmd)                                                   \
  do {                                                                    \
    VENDOR_CHECK(cmd == HCCL_SUCCESS, "HCCL error in: " +                  \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +  \
                ".\n" + "And see details in Ascend logs.\n" +             \
                aclGetRecentErrMsg());                                    \
  } while (0)

const int DICL_UNIQUE_ID_BYTES_SIZE = HCCL_ROOT_INFO_BYTES;

// TODO: not support
DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  VENDOR_CHECK(false, "ascend Not implement diclGetCommAsyncError");
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  HCCL_THROW(HcclGetRootInfo(uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId,
                                        int rank, int localDeviceId) {
  HCCL_THROW(HcclCommInitRootInfo(nranks, &uniqueId, rank, comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
  HCCL_THROW(HcclCommDestroy(comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRawAllReduce(const void *sendBuff, void *recvBuff, size_t count,
                                    DiclDataType datatype, const DiclReduceOp& reduceOp,
                                    diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclAllReduce(const_cast<void *>(sendBuff), recvBuff, count,
                           hcclDataType[datatype], hcclOp[reduceOp], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRawAllGather(const void *sendBuf, void *recvBuf, size_t count,
                                    DiclDataType datatype, diclComm_t comm, 
                                    deviceStream_t stream) {
  HCCL_THROW(HcclAllGather(const_cast<void *>(sendBuf), recvBuf, count, 
                           hcclDataType[datatype], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRawReduce(const void* sendBuf, void* recvBuf, size_t count, 
                                 DiclDataType datatype, const DiclReduceOp& reduceOp,
                                 int root, diclComm_t comm, deviceStream_t stream) {

  HCCL_THROW(HcclReduce(const_cast<void *>(sendBuf), recvBuf, count, hcclDataType[datatype],
                        hcclOp[reduceOp], root, comm, stream));                   
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRawReduceScatter(void *sendBuf, void *recvBuf, size_t recvCount,
                                        DiclDataType datatype, const DiclReduceOp& op, 
                                        diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclReduceScatter(sendBuf, recvBuf, recvCount, hcclDataType[datatype],
                               hcclOp[op], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRawSend(void* sendBuf, size_t count, DiclDataType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclSend(sendBuf, count, hcclDataType[datatype], peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRawRecv(void* recvBuf, size_t count, DiclDataType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclRecv(recvBuf, count, hcclDataType[datatype], peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRawBroadcast(const void *sendBuf, void* recvBuf, size_t count,
                                    DiclDataType datatype, int root, diclComm_t comm,
                                    deviceStream_t stream) {
  HCCL_THROW(HcclBroadcast(const_cast<void *>(sendBuf), count, hcclDataType[datatype],
                           root, comm, stream));
  return DICL_SUCCESS;
}

} // end namespace devapis
} // end namespace dipu