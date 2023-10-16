#pragma once


#include "vendor/vendorapi.h“
#include "deviceapis.h"
#include "../diopirt/diopirt_impl.h"

namespace dipu {

// need add return status.
namespace devapis {
  typedef enum  
  {
    MIN,
    MAX,
    SUM,
    PRODUCT,
    AVG,
  } DiclReduceOp;

  template<typename T>
  static DiclDataType getDiclDataType() {
    DiclDataType dType;
    if (std::is_same<T, float>::value) {
        dType = DiclDataType::TYPE_FP32;
    }
    else if (std::is_same<T, int>::value) {
        dType = DiclDataType::TYPE_INT32;
    }
    else if (std::is_same<T, bool>::value) {
        dType = DiclDataType::TYPE_BOOL;
    }
    else {
        printf("[ERROR] DICL only support float, int, and bool. \n");
        exit(-1);
    }
    return dType;
  };

  extern const int DICL_UNIQUE_ID_BYTES_SIZE;

  // todo:: dipu only export devproxy but not devapis (which move o diopi)
  DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm);

  DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId);

  DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId, int rank, int localDeviceId = -1);

  // DIPU_API void diclCommInitAll(diclComm_t* comms, int ndev, const int* devlist);

  DIPU_API diclResult_t diclCommDestroy(diclComm_t comm);

  // DIPU_API diclResult_t diclCommFinalize(diclComm_t comm);

  // DIPU_API diclResult_t diclCommAbort(diclComm_t comm);

  DIPU_API diclResult_t diclRawAllReduce(const void *sendbuff, void *recvbuff, size_t count, DiclDataType datatype,
                              const DiclReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRawBroadcast(const void *sendbuff, void* recvbuff, size_t count, DiclDataType datatype,
                              int root, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRawAllGather(const void *sendBuf, void *recvBuf, size_t count, DiclDataType datatype,
                              diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRawReduce(const void* sendbuff, void* recvbuff, size_t count, DiclDataType datatype,
                            const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRawReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, DiclDataType dataType, 
                                  const ReduceOp& op, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRawSend(void* sendbuff, size_t count, DiclDataType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRawRecv(void* recvbuff, size_t count, DiclDataType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRawGroupStart();

  DIPU_API diclResult_t diclRawGroupEnd();

} // namespace devapis

} // namespace dipu