/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h

#pragma once

#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModelInstance.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/cuda_utils.h"
// #include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/nccl_utils.h"
// #include <cuda_fp16.h>
#include <mutex>

namespace ft = turbomind;

template<typename T>
struct LlamaTritonSharedModelInstance;

template<typename T>
struct LlamaTritonModel: public AbstractTransformerModel {
    LlamaTritonModel(size_t      tensor_para_size,
                     size_t      pipeline_para_size,
                     int32_t         enable_custom_all_reduce,
                     std::string model_dir);

    ~LlamaTritonModel() = default;

    std::unique_ptr<AbstractTransformerModelInstance>
    createModelInstance(int32_t                                                               deviceId,
                        int32_t                                                               rank,
                        dipu::deviceStream_t                                              stream,
                        std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                        std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr) override;

    void createSharedWeights(int32_t deviceId, int32_t rank) override;

    void createCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                           int32_t                                                   world_size) override;

    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>
    createNcclParams(const int32_t node_id, const int32_t device_id_start, const bool multi_node) override;

    std::unique_ptr<ft::AbstractInstanceComm> createInstanceComm(int32_t size) override;

    void handleMissingParams();

    void setFfiLock(ffi_api_lock_ctrl_t func)
    {
        ffi_lock_ = func;
    }

    std::string toString() override;
    int32_t         getTensorParaSize() override;
    int32_t         getPipelineParaSize() override;

private:
    std::unique_ptr<LlamaTritonSharedModelInstance<T>>
    createSharedModelInstance(int32_t                                                               deviceId,
                              int32_t                                                               rank,
                              std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                              std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr);

    size_t                          head_num_;
    size_t                          kv_head_num_;
    size_t                          size_per_head_;
    size_t                          inter_size_;
    size_t                          num_layer_;
    size_t                          vocab_size_;
    turbomind::LlamaAttentionParams attn_params_;
    float                           norm_eps_;
    int32_t                             max_batch_size_;
    int32_t                             max_context_token_num_;
    int32_t                             session_len_;
    int32_t                             step_length_;
    int32_t                             start_id_;
    int32_t                             end_id_;
    int32_t                             cache_max_entry_count_;
    int32_t                             cache_chunk_size_;
    int32_t                             use_context_fmha_;
    size_t                          tensor_para_size_;
    size_t                          pipeline_para_size_;
    ft::WeightType                  weight_type_;
    bool                            attn_bias_;
    int32_t                             quant_policy_;
    int32_t                             group_size_;

    // shared weights for each device
    std::vector<std::shared_ptr<ft::LlamaWeight<T>>> shared_weights_;

    std::shared_ptr<typename ft::LlamaV2<T>::SharedState> shared_state_;

    std::vector<std::shared_ptr<LlamaTritonSharedModelInstance<T>>> shared_instances_;
    std::deque<std::mutex>                                          shared_mutexes_;  // is locking really needed?

    bool is_fp16_;
    int32_t  enable_custom_all_reduce_ = 0;

    std::string model_name_;
    std::string model_dir_;

    ffi_api_lock_ctrl_t ffi_lock_ = nullptr;
};
