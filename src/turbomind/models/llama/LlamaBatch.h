// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaCacheManager.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/utils/allocator.h"
// #include "src/turbomind/utils/cublasMMWrapper.h"

#include "src/turbomind/runtime/diopirt/diopirt_impl.h"
#include "src/turbomind/utils/indexablelist.h"

namespace turbomind {

template<typename T>
class LlamaV2;

template<typename T>
class LlamaBatch {
public:
    int32_t size() const noexcept
    {
        return batch_size_;
    };

    int32_t maxSize() const noexcept
    {
        return max_batch_size_;
    }

    int32_t finishedCount() const noexcept
    {
        return finished_count_;
    }

    int32_t sessionLen() const noexcept
    {
        return session_len_;
    }

    void verifyRequests(std::vector<std::shared_ptr<Request>>& stop_reqs,
                        std::vector<std::shared_ptr<Request>>& infer_reqs);
    void handleStopRequests(const std::vector<std::shared_ptr<Request>>& requests);

    void allocateBuffer(size_t batch_size, size_t session_len);
    void allocatePersistantBuffer(size_t max_batch_size);
    void freeBuffer();

    void initializeSampling(int32_t infer_request_count);

    void initialize(const std::vector<std::shared_ptr<Request>>& infer_requests);
    void contextDecode();

    void initializeGeneration();
    bool generate();

    void finish();
    void finishRequest(int32_t index, bool force_end);

    void synchronize();

    void setOutputTensors(int32_t max_gen_step);

    void
    outputContextLogits(T* context_decoder_output, const std::vector<int32_t>& indices, const std::vector<int32_t>& lengths);

    explicit LlamaBatch(int32_t max_batch_size, int32_t max_context_token_num, int32_t session_len, LlamaV2<T>* llama);

    ~LlamaBatch()
    {
        freeBuffer();
    }

private:
    const int32_t  max_batch_size_;
    const int32_t  max_context_token_num_;
    const int32_t  session_len_;
    const int32_t  rank_;
    const bool debug_;

    LlamaV2<T>* const llama_;

    // active requests
    std::vector<std::shared_ptr<Request>> requests_;

    T*   context_decoder_input_buf_{};   // CTXDEC
    T*   context_decoder_output_buf_{};  // CTXDEC
    int32_t* context_decoder_ids_buf_{};

    T* decoder_input_buf_{};   // CTXDEC, GENERATE
    T* decoder_output_buf_{};  // CTXDEC, GENERATE

    int32_t* input_ids_buf_{};       // input token ids + cache missed token ids, CTXDEC
    int32_t* input_length_buf_{};    // input + cache missed length, CTXDEC, GENERATE
    int32_t* history_length_buf_{};  // history length, CTXDEC
    int32_t* context_length_buf_{};  // history length + input_length, CTXDEC, GENERATE

    int32_t* total_padding_count_{};  // GENERATE
    int32_t* sequence_lengths_{};     // current sequence length

    uint64_t* k_cache_ptr_buf_{};
    uint64_t* v_cache_ptr_buf_{};

    float* logits_buf_{};        // combined logits
    float* local_logits_buf_{};  // tensor parallel local logits
    float* context_logits_buf_{};
    float* local_context_logits_buf_{};

    // used by dynamic decoder
    int32_t*      token_ids_buf_{};   // all token IDs in [S, B], indexed using `step`
    int32_t*      output_ids_buf_{};  // output ids in [B, S]
    int32_t*      end_ids_buf_{};
    bool*     finished_buf_{};
    uint32_t* seq_limit_len_{};

    // pinned buffers
    int32_t*       h_input_ids_buf_{};
    int32_t*       h_input_length_buf_{};
    int32_t*       h_history_length_buf_{};
    int32_t*       h_context_length_buf_{};
    int32_t*       h_sequence_lengths_{};
    bool*      h_finished_buf_{};
    uintptr_t* h_k_cache_ptr_buf_{};
    uintptr_t* h_v_cache_ptr_buf_{};
    uint32_t*  h_seq_limit_len_{};

    int32_t*      stop_words_buf_{};  // [batch_size, 2, kMaxStopWordsLen]
    int32_t*      bad_words_buf_{};
    int32_t*      h_runtime_top_k_{};
    float*    h_runtime_top_p_{};
    float*    h_temperature_{};
    float*    h_repetition_penalty_{};
    uint64_t* h_random_seed_{};

    IndexableList<dipu::DIPURawGeneratorImpl> topk_curandstate_buf_{};
    IndexableList<dipu::DIPURawGeneratorImpl> topp_curandstate_buf_{};

    // hard limits for persistent buffers
    static constexpr int32_t kMaxStopBadWordsLen = 32;

    using CachedSeq = LlamaCacheManager::Sequence;

    std::vector<CachedSeq> cached_seq_;
    std::vector<int32_t>       request_seq_len_limit_;

    const DataType data_type_{};

    int32_t batch_size_{};
    int32_t max_context_len_{};
    int32_t step_{};
    int32_t finished_count_{};

    bool is_allocate_persistant_buffer_ = false;
    bool is_allocate_buffer_            = false;

    TensorMap inputs_;
    TensorMap outputs_;

    std::unordered_map<std::string, void*> sampling_params_;

    dipu::deviceStream_t     stream_{};
    diopiContext ctx_;
    void* cublas_wrapper_ = nullptr;
    IAllocator*      allocator_{};
};

}  // namespace turbomind
