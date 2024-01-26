/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptContextDecoder.cc

#include "src/turbomind/models/llama/LlamaContextDecoder.h"
// #include "src/turbomind/kernels/bert_preprocess_kernels.h"
// #include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaContextDecoder.h"
// #include "src/turbomind/models/llama/llama_decoder_kernels.h"
// #include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

template<typename T>
void LlamaContextDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void LlamaContextDecoder<T>::allocateBuffer(size_t batch_size, size_t num_token, size_t max_q_len, size_t max_kv_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * max_q_len * max_kv_len, false);
    // padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int32_t) * batch_size * max_q_len, false);
    // cu_seqlens_     = (int*)allocator_->reMalloc(cu_seqlens_, sizeof(int32_t) * (batch_size + 1), false);

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaContextDecoder<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        // allocator_->free((void**)&padding_offset_);
        // allocator_->free((void**)&cu_seqlens_);
        allocator_->free((void**)&attention_mask_);
        allocator_->free((void**)&workspace_);
        // allocator_->free((void**)&h_pinned_token_num_ptr_, true);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void LlamaContextDecoder<T>::initialize(const LlamaAttentionParams& attn_params,
                                        size_t                      kv_head_num,
                                        bool                        use_fmha,
                                        int                         quant_policy)
{
    // h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);

    attn_head_num_ = head_num_;
    attn_size_per_head_ = size_per_head_;
    attn_local_kv_head_num_ = kv_head_num / tensor_para_.world_size_;
    attn_local_head_num_ = head_num_ / tensor_para_.world_size_;
    attn_head_n_rep_ = attn_local_head_num_ / attn_local_kv_head_num_;
    attn_params_ = attn_params;

    // context_attention_layer_ = new LlamaContextAttentionLayer<T>(head_num_,
    //                                                              kv_head_num,
    //                                                              size_per_head_,
    //                                                              attn_params,
    //                                                              tensor_para_,
    //                                                              stream_,
    //                                                              cublas_wrapper_,
    //                                                              allocator_,
    //                                                              is_free_buffer_after_forward_,
    //                                                              use_fmha,
    //                                                              quant_policy);

    // silu_ffn_layer_ = new LlamaFfnLayer<T>(head_num_,
    //                                        size_per_head_,
    //                                        inter_size_,
    //                                        tensor_para_,
    //                                        stream_,
    //                                        cublas_wrapper_,
    //                                        allocator_,
    //                                        is_free_buffer_after_forward_);
}

// template<typename T>
// void LlamaContextDecoder<T>::forwardSelfAttn(const Session&                                 sess,
//                                              T*                                             attn_io,
//                                              const std::unordered_map<std::string, Tensor>* input_tensors,
//                                              int                                            layer,
//                                              bool                                           is_final)
// {
//     // TM_LOG_ERROR(__PRETTY_FUNCTION__);
//     TensorMap self_attention_input_tensors{
//         {"input_query", Tensor{MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_io}},
//         {"attention_mask",
//          {MEMORY_GPU, data_type_, {sess.batch_size, 1, sess.max_query_len, sess.max_key_len}, attention_mask_}},
//         {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &layer}},
//         {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}},
//         {"padding_offset", {MEMORY_GPU, TYPE_INT32, {sess.token_num}, padding_offset_}},
//         {"cu_seqlens", {MEMORY_GPU, TYPE_INT32, {sess.batch_size + 1}, cu_seqlens_}},
//         {"input_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.input_length}},
//         {"history_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.history_length}},
//         {"context_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.context_length}},
//         {"max_seq_len", input_tensors->at("max_seq_len")}};

//     auto& k_cache = *sess.k_cache;
//     auto& v_cache = *sess.v_cache;

//     TensorMap self_attention_output_tensors{
//         {"hidden_features", {MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_io}},
//         {"key_cache", k_cache},
//         {"value_cache", v_cache},
//     };

//     context_attention_layer_->forward(&self_attention_output_tensors,  //
//                                       &self_attention_input_tensors,
//                                       &sess.weights->at(layer)->self_attn_weights);
// }

template<typename T>
LlamaContextDecoder<T>::LlamaContextDecoder(size_t                      head_num,
                                            size_t                      kv_head_num,
                                            size_t                      size_per_head,
                                            size_t                      inter_size,
                                            size_t                      num_layer,
                                            const LlamaAttentionParams& attn_params,
                                            float                       rmsnorm_eps,
                                            NcclParam                   tensor_para,
                                            dipu::deviceStream_t                stream,
                                            void*            cublas_wrapper,
                                            IAllocator*                 allocator,
                                            bool                        is_free_buffer_after_forward,
                                            bool                        use_fmha,
                                            int32_t                         quant_policy):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num * size_per_head),
    num_layer_(num_layer),
    rmsnorm_eps_(rmsnorm_eps),
    tensor_para_(tensor_para),
    data_type_(getTensorType<T>())
{
    initialize(attn_params, kv_head_num, use_fmha, quant_policy);
}

template<typename T>
LlamaContextDecoder<T>::~LlamaContextDecoder()
{
    // delete context_attention_layer_;
    // delete silu_ffn_layer_;
    freeBuffer();
}

template<typename T>
void LlamaContextDecoder<T>::forward(std::vector<Tensor>*                            output_tensors,
                                     const std::vector<Tensor>*                      input_tensors,
                                     const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights)
{
    FT_CHECK(false);
}

template<typename T>
void LlamaContextDecoder<T>::forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                                     const std::unordered_map<std::string, Tensor>*  input_tensors,
                                     const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights)
{
    /**
     * input tensors:
     *   \param decoder_input [num_token, hidden_units], float
     *   \param input_lengths [batch_size], int
     *   \param history_lengths [batch_size], int
     *   \param context_legnths [batch_size], int
     *   \param output_norm_weight [hidden_dims], float
     *   \param max_q_len [1], int on cpu
     *   \param max_kv_len [1], int on cpu
     *   \param max_seq_len [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
     *   \param value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
     *   \param last_token_hidden_units [batch_size, hidden_units]
     */

    Session sess{};
    // std::cout<<"LlamaContextDecoder<T>::forward start! :"<<ctx_.arrays.size()<<std::endl;

    sess.token_num     = input_tensors->at("decoder_input").shape[0];
    sess.batch_size    = input_tensors->at("input_lengths").shape[0];
    sess.max_query_len = input_tensors->at("max_q_len").getVal<int32_t>();
    sess.max_key_len   = input_tensors->at("max_kv_len").getVal<int32_t>();
    sess.weights       = decoder_layer_weights;

    sess.input_length   = input_tensors->at("input_lengths").getPtr<int32_t>();
    sess.history_length = input_tensors->at("history_lengths").getPtr<int32_t>();
    sess.context_length = input_tensors->at("context_lengths").getPtr<int32_t>();

    T* decoder_input_output = input_tensors->at("decoder_input").getPtr<T>();
    T* decoder_output       = output_tensors->at("decoder_output").getPtr<T>();

    sess.k_cache = &output_tensors->at("key_cache");
    sess.v_cache = &output_tensors->at("value_cache");

    allocateBuffer(sess.batch_size, sess.token_num, sess.max_query_len, sess.max_key_len);

    // size_t tmp_token_num{};
    // invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
    //                                    &tmp_token_num,  // updated token num
    //                                    padding_offset_,
    //                                    cu_seqlens_,
    //                                    input_tensors->at("input_lengths").getPtr<int>(),
    //                                    sess.batch_size,
    //                                    sess.max_query_len,
    //                                    stream_);
    // sync_check_cuda_error();
    // FT_CHECK(tmp_token_num == sess.token_num);

    // invokeCreateCausalMasks(attention_mask_,
    //                         sess.input_length,
    //                         sess.context_length,
    //                         sess.max_query_len,
    //                         sess.max_key_len,
    //                         sess.batch_size,
    //                         stream_);
    // sync_check_cuda_error();

    int64_t batch_size{sess.batch_size};
    int64_t local_head_num{attn_local_head_num_};
    int64_t local_kv_head_num{attn_local_kv_head_num_};
    int64_t size_per_head{attn_size_per_head_};
    int64_t max_seq_len{input_tensors->at("max_seq_len").getVal<int32_t>()};
    int64_t max_q_len{input_tensors->at("max_q_len").getVal<int32_t>()};
    int64_t max_kv_len{input_tensors->at("max_kv_len").getVal<int32_t>()};
    float rotary_embedding_base{attn_params_.rotary_embedding_base};
    int64_t rotray_embedding_dim{attn_params_.rotray_embedding_dim};
    diopiScalar_t scarlar_done{diopiDtype_t::diopi_dtype_float64, double(1)};

    turbomind::Tensor decoder_output_tensor = output_tensors->at("decoder_output");
    diopiTensorHandle_t diopi_decoder_output_tensor = dipu::diopi_helper::toDiopiTensorHandle(decoder_output_tensor);
    diopiDevice_t device;
    diopiGetTensorDevice(diopi_decoder_output_tensor, &device);
    diopiDtype_t dtype;
    diopiGetTensorDtype(diopi_decoder_output_tensor, &dtype);
    int64_t itemsize;
    diopiGetTensorElemSize(diopi_decoder_output_tensor, &itemsize);
    int64_t h_kcache_data[batch_size];
    int64_t h_vcache_data[batch_size];
    turbomind::Tensor h_kcache_tensor{MEMORY_CPU, TYPE_INT64, {batch_size}, reinterpret_cast<void*>(h_kcache_data)};
    turbomind::Tensor h_vcache_tensor{MEMORY_CPU, TYPE_INT64, {batch_size}, reinterpret_cast<void*>(h_vcache_data)};
    turbomind::Tensor d_kcache_tensor{MEMORY_GPU, TYPE_INT64, {batch_size}, reinterpret_cast<void*>(sess.k_cache->data)};
    turbomind::Tensor d_vcache_tensor{MEMORY_GPU, TYPE_INT64, {batch_size}, reinterpret_cast<void*>(sess.v_cache->data)};
    diopiTensorHandle_t h_kcache = dipu::diopi_helper::toDiopiTensorHandle(h_kcache_tensor);
    diopiTensorHandle_t h_vcache = dipu::diopi_helper::toDiopiTensorHandle(h_vcache_tensor);
    diopiConstTensorHandle_t kcache = dipu::diopi_helper::toDiopiTensorHandle(d_kcache_tensor);
    diopiConstTensorHandle_t vcache = dipu::diopi_helper::toDiopiTensorHandle(d_vcache_tensor);
    diopiLmdeployCopyD2H(&ctx_, h_kcache, kcache, false);
    diopiLmdeployCopyD2H(&ctx_, h_vcache, vcache, false);

    const turbomind::Tensor& input_lengths_tensor = input_tensors->at("input_lengths");
    diopiConstTensorHandle_t input_lengths = dipu::diopi_helper::toDiopiTensorHandle(input_lengths_tensor);
    // input_lengths_tensor.saveNpy(inputlengths0_path);
    const turbomind::Tensor& history_lengths_tensor = input_tensors->at("history_lengths");
    diopiConstTensorHandle_t history_lengths = dipu::diopi_helper::toDiopiTensorHandle(history_lengths_tensor);
    // history_lengths_tensor.saveNpy(historylengths0_path);
    const turbomind::Tensor& context_lengths_tensor = input_tensors->at("context_lengths");
    diopiConstTensorHandle_t context_lengths = dipu::diopi_helper::toDiopiTensorHandle(context_lengths_tensor);
    // context_lengths_tensor.saveNpy(contextlengths0_path);

    diopiTensorHandle_t key_cache[batch_size];
    diopiTensorHandle_t value_cache[batch_size];
    std::vector<int64_t> shape{int64_t(num_layer_), local_kv_head_num, max_seq_len, size_per_head};
    diopiSize_t newshape{shape.data(), 4};
    // std::cout<<local_head_num<<" "<<local_kv_head_num<<" "<<size_per_head<<" "<<dtype<<" "<<itemsize<<std::endl;
    for (int64_t i = 0; i < batch_size; i++) {
        diopiTensorHandle_t temp_kcache;
        diopiSize_t temp_kcache_ptr_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_kcache_data[i])), -1};
        diopiRequireTensor(&ctx_, &temp_kcache, &newshape, &temp_kcache_ptr_stride, dtype, device);
        diopiTensorHandle_t temp_vcache;
        diopiSize_t temp_vcache_ptr_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_vcache_data[i])), -1};
        diopiRequireTensor(&ctx_, &temp_vcache, &newshape, &temp_vcache_ptr_stride, dtype, device);
        key_cache[i] = temp_kcache;
        value_cache[i] = temp_vcache;
    }

    /////////////////////////////////////////////
    /// RMSNorm
    // invokeRootMeanSquareNorm(decoder_output,
    //                          decoder_input_output,
    //                          decoder_layer_weights->at(0)->self_attn_norm_weights,
    //                          rmsnorm_eps_,
    //                          sess.token_num,
    //                          hidden_units_,
    //                          stream_);
    turbomind::Tensor decoder_input_tensor = input_tensors->at("decoder_input");
    diopiTensorHandle_t diopi_decoder_input_tensor = dipu::diopi_helper::toDiopiTensorHandle(decoder_input_tensor);

    // input_tensors->at("decoder_input").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_input0.npy");
    // output_tensors->at("decoder_output").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_output0.npy");

    int64_t workspace_size = -1;
    int64_t pre_work_size = -1;
    diopiFusedContextAttentionInp(&ctx_, diopi_decoder_output_tensor, nullptr, nullptr, nullptr, &pre_work_size, true, nullptr, &workspace_size, 0,
                    key_cache, value_cache, batch_size, input_lengths, history_lengths, context_lengths,
                    0, int64_t(local_head_num), int64_t(local_kv_head_num), int64_t(size_per_head), int64_t(max_seq_len),
                    int64_t(max_q_len), int64_t(max_kv_len), rotray_embedding_dim, rotary_embedding_base);
    workspace_ = allocator_->reMalloc(workspace_, workspace_size, false);
    pre_work_size = std::max(pre_work_size, int64_t(sizeof(T) * batch_size * max_q_len * max_kv_len));
    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, pre_work_size, false);
    turbomind::Tensor attention_mask_tensor{MEMORY_GPU, data_type_, {batch_size, max_q_len, max_kv_len}, attention_mask_};
    diopiTensorHandle_t diopi_attention_mask = dipu::diopi_helper::toDiopiTensorHandle(attention_mask_tensor);
    diopiTensorHandle_t workspace;
    shape[0] = workspace_size;
    newshape.len = 1;
    diopiSize_t workspace_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(workspace_)), -1};
    diopiRequireTensor(&ctx_, &workspace, &newshape, &workspace_stride, dtype, device);
    diopiFusedContextAttentionInp(&ctx_, diopi_decoder_output_tensor, nullptr, nullptr, diopi_attention_mask, &pre_work_size, false, workspace, &workspace_size, 0,
                    key_cache, value_cache, batch_size, input_lengths, history_lengths, context_lengths,
                    0, int64_t(local_head_num), int64_t(local_kv_head_num), int64_t(size_per_head), int64_t(max_seq_len),
                    int64_t(max_q_len), int64_t(max_kv_len), rotray_embedding_dim, rotary_embedding_base);

    diopiSize_t rms_input_shape;
    diopiGetTensorShape(diopi_decoder_input_tensor, &rms_input_shape);
    diopiTensorHandle_t invRMS;
    diopiSize_t invRMS_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(workspace_)), -1};
    diopiRequireTensor(&ctx_, &invRMS, &rms_input_shape, &invRMS_stride, dtype, device);
    turbomind::Tensor rms_weights{MEMORY_GPU, data_type_, {hidden_units_}, decoder_layer_weights->at(0)->self_attn_norm_weights};
    diopiTensorHandle_t diopi_rms_weights = dipu::diopi_helper::toDiopiTensorHandle(rms_weights);
    int64_t hidden_units_iny64_t = static_cast<int64_t>(hidden_units_);
    diopiSize_t normalized_shape{&hidden_units_iny64_t, 1};
    diopiRMSNorm(&ctx_, diopi_decoder_output_tensor, invRMS, diopi_decoder_input_tensor, normalized_shape, diopi_rms_weights, nullptr, rmsnorm_eps_);
    sync_check_cuda_error();

    diopiSize_t decoder_input_tensor_shape;
    diopiGetTensorShape(diopi_decoder_input_tensor, &decoder_input_tensor_shape);
    diopiTensorHandle_t diopi_decoder_tmp_tensor;
    diopiSize_t diopi_decoder_tmp_tensor_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(workspace_)), -1};
    diopiRequireTensor(&ctx_, &diopi_decoder_tmp_tensor, &decoder_input_tensor_shape, &diopi_decoder_tmp_tensor_stride, dtype, device);

    // input_tensors->at("decoder_input").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_input1.npy");
    // output_tensors->at("decoder_output").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_output1.npy");
    int64_t N = ctx_.arrays.size();
    for (size_t layer = 0; layer < num_layer_; ++layer) {
        /////////////////////////////////////////////
        /// self-attention
        // forwardSelfAttn(sess, decoder_output, input_tensors, layer, false);

        turbomind::Tensor weightqkv_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t((local_head_num+local_kv_head_num*2)*size_per_head)}, sess.weights->at(layer)->self_attn_weights.qkv.kernel};
        turbomind::Tensor weightbias_tensor{MEMORY_GPU, data_type_, {int64_t(1), int64_t((local_head_num+local_kv_head_num*2)*size_per_head)}, sess.weights->at(layer)->self_attn_weights.qkv.bias};
        diopiTensorHandle_t weightqkv = dipu::diopi_helper::toDiopiTensorHandle(weightqkv_tensor);
        diopiTensorHandle_t weightbias = dipu::diopi_helper::toDiopiTensorHandle(weightbias_tensor);
        diopiFusedContextAttentionInp(&ctx_, diopi_decoder_output_tensor, weightqkv, weightbias, diopi_attention_mask, &pre_work_size, true, workspace, &workspace_size, 0,
                        key_cache, value_cache, batch_size, input_lengths, history_lengths, context_lengths,
                        int64_t(layer), int64_t(local_head_num), int64_t(local_kv_head_num), int64_t(size_per_head), int64_t(max_seq_len),
                        int64_t(max_q_len), int64_t(max_kv_len), rotray_embedding_dim, rotary_embedding_base);
        turbomind::Tensor weightoutput_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t(hidden_units_)}, sess.weights->at(layer)->self_attn_weights.output.kernel};
        diopiTensorHandle_t weightoutput = dipu::diopi_helper::toDiopiTensorHandle(weightoutput_tensor);
        diopiMm(&ctx_, diopi_decoder_tmp_tensor, diopi_decoder_output_tensor, weightoutput);

        // invokeFusedAddBiasResidualRMSNorm(decoder_input_output,
        //                                   decoder_output,
        //                                   decoder_layer_weights->at(layer)->self_attn_weights.output.bias,
        //                                   decoder_layer_weights->at(layer)->ffn_norm_weights,
        //                                   rmsnorm_eps_,
        //                                   sess.token_num,
        //                                   hidden_units_,
        //                                   stream_);
        // diopiLmdeployCopyD2D(&ctx_, diopi_decoder_output_tensor, diopi_decoder_tmp_tensor, false); // SH RMSNorm
        // reinterpret_cast<turbomind::Tensor*>(diopi_attention_mask)->saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/diopi_attention_mask.npy");
        // input_tensors->at("decoder_input").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_input2.npy");
        // reinterpret_cast<turbomind::Tensor*>(diopi_decoder_tmp_tensor)->saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_output2.npy");

        turbomind::Tensor rms_attn_bias{MEMORY_GPU, data_type_, {1, hidden_units_}, decoder_layer_weights->at(layer)->self_attn_weights.output.bias};
        diopiTensorHandle_t diopi_rms_attn_bias = dipu::diopi_helper::toDiopiTensorHandle(rms_attn_bias);
        diopiAddInp(&ctx_, diopi_decoder_input_tensor, diopi_decoder_tmp_tensor, &scarlar_done);
        if (rms_attn_bias.data != nullptr) {
            diopiAddInp(&ctx_, diopi_decoder_input_tensor, diopi_rms_attn_bias, &scarlar_done);
        }
        turbomind::Tensor rms_attn_weights{MEMORY_GPU, data_type_, {hidden_units_}, decoder_layer_weights->at(layer)->ffn_norm_weights};
        diopiTensorHandle_t diopi_rms_attn_weights = dipu::diopi_helper::toDiopiTensorHandle(rms_attn_weights);
        diopiRMSNorm(&ctx_, diopi_decoder_output_tensor, invRMS, diopi_decoder_input_tensor, normalized_shape, diopi_rms_attn_weights, nullptr, rmsnorm_eps_);
        sync_check_cuda_error();

        input_tensors->at("decoder_input").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_input3.npy");
        output_tensors->at("decoder_output").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_output3.npy");

        ////////////////////////////////////////////
        /// feed-forward network
        TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, data_type_, {int64_t(sess.token_num), int64_t(hidden_units_)}, decoder_output}}};
        TensorMap ffn_outputs{
            {"ffn_output", {MEMORY_GPU, data_type_, {int64_t(sess.token_num), int64_t(hidden_units_)}, decoder_output}}};
        // silu_ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &decoder_layer_weights->at(layer)->ffn_weights);
        turbomind::Tensor weight1_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t(inter_size_)}, decoder_layer_weights->at(layer)->ffn_weights.gating.kernel};
        diopiTensorHandle_t weight1 = dipu::diopi_helper::toDiopiTensorHandle(weight1_tensor);
        turbomind::Tensor weight3_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t(inter_size_)}, decoder_layer_weights->at(layer)->ffn_weights.intermediate.kernel};
        diopiTensorHandle_t weight3 = dipu::diopi_helper::toDiopiTensorHandle(weight3_tensor);
        turbomind::Tensor weight2_tensor{MEMORY_GPU, data_type_, {int64_t(inter_size_), int64_t(hidden_units_)}, decoder_layer_weights->at(layer)->ffn_weights.output.kernel};
        diopiTensorHandle_t weight2 = dipu::diopi_helper::toDiopiTensorHandle(weight2_tensor);
        diopiFusedSiluFfnInp(&ctx_, diopi_decoder_output_tensor, weight1, weight2, weight3, workspace, &workspace_size, 0);
        sync_check_cuda_error();
        // input_tensors->at("decoder_input").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_input4.npy");
        // output_tensors->at("decoder_output").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_output4.npy");

        auto scale_weight = layer < num_layer_ - 1 ? decoder_layer_weights->at(layer + 1)->self_attn_norm_weights :
                                                     input_tensors->at("output_norm_weight").getPtr<T>();
        // invokeFusedAddBiasResidualRMSNorm(decoder_input_output,  //
        //                                   decoder_output,
        //                                   decoder_layer_weights->at(layer)->ffn_weights.output.bias,
        //                                   scale_weight,
        //                                   rmsnorm_eps_,
        //                                   sess.token_num,
        //                                   hidden_units_,
        //                                   stream_);
        // diopiLmdeployCopyD2D(&ctx_, diopi_decoder_input_tensor, diopi_decoder_output_tensor, false); // SH RMSNorm
        turbomind::Tensor rms_ffn_bias{MEMORY_GPU, data_type_, {1, hidden_units_}, decoder_layer_weights->at(layer)->ffn_weights.output.bias};
        diopiTensorHandle_t diopi_rms_ffn_bias = dipu::diopi_helper::toDiopiTensorHandle(rms_ffn_bias);
        diopiAddInp(&ctx_, diopi_decoder_input_tensor, diopi_decoder_output_tensor, &scarlar_done);
        if (rms_ffn_bias.data != nullptr) {
            diopiAddInp(&ctx_, diopi_decoder_input_tensor, diopi_rms_ffn_bias, &scarlar_done);
        }
        turbomind::Tensor rms_ffn_weights{MEMORY_GPU, data_type_, {hidden_units_}, scale_weight};
        diopiTensorHandle_t diopi_rms_ffn_weights = dipu::diopi_helper::toDiopiTensorHandle(rms_ffn_weights);
        diopiRMSNorm(&ctx_, diopi_decoder_output_tensor, invRMS, diopi_decoder_input_tensor, normalized_shape, diopi_rms_ffn_weights, nullptr, rmsnorm_eps_);
        sync_check_cuda_error();
        dipu::diopi_helper::clearDiopiContextAfterN(ctx_, N);
    }

    dipu::diopi_helper::clearDiopiContextAll(ctx_);
    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    // input_tensors->at("decoder_input").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_inputz.npy");
    // output_tensors->at("decoder_output").saveNpy("/nvme/share/share/shenhao/tis/lmdeploy/data/decoder_outputz.npy");
    // exit(0);
}

template class LlamaContextDecoder<float>;
template class LlamaContextDecoder<half>;

}  // namespace turbomind
