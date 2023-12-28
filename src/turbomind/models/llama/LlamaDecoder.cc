/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptDecoder.cc

#include "src/turbomind/models/llama/LlamaDecoder.h"
#include "src/turbomind/macro.h"
// #include "src/turbomind/models/llama/llama_decoder_kernels.h"
// #include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"

namespace turbomind {

template<typename T>
LlamaDecoder<T>::LlamaDecoder(size_t                      head_num,
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
                              int32_t                         quant_policy):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    hidden_units_(head_num * size_per_head),
    rmsnorm_eps_(rmsnorm_eps),
    tensor_para_(tensor_para),
    data_type_(getTensorType<T>())
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    initialize(attn_params, kv_head_num, quant_policy);
}

template<typename T>
LlamaDecoder<T>::~LlamaDecoder()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // delete self_attention_layer_;
    // delete silu_ffn_layer_;
}

template<typename T>
void LlamaDecoder<T>::initialize(const LlamaAttentionParams& attn_params, size_t kv_head_num, int quant_policy)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    attn_head_num_ = head_num_;
    attn_size_per_head_ = size_per_head_;
    attn_local_kv_head_num_ = kv_head_num / tensor_para_.world_size_;
    attn_local_head_num_ = head_num_ / tensor_para_.world_size_;
    attn_head_n_rep_ = attn_local_head_num_ / attn_local_kv_head_num_;
    attn_params_ = attn_params;

    // self_attention_layer_ = new LlamaDecoderSelfAttentionLayer<T>(head_num_,
    //                                                               kv_head_num,
    //                                                               size_per_head_,
    //                                                               attn_params,
    //                                                               tensor_para_,
    //                                                               stream_,
    //                                                               cublas_wrapper_,
    //                                                               allocator_,
    //                                                               is_free_buffer_after_forward_,
    //                                                               quant_policy);

    // silu_ffn_layer_ = new LlamaFfnLayer<T>(head_num_,
    //                                        size_per_head_,
    //                                        inter_size_,
    //                                        tensor_para_,
    //                                        stream_,
    //                                        cublas_wrapper_,
    //                                        allocator_,
    //                                        is_free_buffer_after_forward_);
}

template<typename T>
void LlamaDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void LlamaDecoder<T>::allocateBuffer(size_t batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaDecoder<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)&workspace_);
        is_allocate_buffer_ = false;
    }
}

// template<typename T>
// void LlamaDecoder<T>::forwardSelfAttn(const LlamaDecoder::Session&                   sess,
//                                       T*                                             attn_io,
//                                       const std::unordered_map<std::string, Tensor>* input_tensors,
//                                       size_t                                         layer)
// {
//     TM_LOG_DEBUG(__PRETTY_FUNCTION__);
//     TensorMap self_attention_input_tensors(*input_tensors);
//     self_attention_input_tensors.insert("input_query",
//                                         {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, attn_io});
//     const int layer_id = layer;
//     self_attention_input_tensors.insert("layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id});
//     auto& k_cache = *sess.k_cache;
//     auto& v_cache = *sess.v_cache;

//     TensorMap self_attention_output_tensors{
//         {"attention_output", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, attn_io}},
//         {"key_cache", k_cache},
//         {"value_cache", v_cache},
//     };

//     self_attention_layer_->forward(&self_attention_output_tensors,  //
//                                    &self_attention_input_tensors,
//                                    &sess.weights->at(layer)->self_attn_weights);
// }

// template<typename T>
// void LlamaDecoder<T>::forwardFfn(const LlamaDecoder::Session& sess, T* ffn_io, size_t layer)
// {
//     TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, ffn_io}}};
//     TensorMap ffn_outputs{{"ffn_output", {MEMORY_GPU, data_type_, {sess.batch_size, hidden_units_}, ffn_io}}};
//     silu_ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &sess.weights->at(layer)->ffn_weights);
// }

template<typename T>
void LlamaDecoder<T>::forward(std::vector<Tensor>*                            output_tensors,
                              const std::vector<Tensor>*                      input_tensors,
                              const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights)
{
    FT_CHECK(false);
}

template<typename T>
void LlamaDecoder<T>::forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                              const std::unordered_map<std::string, Tensor>*  input_tensors,
                              const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    /**
     * input_tensors:
     *   \param decoder_input [batch_size, hidden_dims]
     *   \param sequence_lengths [batch_size] int
     *   \param output_norm_weight [hidden_dims]
     *   \param step [1] on cpu
     *   \param ite [1] on cpu
     *   \param finished [batch_size] bool
     *   \param total_padding_tokens [batch_size], int
     *   \param max_seq_len [1] on cpu
     *   \param masked_tokens [batch_size, memory_len] bool (optional), NOT USED YET
     *
     * output_tensors:
     *   \param decoder_output [batch_size, hidden_dimension]
     *   \param key_cache [batch_size] uint64_t
     *   \param value_cache [batch_size] uint64_t
     */

    // for the shape of key cache, refer to decoder_masked_multihead_attention_template.hpp
    std::cout<<"LlamaDecoder<T>::forward start! :"<<ctx_.arrays.size()<<std::endl;
    Session sess{};
    sess.batch_size = input_tensors->at("decoder_input").shape[0];
    sess.weights    = decoder_layer_weights;

    allocateBuffer(sess.batch_size);

    sess.ite     = input_tensors->at("ite").getVal<const int>();
    sess.k_cache = &output_tensors->at("key_cache");
    sess.v_cache = &output_tensors->at("value_cache");

    sess.max_memory_len = input_tensors->at("max_seq_len").getVal<int>();

    T* decoder_input  = input_tensors->at("decoder_input").getPtr<T>();
    T* decoder_output = output_tensors->at("decoder_output").getPtr<T>();

    ////////////////////////////////////////////
    /// RMSNorm
    // invokeRootMeanSquareNorm(decoder_output,
    //                          decoder_input,
    //                          decoder_layer_weights->at(0)->self_attn_norm_weights,
    //                          rmsnorm_eps_,
    //                          sess.batch_size,
    //                          hidden_units_,
    //                          stream_);
    // sync_check_cuda_error();

    turbomind::Tensor decoder_output_tensor = output_tensors->at("decoder_output");
    diopiTensorHandle_t diopi_decoder_output_tensor = dipu::diopi_helper::toDiopiTensorHandle(decoder_output_tensor);
    diopiDevice_t device;
    diopiGetTensorDevice(diopi_decoder_output_tensor, &device);
    diopiDtype_t dtype;
    diopiGetTensorDtype(diopi_decoder_output_tensor, &dtype);
    int64_t itemsize;
    diopiGetTensorElemSize(diopi_decoder_output_tensor, &itemsize);
    diopiContext ctx(stream_);
    const int64_t step = input_tensors->at("step").getVal<int>();
    const int64_t max_seq_len = input_tensors->at("max_seq_len").getVal<int>();
    int64_t max_memory_len = sess.max_memory_len;
    int64_t ite = sess.ite;
    int64_t batch_size = sess.batch_size;
    int64_t local_head_num = attn_local_head_num_;
    int64_t local_kv_head_num = attn_local_kv_head_num_;
    int64_t size_per_head = size_per_head_;
    diopiScalar_t scarlar_done{diopiDtype_t::diopi_dtype_float64, double(1)};
    // std::cout<<"local_head_num:"<<local_head_num<<std::endl;
    // std::cout<<"local_kv_head_num:"<<local_kv_head_num<<std::endl;
    // std::cout<<"size_per_head:"<<size_per_head<<std::endl;
    // std::cout<<"dtype:"<<dtype<<" itemsize:"<<itemsize<<std::endl;
    // std::cout<<"max_memory_len:"<<max_memory_len<<std::endl;
    // std::cout<<"ite:"<<ite<<std::endl;
    // std::cout<<"batch_size:"<<batch_size<<std::endl;
    // std::cout<<"num_layer:"<<num_layer_<<std::endl;
    // std::cout<<"step:"<<step<<std::endl;
    // std::cout<<"max_seq_len:"<<max_seq_len<<std::endl;
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
    // std::cout<<"kvcache prepare"<<std::endl;

    diopiTensorHandle_t key_cache[batch_size];
    diopiTensorHandle_t value_cache[batch_size];
    std::vector<int64_t> shape{int64_t(num_layer_), local_kv_head_num, max_seq_len, size_per_head};
    diopiSize_t newshape{shape.data(), 4};
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
    std::cout<<"sess.ite:"<<sess.ite<<std::endl;
    float rotary_embedding_base{attn_params_.rotary_embedding_base};
    int64_t rotray_embedding_dim{attn_params_.rotray_embedding_dim};

    const turbomind::Tensor& sequence_lengths_tensor = input_tensors->at("sequence_lengths");
    diopiConstTensorHandle_t sequence_lengths = dipu::diopi_helper::toDiopiTensorHandle(sequence_lengths_tensor);
    const turbomind::Tensor& total_padding_tokens_tensor = input_tensors->at("total_padding_tokens");
    diopiConstTensorHandle_t total_padding_tokens = dipu::diopi_helper::toDiopiTensorHandle(total_padding_tokens_tensor);
    diopiConstTensorHandle_t finished_data = nullptr;
    if (input_tensors->find("finished") != input_tensors->end()) {
        const turbomind::Tensor& finished_data_tensor = input_tensors->at("finished");
        finished_data = dipu::diopi_helper::toDiopiTensorHandle(finished_data_tensor);
    }

    turbomind::Tensor decoder_input_tensor = input_tensors->at("decoder_input");
    diopiTensorHandle_t diopi_decoder_input_tensor = dipu::diopi_helper::toDiopiTensorHandle(decoder_input_tensor);
    // diopiLmdeployCopyD2D(&ctx_, diopi_decoder_output_tensor, diopi_decoder_input_tensor, false); // SH RMSNorm

    int64_t workspace_size = -1;
    diopiFusedDecoderAttentionInp(&ctx_, diopi_decoder_output_tensor, nullptr, nullptr, nullptr, &workspace_size, 0, key_cache, value_cache,
                                    nullptr, nullptr, sequence_lengths, step, 0, local_head_num,
                                    local_kv_head_num, size_per_head, max_seq_len, rotray_embedding_dim, rotary_embedding_base);
    std::cout<<"workspace_size:"<<workspace_size<<std::endl;
    workspace_ = allocator_->reMalloc(workspace_, workspace_size, false);
    diopiTensorHandle_t workspace;
    shape[0] = workspace_size;
    newshape.len = 1;
    diopiSize_t workspace_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(workspace_)), -1};
    diopiRequireTensor(&ctx_, &workspace, &newshape, &workspace_stride, dtype, device);

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

    for (size_t layer = 0; layer < num_layer_; ++layer) {
        // output: self_attn_output_, k_cache, v_cache = self_attn(decoder_normed_input_)
        // forwardSelfAttn(sess, decoder_output, input_tensors, layer);
        turbomind::Tensor weightqkv_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t((local_head_num+local_kv_head_num*2)*size_per_head)}, sess.weights->at(layer)->self_attn_weights.qkv.kernel};
        turbomind::Tensor weightbias_tensor{MEMORY_GPU, data_type_, {int64_t(1), int64_t((local_head_num+local_kv_head_num*2)*size_per_head)}, sess.weights->at(layer)->self_attn_weights.qkv.bias};
        diopiTensorHandle_t weightqkv = dipu::diopi_helper::toDiopiTensorHandle(weightqkv_tensor);
        diopiTensorHandle_t weightbias = dipu::diopi_helper::toDiopiTensorHandle(weightbias_tensor);
        // void* temp_ptr;
        // diopiGetTensorData(workspace, &temp_ptr);
        // std::cout<<"++diopiFusedDecoderAttentionInp++"<<std::endl;
        diopiFusedDecoderAttentionInp(&ctx_, diopi_decoder_output_tensor, weightqkv, weightbias, workspace, &workspace_size, 0, key_cache, value_cache,
                                        finished_data, total_padding_tokens, sequence_lengths, step, layer, local_head_num,
                                        local_kv_head_num, size_per_head, max_seq_len, rotray_embedding_dim, rotary_embedding_base);

        turbomind::Tensor weightoutput_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t(hidden_units_)}, sess.weights->at(layer)->self_attn_weights.output.kernel};
        diopiTensorHandle_t weightoutput = dipu::diopi_helper::toDiopiTensorHandle(weightoutput_tensor);
        diopiMm(&ctx_, diopi_decoder_tmp_tensor, diopi_decoder_output_tensor, weightoutput);

        // invokeFusedAddBiasResidualRMSNorm(decoder_input,
        //                                   decoder_output,
        //                                   decoder_layer_weights->at(layer)->self_attn_weights.output.bias,
        //                                   decoder_layer_weights->at(layer)->ffn_norm_weights,
        //                                   rmsnorm_eps_,
        //                                   sess.batch_size,
        //                                   hidden_units_,
        //                                   stream_);
        // diopiLmdeployCopyD2D(&ctx_, diopi_decoder_output_tensor, diopi_decoder_tmp_tensor, false); // SH RMSNorm
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

        // decoder_layer_output_ = ffn(decoder_normed_input_)
        // forwardFfn(sess, decoder_output, layer);
        turbomind::Tensor weight1_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t(inter_size_)}, sess.weights->at(layer)->ffn_weights.gating.kernel};
        diopiTensorHandle_t weight1 = dipu::diopi_helper::toDiopiTensorHandle(weight1_tensor);
        turbomind::Tensor weight3_tensor{MEMORY_GPU, data_type_, {int64_t(hidden_units_), int64_t(inter_size_)}, sess.weights->at(layer)->ffn_weights.intermediate.kernel};
        diopiTensorHandle_t weight3 = dipu::diopi_helper::toDiopiTensorHandle(weight3_tensor);
        turbomind::Tensor weight2_tensor{MEMORY_GPU, data_type_, {int64_t(inter_size_), int64_t(hidden_units_)}, sess.weights->at(layer)->ffn_weights.output.kernel};
        diopiTensorHandle_t weight2 = dipu::diopi_helper::toDiopiTensorHandle(weight2_tensor);
        diopiFusedSiluFfnInp(&ctx_, diopi_decoder_output_tensor, weight1, weight2, weight3, workspace, &workspace_size, 0);

        auto scale_weight = layer < num_layer_ - 1 ? decoder_layer_weights->at(layer + 1)->self_attn_norm_weights :
                                                     input_tensors->at("output_norm_weight").getPtr<T>();
        // invokeFusedAddBiasResidualRMSNorm(decoder_input,  //
        //                                   decoder_output,
        //                                   decoder_layer_weights->at(layer)->ffn_weights.output.bias,
        //                                   scale_weight,
        //                                   rmsnorm_eps_,
        //                                   sess.batch_size,
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
    }

    dipu::diopi_helper::clearDiopiContextAll(ctx_);
    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    std::cout<<"LlamaDecoder<T>::forward end! :"<<ctx_.arrays.size()<<std::endl;
}

template class LlamaDecoder<half>;
template class LlamaDecoder<float>;

}  // namespace turbomind
