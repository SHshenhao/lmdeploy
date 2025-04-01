# Copyright (c) OpenMMLab. All rights reserved.
try:
    from deep_ep import Buffer

    use_deepep = True
except ImportError:
    use_deepep = False

from typing import Any, List, Tuple

import torch
import torch.distributed as dist

from ..default.token_dispatcher import AlltoAllTokenDispatcher
from ..token_dispatcher import TokenDispatcherImpl

_buffer_normal = None
_buffer_low_latency = None


def get_buffer_normal(group: dist.ProcessGroup, hidden_bytes: int):
    """Copy from DeepEP example usage in model inference prefilling.

    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-model-training-or-inference-prefilling
    """
    global _buffer_normal
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
    if (_buffer_normal is None or _buffer_normal.group != group or _buffer_normal.num_nvl_bytes < num_nvl_bytes
            or _buffer_normal.num_rdma_bytes < num_rdma_bytes):
        _buffer_normal = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer_normal


def get_buffer_low_latency(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
):
    """Copy from DeepEP example usage in model inference decoding.

    https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
    """

    global _buffer_low_latency
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(),
                                                           num_experts)

    if (_buffer_low_latency is None or _buffer_low_latency.group != group or not _buffer_low_latency.low_latency_mode
            or _buffer_low_latency.num_rdma_bytes < num_rdma_bytes):
        assert num_experts % group.size() == 0
        _buffer_low_latency = Buffer(
            group,
            0,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // group.size(),
        )
    return _buffer_low_latency


class DeepEPTokenDispatcher(TokenDispatcherImpl):
    """Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
    https://github.com/NVIDIA/Megatron-
    LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
    ):
        self.group = group
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.num_max_dispatch_tokens_per_rank = 256
        self.hidden_size = hidden_size
        self.hidden_shape = None
        self.params_bytes = params_dtype.itemsize
        self.tokens_per_expert = None
        # Handle used for combine operation
        self.handle = None
        # shared experts
        self.shared_experts = None
        if not use_deepep:
            raise ImportError('DeepEP is not installed. Please install DeepEP package from '
                              'https://github.com/deepseek-ai/deepep.')
        self.buffer_normal = get_buffer_normal(self.group, self.hidden_size * self.params_bytes)
        self.buffer_low_latency = get_buffer_low_latency(
            group=self.group,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            hidden=self.hidden_size,
            num_experts=self.num_experts)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_list: List[int] = None,
        previous_event=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        topk_weights = topk_weights.to(torch.float32)
        (
            hidden_states,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.dispatch_normal(hidden_states, topk_idx, topk_weights, self.num_experts, previous_event)
        self.tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            device=hidden_states.device,
            dtype=torch.int64,
        )
        tokens_per_expert = self.get_number_of_tokens_per_expert()
        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        if hidden_states.shape[0] > 0:
            hidden_states = self.get_permuted_hidden_states_by_experts(hidden_states)
        return hidden_states, topk_idx, topk_weights, tokens_per_expert

    def dispatch_yield(self,
                       hidden_states: torch.Tensor,
                       topk_idx: torch.Tensor,
                       topk_weights: torch.Tensor,
                       expert_list: List[int] = None,
                       previous_event=None,
                       is_prefill: bool = False,
                       is_decoding: bool = False):
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        topk_weights = topk_weights.to(torch.float32)
        # yield for attn1, dis (+share)
        yield
        previous_event = self.buffer_normal.capture()
        (
            recv_hidden_states,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.dispatch_normal_async(hidden_states, topk_idx, topk_weights, self.num_experts, previous_event, True)
        if is_decoding and self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
        else:
            shared_states = None
        # yield for dis (+share), dis_wait
        yield
        event.current_stream_wait()
        # yield for dis_wait, moe
        yield
        self.tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            device=hidden_states.device,
            dtype=torch.int64,
        )
        tokens_per_expert = self.get_number_of_tokens_per_expert()
        self.handle = handle
        self.topk_idx = recv_topk_idx
        self.topk_weights = recv_topk_weights
        if recv_hidden_states.shape[0] > 0:
            recv_hidden_states = self.get_permuted_hidden_states_by_experts(recv_hidden_states)
        return recv_hidden_states, recv_topk_idx, recv_topk_weights, tokens_per_expert, shared_states

    def dispatch_low_latency_yield(self,
                                   hidden_states: torch.Tensor,
                                   topk_idx: torch.Tensor,
                                   topk_weights: torch.Tensor,
                                   expert_list: List[int] = None,
                                   is_prefill: bool = False,
                                   is_decoding: bool = False,
                                   need_cast: bool = False,
                                   need_contiguous: bool = True):
        self.hidden_shape = hidden_states.shape
        topk_idx = topk_idx.to(torch.int64)
        topk_weights = topk_weights.to(torch.float32)
        # yield for attn1, dis (+share)
        yield
        (
            recv_hidden_states,
            recv_expert_count,
            handle,
            event,
            hook,
        ) = self.dispatch_low_latency_async(hidden_states, topk_idx, self.num_max_dispatch_tokens_per_rank,
                                            self.num_experts, True)
        if is_decoding and self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
        else:
            shared_states = None
        # yield for dis (+share), dis_wait
        yield
        event.current_stream_wait()
        # yield for dis_wait, moe
        yield
        if need_cast:
            recv_hidden_states = self.per_token_cast_back(recv_hidden_states[0], recv_hidden_states[1].contiguous())
        else:
            recv_hidden_states = recv_hidden_states[0]
        if need_contiguous:
            recv_hidden_states_list = []
            expert_count = recv_expert_count.tolist()
            for idx, count in enumerate(expert_count):
                if count != 0:
                    recv_hidden_states_list.append(recv_hidden_states[idx][:count])
            if len(recv_hidden_states_list) == 1:
                recv_hidden_states = recv_hidden_states_list[0]
            if len(recv_hidden_states_list) > 1:
                recv_hidden_states = torch.concat(recv_hidden_states_list, dim=0)

        self.tokens_per_expert = recv_expert_count
        self.handle = handle
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights

        return recv_hidden_states, recv_expert_count, shared_states

    def dispatch_normal(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        previous_event=None,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer_normal.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.buffer_normal.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    def dispatch_normal_async(self,
                              x: torch.Tensor,
                              topk_idx: torch.Tensor,
                              topk_weights: torch.Tensor,
                              num_experts: int,
                              previous_event=None,
                              async_finish=True):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer_normal.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.buffer_normal.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    def dispatch_low_latency_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        async_finish: bool,
    ):
        recv_hidden_states, recv_expert_count, handle, event, hook = (self.buffer_low_latency.low_latency_dispatch(
            hidden_states,
            topk_idx,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        ))
        return recv_hidden_states, recv_expert_count, handle, event, hook

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            hidden_states = self.get_restored_hidden_states_by_experts(hidden_states)
        hidden_states, event = self.combine_normal(hidden_states, self.handle)
        self.handle = None
        return hidden_states.view(self.hidden_shape)

    def combine_yield(self,
                      out_states: torch.Tensor,
                      hidden_states: torch.Tensor,
                      is_prefill: bool = False,
                      is_decoding: bool = False):
        if out_states.shape[0] > 0:
            out_states = self.get_restored_hidden_states_by_experts(out_states)
        # yield for moe, comb
        yield
        previous_event = self.buffer_normal.capture()
        out_states, event = self.combine_normal_async(out_states,
                                                      self.handle,
                                                      previous_event=previous_event,
                                                      async_finish=True)
        # yield for comb, (+share) comb_wait,
        yield
        if is_prefill and self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
        else:
            shared_states = None
        event.current_stream_wait()
        # yield for (+share) comb_wait, (+share) attn0
        yield
        self.handle = None
        return out_states.view(self.hidden_shape), shared_states

    def combine_low_latency_yield(self,
                                  out_states: torch.Tensor,
                                  hidden_states: torch.Tensor,
                                  is_prefill: bool = False,
                                  is_decoding: bool = False,
                                  need_contiguous: bool = True):
        if self.tokens_per_expert.sum().item() > 0 and need_contiguous:
            tokens_per_expert_list = self.tokens_per_expert.tolist()
            recv_out_states = torch.zeros(
                [self.num_local_experts, self.num_max_dispatch_tokens_per_rank * self.group.size(), self.hidden_size],
                dtype=hidden_states.dtype,
                device=torch.cuda.current_device())
            count_sum = 0
            for idx, count in enumerate(tokens_per_expert_list):
                if count > 0:
                    recv_out_states[idx, :count, ...] = out_states[count_sum:count_sum + count]
                    count_sum += count
        else:
            recv_out_states = out_states.reshape(self.num_local_experts,
                                                 self.num_max_dispatch_tokens_per_rank * self.group.size(),
                                                 self.hidden_size)
        # yield for moe, comb
        yield
        out_states, event, hook = self.combine_low_latency_async(recv_out_states,
                                                                 self.topk_idx,
                                                                 self.topk_weights,
                                                                 self.handle,
                                                                 async_finish=True)
        # yield for comb, (+share) comb_wait,
        yield
        if is_prefill and self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
        else:
            shared_states = None
        event.current_stream_wait()
        # yield for (+share) comb_wait, (+share) attn0
        yield
        self.handle = None
        return out_states.view(self.hidden_shape), shared_states

    def combine_normal(self, x: torch.Tensor, handle: Tuple, previous_event=None):
        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
        return combined_x, event

    def combine_normal_async(self, x: torch.Tensor, handle: Tuple, previous_event=None, async_finish=True):
        combined_x, _, event = self.buffer_normal.combine(
            x,
            handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )
        return combined_x, event

    def combine_low_latency_async(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Tuple,
        async_finish: bool,
    ):
        combined_hidden_states, event_overlap, hook = (self.buffer_low_latency.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=async_finish,
            return_recv_hook=not async_finish,
        ))
        return combined_hidden_states, event_overlap, hook

    def _indices_to_multihot(self, indices, probs):
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.long,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.topk_idx, self.topk_weights

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """Get the number of tokens per expert."""
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.dispatched_routing_map, self.topk_weights = super().indices_to_multihot(
            self.topk_idx, self.topk_weights, self.num_experts)
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = super().permute(
            hidden_states,
            self.dispatched_routing_map,
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        assert (self.topk_weights.dtype == torch.float32), 'DeepEP only supports float32 probs'
        hidden_states = super().unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.topk_weights,
        )
        return hidden_states.to(input_dtype)

    def set_shared_experts(self, shared_experts: Any = None):
        if self.shared_experts is None:
            self.shared_experts = shared_experts
        return self.shared_experts

    def get_shared_experts(self):
        return self.shared_experts

    def per_token_cast_back(self, x_fp8: torch.Tensor, x_scales: torch.Tensor):
        """Copy from DeepEP example usage in test low latency.

        https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py
        """
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
        x_scales = x_scales.view(x_fp8.size(0), -1, 1)
        return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


class TokenDispatcherBuilder:
    """token dispatcher builder."""

    @staticmethod
    def build(
        group,
        num_experts,
        num_local_experts,
        hidden_size,
        params_dtype,
    ) -> TokenDispatcherImpl:
        """build."""
        if use_deepep is True:
            return DeepEPTokenDispatcher(
                group,
                num_experts,
                num_local_experts,
                hidden_size,
                params_dtype,
            )
        else:
            return AlltoAllTokenDispatcher(
                group,
                num_experts,
                num_local_experts,
            )
