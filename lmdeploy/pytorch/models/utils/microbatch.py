# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from enum import Enum, auto
from typing import Any, List, Optional, Tuple

import torch

import lmdeploy.pytorch.distributed as dist

ENABLE_TWO = True


class ExecType(Enum):
    """batch ecex type."""
    One = auto()
    Two0101 = auto()
    Two0110 = auto()
    TwoLikeOne = auto()
    TwoPrefill = auto()
    TwoDecode = auto()


class BatchWorker:

    def __init__(self, tag: str, generator):
        self._tag = tag
        self._generator = generator
        self._count = 0
        self.output = None

    def next(self):
        assert not self.done

        try:
            next(self._generator)
        except StopIteration as e:
            assert e.value is not None
            self.output = e.value

        self._count += 1

    @property
    def done(self):
        return self.output is not None


def execute_batch(inputs: list, fn, delta_stages: int = 0, exec_type: ExecType = ExecType.One, extern_tag: str = ''):
    worker_list = [BatchWorker(str(idx), fn(**input, tag=str(idx) + extern_tag)) for idx, input in enumerate(inputs)]

    if exec_type == ExecType.One:
        assert len(inputs) == 1
        i = 0
        while not worker_list[0].done:
            worker_list[0].next()
            i += 1

    if exec_type == ExecType.TwoLikeOne:
        assert len(inputs) == 2
        i = 0
        while not worker_list[0].done:
            worker_list[0].next()
            i += 1
        i = 0
        while not worker_list[1].done:
            worker_list[1].next()
            i += 1

    if exec_type == ExecType.Two0101:
        assert len(inputs) == 2

        for _ in range(delta_stages):
            worker_list[0].next()
        i = 0
        while not worker_list[0].done:
            worker_list[0].next()
            worker_list[1].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    if exec_type == ExecType.Two0110:
        assert len(inputs) == 2

        for _ in range(delta_stages):
            worker_list[0].next()
        i = 0
        while not worker_list[0].done:
            if i % 2 == 0:
                worker_list[0].next()
                worker_list[1].next()
            else:
                worker_list[1].next()
                worker_list[0].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    if exec_type == ExecType.TwoPrefill:
        """
        before:
        A-attn0->A-attn1
        roll:
        A-dis->B-attn0->B-attn1->A-dis_wait->B-dis->A-moe->B-dis_wait->A-comb->
        B-moe->(A-share->A-comb_wait)->B-comb->A-attn0->A-attn1->(B-share->B-comb_wait)
        after:
        B-dis_wait->B-moe->B-comb->B-comb_wait and end
        """
        assert len(inputs) == 2 and delta_stages in [0, 2]

        for _ in range(2):
            worker_list[0].next()

        pipeline = [
            '0-dis', '1-attn0', '1-attn1', '0-dis_wait', '1-dis', '0-moe', '1-dis_wait', '0-comb', '1-moe',
            '0-share+0-comb_wait', '1-comb', '0-attn0', '0-attn1', '1-share+1-comb_wait'
        ]
        pipline_length = len(pipeline)
        i = 0
        while not worker_list[0].done:
            worker_list[int(pipeline[i % pipline_length][0])].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    if exec_type == ExecType.TwoDecode:
        """
        before:
        A-attn0->A-attn1->(A-dis->A-share)
        roll:
        B-attn0->A-dis_wait->A-moe->A-comb->B-attn1->A-comb_wait->(B-dis->B-share)->
        A-attn0->B-dis_wait->B-moe->B-comb->A-attn1->B-comb_wait->(A-dis->A-share)
        after:
        B-dis_wait->B-moe->B-comb->B-comb_wait and end
        """
        assert len(inputs) == 2 and delta_stages in [0, 3]

        for _ in range(3):
            worker_list[0].next()

        pipeline = [
            '1-attn0', '0-dis_wait', '0-moe', '0-comb', '1-attn1', '0-comb_wait', '1-dis+1-share', '0-attn0',
            '1-dis_wait', '1-moe', '1-comb', '0-attn1', '1-comb_wait', '0-dis+0-share'
        ]
        pipline_length = len(pipeline)
        i = 0
        while not worker_list[0].done:
            worker_list[int(pipeline[i % pipline_length][0])].next()
            i += 1

        while not worker_list[1].done:
            worker_list[1].next()

    for worker in worker_list:
        assert worker.done
    return [worker.output for worker in worker_list]


def can_two_batch(attn_metadata):
    if attn_metadata.q_start_loc.size(dim=0) < 2:
        return False
    else:
        return True


def get_new_meta(attn_metadata, start_idx: int, end_idx: int):
    new_attn_metadata = deepcopy(attn_metadata)
    new_attn_metadata.block_offsets = attn_metadata.block_offsets[start_idx:end_idx, ...]
    q_start_loc = int(attn_metadata.q_start_loc[start_idx].item())
    new_attn_metadata.q_start_loc = attn_metadata.q_start_loc[start_idx:end_idx] - q_start_loc
    k_start_loc = int(attn_metadata.kv_start_loc[start_idx].item()) if attn_metadata.kv_start_loc is not None else 0
    new_attn_metadata.kv_start_loc = attn_metadata.kv_start_loc[start_idx:end_idx] - k_start_loc \
        if attn_metadata.kv_start_loc is not None else None
    new_attn_metadata.q_seqlens = attn_metadata.q_seqlens[start_idx:end_idx]
    new_attn_metadata.kv_seqlens = attn_metadata.kv_seqlens[start_idx:end_idx] \
        if attn_metadata.kv_seqlens is not None else None
    new_attn_metadata.kv_flatten_size = sum(new_attn_metadata.kv_seqlens.tolist()) \
        if attn_metadata.kv_flatten_size is not None else None
    return new_attn_metadata, q_start_loc, k_start_loc


def get_new_rotary_pos_emb(rotary_pos_emb, start_loc, end_loc):
    new_rotary_pos_emb = (rotary_pos_emb[0][start_loc:end_loc, ...].contiguous(), rotary_pos_emb[1][start_loc:end_loc,
                                                                                                    ...].contiguous())
    return new_rotary_pos_emb


def get_new_input(hidden_states, rotary_pos_emb, past_key_values, residual, attn_metadata, start_idx, end_idx,
                  start_loc, end_loc):
    new_hidden_states = hidden_states[:, start_loc:end_loc, :].contiguous()
    new_rotary_pos_emb = get_new_rotary_pos_emb(rotary_pos_emb, start_loc, end_loc)
    new_past_key_values = past_key_values
    new_residual = residual[:, start_loc:end_loc, :].contiguous() if residual is not None else None
    new_attn_metadata, _, _ = get_new_meta(attn_metadata, start_idx, end_idx)
    return new_hidden_states, new_rotary_pos_emb, new_past_key_values, new_residual, new_attn_metadata


def split_seqlens_and_startloc(attn_metadata, num=2):
    """split seqlens and startloc, support 2 only."""
    assert num == 2
    q_start_loc = attn_metadata.q_start_loc.tolist()
    q_seqlens = attn_metadata.q_seqlens.tolist()
    kv_start_loc = attn_metadata.kv_start_loc.tolist() if attn_metadata.kv_start_loc is not None else None
    kv_seqlens = attn_metadata.kv_seqlens.tolist() if attn_metadata.kv_seqlens is not None else None
    assert len(q_start_loc) == len(q_seqlens)
    assert len(q_start_loc) >= 2
    assert len(q_seqlens) >= 2
    total_len = sum(q_seqlens)
    min_diff = total_len
    split_flag = 1
    for idx in range(1, len(q_seqlens)):
        diff = abs(sum(q_seqlens[:idx]) - sum(q_seqlens[idx:]))
        if diff < min_diff:
            min_diff = diff
            split_flag = idx
    q_start_loc_a = q_start_loc[:split_flag]
    q_start_loc_b = q_start_loc[split_flag:]
    q_seqlens_a = q_seqlens[:split_flag]
    q_seqlens_b = q_seqlens[split_flag:]
    kv_start_loc_a = kv_start_loc[:split_flag] if kv_start_loc is not None else None
    kv_start_loc_b = kv_start_loc[split_flag:] if kv_start_loc is not None else None
    kv_seqlens_a = kv_seqlens[:split_flag] if kv_seqlens is not None else None
    kv_seqlens_b = kv_seqlens[split_flag:] if kv_seqlens is not None else None
    assert sum(q_seqlens_a) + sum(q_seqlens_b) == total_len
    return (q_seqlens, q_seqlens_a, q_seqlens_b, q_start_loc, q_start_loc_a, q_start_loc_b, kv_seqlens, kv_seqlens_a,
            kv_seqlens_b, kv_start_loc, kv_start_loc_a, kv_start_loc_b)


def split_input(hidden_states,
                rotary_pos_emb,
                past_key_values,
                residual,
                attn_metadata,
                moe_start_idx,
                moe_end_idx,
                num=2):
    """split input, support 1 or 2 only."""
    # one batch
    if num == 1:
        input = {
            'hidden_states': hidden_states,
            'rotary_pos_emb': rotary_pos_emb,
            'past_key_values': past_key_values,
            'residual': residual,
            'attn_metadata': attn_metadata,
            'start_idx': moe_start_idx,
            'end_idx': moe_end_idx
        }
        extern_tag = 'D' if attn_metadata.is_decoding else 'P'
        return [input], ExecType.One, 0, extern_tag
    # two batch or more
    assert num == 2
    (q_seqlens, q_seqlens_a, q_seqlens_b, q_start_loc, q_start_loc_a, q_start_loc_b, kv_seqlens, kv_seqlens_a,
     kv_seqlens_b, kv_start_loc, kv_start_loc_a, kv_start_loc_b) = split_seqlens_and_startloc(attn_metadata, 2)

    start_idx_a = 0
    end_idx_a = len(q_seqlens_a)
    start_idx_b = end_idx_a
    end_idx_b = len(q_seqlens)

    hidden_states_a, rotary_pos_emb_a, past_key_values_a, residual_a, attn_metadata_a = get_new_input(
        hidden_states, rotary_pos_emb, past_key_values, residual, attn_metadata, start_idx_a, end_idx_a,
        q_start_loc_a[0], q_start_loc_a[-1] + q_seqlens_a[-1])
    hidden_states_b, rotary_pos_emb_b, past_key_values_b, residual_b, attn_metadata_b = get_new_input(
        hidden_states, rotary_pos_emb, past_key_values, residual, attn_metadata, start_idx_b, end_idx_b,
        q_start_loc_b[0], q_start_loc_b[-1] + q_seqlens_b[-1])

    input_a = {
        'hidden_states': hidden_states_a,
        'rotary_pos_emb': rotary_pos_emb_a,
        'past_key_values': past_key_values,
        'residual': residual_a,
        'attn_metadata': attn_metadata_a,
        'start_idx': moe_start_idx,
        'end_idx': moe_end_idx
    }
    input_b = {
        'hidden_states': hidden_states_b,
        'rotary_pos_emb': rotary_pos_emb_b,
        'past_key_values': past_key_values,
        'residual': residual_b,
        'attn_metadata': attn_metadata_b,
        'start_idx': moe_start_idx,
        'end_idx': moe_end_idx
    }

    if attn_metadata.is_decoding:
        exec_type = ExecType.TwoDecode
        delta_stages = 0
        extern_tag = 'D'
    else:
        exec_type = ExecType.TwoPrefill
        delta_stages = 0
        extern_tag = 'P'

    return [input_a, input_b], exec_type, delta_stages, extern_tag


def merge_output(output_list):
    # one batch
    if len(output_list) == 1:
        return output_list[0]
    # two batch or more
    hidden_states = torch.concat([output[0] for output in output_list], dim=1)
    residual = None
    if output_list[0][0] is not None:
        residual = torch.concat([output[1] for output in output_list], dim=1)
    return hidden_states, residual


def patch_twomicrobatch():
    if not ENABLE_TWO:
        return
    # Hack model FusedMoEBlockedF8
    from lmdeploy.pytorch.nn.moe import FusedMoEBlockedF8

    def FusedMoEBlockedF8_forward_yield(self,
                                        hidden_states: torch.Tensor,
                                        topk_weights: torch.Tensor,
                                        topk_ids: torch.LongTensor,
                                        tag: Any = None,
                                        shared_experts: Any = None):
        ret = yield from self.impl.forward_yield(hidden_states, topk_weights, topk_ids, self.gate_up.weight,
                                                 self.gate_up.scale, self.down.weight, self.down.scale,
                                                 self.expert_list, tag, shared_experts)
        return ret

    FusedMoEBlockedF8.forward_yield = FusedMoEBlockedF8_forward_yield

    from lmdeploy.pytorch.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2Model, DeepseekV2MoE

    # Hack model DeepseekV2MoE
    def DeepseekV2MoE_forward_yield(self, hidden_states: torch.Tensor, tag: Any = None):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self.gate(hidden_states)

        out_states, shared_states = yield from self.experts.forward_yield(hidden_states, topk_weights, topk_ids, tag,
                                                                          self.shared_experts)

        if shared_states is not None:
            out_states += shared_states
        elif self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
            out_states += shared_states
        else:
            pass

        out_states = out_states.reshape(batch_size, sequence_length, -1)

        if self._all_reduce:
            dist.all_reduce(out_states)

        return out_states

    DeepseekV2MoE.forward_yield = DeepseekV2MoE_forward_yield

    # Hack model DeepseekV2DecoderLayer
    def DeepseekV2DecoderLayer_forward_yield(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        tag: Any = None,
    ):
        """DeepseekV2DecoderLayer_forward_yield."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # yield for attn0 and attn1
        yield
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # yield for attn1, dis (+share), dis_wait, moe, comb, (+share) comb_wait, (+share) attn0
        hidden_states = yield from self.mlp.forward_yield(hidden_states, tag)

        outputs = (hidden_states, residual)
        return outputs

    DeepseekV2DecoderLayer.forward_yield = DeepseekV2DecoderLayer_forward_yield

    # Hack model DeepseekV2Model
    def DeepseekV2Model_forward_yieldlayers(self,
                                            hidden_states: torch.Tensor,
                                            rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
                                            past_key_values: Optional[List[torch.FloatTensor]] = None,
                                            residual: Optional[torch.Tensor] = None,
                                            attn_metadata: Any = None,
                                            start_idx: int = -1,
                                            end_idx: int = -1,
                                            tag: Any = None):
        """DeepseekV2Model_forward_yieldlayers."""
        assert start_idx >= 0 and start_idx < len(self.layers) and end_idx > 0 and end_idx <= len(self.layers),\
            f'forward_yieldlayers error !!! start_idx:{start_idx},end_idx:{end_idx}, layer num:{len(self.layers)}'
        for idx in range(start_idx, end_idx):
            past_key_value = past_key_values[idx]
            hidden_states, residual = yield from self.layers[idx].forward_yield(hidden_states,
                                                                                rotary_pos_emb=rotary_pos_emb,
                                                                                past_key_value=past_key_value,
                                                                                residual=residual,
                                                                                attn_metadata=attn_metadata,
                                                                                tag=tag)
        return hidden_states, residual

    def DeepseekV2Model_forward_batch(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """DeepseekV2Model_forward_batch."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        assert self.config.moe_layer_freq == 1

        moe_start_idx = min(self.config.first_k_dense_replace, len(self.layers))

        for idx, decoder_layer in enumerate(self.layers[:moe_start_idx]):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        if moe_start_idx < len(self.layers):
            num = 1
            if can_two_batch(attn_metadata):
                num = 2
            # split
            input_list, exec_type, delta_stages, extern_tag = split_input(hidden_states,
                                                                          rotary_pos_emb,
                                                                          past_key_values,
                                                                          residual,
                                                                          attn_metadata,
                                                                          moe_start_idx,
                                                                          len(self.layers),
                                                                          num=num)

            output_list = execute_batch(inputs=input_list,
                                        fn=self.forward_yieldlayers,
                                        delta_stages=delta_stages,
                                        exec_type=exec_type,
                                        extern_tag=extern_tag)
            hidden_states, residual = merge_output(output_list)

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def DeepseekV2Model_forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if ENABLE_TWO:
            return self.forward_batch(input_ids, position_ids, past_key_values, attn_metadata, inputs_embeds)
        else:
            return self.forward_raw(input_ids, position_ids, past_key_values, attn_metadata, inputs_embeds)

    DeepseekV2Model.forward_raw = DeepseekV2Model.forward
    DeepseekV2Model.forward_yieldlayers = DeepseekV2Model_forward_yieldlayers
    DeepseekV2Model.forward_batch = DeepseekV2Model_forward_batch
    DeepseekV2Model.forward = DeepseekV2Model_forward
