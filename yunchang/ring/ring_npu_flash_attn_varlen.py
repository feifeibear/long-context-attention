from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from .utils import RingComm, ring_npu_attention_out_update
from yunchang.kernels import AttnType, select_flash_attn_impl

CuSeqLen = Union[torch.Tensor, Sequence[int]]
RNGState = Tuple[int, int, int]


def ring_npu_flash_attn_varlen_forward(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_len: CuSeqLen,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[RNGState]]:
    comm = RingComm(process_group)
    npu_fa_forward = select_flash_attn_impl(
        attn_type, stage="fwd-only", attn_processor=attn_processor
    )
    if comm.world_size == 1:
        block_out, block_softmax_max, block_softmax_sum, rng_state = npu_fa_forward(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            input_layout="TND",
            actual_seq_qlen=cu_seq_len,
            actual_seq_kvlen=cu_seq_len,
            softmax_layout="TND",
        )
        return (
            block_out,
            block_softmax_max[:, :, 0].contiguous(),
            block_softmax_sum[:, :, 0].contiguous(),
            [rng_state],
        )

    global_out, global_softmax_max, global_softmax_sum = None, None, None
    next_k, next_v = None, None

    rng_states: List[RNGState] = [(0, 0, 0) for _ in range(comm.world_size)]
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        rng_state = (0, 0, 0)
        if not causal or step <= comm.rank:
            block_out, block_softmax_max, block_softmax_sum, rng_state = npu_fa_forward(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal and step == 0,
                input_layout="TND",
                actual_seq_qlen=cu_seq_len,
                actual_seq_kvlen=cu_seq_len,
                softmax_layout="TND",
            )
            global_out, global_softmax_max, global_softmax_sum = (
                ring_npu_attention_out_update(
                    global_out,
                    global_softmax_max,
                    global_softmax_sum,
                    block_out,
                    block_softmax_max,
                    block_softmax_sum,
                )
            )

        rng_states[step] = rng_state
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    global_out = global_out.to(q.dtype)
    return global_out, global_softmax_max, global_softmax_sum, rng_states


def ring_npu_flash_attn_varlen_backward(
    process_group: Optional[dist.ProcessGroup],
    grad_attention_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_in: torch.Tensor,
    rng_states: Sequence[RNGState],
    cu_seq_len: CuSeqLen,
    softmax_scale: float = 1.0,
    dropout_p: float = 0.0,
    causal: bool = False,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    npu_fa_backward = select_flash_attn_impl(
        attn_type, stage="bwd-only", attn_processor=attn_processor
    )
    if kv_comm.world_size == 1:
        grad_query, grad_key, grad_value = npu_fa_backward(
            grad_attention_out,
            q,
            k,
            v,
            attention_in=attention_in,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            rng_state=rng_states[0],
            input_layout="TND",
            actual_seq_qlen=cu_seq_len,
            actual_seq_kvlen=cu_seq_len,
            softmax_layout="TND",
        )

        return grad_query, grad_key, grad_value
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    next_k, next_v = None, None
    next_dk, next_dv = None, None

    softmax_max = softmax_max.unsqueeze(-1).expand(-1, -1, 8).contiguous()
    softmax_sum = softmax_sum.unsqueeze(-1).expand(-1, -1, 8).contiguous()
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        rng_state = rng_states[step]
        if step <= kv_comm.rank or not causal:
            grad_query, grad_key, grad_value = npu_fa_backward(
                grad_attention_out,
                q,
                k,
                v,
                attention_in=attention_in,
                softmax_max=softmax_max,
                softmax_sum=softmax_sum,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=(causal and step == 0),
                rng_state=rng_state,
                input_layout="TND",
                actual_seq_qlen=cu_seq_len,
                actual_seq_kvlen=cu_seq_len,
                softmax_layout="TND",
            )
            dq += grad_query.to(torch.float32)

            if step > 0:
                d_kv_comm.wait()
                dk = grad_key.to(torch.float32) + next_dk
                dv = grad_value.to(torch.float32) + next_dv
            else:
                dk = grad_key.to(torch.float32)
                dv = grad_value.to(torch.float32)
        else:
            if step > 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class RingNPUFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seq_len: CuSeqLen,
        softmax_scale: Optional[float],
        dropout_p: float = 0.0,
        causal: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        attn_type: AttnType = AttnType.NPU,
        attn_processor: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, softmax_max, softmax_sum, rng_states = ring_npu_flash_attn_varlen_forward(
            process_group,
            q,
            k,
            v,
            cu_seq_len,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            attn_type=attn_type,
            attn_processor=attn_processor,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)

        ctx.group = process_group
        ctx.cu_seq_len = cu_seq_len
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.rng_states = rng_states
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        return out

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor, *args: Any
    ) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, out, softmax_max, softmax_sum = ctx.saved_tensors
        dout = dout.contiguous()
        dq, dk, dv = ring_npu_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=out,
            rng_states=ctx.rng_states,
            cu_seq_len=ctx.cu_seq_len,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            attn_type=ctx.attn_type,
            attn_processor=ctx.attn_processor,
        )
        return (dq, dk, dv) + (None,) * 7


def ring_npu_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: CuSeqLen,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return RingNPUFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        softmax_scale,
        dropout_p,
        causal,
        group,
        attn_type,
        attn_processor,
    )


def ring_npu_flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: CuSeqLen,
    max_seqlen: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return ring_npu_flash_attn_varlen_func(
        qkv[:, 0].contiguous(),
        qkv[:, 1].contiguous(),
        qkv[:, 2].contiguous(),
        cu_seqlens=cu_seqlens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        attn_type=attn_type,
        attn_processor=attn_processor,
        group=group,
        **kwargs,
    )
