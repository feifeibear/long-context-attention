from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from .utils import (
    RingComm,
    ring_npu_attention_out_update,
)
from yunchang.kernels import AttnType, select_flash_attn_impl

CuSeqLen = Union[torch.Tensor, Sequence[int]]
HalfIndex = Union[slice, torch.Tensor]
RNGState = Tuple[int, int, int]


def _cu_seq_len_tensor(cu_seqlens: CuSeqLen) -> torch.Tensor:
    """Return CPU cumulative lengths with the leading zero for indexing."""
    if isinstance(cu_seqlens, torch.Tensor):
        values = cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    else:
        values = torch.as_tensor(cu_seqlens, dtype=torch.long, device="cpu")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("cu_seqlens must be a non-empty 1-D sequence")
    if values[0].item() != 0:
        values = torch.cat(
            [torch.zeros(1, dtype=values.dtype), values]
        )
    if (
        values.numel() < 2
        or (values[1:] <= 0).any()
        or (values[1:] < values[:-1]).any()
    ):
        raise ValueError(
            "cu_seqlens must contain positive, non-decreasing endpoints"
        )
    return values.contiguous()


def get_half_index(cu_seqlens: torch.Tensor, *, front: bool) -> HalfIndex:
    if len(cu_seqlens) == 2:
        midpoint = int(cu_seqlens[-1].item()) // 2
        if front:
            return slice(None, midpoint)
        else:
            return slice(midpoint, None)

    index = torch.zeros(
        (int(cu_seqlens[-1].item()),),
        dtype=torch.bool,
        device=cu_seqlens.device,
    )
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


def zigzag_ring_npu_flash_attn_varlen_forward(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    half_index0: HalfIndex,
    half_index1: HalfIndex,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[RNGState]]:
    if not causal:
        raise ValueError("zigzag ring requires causal=True")
    full_cu_seqlens = _cu_seq_len_tensor(cu_seqlens)
    comm = RingComm(process_group)
    npu_fa_forward = select_flash_attn_impl(
        impl_type=attn_type,
        stage="fwd-only",
        attn_processor=attn_processor,
    )

    block_seq_len = q.shape[0] // 2
    q1 = q[half_index1].contiguous()

    global_out, global_softmax_max, global_softmax_sum = None, None, None
    next_k, next_v = None, None
    half_cu_seqlens = full_cu_seqlens // 2

    # Keep cumulative lengths on CPU for indexing. The NPU wrapper converts
    # them to endpoint lists at the operator boundary.
    actual_seqlen = full_cu_seqlens
    actual_half_seqlen = half_cu_seqlens
    rng_states: List[RNGState] = [(0, 0, 0) for _ in range(comm.world_size)]

    def _forward_block(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNGState]:
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = (
            actual_half_seqlen if seqlen_q == block_seq_len else actual_seqlen
        )
        cu_seqlens_kv = (
            actual_half_seqlen if seqlen_kv == block_seq_len else actual_seqlen
        )
        block_out, block_softmax_max, block_softmax_sum, rng_state = (
            npu_fa_forward(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q,
                actual_seq_kvlen=cu_seqlens_kv,
                softmax_layout="TND",
            )
        )
        return block_out, block_softmax_max, block_softmax_sum, rng_state

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_softmax_max, block_softmax_sum, rng_state = (
                _forward_block(q, k, v, causal=True)
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
        elif step <= comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            block_out, block_softmax_max, block_softmax_sum, rng_state = (
                _forward_block(q, k0, v0, causal=False)
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
        else:
            block_out, block_softmax_max, block_softmax_sum, rng_state = (
                _forward_block(q1, k, v, causal=False)
            )
            (
                global_out[half_index1],
                global_softmax_max[half_index1],
                global_softmax_sum[half_index1],
            ) = ring_npu_attention_out_update(
                global_out[half_index1],
                global_softmax_max[half_index1],
                global_softmax_sum[half_index1],
                block_out,
                block_softmax_max,
                block_softmax_sum,
            )
        rng_states[step] = rng_state
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    global_out = global_out.to(q.dtype)
    return global_out, global_softmax_max, global_softmax_sum, rng_states


def zigzag_ring_npu_flash_attn_varlen_backward(
    process_group: Optional[dist.ProcessGroup],
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    rng_states: Sequence[RNGState],
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    cu_seqlens: torch.Tensor,
    half_index0: HalfIndex,
    half_index1: HalfIndex,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not causal:
        raise ValueError("zigzag ring requires causal=True")
    full_cu_seqlens = _cu_seq_len_tensor(cu_seqlens)
    npu_fa_backward = select_flash_attn_impl(
        impl_type=attn_type,
        stage="bwd-only",
        attn_processor=attn_processor,
    )

    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None

    dout1 = dout[half_index1]
    q1 = q[half_index1].contiguous()
    out1 = out[half_index1]

    softmax_max1 = (
        softmax_max[half_index1].unsqueeze(-1).expand(-1, -1, 8).contiguous()
    )
    softmax_sum1 = (
        softmax_sum[half_index1].unsqueeze(-1).expand(-1, -1, 8).contiguous()
    )

    softmax_max = softmax_max.unsqueeze(-1).expand(-1, -1, 8).contiguous()
    softmax_sum = softmax_sum.unsqueeze(-1).expand(-1, -1, 8).contiguous()

    block_seq_len = q.shape[0] // 2
    half_cu_seqlens = full_cu_seqlens // 2

    actual_seqlen = full_cu_seqlens
    actual_half_seqlen = half_cu_seqlens

    def _backward_block(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        causal: bool,
        rng_state: RNGState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = (
            actual_half_seqlen if seqlen_q == block_seq_len else actual_seqlen
        )
        cu_seqlens_kv = (
            actual_half_seqlen if seqlen_kv == block_seq_len else actual_seqlen
        )

        grad_query, grad_key, grad_value, *_ = npu_fa_backward(
            dout,
            q,
            k,
            v,
            attention_in=out,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            rng_state=rng_state,
            input_layout="TND",
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_kv,
            softmax_layout="TND",
        )
        return grad_query, grad_key, grad_value

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        rng_state = rng_states[step]
        if step == 0:
            dq, dk, dv = _backward_block(
                dout,
                q,
                k,
                v,
                out,
                softmax_max,
                softmax_sum,
                causal=True,
                rng_state=rng_state,
            )
            dq = dq.to(torch.float32)
            dk = dk.to(torch.float32)
            dv = dv.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[half_index0]
                v0 = v[half_index0]
                grad_query, grad_key, grad_value = _backward_block(
                    dout,
                    q,
                    k0,
                    v0,
                    out,
                    softmax_max,
                    softmax_sum,
                    causal=False,
                    rng_state=rng_state,
                )
                dq += grad_query
            else:
                grad_query, grad_key, grad_value = _backward_block(
                    dout1,
                    q1,
                    k,
                    v,
                    out1,
                    softmax_max1,
                    softmax_sum1,
                    causal=False,
                    rng_state=rng_state,
                )
                dq[half_index1] += grad_query

            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[half_index0] += grad_key
                dv[half_index0] += grad_value
            else:
                dk += grad_key
                dv += grad_value

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class ZigZagRingNPUFlashAttnVarlenFunc(torch.autograd.Function):
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

        cu_seq_len = _cu_seq_len_tensor(cu_seq_len)
        full_cu_seq_len = cu_seq_len
        half_index0 = get_half_index(full_cu_seq_len, front=True)
        half_index1 = get_half_index(full_cu_seq_len, front=False)
        if isinstance(half_index0, torch.Tensor):
            # Boolean masks used for advanced indexing must live with the
            # tensors being indexed on NPU; cumulative lengths stay on CPU for
            # torch_npu's actual_seq_* contract.
            half_index0 = half_index0.to(device=q.device)
            half_index1 = half_index1.to(device=q.device)
        global_out, global_softmax_max, global_softmax_sum, rng_states = (
            zigzag_ring_npu_flash_attn_varlen_forward(
                process_group,
                q,
                k,
                v,
                cu_seq_len,
                half_index0,
                half_index1,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
                attn_type=attn_type,
                attn_processor=attn_processor,
            )
        )
        # this should be out_padded
        is_half_index_tensor = isinstance(half_index0, torch.Tensor)
        ctx.is_half_index_tensor = is_half_index_tensor
        if is_half_index_tensor:
            ctx.save_for_backward(
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
                half_index0,
                half_index1,
            )
        else:
            ctx.save_for_backward(
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
            )
            ctx.half_index0 = half_index0
            ctx.half_index1 = half_index1

        ctx.group = process_group
        ctx.rng_states = rng_states
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        return global_out

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor, *args: Any
    ) -> Tuple[Optional[torch.Tensor], ...]:
        if ctx.is_half_index_tensor:
            (
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
                half_index0,
                half_index1,
            ) = ctx.saved_tensors
        else:
            (
                q,
                k,
                v,
                global_out,
                global_softmax_max,
                global_softmax_sum,
                cu_seq_len,
            ) = ctx.saved_tensors
            half_index0 = ctx.half_index0
            half_index1 = ctx.half_index1
        dq, dk, dv = zigzag_ring_npu_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            global_out,
            ctx.rng_states,
            global_softmax_max,
            global_softmax_sum,
            cu_seq_len,
            half_index0,
            half_index1,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            attn_type=ctx.attn_type,
            attn_processor=ctx.attn_processor,
        )
        return (dq, dk, dv) + (None,) * 7


def zigzag_ring_npu_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: CuSeqLen,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return ZigZagRingNPUFlashAttnVarlenFunc.apply(
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


def zigzag_ring_npu_flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: CuSeqLen,
    max_seqlen: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return zigzag_ring_npu_flash_attn_varlen_func(
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
