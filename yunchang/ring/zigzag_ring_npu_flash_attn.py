from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .utils import RingComm, ring_npu_attention_out_update
from yunchang.kernels import AttnType, select_flash_attn_impl

RNGState = Tuple[int, int, int]


def zigzag_npu_ring_flash_attn_forward(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[RNGState]]:
    if not causal:
        raise ValueError("zigzag ring requires causal=True")
    npu_fa_forward = select_flash_attn_impl(
        impl_type=attn_type,
        stage="fwd-only",
        attn_processor=attn_processor,
    )
    comm = RingComm(process_group)
    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:].contiguous()

    out = None
    softmax_max = None
    softmax_sum = None
    rng_states: List[RNGState] = [(0, 0, 0) for _ in range(comm.world_size)]
    k = k.contiguous()
    v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_max, block_sum, rng_state = npu_fa_forward(
                q, k, v, softmax_scale, dropout_p, causal=True, input_layout="BSND"
            )
            out, softmax_max, softmax_sum = ring_npu_attention_out_update(
                out, softmax_max, softmax_sum, block_out, block_max, block_sum
            )
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_max, block_sum, rng_state = npu_fa_forward(
                q, k0, v0, softmax_scale, dropout_p, causal=False
            )
            out, softmax_max, softmax_sum = ring_npu_attention_out_update(
                out, softmax_max, softmax_sum, block_out, block_max, block_sum
            )
        else:
            block_out, block_max, block_sum, rng_state = npu_fa_forward(
                q1, k, v, softmax_scale, dropout_p, causal=False
            )
            (
                out[:, block_seq_len:],
                softmax_max[:, :, block_seq_len:],
                softmax_sum[:, :, block_seq_len:],
            ) = ring_npu_attention_out_update(
                out[:, block_seq_len:],
                softmax_max[:, :, block_seq_len:],
                softmax_sum[:, :, block_seq_len:],
                block_out,
                block_max,
                block_sum,
            )

        rng_states[step] = rng_state

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    return out, softmax_max, softmax_sum, rng_states


def zigzag_npu_ring_flash_attn_backward(
    process_group: Optional[dist.ProcessGroup],
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    rng_states: Sequence[RNGState],
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not causal:
        raise ValueError("zigzag ring requires causal=True")
    npu_fa_backward = select_flash_attn_impl(
        impl_type=attn_type,
        stage="bwd-only",
        attn_processor=attn_processor,
    )
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:].contiguous()
    dout1 = dout[:, block_seq_len:].contiguous()
    out1 = out[:, block_seq_len:].contiguous()
    # BSND output uses S at dim=1; NPU softmax stats use BNS8, with S at dim=2.
    softmax_max1 = (
        softmax_max[:, :, block_seq_len:]
        .unsqueeze(-1)
        .expand(-1, -1, -1, 8)
        .contiguous()
    )
    softmax_sum1 = (
        softmax_sum[:, :, block_seq_len:]
        .unsqueeze(-1)
        .expand(-1, -1, -1, 8)
        .contiguous()
    )

    softmax_max = softmax_max.unsqueeze(-1).expand(-1, -1, -1, 8).contiguous()
    softmax_sum = softmax_sum.unsqueeze(-1).expand(-1, -1, -1, 8).contiguous()
    dq = dk = dv = None
    next_dk = next_dv = None
    dk_comm_buffer = dv_comm_buffer = None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            dq_block, dk_block, dv_block = npu_fa_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_max,
                softmax_sum,
                softmax_scale,
                dropout_p,
                causal=True,
                rng_state=rng_states[step],
                input_layout="BSND",
            )
            dq = dq_block.float().clone()
            dk = dk_block.float().clone()
            dv = dv_block.float().clone()
        else:
            if step <= kv_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                dq_block, dk_block, dv_block = npu_fa_backward(
                    dout,
                    q,
                    k0,
                    v0,
                    out,
                    softmax_max,
                    softmax_sum,
                    softmax_scale,
                    dropout_p,
                    causal=False,
                    rng_state=rng_states[step],
                )
                dq += dq_block.float()
            else:
                dq_block, dk_block, dv_block = npu_fa_backward(
                    dout1,
                    q1,
                    k,
                    v,
                    out1,
                    softmax_max1,
                    softmax_sum1,
                    softmax_scale,
                    dropout_p,
                    causal=False,
                    rng_state=rng_states[step],
                )
                dq[:, block_seq_len:] += dq_block.float()

            # Receive gradients already accumulated for the current rotating KV
            # block, then add this rank's partial gradients.
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_block.float()
                dv[:, :block_seq_len] += dv_block.float()
            else:
                dk += dk_block.float()
                dv += dv_block.float()

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()
    assert dq is not None and next_dk is not None and next_dv is not None
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class ZigZagRingNpuFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        group: Optional[dist.ProcessGroup],
        attn_type: AttnType,
        attn_processor: Optional[torch.nn.Module],
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # dummy = q.sum()
        # _ = q.cpu()
        out, softmax_max, softmax_sum, rng_states = (
            zigzag_npu_ring_flash_attn_forward(
                group,
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
                attn_type=attn_type,
                attn_processor=attn_processor,
            )
        )
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)
        ctx.rng_states = rng_states
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.group = group
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        return out

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor, *args: Any
    ) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, out, softmax_max, softmax_sum = ctx.saved_tensors
        dq, dk, dv = zigzag_npu_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum,
            ctx.rng_states,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            attn_type=ctx.attn_type,
            attn_processor=ctx.attn_processor,
        )
        return (dq, dk, dv) + (None,) * 6


def zigzag_ring_flash_attn_npu_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return ZigZagRingNpuFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        group,
        attn_type,
        attn_processor,
    )


def zigzag_ring_npu_flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    attn_type: AttnType = AttnType.NPU,
    attn_processor: Optional[torch.nn.Module] = None,
    group: Optional[dist.ProcessGroup] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return zigzag_ring_flash_attn_npu_func(
        qkv[:, :, 0].contiguous(),
        qkv[:, :, 1].contiguous(),
        qkv[:, :, 2].contiguous(),
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        attn_type=attn_type,
        attn_processor=attn_processor,
        group=group,
        **kwargs,
    )
