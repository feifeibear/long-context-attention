import os

import torch
from torch import distributed as dist

from test_utils import assert_close, eager_attn, make_varlen_input, sequential_print
from yunchang.ring.zigzag_ring_npu_flash_attn_varlen import (
    zigzag_ring_npu_flash_attn_varlen_func,
)


def zigzag_shard_varlen_seq(
    data: torch.Tensor, cu_seq_len: torch.Tensor
) -> torch.Tensor:
    n = len(cu_seq_len)
    local_data = []
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for i in range(n - 1):
        st, ed = cu_seq_len[i], cu_seq_len[i + 1]
        length = ed - st
        chunk_size = length // (2 * world_size)
        assert chunk_size * (2 * world_size) == length

        chunk1 = data[st + rank * chunk_size : st + (rank + 1) * chunk_size]
        chunk2 = data[ed - (rank + 1) * chunk_size : ed - rank * chunk_size]
        cur_seq_shard = torch.concat([chunk1, chunk2], dim=0)
        local_data.append(cur_seq_shard)

    return torch.concat(local_data, dim=0)


if __name__ == "__main__":
    seq_length = [16, 64, 128]
    num_heads = 4
    dim = 128
    forward_only = False
    causal = True
    softmax_scale = dim ** (-0.5)
    dtype = torch.bfloat16
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.set_default_device(f"npu:{local_rank}")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")

    q, k, v, dout = make_varlen_input(seq_length, num_heads, dim, dtype)
    q.requires_grad = not forward_only
    k.requires_grad = not forward_only
    v.requires_grad = not forward_only

    cu_seq_qlen = torch.cumsum(
        torch.as_tensor([0] + seq_length, dtype=torch.long), dim=0
    )

    local_q = zigzag_shard_varlen_seq(q.detach(), cu_seq_qlen).requires_grad_()
    local_k = zigzag_shard_varlen_seq(k.detach(), cu_seq_qlen).requires_grad_()
    local_v = zigzag_shard_varlen_seq(v.detach(), cu_seq_qlen).requires_grad_()
    local_out = zigzag_shard_varlen_seq(dout.clone(), cu_seq_qlen).requires_grad_()

    # Match the GPU FA convention: cumulative lengths include the leading 0.
    local_cu_seq_len = (cu_seq_qlen // dist.get_world_size()).long()

    attn_out = zigzag_ring_npu_flash_attn_varlen_func(
        local_q,
        local_k,
        local_v,
        softmax_scale=softmax_scale,
        cu_seqlens=local_cu_seq_len,
        causal=causal,
    )

    out_ref = eager_attn(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        cu_seq_qlen=cu_seq_qlen,
        cu_seq_kv_len=cu_seq_qlen,
        dtype=dtype,
    )

    shard_out = zigzag_shard_varlen_seq(out_ref, cu_seq_qlen)
    diff_abs = (shard_out.to(attn_out.dtype) - attn_out).abs()
    assert_close("forward", attn_out, shard_out.to(attn_out.dtype))

    sequential_print(
        f"[attn out] rank:{dist.get_rank()}, diff mean:{diff_abs.mean()}, "
        f"diff max:{diff_abs.max()}, norm:{shard_out.norm(p=2)}"
    )

    if not forward_only:
        attn_out.backward(local_out)
        out_ref.backward(dout)

        ref_dq = zigzag_shard_varlen_seq(q.grad, cu_seq_qlen)
        ref_dk = zigzag_shard_varlen_seq(k.grad, cu_seq_qlen)
        ref_dv = zigzag_shard_varlen_seq(v.grad, cu_seq_qlen)

        for name, local_grad, ref in [
            ("dq", local_q.grad, ref_dq),
            ("dk", local_k.grad, ref_dk),
            ("dv", local_v.grad, ref_dv),
        ]:
            d = (local_grad - ref).abs()
            assert_close(name, local_grad, ref)
            sequential_print(
                f"[{name}], rank:{dist.get_rank()}, diff mean:{d.mean()}, "
                f"diff max:{d.max()}, norm:{ref.norm(p=2)}"
            )

    dist.destroy_process_group()
