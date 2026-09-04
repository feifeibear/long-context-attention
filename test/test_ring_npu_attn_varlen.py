import os

import torch
from torch import distributed as dist

from test_utils import assert_close, eager_attn, make_varlen_input, sequential_print
from yunchang.ring.ring_npu_flash_attn_varlen import ring_npu_flash_attn_varlen_func


def shard_varlen_seq(
    data: torch.Tensor, cu_seq_len: torch.Tensor
) -> torch.Tensor:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    sharded_data = []
    length = len(cu_seq_len)

    for i in range(length - 1):
        st = cu_seq_len[i]
        ed = cu_seq_len[i + 1]

        seq_len = ed - st
        shard_size = seq_len // world_size
        assert shard_size * world_size == seq_len

        sharded_data.append(
            data[st:ed][rank * shard_size : (rank + 1) * shard_size]
        )

    return torch.concat(sharded_data, dim=0)


if __name__ == "__main__":
    seq_length = [16, 64, 128]
    num_heads = 4
    dim = 128
    forward_only = False
    causal = True
    softmax_scale = dim ** (-0.5)
    dtype = torch.bfloat16

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("hccl")
    torch.npu.set_device(local_rank)
    torch.set_default_device(f"npu:{local_rank}")

    q, k, v, dout = make_varlen_input(seq_length, num_heads, dim, dtype)
    q.requires_grad = not forward_only
    k.requires_grad = not forward_only
    v.requires_grad = not forward_only
    cu_seq_qlen = torch.cumsum(
        torch.as_tensor([0] + seq_length, dtype=torch.long), dim=0
    )

    local_q = shard_varlen_seq(q.detach(), cu_seq_qlen).requires_grad_()
    local_k = shard_varlen_seq(k.detach(), cu_seq_qlen).requires_grad_()
    local_v = shard_varlen_seq(v.detach(), cu_seq_qlen).requires_grad_()
    local_out = shard_varlen_seq(dout.clone(), cu_seq_qlen).requires_grad_()

    # Match the GPU FA convention: cumulative lengths include the leading 0.
    local_cu_seq_len = (cu_seq_qlen // dist.get_world_size()).long()

    attn_out = ring_npu_flash_attn_varlen_func(
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

    shard_out = shard_varlen_seq(out_ref, cu_seq_qlen)
    diff_abs = (shard_out.to(attn_out.dtype) - attn_out).abs()
    assert_close("forward", attn_out, shard_out.to(attn_out.dtype))
    dist.barrier()
    sequential_print(
        f"[attn out] rank:{dist.get_rank()}, diff mean:{diff_abs.mean()}, "
        f"diff max:{diff_abs.max()}, norm:{shard_out.norm(p=2)}"
    )

    if not forward_only:
        attn_out.backward(local_out)
        out_ref.backward(dout)

        ref_dq = shard_varlen_seq(q.grad, cu_seq_qlen)
        ref_dk = shard_varlen_seq(k.grad, cu_seq_qlen)
        ref_dv = shard_varlen_seq(v.grad, cu_seq_qlen)

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

            dist.barrier()

    dist.destroy_process_group()
