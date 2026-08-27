import os

import torch
import torch_npu  # noqa: F401 - registers the torch.npu backend
from torch import distributed as dist

from test_utils import assert_close, eager_attn, make_bsnd_input, sequential_print
from yunchang.ring.zigzag_ring_npu_flash_attn import zigzag_ring_flash_attn_npu_func


def zigzag_shard_seq(data: torch.Tensor) -> torch.Tensor:
    seq_len = data.shape[1]
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    chunk_size = seq_len // (2 * world_size)
    chunk1 = data[:, rank * chunk_size : (rank + 1) * chunk_size]
    chunk2 = data[
        :, seq_len - (rank + 1) * chunk_size : seq_len - rank * chunk_size
    ]

    return torch.concat([chunk1, chunk2], dim=1).contiguous()


if __name__ == "__main__":
    batch_size = 2
    seq_length = 128
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
    q, k, v, dout = make_bsnd_input(
        batch_size, seq_length, num_heads, dim, dtype
    )
    q.requires_grad = not forward_only
    k.requires_grad = not forward_only
    v.requires_grad = not forward_only

    local_q = zigzag_shard_seq(q.detach()).requires_grad_()
    local_k = zigzag_shard_seq(k.detach()).requires_grad_()
    local_v = zigzag_shard_seq(v.detach()).requires_grad_()
    local_out = zigzag_shard_seq(dout.clone()).requires_grad_()
    attn_out = zigzag_ring_flash_attn_npu_func(
        local_q,
        local_k,
        local_v,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    out_ref = eager_attn(
        q, k, v, softmax_scale=softmax_scale, causal=causal, dtype=dtype
    )
    shard_out = zigzag_shard_seq(out_ref)
    diff_abs = (shard_out.to(attn_out.dtype) - attn_out).abs()
    assert_close("forward", attn_out, shard_out.to(attn_out.dtype))

    sequential_print(
        f"[attn out] rank:{dist.get_rank()}, diff mean:{diff_abs.mean()}, "
        f"diff max:{diff_abs.max()}, norm:{shard_out.norm(p=2)}"
    )

    if not forward_only:
        attn_out.backward(local_out)
        out_ref.backward(dout)

        ref_dq = zigzag_shard_seq(q.grad)
        ref_dk = zigzag_shard_seq(k.grad)
        ref_dv = zigzag_shard_seq(v.grad)

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
