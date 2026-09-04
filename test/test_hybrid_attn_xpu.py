import os
from yunchang import LongContextAttention, set_seq_parallel_pg, EXTRACT_FUNC_DICT
import torch
import torch.distributed as dist

from yunchang.kernels import AttnType
from test_utils import attention_ref
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test hybrid attention with XPU backend"
    )
    parser.add_argument(
        "--seqlen", type=int, default=1024, help="sequence length (default: 1024)"
    )
    parser.add_argument(
        "--sp_ulysses_degree",
        type=int,
        default=None,
        help="sp_ulysses_degree (default: world_size)",
    )
    parser.add_argument(
        "--ring_impl_type",
        type=str,
        default="basic_xpu",
        choices=["basic_xpu"],
        help="ring implementation type (default: basic_xpu)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="whether to use causal attention (default: False)",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="xpu",
        choices=["xpu"],
        help="attention implementation type (default: xpu)",
    )
    return parser.parse_args()


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"[Rank#0] {msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[Rank#{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


# test it with:
# torchrun --nproc_per_node=4  test/test_hybrid_attn_xpu.py
if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)

    dist.init_process_group("xccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    device = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(device)

    batch_size = 1
    seqlen = args.seqlen
    nheads = 32
    d = 2048 // 32
    causal = args.causal

    assert seqlen % world_size == 0
    assert d % 8 == 0

    ring_impl_type = args.ring_impl_type

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    sp_ulysses_degree = (
        args.sp_ulysses_degree if args.sp_ulysses_degree is not None else world_size
    )
    sp_ring_degree = world_size // sp_ulysses_degree

    print(
        f"rank {rank}, sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}"
    )

    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    local_q = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            q, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )
    local_k = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            k, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )
    local_v = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            v, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    attn_impl_map = {"xpu": AttnType.XPU}

    usp_attn = LongContextAttention(
        ring_impl_type=ring_impl_type,
        attn_type=attn_impl_map[args.attn_impl],
    )

    if rank == 0:
        print("#" * 30)
        print("# usp attn forward (XPU):")
        print("#" * 30)

    local_out = usp_attn(
        local_q,
        local_k,
        local_v,
        causal=causal,
    )

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print("# reference forward (SDPA):")
        print("#" * 30)

    # Reference: PyTorch SDPA which runs on XPU
    softmax_scale = q.shape[-1] ** -0.5
    q_t = q.transpose(1, 2)  # (bs, nheads, seqlen, headdim)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, scale=softmax_scale, is_causal=causal
    ).transpose(1, 2)  # (bs, seqlen, nheads, headdim)

    dist.barrier()

    local_out_ref = EXTRACT_FUNC_DICT[ring_impl_type](
        out_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    )

    log("local (rank) out", local_out, rank0_only=True)
    log("out (distributed) - out_ref (non-distributed) diff", local_out_ref - local_out)

    torch.testing.assert_close(local_out, local_out_ref, atol=1e-1, rtol=0)

    if rank == 0:
        print("XPU ring attention correctness check passed!", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()
