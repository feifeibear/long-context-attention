import os
import torch
import torch.distributed as dist
from yunchang import (
    LongContextAttention,
    set_seq_parallel_pg,
    UlyssesAttention,
)
from yunchang.comm import EXTRACT_FUNC_DICT
import argparse

parser = argparse.ArgumentParser(description="Benchmark long-context attention on XPU.")

parser.add_argument(
    "--ring_impl_type",
    type=str,
    default="basic_xpu",
    choices=["basic_xpu"],
    help="ring attn implementation type",
)
parser.add_argument("--nheads", type=int, default=2, help="head number")
parser.add_argument("--head_size", type=int, default=128, help="head size")
parser.add_argument("--seq_len", type=int, default=4 * 1024, help="sequence length")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument(
    "--use_ulysses_lowdim",
    action="store_true",
    default=True,
    help="ulysses process group on low dimension",
)
parser.add_argument(
    "--ulysses_degree",
    type=int,
    default=1,
    help="ulysses attention sequence parallel degree",
)
parser.add_argument(
    "--use_ulysses",
    action="store_true",
    default=False,
    help="use ulysses only (no ring)",
)
parser.add_argument(
    "--no_causal",
    action="store_true",
    default=False,
    help="use non-causal attention",
)

args = parser.parse_args()


def color_print(text):
    print("\033[91m {}\033[00m".format(text))


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))


def benchmark(num_iter=10, forward_only=True, log=True):
    from yunchang.kernels import AttnType

    dtype = torch.bfloat16
    rank = dist.get_rank()
    local_rank = get_local_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(device)

    batch_size = args.batch_size
    seqlen = args.seq_len
    nheads = args.nheads
    d = args.head_size
    causal = not args.no_causal

    assert seqlen % (2 * world_size) == 0, f"seqlen {seqlen} world_size {world_size}"
    assert d % 8 == 0

    sp_ulysses_degree = min(args.ulysses_degree, world_size)
    sp_ring_degree = world_size // sp_ulysses_degree

    set_seq_parallel_pg(
        sp_ulysses_degree, sp_ring_degree, rank, world_size, args.use_ulysses_lowdim
    )

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    if args.use_ulysses:
        longctx_attn = UlyssesAttention(attn_type=AttnType.XPU)
    else:
        longctx_attn = LongContextAttention(
            ring_impl_type=args.ring_impl_type, attn_type=AttnType.XPU
        )

    # warmup
    for _ in range(2):
        with torch.no_grad():
            longctx_attn(q, k, v, causal=causal)
        torch.xpu.synchronize(device=device)

    begin = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)

    begin.record()
    with torch.no_grad():
        for _ in range(num_iter):
            longctx_attn(q, k, v, causal=causal)
            torch.xpu.synchronize(device=device)
    end.record()

    torch.xpu.synchronize(device=device)
    elapse = begin.elapsed_time(end) / 1000.0  # ms -> s

    if rank == 0 and log:
        color_print(f"{num_iter / elapse:.3f} iter/s, {elapse:.3f} sec")


if __name__ == "__main__":
    # Run with: torchrun --nproc_per_node=<N> benchmark/benchmark_longctx_xpu.py
    dist.init_process_group("xccl")
    rank = dist.get_rank()

    if rank == 0:
        color_print(
            f"ring_impl_type: {args.ring_impl_type}. "
            f"nheads: {args.nheads} head_size: {args.head_size} seq_len: {args.seq_len} "
            f"ulysses_degree: {args.ulysses_degree} use_ulysses_lowdim: {args.use_ulysses_lowdim} "
            f"use_ulysses: {args.use_ulysses} causal: {not args.no_causal}"
        )

    benchmark(log=False)
    benchmark(log=True)
    dist.destroy_process_group()
