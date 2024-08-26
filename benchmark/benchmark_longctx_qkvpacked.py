from flash_attn import flash_attn_varlen_qkvpacked_func
import torch
import torch.distributed as dist
from yunchang import set_seq_parallel_pg, LongContextAttentionQKVPacked
from yunchang.comm import EXTRACT_FUNC_DICT
import torch.cuda

import argparse

parser = argparse.ArgumentParser(description="args for benchmark.")

parser.add_argument(
    "--ring_impl_type",
    type=str,
    default="basic",
    choices=["basic", "zigzag", "strip"],
    help="ring attn implementation type",
)
parser.add_argument("--nheads", type=int, default=2, help="head number")
parser.add_argument("--head_size", type=int, default=128, help="head number")
parser.add_argument(
    "--seq_len",
    type=int,
    default=4 * 1024,
    help="local sequence length, the global sequence length is seq_len * world_size",
)
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument(
    "--fwd_only", action="store_true", help="benchmark forward pass only"
)
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
args = parser.parse_args()


def color_print(text):
    print("\033[91m {}\033[00m".format(text))

import os
def get_local_rank():
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    return local_rank


def benchmark(num_iter=100, forward_only=True, log=True):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    local_rank = get_local_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    batch_size = args.batch_size
    seqlen = args.seq_len
    nheads = args.nheads
    d = args.head_size

    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0, f"seqlen {seqlen} world_size {world_size}"
    assert d % 8 == 0

    qkv = torch.randn(
        batch_size,
        seqlen * world_size,
        3,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout = torch.randn(
        batch_size, seqlen * world_size, nheads, d, device=device, dtype=dtype
    )

    sp_ulysses_degree = min(args.ulysses_degree, world_size)
    sp_ring_degree = world_size // sp_ulysses_degree

    set_seq_parallel_pg(
        sp_ulysses_degree, sp_ring_degree, rank, world_size, args.use_ulysses_lowdim
    )

    longctx_attn = LongContextAttentionQKVPacked(ring_impl_type=args.ring_impl_type)

    # NOTE() using zigzag and stripe have a special layout.
    qkv = (
        EXTRACT_FUNC_DICT[args.ring_impl_type](
            qkv, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )
    qkv.requires_grad = True
    dout = (
        EXTRACT_FUNC_DICT[args.ring_impl_type](
            dout, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    out = longctx_attn(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
    )
    out.backward(dout)

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = longctx_attn(
                    qkv,
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                )

    else:
        for _ in range(num_iter):
            qkv.grad = None
            out = longctx_attn(
                qkv,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
            out.backward(dout)
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if rank == 0 and log:
        color_print(f"{num_iter / time:.3f} iter/s, {time:.3f} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = args.fwd_only

    torch.cuda.empty_cache()
    if rank == 0:
        color_print(vars(args))
        color_print(
            f"# long context attention qkvpacked {args.ring_impl_type}. ulysses_degree : {args.ulysses_degree} "
            f"fwd_only {forward_only} "
            f"use_ulysses_lowdim {args.use_ulysses_lowdim} "
        )
    benchmark(forward_only=forward_only, log=False)
    benchmark(forward_only=forward_only, log=True)
