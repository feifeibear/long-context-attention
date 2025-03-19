import torch
import torch.distributed as dist
from yunchang import (
    AsyncLongContextAttention,
    LongContextAttention,
    set_seq_parallel_pg,
    UlyssesAttention,
)
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
parser.add_argument("--head_size", type=int, default=128, help="head size")
parser.add_argument("--seq_len", type=int, default=4 * 1024, help="sequence length")
parser.add_argument("--group_num", type=int, default=1, help="group number")
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
    "--use_qkvpack",
    action="store_true",
    default=False,
    help="pack qkv before all-to-all",
)
parser.add_argument(
    "--ulysses_degree",
    type=int,
    default=1,
    help="ulysses attention sequence parallel degree",
)
parser.add_argument(
    "--use_profiler",
    action="store_true",
    default=False,
    help="use torch profiler",
)
parser.add_argument(
    "--use_ulysses",
    action="store_true",
    default=False,
    help="use ulysses",
)
parser.add_argument(
    "--attn_type",
    type=str,
    default="fa",
    choices=["fa", "fa3", "torch"],
    help="attention type",
)
# decault causal=True for LLM. no_causal is for DiT.
parser.add_argument(
    "--no_causal",
    action="store_true",
    default=False,
    help="use no causal attention",
)

args = parser.parse_args()


def color_print(text):
    print("\033[91m {}\033[00m".format(text))


def init_prof(use_profiler):
    activities = []
    activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    from contextlib import nullcontext

    ctx = (
        torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/"),
            record_shapes=True,
            with_stack=True,
        )
        if use_profiler
        else nullcontext()
    )
    return ctx

import os

def get_local_rank():
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    return local_rank

def benchmark(num_iter=10, forward_only=True, log=True, profile=False):
    dtype = torch.float16
    rank = dist.get_rank()
    local_rank = get_local_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    batch_size = args.batch_size
    seqlen = args.seq_len
    nheads = args.nheads
    group_num = args.group_num
    d = args.head_size

    dropout_p = 0.0
    causal = not args.no_causal
    deterministic = False

    assert seqlen % (2 * world_size) == 0, f"seqlen {seqlen} world_size {world_size}"
    assert d % 8 == 0
    assert nheads % group_num == 0, f"nheads {nheads} group_num {group_num}"
    assert (
        nheads // group_num % args.ulysses_degree == 0
    ), f"nheads {nheads}, group_num {group_num}, ulysses_degree {args.ulysses_degree}"

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen,
        nheads // group_num,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen,
        nheads // group_num,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    sp_ulysses_degree = min(args.ulysses_degree, world_size)
    sp_ring_degree = world_size // sp_ulysses_degree

    set_seq_parallel_pg(
        sp_ulysses_degree, sp_ring_degree, rank, world_size, args.use_ulysses_lowdim
    )

    from yunchang.kernels import AttnType
    attn_type = AttnType.from_string(args.attn_type) 
    if args.use_ulysses:
        longctx_attn = UlyssesAttention(attn_type=attn_type)
    else:
        longctx_attn = LongContextAttention(ring_impl_type=args.ring_impl_type, attn_type=attn_type)
        
    out = longctx_attn(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
    )
    if not args.fwd_only:
        out.backward(dout)

    out = longctx_attn(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
    )
    if not args.fwd_only:
        out.backward(dout)

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    ctx = init_prof(profile)

    with ctx as prof:
        if forward_only:
            with torch.no_grad():
                for _ in range(num_iter):
                    _ = longctx_attn(
                        q,
                        k,
                        v,
                        dropout_p=dropout_p,
                        causal=causal,
                        window_size=(-1, -1),
                        softcap=0.0,
                        alibi_slopes=None,
                        deterministic=deterministic,
                        return_attn_probs=False,
                    )

                    torch.cuda.synchronize(device=device)

                    if profile:
                        prof.step()
        else:
            for _ in range(num_iter):
                q.grad = None
                k.grad = None
                v.grad = None
                out = longctx_attn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                )
                out.backward(dout)

                if profile:
                    prof.step()

    end = torch.cuda.Event(enable_timing=True)
    end.record()

    torch.cuda.synchronize(device=device)
    elapse = begin.elapsed_time(end) / 1000.0

    if rank == 0 and log:
        color_print(f"{num_iter / elapse:.3f} iter/s, {elapse:.3f} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = args.fwd_only

    torch.cuda.empty_cache()
    if rank == 0:
        color_print(
            f"ring_impl_type: {args.ring_impl_type}. "
            f"nheads: {args.nheads} head_size: {args.head_size} seq_len: {args.seq_len} "
            f"ulysses_degree : {args.ulysses_degree} fwd_only {forward_only} use_ulysses_lowdim {args.use_ulysses_lowdim}. "
            f"use_qkvpack: {args.use_qkvpack} "
            f"use_ulysses: {args.use_ulysses} "
            f"causal: {not args.no_causal} "
            f"attn_type: {args.attn_type} "
        )
    torch.cuda.empty_cache()
    benchmark(forward_only=forward_only, log=False)
    benchmark(forward_only=forward_only, log=True, profile=args.use_profiler)
    dist.destroy_process_group()