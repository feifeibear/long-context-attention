from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
from yunchang import (
    ring_flash_attn_func,
    zigzag_ring_flash_attn_func,
    stripe_flash_attn_func,
)
import torch.cuda

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--nheads", type=int, default=2, help="head number")
parser.add_argument("--head_size", type=int, default=128, help="head number")
parser.add_argument("--seq_len", type=int, default=4 * 1024, help="head number")
parser.add_argument("--group_num", type=int, default=1, help="group number")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument(
    "--fwd_only", action="store_true", help="benchmark forward pass only"
)

args = parser.parse_args()


def color_print(text):
    print("\033[91m {}\033[00m".format(text))


def benchmark(f, num_iter=100, forward_only=True, log=True):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    batch_size = args.batch_size
    seqlen = args.seq_len
    nheads = args.nheads
    d = args.head_size
    group_num = args.group_num

    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0, f"seqlen {seqlen} world_size {world_size}"
    assert d % 8 == 0
    assert nheads % group_num == 0, f"nheads {nheads} group_num {group_num}"

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

    _ = f(
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
    out = f(
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

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(
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

    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            out = f(
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

    for f in [
        flash_attn_func,
        ring_flash_attn_func,
        zigzag_ring_flash_attn_func,
        stripe_flash_attn_func,
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            color_print(f"# {f.__name__} fwd_only {forward_only}")
        benchmark(f, forward_only=forward_only, log=False)
        benchmark(f, forward_only=forward_only, log=True)
