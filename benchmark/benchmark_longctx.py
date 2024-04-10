import torch
import torch.distributed as dist
from yunchang import LongContextAttention, set_seq_parallel_pg
import torch.cuda
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument(
    "--ring_impl_type",
    type=str,
    default="basic",
    choices=["basic", "zigzag", "strip"],
    help="ring attn implementation type",
)
parser.add_argument("--nheads", type=int, default=2, help="head number")
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


def benchmark(num_iter=100, forward_only=True, log=True):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    batch_size = 1
    seqlen = 1024 * 8
    nheads = args.nheads
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    q, k, v = torch.randn(
        3, batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    ).chunk(3, dim=0)
    q = q.squeeze(0)
    k = k.squeeze(0)
    v = v.squeeze(0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    sp_ulysses_degree = min(args.ulysses_degree, world_size)
    sp_ring_degree = world_size // sp_ulysses_degree

    set_seq_parallel_pg(
        sp_ulysses_degree, sp_ring_degree, rank, world_size, args.use_ulysses_lowdim
    )
    longctx_attn = LongContextAttention()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

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
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                )

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
        print(f"{num_iter / time:.3f} iter/s, {time:.3f} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = args.fwd_only

    torch.cuda.empty_cache()
    if rank == 0:
        color_print(
            f"# long context attention. ulysses_degree : {args.ulysses_degree} fwd_only {forward_only} use_ulysses_lowdim {args.use_ulysses_lowdim}"
        )
    torch.cuda.empty_cache()
    benchmark(forward_only=forward_only, log=False)
    benchmark(forward_only=forward_only, log=True)
