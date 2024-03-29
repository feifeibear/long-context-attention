import os

from ring_flash_attn import ring_flash_attn_qkvpacked_func
import torch
import torch.distributed as dist
from long_context_attn import LongContextAttentionQKVPacked, set_seq_parallel_pg


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
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
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 2
    seqlen = 3816
    nheads = 1
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    sp_ulysses_degree = min(world_size, nheads)
    sp_ring_degree = world_size // sp_ulysses_degree

    ulysses_pg, ring_pg = set_seq_parallel_pg(
        sp_ulysses_degree, sp_ring_degree, rank, world_size
    )
    longctx_attn = LongContextAttentionQKVPacked(ulysses_pg, ring_pg)

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True

    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    local_qkv_ref = local_qkv.detach().clone()
    local_qkv_ref.requires_grad = True

    local_dout_ref = local_dout.detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    print(f"local_qkv shape {local_qkv.shape}")
    out = longctx_attn(
        local_qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    # local_out = out.chunk(world_size, dim=1)[rank]
    # local_lse = lse.chunk(world_size, dim=-1)[rank]

    fn = ring_flash_attn_qkvpacked_func

    ring_out, ring_lse, _ = fn(
        local_qkv_ref,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    log("out", out, rank0_only=True)
    # log("lse", lse, rank0_only=True)
    log("out diff", out - ring_out)
    # log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(local_dout)
    dqkv = local_qkv.grad

    ring_out.backward(local_dout_ref)
    ring_dqkv = local_qkv_ref.grad

    log("load_dq", ring_dqkv)
    log("dq diff", dqkv - ring_dqkv)
