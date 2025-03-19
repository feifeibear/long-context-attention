import torch
import torch.distributed as dist
from yunchang import (
    LongContextAttentionQKVPacked, 
    set_seq_parallel_pg, 
    EXTRACT_FUNC_DICT, 
    RING_IMPL_QKVPACKED_DICT
)
from yunchang.kernels import AttnType


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

import os

def get_local_rank():
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    return local_rank

def test(ring_impl_type="zigzag"):

    rank = dist.get_rank()
    local_rank = get_local_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")
    print(f"rank {rank} local_rank {local_rank} world_size {world_size}")

    batch_size = 2
    seqlen = 1024
    nheads = 8
    d = 32
    dropout_p = 0.0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    sp_ulysses_degree = 2 # min(world_size, nheads)
    sp_ring_degree = world_size // sp_ulysses_degree

    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    longctx_attn = LongContextAttentionQKVPacked(ring_impl_type=ring_impl_type, 
                                                attn_type=AttnType.FA)

    ## prepare input and output tensors

    # global tensors
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    with torch.no_grad():
        dist.broadcast(qkv, src=0)
        dist.broadcast(dout, src=0)

    # sharded tensors for long context attn
    local_qkv = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            qkv, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )
    local_qkv.requires_grad = True

    local_dout = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            dout, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )
    # shared tensors for reference
    local_qkv_ref = local_qkv.detach().clone()
    local_qkv_ref.requires_grad = True

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    print(f"local_qkv shape {local_qkv.shape}")
    local_out = longctx_attn(
        local_qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    from flash_attn import flash_attn_qkvpacked_func
    # local_out = out.chunk(world_size, dim=1)[rank]
    # local_lse = lse.chunk(world_size, dim=-1)[rank]

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out_ref = EXTRACT_FUNC_DICT[ring_impl_type](
        out, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    )

    log("out_ref", local_out_ref, rank0_only=True)
    log("out", local_out, rank0_only=True)

    # log("lse", lse, rank0_only=True)
    log("out diff", local_out - local_out_ref)
    # log("lse diff", local_lse - ring_lse)

    dist.barrier()

    # if rank == 0:
    #     print(local_out_ref)
    #     print(local_out)

    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    # long context attn backward
    local_out.backward(local_dout)
    local_dqkv = local_qkv.grad

    # local ring backward
    out.backward(dout)
    dqkv = qkv.grad

    local_dqkv_ref = EXTRACT_FUNC_DICT[ring_impl_type](
        dqkv, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    )

    log("load_dq", local_dqkv_ref)
    log("dq diff", local_dqkv - local_dqkv_ref)



if __name__ == "__main__":
    dist.init_process_group("nccl")
    for ring_impl_type in ["basic", "zigzag"]:
        print(f"ring_impl_type: {ring_impl_type}")
        test(ring_impl_type)
    if dist.is_initialized():
        dist.destroy_process_group()
