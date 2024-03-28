import os

from ring_flash_attn.ring_flash_attn import ring_flash_attn_func
import torch
import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_qkvpacked_func
from ds_ulysses_attn.ulysses_attn_layer import DistributedAttention
from deepspeed import init_distributed

from flash_attn import flash_attn_func


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

    init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 2
    seqlen = 3816
    nheads = 4
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0
    # assert batch_size == 1

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_q.requires_grad = True
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_k.requires_grad = True
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
    local_v.requires_grad = True

    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    # prcess_group == sequence_process_group
    sp_pg = dist.new_group(ranks=[i for i in range(world_size)])

    mha = (
        torch.nn.MultiheadAttention(d * nheads // world_size, nheads // world_size)
        .to(device)
        .to(dtype)
    )

    def torch_attn(query_layer, key_layer, value_layer, *args):
        """
        local attn implementations
        Args:
            query_layer : (bs, seqlen, hc/P, hs)
            key_layer : (bs, seqlen, hc/P, hs)
            value_layer : (bs, seqlen, hc/P, hs)
        Returns:
            context_layer : (bs, seqlen, hc/P, hs)
        """
        print(f"query_layer.shape {query_layer.shape}")
        bs, seqlen, split_hc, hs = query_layer.shape
        query_layer = query_layer.reshape(bs, seqlen, -1)
        key_layer = key_layer.reshape(bs, seqlen, -1)
        value_layer = value_layer.reshape(bs, seqlen, -1)

        context_layer, _ = mha(query_layer, key_layer, value_layer, *args)

        context_layer = context_layer.reshape(bs, seqlen, -1, hs)

        return context_layer

    # attn = torch_attn
    # attn = ring_flash_attn_func

    # warp flash_attn to match the attn signature in `DistributedAttention`
    def flash_attn_impl(q, k, v, **args):
        out, _, _ = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        return out

    attn = flash_attn_impl

    dist_attn = DistributedAttention(attn, sp_pg, scatter_idx=2, gather_idx=1)

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses forward:")
        print("#" * 30)

    local_out = dist_attn(
        local_q.reshape(batch_size, seqlen // world_size, nheads, d),
        local_k.reshape(batch_size, seqlen // world_size, nheads, d),
        local_v.reshape(batch_size, seqlen // world_size, nheads, d),
    )

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses backward:")
        print("#" * 30)

    local_out.backward(local_dout)

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)
    # reference, a local flash attn
    out_ref, _, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)

    out_ref.backward(dout)

    dist.barrier()

    # check correctness

    local_out_ref = out_ref.chunk(world_size, dim=1)[rank]

    log("out", local_out, rank0_only=True)
    log("out diff", local_out_ref - local_out)

    local_dq_ref = q.grad.chunk(world_size, dim=1)[rank]
    log("load_dq", local_q.grad)
    log("dq diff", local_dq_ref - local_q.grad)

    local_dk_ref = k.grad.chunk(world_size, dim=1)[rank]
    log("load_dk", local_k.grad)
    log("dk diff", local_dk_ref - local_k.grad)

    local_dv_ref = v.grad.chunk(world_size, dim=1)[rank]
    log("load_dk", local_v.grad)
    log("dv diff", local_dv_ref - local_v.grad)
