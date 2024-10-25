import torch
import torch.distributed as dist
from yunchang import UlyssesAttention

from flash_attn import flash_attn_func
import torch.cuda


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
    nheads = 8
    d = 64
    dropout_p = 0
    causal = False
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

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires GPU.")

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_q.requires_grad = True
    local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
    local_k.requires_grad = True
    local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
    local_v.requires_grad = True

    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    # Move tensors to the correct device and ensure they're contiguous
    local_q = local_q.to(device).contiguous()
    local_k = local_k.to(device).contiguous()
    local_v = local_v.to(device).contiguous()
    local_dout = local_dout.to(device).contiguous()

    # prcess_group == sequence_process_group
    sp_pg = None #dist.new_group(ranks=[i for i in range(world_size)])

    dist_attn = UlyssesAttention(sp_pg, use_fa=True, use_sage=True)

    # Add mode parameter
    mode = 'fwd-only'
    # mode = 'fwd-bwd'

    if rank == 0:
        print("#" * 30)
        print(f"# ds-ulysses {mode}:")
        print("#" * 30)

    if mode == 'fwd-only':
        with torch.no_grad():
            local_out = dist_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=True,
            )
    else:
        local_out = dist_attn(
            local_q,
            local_k,
            local_v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )

        if rank == 0:
            print("#" * 30)
            print("# ds-ulysses backward:")
            print("#" * 30)

        local_out.backward(local_dout)

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print(f"# local {mode}:")
        print("#" * 30)

    # Ensure reference tensors are on the correct device
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    dout = dout.to(device)

    # reference, a local flash attn
    if mode == 'fwd-only':
        with torch.no_grad():
            out_ref, _, _ = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=True,
            )
    else:
        out_ref, _, _ = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        out_ref.backward(dout)

    dist.barrier()

    # check correctness
    local_out_ref = out_ref.chunk(world_size, dim=1)[rank]

    log("out", local_out, rank0_only=True)
    log("out diff", local_out_ref - local_out)

    if mode != 'fwd-only':
        local_dq_ref = q.grad.chunk(world_size, dim=1)[rank]
        log("load_dq", local_q.grad)
        log("dq diff", local_dq_ref - local_q.grad)

        local_dk_ref = k.grad.chunk(world_size, dim=1)[rank]
        log("load_dk", local_k.grad)
        log("dk diff", local_dk_ref - local_k.grad)

        local_dv_ref = v.grad.chunk(world_size, dim=1)[rank]
        log("load_dk", local_v.grad)
        log("dv diff", local_dv_ref - local_v.grad)
