from yunchang import (
    AsyncLongContextAttention,
    LongContextAttention,
    set_seq_parallel_pg,
)
import torch
import torch.distributed as dist
from flash_attn import flash_attn_func

from test_utils import attention_ref

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


if __name__ == "__main__":
    torch.random.manual_seed(0)

    use_bwd = False
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Inference mainly uses fp16; ROCM flash attention with bf16 precision is slightly larger, will be fixed soon 
    dtype = torch.float16
    device = torch.device(f"cuda:{rank}")

    batch_size = 2
    seqlen = 3816
    nheads = 2
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    use_async_all_to_all = True
    assert seqlen % world_size == 0
    assert d % 8 == 0
    # assert batch_size == 1

    # Prepare inputs
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

    # prepare process group for hybrid sequence parallelism
    use_ring_low_dim = True

    sp_ulysses_degree = min(nheads, world_size)
    sp_ring_degree = world_size // sp_ulysses_degree
    print(
        f"rank {rank}, sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}"
    )

    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    if use_async_all_to_all:
        hybrid_seq_parallel_attn = AsyncLongContextAttention()
    else:
        hybrid_seq_parallel_attn = LongContextAttention()

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses forward:")
        print("#" * 30)

    # common test parameters
    window_size=(-1, -1)
    alibi_slopes, attn_bias = None, None
    dropout_mask = None

    local_out = hybrid_seq_parallel_attn(
        local_q,
        local_k,
        local_v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses backward:")
        print("#" * 30)

    if use_bwd:
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
        window_size=window_size,
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    out_pt_ref, attn_pt_ref = attention_ref(
        q,
        k,
        v,
        None,
        None,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
    )

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)

    if use_bwd:
        out_ref.backward(dout)

    dist.barrier()

    # check correctness

    local_out_ref = out_ref.chunk(world_size, dim=1)[rank]
    local_out_pt_ref = out_ref.chunk(world_size, dim=1)[rank]

    log("local (rank) out", local_out, rank0_only=True)
    log("out (distributed) - out_ref (non-distributed) diff", local_out_ref - local_out)
    log("out_ref (non-distributed) - out_pt_ref (gpu) diff", local_out_ref - local_out_pt_ref)

    torch.testing.assert_close(local_out, local_out_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(out_ref, out_pt_ref, atol=1e-2, rtol=0)

    if use_bwd:
        local_dq_ref = q.grad.chunk(world_size, dim=1)[rank]
        log("load_dq", local_q.grad)
        log("dq diff", local_dq_ref - local_q.grad)

        local_dk_ref = k.grad.chunk(world_size, dim=1)[rank]
        log("load_dk", local_k.grad)
        log("dk diff", local_dk_ref - local_k.grad)

        local_dv_ref = v.grad.chunk(world_size, dim=1)[rank]
        log("load_dk", local_v.grad)
        log("dv diff", local_dv_ref - local_v.grad)
