from yunchang import (
    AsyncLongContextAttention,
    LongContextAttention,
    set_seq_parallel_pg,
    EXTRACT_FUNC_DICT
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

# test it with:
# torchrun --nproc_per_node=4  test/test_hybrid_attn_v2.py
if __name__ == "__main__":
    torch.random.manual_seed(0)

    use_bwd = True
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size == 4, f"torchrun --nproc_per_node=4  test/test_hybrid_attn_v2.py"
    # Inference mainly uses fp16; ROCM flash attention with bf16 precision is slightly larger, will be fixed soon 
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 2
    seqlen = 1024
    nheads = 4
    d = 128
    dropout_p = 0
    causal = True

    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    ring_impl_type = "zigzag"  # You can change this to "basic" or "zigzag" if needed

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

    # prepare process group for hybrid sequence parallelism
    use_ring_low_dim = True

    sp_ulysses_degree = 2
    sp_ring_degree = world_size // sp_ulysses_degree

    print(
        f"rank {rank}, sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}"
    )

    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    # Use EXTRACT_FUNC_DICT to shard the tensors
    local_q = EXTRACT_FUNC_DICT[ring_impl_type](
        q, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    ).detach().clone()
    local_q.requires_grad = True

    local_k = EXTRACT_FUNC_DICT[ring_impl_type](
        k, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    ).detach().clone()
    local_k.requires_grad = True

    local_v = EXTRACT_FUNC_DICT[ring_impl_type](
        v, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    ).detach().clone()
    local_v.requires_grad = True
    usp_attn = LongContextAttention(ring_impl_type=ring_impl_type)

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses forward:")
        print("#" * 30)

    # common test parameters
    window_size=(-1, -1)
    alibi_slopes, attn_bias = None, None
    dropout_mask = None

    print(f"before usp attn forward: {local_q.shape} {local_k.shape} {local_v.shape}")
    # usp attn forward
    local_out = usp_attn(
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

    # extract local dout
    local_dout = EXTRACT_FUNC_DICT[ring_impl_type](
        dout, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    ).detach().clone()

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses backward:")
        print("#" * 30)

    # usp attn backward
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
    # When checking correctness, use EXTRACT_FUNC_DICT for reference outputs
    local_out_ref = EXTRACT_FUNC_DICT[ring_impl_type](
        out_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    )
    local_out_pt_ref = EXTRACT_FUNC_DICT[ring_impl_type](
        out_pt_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    )

    log("local (rank) out", local_out, rank0_only=True)
    log("out (distributed) - out_ref (non-distributed) diff", local_out_ref - local_out)

    # log("out_ref (non-distributed) - out_pt_ref (gpu) diff", local_out_ref - local_out_pt_ref)

    torch.testing.assert_close(local_out, local_out_ref, atol=1e-2, rtol=0)
    # torch.testing.assert_close(out_ref, out_pt_ref, atol=1e-2, rtol=0)

    if use_bwd:
        local_dq_ref = EXTRACT_FUNC_DICT[ring_impl_type](
            q.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        log("load_dq", local_q.grad)
        log("dq diff", local_dq_ref - local_q.grad)

        local_dk_ref = EXTRACT_FUNC_DICT[ring_impl_type](
            k.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        log("load_dk", local_k.grad)
        log("dk diff", local_dk_ref - local_k.grad)

        local_dv_ref = EXTRACT_FUNC_DICT[ring_impl_type](
            v.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        log("load_dv", local_v.grad)
        log("dv diff", local_dv_ref - local_v.grad)

    if dist.is_initialized():
        dist.destroy_process_group()
