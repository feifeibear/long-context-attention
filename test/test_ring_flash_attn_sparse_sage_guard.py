import os
import sys

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    import yunchang.ring.ring_flash_attn as ring_flash_attn
    from yunchang.kernels import AttnType

    # SparseAttentionMeansim is GPU-only, so stub the kernel selection and
    # check that this guard rejects the unsupported combination before it
    # would ever be reached.
    called = []

    def stub_select(*args, **kwargs):
        called.append(1)

        def fn(*a, **k):
            t = torch.zeros(1, 2, 1, 4)
            return t, t

        return fn

    ring_flash_attn.select_flash_attn_impl = stub_select

    q = k = v = torch.randn(1, 2, 1, 4)
    pg = dist.new_group(list(range(world_size)))

    raised = None
    try:
        ring_flash_attn.ring_flash_attn_forward(
            pg, q, k, v, softmax_scale=1.0, attn_type=AttnType.SPARSE_SAGE
        )
    except RuntimeError as e:
        raised = str(e)

    dist.barrier()
    dist.destroy_process_group()

    ok = raised == "Sparse Sage attention does not support ring degree > 1." and not called
    print(f"rank {rank}: raised={raised!r} kernel_called={bool(called)} ok={ok}")
    if rank == 0:
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
