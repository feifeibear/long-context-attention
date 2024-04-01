import torch
from ring_flash_attn import (
    ring_flash_attn_func,
    ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_func,
    stripe_flash_attn_qkvpacked_func,
)


def set_seq_parallel_pg(
    sp_ulysses_degree, sp_ring_degree, rank, world_size, use_ulysses_low=True
):
    num_ulysses_pgs = world_size // sp_ulysses_degree
    num_ring_pgs = world_size // sp_ring_degree

    if use_ulysses_low:
        for i in range(num_ulysses_pgs):
            ulysses_ranks = list(
                range(i * sp_ulysses_degree, (i + 1) * sp_ulysses_degree)
            )
            group = torch.distributed.new_group(ulysses_ranks)
            if rank in ulysses_ranks:
                ulyssess_pg = group

        for i in range(num_ring_pgs):
            ring_ranks = list(range(i, world_size, num_ring_pgs))
            group = torch.distributed.new_group(ring_ranks)
            if rank in ring_ranks:
                ring_pg = group

        return ulyssess_pg, ring_pg

    else:
        for i in range(num_ring_pgs):
            ring_ranks = list(range(i * sp_ring_degree, (i + 1) * sp_ring_degree))
            group = torch.distributed.new_group(ring_ranks)
            if rank in ring_ranks:
                ring_pg = group

        for i in range(num_ulysses_pgs):
            ulysses_ranks = list(range(i, world_size, num_ulysses_pgs))
            group = torch.distributed.new_group(ulysses_ranks)
            if rank in ulysses_ranks:
                ulyssess_pg = group

        return ulyssess_pg, ring_pg


RING_IMPL_DICT = {
    "basic": ring_flash_attn_func,
    "zigzag": zigzag_ring_flash_attn_func,
    "strip": stripe_flash_attn_func,
}

RING_IMPL_QKVPACKED_DICT = {
    "basic": ring_flash_attn_qkvpacked_func,
    "zigzag": zigzag_ring_flash_attn_qkvpacked_func,
    "strip": stripe_flash_attn_qkvpacked_func,
}
