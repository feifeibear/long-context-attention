import torch


def set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size):
    num_ulysses_pgs = world_size // sp_ulysses_degree
    for i in range(num_ulysses_pgs):
        ulysses_ranks = list(range(i * sp_ulysses_degree, (i + 1) * sp_ulysses_degree))
        group = torch.distributed.new_group(ulysses_ranks)
        if rank in ulysses_ranks:
            ulyssess_pg = group

    num_ring_pgs = world_size // sp_ring_degree
    for i in range(num_ring_pgs):
        ring_ranks = list(range(i, world_size, num_ring_pgs))
        group = torch.distributed.new_group(ring_ranks)
        if rank in ring_ranks:
            ring_pg = group
            print(f"{rank} in {ring_ranks}")

    return ulyssess_pg, ring_pg
