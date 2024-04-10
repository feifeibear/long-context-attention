
import torch

class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance

class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.ULYSSES_PG = None
        self.RING_PG = None

PROCESS_GROUP = ProcessGroupSingleton()

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
    
    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.RING_PG = ring_pg



