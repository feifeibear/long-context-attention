import torch
import torch.distributed as dist

from yunchang.globals import PROCESS_GROUP


def stripe_extract_local(value, rank, world_size, rd, ud, *args, **kwargs):
    # ud at the highest dim
    input_dim = value.dim()
    if input_dim == 5:
        batch_size, seqlen, _, nheads, d = value.shape
    elif input_dim == 4:
        batch_size, seqlen, nheads, d = value.shape
    else:
        raise ValueError("value dim should be 4 or 5")

    # (ud, L, rd)
    value = value.reshape(batch_size, seqlen // rd, rd, -1).contiguous()
    value = value.transpose(1, 2).reshape(batch_size, seqlen, -1).contiguous()
    value = value.chunk(world_size, dim=1)[rank]

    if input_dim == 5:
        value = value.reshape(batch_size, seqlen // world_size, 3, nheads, d)
    elif input_dim == 4:
        value = value.reshape(batch_size, seqlen // world_size, nheads, d)
    return value


def basic_extract_local(value, rank, world_size, *args, **kwargs):
    return value.chunk(world_size, dim=1)[rank].detach().clone()


def zigzag_extract_local(value, rank, world_size, rd, ud, dim=1, *args, **kwargs):
    input_dim = value.dim()
    if input_dim == 5:
        batch_size, seqlen, _, nheads, d = value.shape
    elif input_dim == 4:
        batch_size, seqlen, nheads, d = value.shape
    else:
        raise ValueError("value dim should be 4 or 5")

    value_chunks = value.chunk(2 * rd, dim=dim)

    # TODO assert ulyssess on low dim
    r_rank = dist.get_rank(group=PROCESS_GROUP.RING_PG)
    u_rank = dist.get_rank(group=PROCESS_GROUP.ULYSSES_PG)

    local_value = torch.cat(
        [value_chunks[r_rank], value_chunks[2 * rd - r_rank - 1]], dim=dim
    ).chunk(ud, dim=dim)[u_rank]

    if input_dim == 5:
        local_value = local_value.reshape(
            batch_size, seqlen // world_size, 3, nheads, d
        )
    elif input_dim == 4:
        local_value = local_value.reshape(batch_size, seqlen // world_size, nheads, d)

    return local_value.contiguous()


EXTRACT_FUNC_DICT = {
    "basic": basic_extract_local,
    "strip": stripe_extract_local,
    "zigzag": zigzag_extract_local,
}
