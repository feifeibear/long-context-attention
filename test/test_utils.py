import math
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from einops import rearrange, repeat


# adpated from flash-attention
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

# adpated from flash-attention
def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def get_causal_mask(
    q_len: int, kv_len: int, device: torch.device
) -> torch.Tensor:
    assert q_len == kv_len
    return torch.triu(
        torch.ones(q_len, kv_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


def _bsnd_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
) -> torch.Tensor:
    dtype = query.dtype
    # [B, S, N, D] -> [B, N, S, D]
    # use fp32 as reference
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    attn = torch.matmul(
        q,
        k.transpose(-1, -2),
    )

    attn *= softmax_scale

    if causal:
        q_len = q.size(-2)
        kv_len = k.size(-2)

        mask = get_causal_mask(
            q_len,
            kv_len,
            query.device,
        )[None, None, :, :]

        attn = attn.masked_fill(mask, float("-inf"))

    score = torch.softmax(attn, dim=-1)

    out = torch.matmul(score, v)

    return out.to(dtype).permute(0, 2, 1, 3).contiguous()


def eager_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
    cu_seq_qlen: Optional[torch.Tensor] = None,
    cu_seq_kv_len: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    print(f"compared with eager attention ({dtype})")
    query = query.to(dtype)
    key = key.to(dtype)
    value = value.to(dtype)

    if query.ndim == 4:
        # BSND
        return _bsnd_attn(
            query,
            key,
            value,
            softmax_scale,
            causal,
        )

    # TND
    assert query.ndim == 3
    assert cu_seq_qlen is not None
    assert cu_seq_kv_len is not None

    assert len(cu_seq_qlen) == len(cu_seq_kv_len)
    assert cu_seq_qlen[0].item() == 0
    assert cu_seq_kv_len[0].item() == 0

    outputs = []

    for i in range(len(cu_seq_qlen) - 1):

        qst = int(cu_seq_qlen[i].item())
        qed = int(cu_seq_qlen[i + 1].item())

        kvst = int(cu_seq_kv_len[i].item())
        kved = int(cu_seq_kv_len[i + 1].item())

        local_q = query[qst:qed].unsqueeze(0)
        local_k = key[kvst:kved].unsqueeze(0)
        local_v = value[kvst:kved].unsqueeze(0)

        local_out = _bsnd_attn(
            local_q,
            local_k,
            local_v,
            softmax_scale,
            causal,
        )

        outputs.append(local_out.squeeze(0))

    return torch.cat(outputs, dim=0)


def make_varlen_input(
    seq_length: Sequence[int],
    num_heads: int,
    dim: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.concat(
        [torch.randn(length, num_heads, dim, dtype=dtype) for length in seq_length]
    )
    k = torch.concat(
        [torch.randn(length, num_heads, dim, dtype=dtype) for length in seq_length]
    )
    v = torch.concat(
        [torch.randn(length, num_heads, dim, dtype=dtype) for length in seq_length]
    )
    dout = torch.concat(
        [torch.randn(length, num_heads, dim, dtype=dtype) for length in seq_length]
    )

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    return q, k, v, dout


def make_bsnd_input(
    batch_size: int,
    seq_length: int,
    num_heads: int,
    dim: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        batch_size, seq_length, num_heads, dim, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seq_length, num_heads, dim, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seq_length, num_heads, dim, dtype=dtype, requires_grad=True
    )
    dout = torch.randn(batch_size, seq_length, num_heads, dim, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)
    return q, k, v, dout


def sequential_print(text: str) -> None:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    for r in range(world_size):
        if r == rank:
            print(text)
        dist.barrier()


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-1,
    rtol: float = 1e-1,
) -> None:
    is_close = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    if dist.is_available() and dist.is_initialized():
        status = torch.tensor(
            int(is_close), dtype=torch.int32, device=actual.device
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        is_close = bool(status.item())
    if not is_close:
        diff = (actual - expected).abs()
        raise AssertionError(
            f"{name} mismatch: max_abs={diff.max().item()}, "
            f"mean_abs={diff.mean().item()}, atol={atol}, rtol={rtol}"
        )
