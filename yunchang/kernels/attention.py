try:
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn_interface import flash_attn_func as flash3_attn_func
    HAS_FLASH_ATTN_HOPPER = True
except ImportError:
    HAS_FLASH_ATTN_HOPPER = False

import torch.nn.functional as F

def torch_attn(q, k, v, dropout_p = 0.0, 
            softmax_scale = None, 
            causal=False, 
            *args, **kwargs):
    batch_size, seq_len, hs, hd = q.size()
    query = q.view(batch_size, -1, hs, hd).transpose(1, 2)
    key = k.view(batch_size, -1, hs, hd).transpose(1, 2)
    value = v.view(batch_size, -1, hs, hd).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=causal
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, hs, hd
    )
    hidden_states = hidden_states.to(query.dtype)
    return hidden_states

def flash_attn_forward(q, k, v, 
        dropout_p = 0.0, 
        softmax_scale = None, 
        causal=False, 
        window_size=(-1, -1), 
        softcap=None, 
        alibi_slopes=None, 
        return_softmax=False):
    assert HAS_FLASH_ATTN
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p = dropout_p,
        softmax_scale = softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax,
    )
    return block_out, block_lse

def flash_attn_backward(dout, q, k, v, out, softmax_lse, block_dq_buffer, block_dk_buffer, block_dv_buffer, dropout_p, softmax_scale, 
    bwd_causal, window_size, softcap, alibi_slopes, deterministic, rng_state):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    assert HAS_FLASH_ATTN
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        block_dq_buffer,
        block_dk_buffer,
        block_dv_buffer,
        dropout_p,
        softmax_scale,
        bwd_causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        rng_state,
    )

def flash_attn3_func_forward(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax):
    assert HAS_FLASH_ATTN_HOPPER
    # current signature of flash_attn_forward_hopper:
    # (q, k, v, softmax_scale, causal, window_size, descale_q=None, descale_k=None, descale_v=None, gqa_parallel=False)
    out, q, k, v, out_padded, softmax_lse, S_dmask = flash_attn_forward_hopper(
        q, k, v, softmax_scale, causal, window_size
    )
    return out, softmax_lse

def flash_attn3_func_backward(dout, q, k, v, out, softmax_lse, 
                                    block_dq_buffer, block_dk_buffer, block_dv_buffer, 
                                    dropout_p, softmax_scale, 
                                    bwd_causal, window_size, softcap, alibi_slopes, deterministic, rng_state):
    # (dout, q, k, v, out, softmax_lse, dq, dk, dv, softmax_scale, causal):
    assert HAS_FLASH_ATTN_HOPPER

    flash_attn_func_hopper_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        block_dq_buffer,
        block_dk_buffer,
        block_dv_buffer,
        softmax_scale,
        bwd_causal,
        window_size,
        deterministic,
    )
