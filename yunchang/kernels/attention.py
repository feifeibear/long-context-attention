from yunchang.globals import HAS_FLASH_ATTN, HAS_FLASH_ATTN_HOPPER
import math
import torch
if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


if HAS_FLASH_ATTN_HOPPER:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn_interface import flash_attn_func as flash3_attn_func
else:
    flash_attn_forward_hopper = None
    flash_attn_func_hopper_backward = None
    flash3_attn_func = None

import torch.nn.functional as F


def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
# def pytorch_attn_forward(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     softmax_scale,
#     causal=True,
# ):
    # TODO(optimize) preprocess to reuse the original code
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    batch_size, nheads, seqlen, d = q.shape
    S = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nheads, seqlen, seqlen)
        S.masked_fill_(causal_mask, float('-inf'))

    # Online softmax
    S_max = torch.max(S, dim=-1, keepdim=True)[0]
    exp_S = torch.exp(S - S_max)
    exp_sum = torch.sum(exp_S, dim=-1, keepdim=True)
    log_sum_exp = torch.log(exp_sum) + S_max
    P = exp_S / exp_sum
    O = torch.matmul(P, v)

    # TODO(optimize) post process to reuse the original code
    O = O.transpose(1, 2)

    lse = log_sum_exp.squeeze(-1)
    return O, lse

# def pytorch_attn_backward(
#     dout,
#     q,
#     k,
#     v,
#     out,
#     softmax_lse,
#     softmax_scale = None,
#     causal=True,
#     *args,
#     **kwargs,
# ):
def pytorch_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    block_dq_buffer=None,  # Add new parameters with default values
    block_dk_buffer=None,
    block_dv_buffer=None,
    dropout_p=0.0,
    softmax_scale=None,
    bwd_causal=None,  # This will replace the original causal parameter
    window_size=None,
    softcap=None,
    alibi_slopes=None,
    deterministic=True,
    rng_state=None,
    *args,
    **kwargs,
):
    # TODO() preprocess to reuse the original code
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = out.transpose(1, 2)
    dout = dout.transpose(1, 2)

    batch_size, nheads, seqlen, d = q.shape
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    # Recreate S and P from log_sum_exp
    S = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    if bwd_causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

    P = torch.exp(S - softmax_lse.unsqueeze(-1))
    # Step 1: Compute dV
    dV = torch.matmul(P.transpose(-2, -1), dout)
    # Step 2: Compute dP
    dP = torch.matmul(dout, v.transpose(-2, -1))
    # Step 3: Compute D
    D = torch.sum(dout * out, dim=-1, keepdim=True)
    # Step 4: Compute dS
    dS = P * (dP - D)
    # Apply causal mask to dS if is_causal is True
    if bwd_causal:
        dS = dS.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), 0)
    # Step 5: Compute dQ
    dQ = torch.matmul(dS, k) * softmax_scale
    # Step 6: Compute dK
    dK = torch.matmul(dS.transpose(-2, -1), q) * softmax_scale

    # TODO() post process to reuse origina; code
    dQ = dQ.transpose(1, 2)
    dK = dK.transpose(1, 2)
    dV = dV.transpose(1, 2)

    return dQ, dK, dV

def flash_attn_forward(q, k, v, 
        dropout_p = 0.0, 
        softmax_scale = None, 
        causal=False, 
        window_size=(-1, -1), 
        softcap=None, 
        alibi_slopes=None, 
        return_softmax=False):
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < '2.6.3':
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
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p = dropout_p,
            softmax_scale = softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
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
    if flash_attn.__version__ < '2.6.3':
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
    else:
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
            window_size[0],  # Pass window_size_left
            window_size[1],  # Pass window_size_right
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
    assert HAS_FLASH_ATTN_HOPPER, f"FlashAttention Hopper is not available"

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
