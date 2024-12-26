# adapted from https://github.com/huggingface/picotron/blob/main/picotron/context_parallel/context_parallel.py
import math
import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple
from yunchang.kernels import select_flash_attn_impl, FlashAttentionImpl
from .utils import RingComm

def ring_pytorch_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: FlashAttentionImpl = FlashAttentionImpl.FA,
):
# def ring_attention(process_group, q, k, v, sm_scale, is_causal):
    return RingAttentionFunc.apply(group, q, k, v, softmax_scale, causal)

class RingAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, q, k, v, sm_scale, is_causal):
        comm = RingComm(group)
        #TODO(fmom): add flex attention
        #TODO(fmom): add flash attention
        #TODO(fmom): Find a better to save these tensors without cloning
        k_og = k.clone()
        v_og = v.clone()
        out, lse = None, None
        next_k, next_v = None, None

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            if not is_causal or step <= comm.rank:
                block_out, block_lse  = ring_pytorch_attn_forward(
                    q, k, v, softmax_scale = sm_scale, causal = is_causal and step == 0
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                
            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)
        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout, *args):

        q, k, v, out, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        kv_comm = RingComm(ctx.group)
        d_kv_comm = RingComm(ctx.group)

        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None
        
        block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        next_dk, next_dv = None, None
        next_k, next_v = None, None

        for step in range(kv_comm.world_size):
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            if step <= kv_comm.rank or not is_causal:
                bwd_causal = is_causal and step == 0

                block_dq_buffer, block_dk_buffer, block_dv_buffer = ring_pytorch_attn_backward(
                    dout, q, k, v, out, softmax_lse = softmax_lse, softmax_scale = sm_scale, causal = bwd_causal
                )

                if dq is None:
                    dq = block_dq_buffer.to(torch.float32)
                    dk = block_dk_buffer.to(torch.float32)
                    dv = block_dv_buffer.to(torch.float32)
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        return dq, next_dk, next_dv, None, None

def ring_pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: FlashAttentionImpl = None,
):
# def ring_attention_forward(q, k, v, sm_scale, is_causal):
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
    return O, log_sum_exp.squeeze(-1)

def ring_pytorch_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: FlashAttentionImpl = FlashAttentionImpl.TORCH,
):
# def ring_attention_backward(dO, Q, K, V, O, softmax_lse, sm_scale, is_causal):
    batch_size, nheads, seqlen, d = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))
    
    # Recreate S and P from log_sum_exp
    S = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    if causal:
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
    if causal:
        dS = dS.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), 0)
    # Step 5: Compute dQ
    dQ = torch.matmul(dS, k) * softmax_scale
    # Step 6: Compute dK
    dK = torch.matmul(dS.transpose(-2, -1), q) * softmax_scale
    return dQ, dK, dV

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    def _update(current_out, current_lse):
        # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
        # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
        # For additional context and discussion, please refer to:
        # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
        current_out = current_out - F.sigmoid(block_lse - current_lse) * (current_out - block_out)
        current_lse = current_lse - F.logsigmoid(current_lse - block_lse)
        return current_out, current_lse
    
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(dim=-1)

    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        return block_out, block_lse

    if slice_ is not None:
        out[slice_], lse[slice_] = _update(out[slice_], lse[slice_])
    else:
        out, lse = _update(out, lse)
        
    return out, lse
