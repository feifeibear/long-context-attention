import torch
import torch.distributed as dist
from .utils import RingComm, update_out_and_lse
from yunchang.kernels import select_flash_attn_impl, AttnType


def ring_xpu_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    attn_type: AttnType = AttnType.XPU,
    attn_processor=None,
):
    comm = RingComm(process_group)

    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            block_out, block_lse = fn(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class RingXpuFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        softcap,
        return_softmax,
        group,
        attn_type,
        attn_processor,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ring_xpu_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            attn_type=attn_type,
            attn_processor=attn_processor,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.group = group
        ctx.attn_type = attn_type
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        raise RuntimeError("Backward pass is not supported for XPU flash attention")


def ring_xpu_flash_attn_func(
    q,
    k,
    v,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.XPU,
    attn_processor=None,
):
    assert dropout_p == 0.0, "dropout not supported"
    assert alibi_slopes is None, "alibi_slopes not supported"
    return RingXpuFlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        softcap,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
    )
