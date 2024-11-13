from .attention import (
    flash_attn_forward, 
    flash_attn_backward, 
    flash_attn3_func_forward, 
    flash_attn3_func_backward, 
    flash3_attn_func,
    torch_attn
)
from enum import Enum, auto
from flash_attn import flash_attn_func

class FlashAttentionImpl(Enum):
    FA = "fa"
    FA3 = "fa3"
    TORCH = "torch"

def select_flash_attn_impl(impl_type: FlashAttentionImpl, stage : str = "fwd-bwd"):
    if impl_type == FlashAttentionImpl.FA:
        if stage == "fwd-only":
            return flash_attn_forward
        elif stage == "bwd-only":
            return flash_attn_backward
        elif stage == "fwd-bwd":
            print(f"flash_attn_func: {flash_attn_func} here")
            return flash_attn_func
        
    elif impl_type == FlashAttentionImpl.FA3:
        if stage == "fwd-only":
            return flash_attn3_func_forward
        elif stage == "bwd-only":
            return flash_attn3_func_backward
        elif stage == "fwd-bwd":
            def fn(q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                *args, **kwargs
            ):
                # (q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1),
                # deterministic=False, descale_q=None, descale_k=None, descale_v=None, gqa_parallel=False)
                assert softmax_scale is not None, f"softmax_scale is required for FA3"
                assert dropout_p == 0.0, f"dropout_p: {dropout_p} is not supported for FA3"
                return flash3_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
        
            return fn

    elif impl_type == FlashAttentionImpl.TORCH:
        if stage == "fwd-bwd":
            return torch_attn
        else:
            raise ValueError(f"Torch fwd-only and bwd-only is not supported")
    else:
        raise ValueError(f"Unknown flash attention implementation: {impl_type}")

__all__ = ["flash_attn_forward", "flash_attn_backward", "flash_attn3_func_forward", "flash_attn3_func_forward", "FlashAttentionImpl"]