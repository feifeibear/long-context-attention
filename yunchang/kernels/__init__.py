from .attention import (
    flash_attn_forward, 
    flash_attn_backward, 
    flash_attn3_func_forward, 
    flash_attn3_func_backward, 
    torch_attn,
    HAS_FLASH_ATTN_HOPPER
)
from enum import Enum, auto

from yunchang.globals import HAS_FLASH_ATTN

if HAS_FLASH_ATTN:
    from flash_attn import flash_attn_func

class FlashAttentionImpl(Enum):
    FA = "fa"
    FA3 = "fa3"
    TORCH = "torch"

    @classmethod
    def from_string(cls, s: str):
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"'{s}' is not a valid {cls.__name__}")

def select_flash_attn_impl(impl_type: FlashAttentionImpl, stage : str = "fwd-bwd"):
    if impl_type == FlashAttentionImpl.FA:
        if stage == "fwd-only":
            return flash_attn_forward
        elif stage == "bwd-only":
            return flash_attn_backward
        elif stage == "fwd-bwd":
            assert HAS_FLASH_ATTN, "FlashAttention is not available"
            return flash_attn_func
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
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
                assert HAS_FLASH_ATTN_HOPPER, "FlashAttention3 is not available! install it from https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release"
                # (q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1),
                # deterministic=False, descale_q=None, descale_k=None, descale_v=None, gqa_parallel=False)
                from .attention import flash3_attn_func
                assert softmax_scale is not None, f"softmax_scale is required for FA3"
                assert dropout_p == 0.0, f"dropout_p: {dropout_p} is not supported for FA3"
                return flash3_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
        
            return fn
        else:
            raise ValueError(f"Unknown stage: {stage}")

    elif impl_type == FlashAttentionImpl.TORCH:
        if stage == "fwd-bwd" or stage == "fwd-only":
            return torch_attn
        else:
            raise ValueError(f"FlashAttentionImpl.TORCH: bwd-only is not supported")
    else:
        raise ValueError(f"Unknown flash attention implementation: {impl_type}")

__all__ = ["flash_attn_forward", "flash_attn_backward", "flash_attn3_func_forward", "flash_attn3_func_forward", "FlashAttentionImpl"]