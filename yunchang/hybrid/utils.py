from yunchang.ring import (
    ring_flash_attn_func,
    ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_func,
    stripe_flash_attn_qkvpacked_func,
    ring_pytorch_attn_func,
    ring_flashinfer_attn_func,
    ring_flashinfer_attn_qkvpacked_func,
    ring_npu_flash_attn_func,
    ring_npu_flash_attn_qkvpacked_func,
    ring_npu_flash_attn_varlen_func,
    ring_npu_flash_attn_varlen_qkvpacked_func,
    zigzag_ring_flash_attn_npu_func,
    zigzag_ring_npu_flash_attn_qkvpacked_func,
    zigzag_ring_npu_flash_attn_varlen_func,
    zigzag_ring_npu_flash_attn_varlen_qkvpacked_func,
    ring_xpu_flash_attn_func,
)

RING_IMPL_DICT = {
    "basic": ring_flash_attn_func,
    "zigzag": zigzag_ring_flash_attn_func,
    "strip": stripe_flash_attn_func,
    "basic_pytorch": ring_pytorch_attn_func,
    "basic_flashinfer": ring_flashinfer_attn_func,
    "basic_npu": ring_npu_flash_attn_func,
    "zigzag_npu": zigzag_ring_flash_attn_npu_func,
    "basic_xpu": ring_xpu_flash_attn_func,
}

# Varlen ring functions have a different required argument (cu_seqlens), so
# keep them out of the fixed-length Hybrid API registry.
RING_IMPL_VARLEN_DICT = {
    "basic_npu": ring_npu_flash_attn_varlen_func,
    "zigzag_npu": zigzag_ring_npu_flash_attn_varlen_func,
}

RING_IMPL_VARLEN_QKVPACKED_DICT = {
    "basic_npu": ring_npu_flash_attn_varlen_qkvpacked_func,
    "zigzag_npu": zigzag_ring_npu_flash_attn_varlen_qkvpacked_func,
}

RING_IMPL_QKVPACKED_DICT = {
    "basic": ring_flash_attn_qkvpacked_func,
    "zigzag": zigzag_ring_flash_attn_qkvpacked_func,
    "strip": stripe_flash_attn_qkvpacked_func,
    "basic_flashinfer": ring_flashinfer_attn_qkvpacked_func,
    "basic_npu": ring_npu_flash_attn_qkvpacked_func,
    "zigzag_npu": zigzag_ring_npu_flash_attn_qkvpacked_func,
}
