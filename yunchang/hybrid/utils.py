from yunchang.ring import (
    ring_flash_attn_func,
    ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_func,
    stripe_flash_attn_qkvpacked_func,
)

RING_IMPL_DICT = {
    "basic": ring_flash_attn_func,
    "zigzag": zigzag_ring_flash_attn_func,
    "strip": stripe_flash_attn_func,
}

RING_IMPL_QKVPACKED_DICT = {
    "basic": ring_flash_attn_qkvpacked_func,
    "zigzag": zigzag_ring_flash_attn_qkvpacked_func,
    "strip": stripe_flash_attn_qkvpacked_func,
}
