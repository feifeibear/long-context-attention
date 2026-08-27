from importlib import import_module
from typing import Any
from unittest.mock import patch

import torch

from yunchang.hybrid.utils import (
    RING_IMPL_QKVPACKED_DICT,
    RING_IMPL_VARLEN_QKVPACKED_DICT,
)

basic_fixed = import_module("yunchang.ring.ring_npu_flash_attn")
basic_varlen = import_module("yunchang.ring.ring_npu_flash_attn_varlen")
zigzag_fixed = import_module("yunchang.ring.zigzag_ring_npu_flash_attn")
zigzag_varlen = import_module("yunchang.ring.zigzag_ring_npu_flash_attn_varlen")


def _fake_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    return q + 2 * k + 3 * v


def _assert_packed_grad(qkv: torch.Tensor, packed_dim: int) -> None:
    if qkv.grad is None:
        raise AssertionError("packed QKV gradient was not populated")
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=qkv.dtype)
    shape = [1] * qkv.ndim
    shape[packed_dim] = 3
    torch.testing.assert_close(qkv.grad, expected.reshape(shape).expand_as(qkv))


def test_fixed_npu_qkvpacked_wrappers() -> None:
    cases = (
        (
            basic_fixed,
            "ring_npu_flash_attn_func",
            basic_fixed.ring_npu_flash_attn_qkvpacked_func,
        ),
        (
            zigzag_fixed,
            "zigzag_ring_flash_attn_npu_func",
            zigzag_fixed.zigzag_ring_npu_flash_attn_qkvpacked_func,
        ),
    )
    for module, target, wrapper in cases:
        qkv = torch.randn(2, 4, 3, 2, 8, requires_grad=True)
        with patch.object(module, target, side_effect=_fake_attention):
            out = wrapper(qkv, dropout_p=0.1, softmax_scale=0.5, causal=True)
            out.sum().backward()
        _assert_packed_grad(qkv, packed_dim=2)


def test_varlen_npu_qkvpacked_wrappers() -> None:
    cases = (
        (
            basic_varlen,
            "ring_npu_flash_attn_varlen_func",
            basic_varlen.ring_npu_flash_attn_varlen_qkvpacked_func,
        ),
        (
            zigzag_varlen,
            "zigzag_ring_npu_flash_attn_varlen_func",
            zigzag_varlen.zigzag_ring_npu_flash_attn_varlen_qkvpacked_func,
        ),
    )
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.long)
    for module, target, wrapper in cases:
        qkv = torch.randn(6, 3, 2, 8, requires_grad=True)
        with patch.object(module, target, side_effect=_fake_attention) as mocked:
            out = wrapper(
                qkv,
                cu_seqlens,
                max_seqlen=4,
                dropout_p=0.1,
                softmax_scale=0.5,
                causal=True,
            )
            assert mocked.call_args.kwargs["cu_seqlens"] is cu_seqlens
            out.sum().backward()
        _assert_packed_grad(qkv, packed_dim=1)


def test_npu_qkvpacked_registries() -> None:
    assert (
        RING_IMPL_QKVPACKED_DICT["basic_npu"]
        is basic_fixed.ring_npu_flash_attn_qkvpacked_func
    )
    assert (
        RING_IMPL_QKVPACKED_DICT["zigzag_npu"]
        is zigzag_fixed.zigzag_ring_npu_flash_attn_qkvpacked_func
    )
    assert (
        RING_IMPL_VARLEN_QKVPACKED_DICT["basic_npu"]
        is basic_varlen.ring_npu_flash_attn_varlen_qkvpacked_func
    )
    assert (
        RING_IMPL_VARLEN_QKVPACKED_DICT["zigzag_npu"]
        is zigzag_varlen.zigzag_ring_npu_flash_attn_varlen_qkvpacked_func
    )
