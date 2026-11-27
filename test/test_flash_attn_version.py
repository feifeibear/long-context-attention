"""Check that flash_attn_forward/backward pick the right flash-attn call signature.

flash-attn 2.7.0 replaced `window_size` with `window_size_left` and
`window_size_right`. Runs on CPU without flash-attn installed:

    python test/test_flash_attn_version.py
"""
import sys
import types

import torch

import yunchang.kernels.attention as attention

# (installed version, whether it takes window_size_left/window_size_right)
CASES = [
    ("2.6.0.post1", False),
    ("2.6.3", False),
    ("2.7.0.post2", True),
    ("2.8.3", True),
    ("2.10.0", True),
]


def forward_kwargs(version):
    seen = {}

    def fake_forward(*args, **kwargs):
        seen.update(kwargs)
        out = torch.zeros(1)
        if "window_size_left" in kwargs:
            return out, out, None, None
        return out, None, None, None, None, out, None, None

    attention.flash_attn = types.SimpleNamespace(__version__=version)
    attention._flash_attn_forward = fake_forward
    q = torch.zeros(1, 1, 1, 4)
    attention.flash_attn_forward(q, q, q, window_size=(-1, -1))
    return seen


def backward_args(version):
    seen = []
    attention.flash_attn = types.SimpleNamespace(__version__=version)
    attention._flash_attn_backward = lambda *args: seen.extend(args)
    q = torch.zeros(1, 1, 1, 4)
    attention.flash_attn_backward(
        q, q, q, q, q, q, q, q, q, 0.0, None, False, (-1, -1), 0.0, None, False, None
    )
    return seen


def main():
    attention.HAS_FLASH_ATTN = True
    failures = []
    for version, split in CASES:
        kwargs = forward_kwargs(version)
        forward_ok = ("window_size_left" in kwargs) == split and ("window_size" in kwargs) != split
        args = backward_args(version)
        # The window size sits right after the causal flag: one tuple, or two ints.
        backward_ok = (args[12] == -1 and args[13] == -1) if split else args[12] == (-1, -1)
        ok = forward_ok and backward_ok
        print(f"flash-attn {version}: split window_size expected={split} "
              f"forward_ok={forward_ok} backward_ok={backward_ok}")
        if not ok:
            failures.append(version)
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
