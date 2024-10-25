import pytest
import torch

import triton
import triton.language as tl

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [32, 64, 128, 256]\
#     for BN in [32, 64, 128, 256]\
#     for s in ([1] if is_hip() else [3, 4, 7])\
#     for w in [4, 8]\
# ]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [32]\
    for BN in [32]\
    for s in [3]\
    for w in [4]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True