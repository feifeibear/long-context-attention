import pytest
import torch
import pandas as pd
import numpy as np

import triton
import triton.language as tl

from configs import *

from flash_atten_fp import attention
from flash_atten_int8 import attention_int8
from flash_atten_full_int8 import attention_full_int8

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for causal in [False]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) + ["triton-int8"] +
            ["triton-full-int8"] + (["flash"] if HAS_FLASH else []),
            line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) + ["Triton [int8]"] +
            ["Triton [full int8]"] + (["Flash-2"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-"), ("m", "-")],
            ylabel="ms",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-causal={causal}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "causal": causal,
            },
        ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, provider, device="cuda"):
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        if "int8" in provider:
            if "full" in provider:
                q = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                k = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                v = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                q_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                k_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                v_scale = torch.randn((BATCH, H),        dtype=dtype, device=device, requires_grad=False)
                sm_scale = 1.3
                fn = lambda: attention_full_int8(q, k, v, q_scale, k_scale, v_scale, causal, sm_scale)
            else:
                q = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                k = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
                q_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                k_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                sm_scale = 1.3
                fn = lambda: attention_int8(q, k, v, q_scale, k_scale, causal, sm_scale)
        else:
            q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
            k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
            v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
            if "fp8" in provider:
                q = q.to(torch.float8_e5m2)
                k = k.to(torch.float8_e5m2)
                v = v.permute(0, 1, 3, 2).contiguous()
                v = v.permute(0, 1, 3, 2)
                v = v.to(torch.float8_e5m2)
            sm_scale = 1.3
            fn = lambda: attention(q, k, v, causal, sm_scale)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops / ms * 1e-9

def quant_pertoken(X):
    X_max, _ = torch.abs(X).max(dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, :, :, None]).to(torch.int8)
    return ret, X_scale

def quant_pertensor(X):
    X_max, _ = torch.abs(X).max(dim=-1)
    X_max, _ = torch.max(X_max, dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, :, None, None]).to(torch.int8)
    return ret, X_scale

CALC_NUM = 4
def mean_relative_error(matrix2, matrix1):
    assert matrix1.shape == matrix2.shape
    m1 = matrix1.view(-1)[-CALC_NUM:]
    m2 = matrix2.view(-1)[-CALC_NUM:]
    absolute_error = torch.abs(m1 - m2)
    relative_error = absolute_error / (torch.abs(m1) + 1e-5)
    # print("relative_error", relative_error)
    # for i, t in enumerate(relative_error):
    #     if t > 1:
    #         print("i", i, "m1[i]", m1[i], "m2[i]", m2[i], "t", t)
    mean_relative_error = relative_error.mean()
    
    return mean_relative_error.item()

def generate_alternating_tensor(rows, cols):
    # 生成一个形状为 (cols,) 的张量，其元素为 [0, 1, 0, 1, ...]
    alternating_row = torch.arange(cols) % 2

    # 将该行重复 rows 次，生成一个形状为 (rows, cols) 的张量
    alternating_tensor = torch.tile(alternating_row, (rows, 1))

    return alternating_tensor

def acc_test(BATCH, H, N_CTX, HEAD_DIM, causal, device="cuda"):
    sm_scale = 1
    dtype = torch.float16
    # q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    # k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    # v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)

    q = torch.rand((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    k = torch.rand((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    v = torch.rand((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)

    # q = torch.ones((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)*0.5
    # k = torch.ones((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    # v = torch.ones((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)*0.5

    # q = generate_alternating_tensor(BATCH * H * N_CTX, HEAD_DIM).view(BATCH, H, N_CTX, HEAD_DIM).to(dtype=dtype, device=device)
    # k = generate_alternating_tensor(BATCH * H * N_CTX, HEAD_DIM).view(BATCH, H, N_CTX, HEAD_DIM).to(dtype=dtype, device=device)
    # v = generate_alternating_tensor(BATCH * H * N_CTX, HEAD_DIM).view(BATCH, H, N_CTX, HEAD_DIM).to(dtype=dtype, device=device)

    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)

    atten_out = attention(q, k, v, causal, sm_scale)

    q8, qs8 = quant_pertoken(q)
    k8, ks8 = quant_pertoken(k)
    v8, vs8 = quant_pertensor(v)
    int8_out = attention_int8(q8, k8, v, qs8, ks8, causal, sm_scale)
    full_int8_out = attention_full_int8(q8, k8, v8, qs8, ks8, vs8, causal, sm_scale)

    q = q.to(torch.float8_e5m2)
    k = k.to(torch.float8_e5m2)
    v = v.to(torch.float8_e5m2)
    fp8_out = attention(q, k, v, causal, sm_scale)

    print("------------------------------------")
    print("ref_out      ", ref_out.view(-1)[-CALC_NUM:])
    print("atten_out    ", atten_out.view(-1)[-CALC_NUM:])
    print("fp8_out      ", fp8_out.view(-1)[-CALC_NUM:])
    print("int8_out     ", int8_out.view(-1)[-CALC_NUM:])
    print("full_int8_out", full_int8_out.view(-1)[-CALC_NUM:])

    t0 = mean_relative_error(atten_out, ref_out)
    t1 = mean_relative_error(fp8_out.to(torch.float16), ref_out)
    t2 = mean_relative_error(int8_out, ref_out)
    t3 = mean_relative_error(full_int8_out, ref_out)
    print("------------------------------------")
    # print([t0, t1, t2, t3])
    print("MRE(atten_out,      ref_out)", t0)
    print("MRE(fp8_out,        ref_out)", t1)
    print("MRE(int8_out,       ref_out)", t2)
    print("MRE(full_int8_out,  ref_out)", t3)

    # return [t0, t1, t2, t3]



if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # bench_flash_attention.run(save_path=".", print_data=True)

    # acc_df = pd.DataFrame(columns=['N_CTX', 'Triton [FP16]', 'Triton [FP8]', 'Triton [int8]', 'Triton [full int8]'])
    # REPEAT_TIMES = 1
    # for t in [2**i for i in range(10, 15)]:
    #     print(t)
    #     res = []
    #     for i in range(REPEAT_TIMES):
    #         while(1):
    #             tmp = acc_test(BATCH=2, H=2, N_CTX=t, HEAD_DIM=64, causal=False)
    #             if True:
    #                 break
    #         res.append(tmp)
    #     t0, t1, t2, t3 = np.mean(res, axis=0)
    #     new_row = {'N_CTX': t, 'Triton [FP16]': t0, 'Triton [FP8]': t1, 'Triton [int8]': t2, 'Triton [full int8]': t3}
    #     print(new_row)
    #     acc_df = pd.concat([acc_df, pd.DataFrame([new_row])], ignore_index=True)
    # print(acc_df)

    acc_test(BATCH=1, H=1, N_CTX=32, HEAD_DIM=32, causal=False)

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 1
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)