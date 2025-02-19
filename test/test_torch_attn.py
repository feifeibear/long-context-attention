import argparse
import time

import torch
import torch.distributed as dist
from yunchang.kernels.attention import pytorch_attn_forward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seq_len", type=int, default=16384)
    parser.add_argument("-n", "--num_heads", type=int, default=24)
    parser.add_argument("-d", "--attention_head_dim", type=int, default=128)
    return parser.parse_args()


def main():
    # init
    args = parse_args()
    torch.manual_seed(1024)

    # prepare qkv
    q = torch.rand(1, args.seq_len, args.num_heads, args.attention_head_dim).cuda().to(torch.float16)
    k = torch.rand(1, args.seq_len, args.num_heads, args.attention_head_dim).cuda().to(torch.float16)
    v = torch.rand(1, args.seq_len, args.num_heads, args.attention_head_dim).cuda().to(torch.float16)

    # measure time
    def time_func(func, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        out = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        return out, end - start
    
    # run simple profiling
    flash_total_time = 0
    efficient_total_time = 0

    torch.cuda.reset_peak_memory_stats()
    for i in range(10):
        flash_out, flash_time = time_func(pytorch_attn_forward, q, k, v, op_type="flash")

        if i >= 3:
            # skip 3 warmup runs
            flash_total_time += flash_time
    flash_max_memory_allocated = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    for i in range(10):
        efficient_out, efficient_time = time_func(pytorch_attn_forward, q, k, v, op_type="efficient")

        if i >= 3:
            # skip 3 warmup runs
            efficient_total_time += efficient_time
    efficient_max_memory_allocated = torch.cuda.max_memory_allocated()
    

    print(f"Flash time: {flash_total_time / 7}")
    print(f"Efficient time: {efficient_total_time / 7}")
    print(f"Flash max memory allocated: {flash_max_memory_allocated}")
    print(f"Efficient max memory allocated: {efficient_max_memory_allocated}")
    
    # ensure output is close
    torch.testing.assert_close(flash_out, efficient_out)

if __name__ == "__main__":
    main()
