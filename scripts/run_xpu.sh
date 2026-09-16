#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600
GPU_NUM=$(python -c "import torch; print(torch.xpu.device_count());")

torchrun --nproc_per_node $GPU_NUM test/test_hybrid_attn_xpu.py --seqlen 2048 --causal
