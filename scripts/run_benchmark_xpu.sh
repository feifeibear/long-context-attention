#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600

SEQLEN=8192
NHEADS=32
HEAD_SIZE=64
BS=1
GPU_NUM=$(python -c "import torch; print(torch.xpu.device_count());")

for ULYSSES_DEGREE in 4 2 1; do
    if [[ ${ULYSSES_DEGREE} -gt ${GPU_NUM} ]]; then
        echo "Not enpugh GPU cards for this ULYSSES_DEGREE=${ULYSSES_DEGREE} config"
        continue
    fi
    torchrun --nproc_per_node $GPU_NUM benchmark/benchmark_longctx_xpu.py \
        --nheads $NHEADS \
        --batch_size $BS \
        --seq_len $SEQLEN \
        --head_size $HEAD_SIZE \
        --ulysses_degree $ULYSSES_DEGREE \
        --ring_impl_type basic_xpu \
        --no_causal
done
