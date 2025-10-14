#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=21289
export HCCL_IF_BASE_PORT=64199

torchrun  --master_port=29600 --nproc_per_node 4 test/test_ulysses_attn_npu.py
