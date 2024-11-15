export PYTHONPATH=$PWD:$PYTHONPATH# export NCCL_PXN_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_TC=160
# export NCCL_IB_TIMEOUT=22

export CUDA_DEVICE_MAX_CONNECTIONS=1

# nccl settings
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22

export GLOO_SOCKET_IFNAME=eth0


# comment this line fwd+bwd
# FWD_FLAG="--fwd_only"

NHEADS=24
SEQLEN=1024
GROUP_NUM=1
GPU_NUM=2
HEAD_SIZE=128
ULYSSES_DEGREE=8

NRANK=${NRANK:-0}
# RING_IMPL_TYPE="zigzag"

# make sure NHEADS // GROUP_NUM % ULYSSES_DEGREE == 0
for attn_type in "torch" "fa" "fa3"; do
for ULYSSES_DEGREE in 2; do
for RING_IMPL_TYPE in "basic"; do
torchrun --nproc_per_node $GPU_NUM --node_rank $NRANK benchmark/benchmark_longctx.py \
--nheads $NHEADS --group_num $GROUP_NUM --batch_size 1 $FWD_FLAG --seq_len $SEQLEN --head_size $HEAD_SIZE \
--ulysses_degree $ULYSSES_DEGREE --ring_impl_type $RING_IMPL_TYPE --no_causal --attn_type $attn_type --use_ulysses
done
done
done
