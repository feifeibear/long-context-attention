export PYTHONPATH=$PWD:$PYTHONPATH
# export NCCL_PXN_DISABLE=1
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

NHEADS=64
SEQLEN=131072
GROUP_NUM=8
GPU_NUM=8
ULYSSES_DEGREE=1

NRANK=${NRANK:-0}
# RING_IMPL_TYPE="zigzag"

# make sure NHEADS // GROUP_NUM % ULYSSES_DEGREE == 0
for ULYSSES_DEGREE in 8 4 2 1; do
for RING_IMPL_TYPE in "zigzag"; do
torchrun --nproc_per_node $GPU_NUM --node_rank $NRANK benchmark/benchmark_longctx.py --nheads $NHEADS --group_num $GROUP_NUM --batch_size 1 $FWD_FLAG --seq_len $SEQLEN --ulysses_degree $ULYSSES_DEGREE --ring_impl_type $RING_IMPL_TYPE
done
done

torchrun --nproc_per_node $GPU_NUM  --node_rank $NRANK benchmark/benchmark_ring_func.py --nheads $NHEADS --group_num $GROUP_NUM --batch_size 1 $FWD_FLAG --seq_len $SEQLEN

