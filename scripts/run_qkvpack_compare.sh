export PYTHONPATH=$PWD:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export NCCL_PXN_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_TC=160
# export NCCL_IB_TIMEOUT=22
# export NCCL_P2P=0

# torchrun --nproc_per_node 8 test/test_hybrid_attn.py

FWD_FLAG="--fwd_only"



# SEQLEN=512
# SEQLEN=1024
# SEQLEN=4096
# SEQLEN=512
# SEQLEN=16384
SEQLEN=32768 #128K

NHEADS=32

# HEAD_SIZE=128
HEAD_SIZE=32
GROUP_NUM=4
BS=1

GPU_NUM=8

# USE_PROFILE="--use_profiler"

# NHEADS // GROUP_NUM > ulysses_degree

for RING_IMPL_TYPE in "basic" "zigzag" "strip"; do
for ULYSSES_DEGREE in 8 4 2 1; do

torchrun --nproc_per_node $GPU_NUM benchmark/benchmark_longctx_qkvpacked.py \
--nheads $NHEADS \
--batch_size $BS \
--seq_len $SEQLEN \
--head_size $HEAD_SIZE \
--ulysses_degree $ULYSSES_DEGREE \
--ring_impl_type $RING_IMPL_TYPE \
$FWD_FLAG

torchrun --nproc_per_node $GPU_NUM benchmark/benchmark_longctx.py \
--nheads $NHEADS \
--group_num $GROUP_NUM \
--batch_size $BS \
--seq_len $SEQLEN \
--head_size $HEAD_SIZE \
--ulysses_degree $ULYSSES_DEGREE \
--ring_impl_type $RING_IMPL_TYPE \
$FWD_FLAG

done
done


