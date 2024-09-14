## Install for AMD GPU

Supported GPU : MI300X, MI308X

GPU arch : gfx942

Step 1: prepare docker envrionment

Tow recommended docker container to start with 

- rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0 : hosted in dockerhub, no conda
- [dockerhub repo](https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub/blob/main/rocm/Dockerfile.rocm62.ubuntu-22.04) : Customerized Dockerfile with conda virtual env and develop kit support

An example to create an docker container :

```bash
# create docker container
IMG=rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
tag=py310-rocm6.2-distattn-dev

docker_args=$(echo -it --privileged \
 --name $tag \
 --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
 --device=/dev/kfd --device=/dev/dri \
 --ipc=host \
 --security-opt seccomp=unconfined \
 --shm-size 16G \
 --group-add video \
 -v $(readlink -f `pwd`):/workspace \
 --workdir /workspace \
 --cpus=$((`nproc` / 2  - 1)) \
 $IMG
)

docker_args=($docker_args)

docker container create "${docker_args[@]}"

# start it
docker start -a -i $tag
```

Update ROCM SDK using this [script](https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub/blob/main/rocm/update_sdk.sh):

```bash
# e.g.:
ROCM_VERSION=6.3 bash rocm/update_sdk.sh
```

Step 2 : build from local.

install flash_attn from source

```bash
pip install flash_attn@git+https://git@github.com/Dao-AILab/flash-attention.git
```

then install yunchang

> MAX_JOBS=$(nproc) pip install . -verbose

**Features:**

1. No Limitation on the Number of Heads: Our approach does not impose a restriction on the number of heads, providing greater flexibility for various attention mechanisms.

2. Cover the Capability of either Ulysses and Ring: By setting the ulysses_degree to the sequence parallel degree, the system operates identically to Ulysses. Conversely, setting the ulysses_degree to 1 mirrors the functionality of Ring.

3. Enhanced Performance: We achieve superior performance benchmarks over both Ulysses and Ring, offering a more efficient solution for attention mechanism computations.

4. Compatibility with Advanced Parallel Strategies: LongContextAttention is fully compatible with other sophisticated parallelization techniques, including Tensor Parallelism, ZeRO, and Pipeline Parallelism, ensuring seamless integration with the latest advancements in parallel computing.
