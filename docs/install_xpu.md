## Install for Intel XPU (Xe GPU / BMG)

Supported hardware: Intel Battlemage (BMG), Arc series, and later Xe-class GPUs that support FP64.

### Step 1: Prepare the environment

A PyTorch build with XPU support is required. Use the Intel PyTorch XPU docker image or a pre-built wheel from Intel:

```bash
# Install PyTorch with XPU support (example using pip index from Intel)
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Or use the official Intel PyTorch XPU docker image as a starting point:

```bash
IMG=intel/pytorch:xpu-2.13.0-ubuntu24.04-20260907
docker run -it --privileged --device /dev/dri \
  --ipc=host --net=host \
  --group-add $(getent group video | cut -d: -f3) \
  --group-add $(getent group render | cut -d: -f3) \
  -v $(pwd):/workspace --workdir /workspace \
  $IMG bash
```

### Step 2: Install sgl-kernel-xpu

XPU flash attention is provided by `sgl-kernel-xpu`, which exposes `sgl_kernel.flash_attn.flash_attn_varlen_func`:

```bash
pip install sglang-kernel-xpu
# or build from source:
# git clone https://github.com/intel/sgl-kernel-xpu && cd sgl-kernel-xpu
# source oneapi setvars.sh script
# based on RAM memory set CMAKE_BUILD_PARALLEL_LEVEL when host out of memory observed
# pip install -e .
```

### Step 3: Install yunchang

```bash
pip install yunchang
# or from source:
# pip install -e .
```

### Verify XPU detection

```python
from yunchang.globals import HAS_XPU
print("XPU flash attention available:", HAS_XPU)
```
> **Note:** Backward pass is not supported for `AttnType.XPU`. Use it for inference or forward-only training scenarios.

### Test

```bash
torchrun --nproc_per_node=<NUM_GPUS> test/test_hybrid_attn_xpu.py --seqlen 2048 --causal
```

### Benchmark

```bash
torchrun --nproc_per_node=<NUM_GPUS> benchmark/benchmark_longctx_xpu.py \
    --seq_len 8192 --nheads 32 --head_size 64
```
