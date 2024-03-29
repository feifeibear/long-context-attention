# Long-Context-Attention: Distributed Attention Implementations for Long Context LLM Model Training.
This repo contains three sequence parallel approaches. DeepSpeed-Ulysses Attention, Ring-Attention and a hybrid Long-Context-Attention.

## LongContextAttention (Hybrid Ulysses-Ring Attention)
Applying a hybrid sequence parallelism, this method scales the sequence length across multiple GPUs. 
It overcomes the limitations of both Ulysses and Ring attention approaches.

1. Architectural Robustness: Ulysses encounters challenges when the number of heads exceeds the world size, whereas hybrid sequence parallelism imposes no such restrictions.

2. A hybrid communication pattern: Ring-attention leveraging computation to overlap the P2P communication costs, however potientally leads to poor bandwidth utilizations when blocks sizes are set ineffienctly.
Ulysses employs All-to-All communications, ensuring the communication cost scales with sequence length rather than the number of GPUs. 
The hybrid sequence parallelism integrates the best of both approaches.



### Test

```bash
torchrun --nproc_per_node 8 test/test_long_context_qkvpacked_attn.py
```

### Benchmark
```
FWD_FLAG=0
torchrun --nproc_per_node 2 benchmark/benchmark_longctx_qkvpacked.py --nheads 2 --batch_size 2 --fwd_only $FWD_FLAG --ulysses_degree 1
torchrun --nproc_per_node 2 benchmark/benchmark_longctx_qkvpacked.py --nheads 2 --batch_size 2 --fwd_only $FWD_FLAG --ulysses_degree 2
torchrun --nproc_per_node 2 benchmark/benchmark_qkvpacked_func.py --nheads 2 --batch_size 2 --fwd_only $FWD_FLAG
```

![head=8](./media/long_ctx_h8.png)
![head=8](./media/long_ctx_h2.png)

## Ulysses Attention
This repository re-implements the all-to-all communication pattern for inputs as 4D tensors, following the principles of [DeepSpeed-Ulysses](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md).
It is important to note that DeepSpeed-Ulysses does not accommodate scenarios where the number of attention heads surpasses the size of the world (i.e., the total number of GPUs in the distributed setup).


### Test

```bash
torchrun --nproc_per_node 8 test/test_ulysses_attn.py
```

## Ring Flash Attention


Ring-Attention use the code from repo [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention), which implements the [RingAttention](https://github.com/lhao499/RingAttention) with [FlashAttention](https://github.com/Dao-AILab/flash-attention). We reuse the APIs:

- `ring_flash_attn_func`: ring attention version of `flash_attn_func`
- `ring_flash_attn_varlen_func`: ring attention version of `flash_attn_varlen_func`
- `zigzag_ring_flash_attn_func`: an optimized version of `ring_flash_attn_func`, see [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2)
- `zigzag_ring_flash_attn_varlen_func`: an optimized version of `ring_flash_attn_varlen_func`
- `stripe_flash_attn_func`: stripe attention version of `ring_flash_attn_func`, the block size is set to 1 to use flash_attn api.

Note that

- all function has the `*_func`, `*_kvpacked_func`, `*_qkvpacked_func` variant implemented.
- the varlen versions only support passing one `cu_seqlens`.

The main idea is to use the `softmax_lse` output from the flash attention kernels.

### Limits

There are some arithmetic errors with the current implementation. The reason for them is probably that flash attention will return bf16 value for each block, so we cannot accumluate the values with the original fp32 ones.

And also because we need to save extra fp32 buffer during computation, the memory usage would be higher than theoretic limit.

### TODOs

- [x] Implement `ring_flash_attn_varlen_qkvpacked_func`
- [x] Implement `zigzag_ring_flash_attn_qkvpacked_func` [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2)
- [x] Implement `stripe_flash_attn_qkvpacked_func`
- [x] Implement `zigzag_ring_flash_attn_varlen_qkvpacked_func`
- [x] Implement `*_kvpacked_func` and `*_func` variant for all APIs
- [ ] Optimize `*_varlen_func`
- [ ] Try to upstream to flash attention.

### Test

```bash
torchrun --nproc_per_node 8 test/test_ring_flash_attn_func.py
torchrun --nproc_per_node 8 test/test_ring_flash_attn_varlen_func.py
torchrun --nproc_per_node 8 test/test_zigzag_ring_flash_attn_func.py
torchrun --nproc_per_node 8 test/test_zigzag_ring_flash_attn_varlen_func.py
torchrun --nproc_per_node 8 test/test_stripe_flash_attn_func.py
```

## Citation
```
@misc{fang2024long,
      title={Long-Context-Attention: Distributed Attention Implementations for Long Context LLM Model Training},
      author={Jiarui Fang, Zilin Zhu, Yang Yu},
      year={2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/feifeibear/long-context-attention}},
}
@article{jacobs2023deepspeed,
      title={Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models},
      author={Jacobs, Sam Ade and Tanaka, Masahiro and Zhang, Chengming and Zhang, Minjia and Song, Leon and Rajbhandari, Samyam and He, Yuxiong},
      journal={arXiv preprint arXiv:2309.14509},
      year={2023}
}
@article{liu2023ring,
      title={Ring attention with blockwise transformers for near-infinite context},
      author={Liu, Hao and Zaharia, Matei and Abbeel, Pieter},
      journal={arXiv preprint arXiv:2310.01889},
      year={2023}
}
```
