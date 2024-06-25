# Long-Context-Attention (YunChang-云长): A Unified Sequence Parallel (USP) Attention for Long Context LLM Model Training and Inference

[\[Tech Report\] USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)


<p align="center">
    <img src="./media/yun_chang.jpg" width="200" />
</p>

This repo provides a sequence parallel approach that synergizes the strengths of two popular distributed attentions, i.e. DeepSpeed-Ulysses-Attention and Ring-Attention, delivering a more general and stronger versatility and better performance. 
The project is built on [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) and refers to the [DeepSpeed-Ulysses](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md).



## What's wrong with Ulysses and Ring?

- Ulysses is sensitive to the number of attention heads. 
The parallelism degree in Ulysses cannot exceed the number of heads. 
Consequently, it is not suitable for GQA (Grouped Query Attention) and MQA (Multi-Query Attention) scenarios. For instance, Ulysses does not operate effectively with a single head. 
In addition, since Tensor Parallelism also requires division across the head number dimension, achieving compatibility between Ulysses and TP can be challenging.

- Ring-Attention is ineffient than Ulysses in computation and communication.
Ring-Attention segments the Query, Key, and Value (QKV) into smaller blocks, which can lead to a decrease in efficiency when using FlashAttention.
Even with the communication and computation processes fully overlapped, the total execution time lags behind that of Ulysses. 
Furthermore, Ring-Attention utilizes asynchronous peer-to-peer communication, which not only has a lower bandwidth utilization compared to collective communication methods but also poses the risk of potential communication deadlocks in large-scale deployments.


## LongContextAttention, a.k.a Unified Sequence Parallelism and Hybrid Sequence Parallelism

`LongContextAttention` is a **unified sequence parallel** , also known as **hybrid sequence parallel** ,that hybrid DeepSpeed-Ulysses-Attention and Ring-Attention therefore addressing the limitations of both methods.

<p align="center">
    <img src="./media/hybrid_seqparallel.png">
</p>


### Install

Option 1: pip install from pypi. 

`pip install yunchang==0.2`

Option 2: build from local.

`pip install .`


**Features:**

1. No Limitation on the Number of Heads: Our approach does not impose a restriction on the number of heads, providing greater flexibility for various attention mechanisms.

2. Cover the Capability of either Ulysses and Ring: By setting the ulysses_degree to the sequence parallel degree, the system operates identically to Ulysses. Conversely, setting the ulysses_degree to 1 mirrors the functionality of Ring.

3. Enhanced Performance: We achieve superior performance benchmarks over both Ulysses and Ring, offering a more efficient solution for attention mechanism computations.

4. Compatibility with Advanced Parallel Strategies: LongContextAttention is fully compatible with other sophisticated parallelization techniques, including Tensor Parallelism, ZeRO, and Pipeline Parallelism, ensuring seamless integration with the latest advancements in parallel computing.

### Verified in Megatron-LM
The loss curves for Data Parallel (DP) and Unified Sequence Parallel (ulysses=2+ring=2) are closely aligned, as illustrated in the figure. This alignment confirms the accuracy of the unified sequence parallel.

<p align="center">
    <img src="./media/loss.png">
</p>

You should reorder Query tensors with [EXTRACT_FUNC_DICT](./yunchang/comm/extract_local.py) when using load-balance Ring Attention when applying the causal mask.
In the Megatron-LM, you can reorder the input tokens before feed them into the model and apply the same reordering to RoPE parameters. See our paper for detailed instructions.

## Best Practice for 4D Parallelism


We analyze the impact of introducing Sequnce Parallelism to Data/ZeRO/Tensor/Pipeline Parallelism in a technique report, which can be found at [here](https://arxiv.org/abs/2405.07719).

Some best practices are listed here:

1. We suggest using Unified-SP in place of SP-Ring and SP-Ulysses, as it encompasses the capabilities of both while offering additional benefits.

2. DP (data parallelism) vs SP: We suggest prioritizing the use of DP over SP if possible. 
Only when the batch size (bs) is insufficient for partitioning should one consider whether to employ SP

3. Utilizing SP, it should always be used in conjunction wit ZeRO-1/2.

4. Unified-SP has lower communication cost than Tensor Parallel with megatron-lm sequence parallelism (TP-sp)! You can use Unified-SP to replace TP for better speed. However, now switching TP (tensor parallelism) to SP+ZeRO2 cannot increase the sequence length in training. SP+ZeRO3 can train a similar sequence length as TP-sp. We suggest that SP may have an advantage over TP when employing GQA in terms of communication cost, as GQA can reduce the communication cost of SP without affecting TP.

5. Setting a higher parallel degree of SP parallelism is possible, which may need to set a large ring degree when the head number is limited, to train a long sequence across a greater number of computational devices. But TP could not be set a high parallel.

### Test

```bash
torchrun --nproc_per_node 8 test/test_hybrid_qkvpacked_attn.py
```

### Benchmark


```bash
bash ./scripts/run_qkvpack_compare.sh
```

On an 8xA100 NVLink machine, the benchmark results are as follows:

<p align="center">
    <img src="./media/benchmark_results.png">
</p>

On an 8xL20 PCIe machine and a 4xA100 PCIe machine, the benchmark results are as follows:

<p align="center">
    <img src="./media/pcie_machine.jpg">
</p>

Some Conclusions:

1. If the head number is enough, Ulysses outperforms Ring-Attention. The All-to-All communication of Ulysses is highly efficient within a single machine, with a very low overhead ratio. In contrast, Ring splits computation and communication, which increases the overall of computation time, and even with complete overlap, it is slower than Ulysses.

2. QKV packed (`LongContextAttentionQKVPacked`) is better than the QKV no packed (`LongContextAttention`) version, with the difference becoming more pronounced as the sequence length decreases. MAQ and GQA can only use the no packed version.

3. Among the variants of the Ring-Attention implementation, `zigzag` and `stripe` perform better than `basic`. Typically, zigzag is slightly better than stripe, but as the sequence length increases, the difference between zigzag and stripe becomes less noticeable. It is worth noting that both zigzag and stripe have specific layout requirements for the sequence dimension.

4. Hybrid parallelism works well to heterogeneous network devices. For example, on an 8-GPU L20 setup, the optimal performance is achieved when ulysess_degree is set to 2 and ring_degree is set to 4.

## Projects apply our methods

[Ascend/AscendSpeed](https://gitee.com/ascend/AscendSpeed/blob/master/docs/features/hybrid-context-parallel.md)

[EasyContext](https://github.com/jzhang38/EasyContext)

## Citation
```
@article{fang2024unified,
  title={USP: A Unified Sequence Parallelism Approach for Long Context Generative AI},
  author={Fang, Jiarui and Zhao, Shangchun},
  journal={arXiv preprint arXiv:2405.07719},
  year={2024}
}
```
