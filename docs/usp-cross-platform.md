# From GPUs to TPUs: How Unified Sequence Parallelism Became a Cross-Platform Attention Strategy

*November 26, 2026 · YunChang / USP project*

In May 2024, [Jiarui Fang and Shangchun Zhao introduced Unified Sequence Parallelism (USP)](https://arxiv.org/abs/2405.07719), a way to combine DeepSpeed-Ulysses and Ring Attention in one two-dimensional parallelism scheme. The motivation was simple: neither technique alone makes the right trade-off for every model or interconnect.

Two years later, variations of that two-axis design can be found in GPU training stacks, diffusion inference libraries, Ascend NPU software, and Google's TPU ecosystem. The interesting lesson is not that one implementation has won everywhere. It is that the same decomposition of distributed attention is useful across quite different systems.

**Paper:** [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI (Fang & Zhao, 2024)](https://arxiv.org/abs/2405.07719) · [DOI](https://doi.org/10.48550/arXiv.2405.07719) · [YunChang implementation](https://github.com/feifeibear/long-context-attention)

## 1. Why combine Ulysses and Ring Attention?

[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) uses all-to-all communication to exchange sequence shards for attention-head shards. Each rank computes local attention for its assigned heads across the sequence, then another exchange restores the original sequence layout. This can be highly efficient on a fast interconnect, but the usable degree depends on how the query and key/value heads can be partitioned, including any existing tensor parallelism.

[Ring Attention](https://arxiv.org/abs/2310.01889) instead partitions sequence tokens and streams key/value blocks around the ranks. Each rank accumulates its local attention result using a numerically stable online reduction. It can scale past head-count restrictions, and communication may overlap with computation. A long ring, however, introduces more communication stages and can leave smaller work units for the local attention kernel.

USP makes these choices composable. Define a two-dimensional process mesh with degrees U (Ulysses) and R (Ring), so the total sequence-parallel degree is:

```
SP = U × R
```

Within each Ulysses group, all-to-all communication converts sequence sharding into head sharding. Across each ring group, the ranks exchange KV blocks for their assigned heads. After attention, the inverse all-to-all restores the sequence partition. The endpoints U=1 and R=1 recover pure Ring and pure Ulysses, respectively.

For eight devices, the possible layouts include:

| U | R | Strategy |
|---:|---:|---|
| 8 | 1 | Pure Ulysses |
| 4 | 2 | Hybrid |
| 2 | 4 | Hybrid |
| 1 | 8 | Pure Ring |

The actual choices depend on head divisibility, sequence layout, kernel support, and topology. On a multi-node GPU cluster, it may make sense to place the all-to-all group inside a high-bandwidth domain and use ring exchanges between domains. Other platforms can map those axes differently.

## 2. What the original USP paper established

The [2024 report](https://arxiv.org/html/2405.07719v5) did more than combine communication primitives. It analyzed sequence parallelism together with data, tensor, ZeRO and pipeline parallelism; discussed causal load balancing and topology-aware group placement; and evaluated end-to-end LLM training.

A useful result in Table 6: on two RDMA-connected eight-GPU A800 nodes, Llama 3 8B training at 64K and 80K tokens used U=4 and R=4 for the best SP-only configurations reported. Compared with pure Ring at the same SP degree, the paper reports approximately 13% and 12% higher throughput, respectively. These figures are specific to that experiment, not universal speedup guarantees.

Table 7 also reports a 208K-token configuration at 47% model FLOPs utilization. That particular run used TP=8, U=1 and R=2, not the fully hybrid U>1, R>1 configuration. It illustrates the importance of choosing parallelism dimensions together rather than attributing every scaling result to mixed Ulysses and Ring.

## 3. The same pattern across accelerator ecosystems

The following examples are deliberately separated into actual code reuse and implementations of the same algorithmic pattern.

**GPU and multimodal systems.** [xDiT's documentation](https://github.com/xdit-project/xDiT/blob/main/docs/methods/usp.md) identifies YunChang as its USP implementation for diffusion transformers. [LongVILA's project documentation](https://github.com/NVlabs/VILA/blob/main/longvila/README.md#acknowledgement) acknowledges the use of YunChang's 2D attention implementation, and its [research paper](https://arxiv.org/abs/2408.10188) cites USP in its discussion of multimodal sequence parallelism. It also credits [LoongTrain](https://arxiv.org/abs/2406.18485), related work on head-context parallelism. [EasyContext](https://github.com/jzhang38/EasyContext) likewise acknowledges YunChang for sequence-parallel code.

**NVIDIA's implementation.** [Transformer Engine's API](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html) exposes hierarchical context parallelism through the `a2a+p2p` communication choice. [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html) documents the same pattern. These are analogous vendor implementations; the cited documentation does not establish reuse of YunChang code.

**Ascend NPUs.** [MindSpeed's hybrid context parallelism documentation](https://gitee.com/ascend/MindSpeed/blob/cf7f0092061caef91e17f016b3864639dad5a008/docs/features/hybrid-context-parallel.md) describes mixed Ulysses/Ring parallelism and explicitly links the 2024 USP paper.

**Google Cloud TPUs.** On November 9, 2026, [Google Cloud announced long-context training in MaxText](https://developers.googleblog.com/announcing-long-context-training-on-tpus-with-maxtext-context-parallelism/), including USP as one of four context-parallel strategies. Its [MaxText sharding guide](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/optimization/sharding.html) links directly to the USP paper and exposes `context_parallel_strategy=usp`. Its JAX/TPU implementation uses separate mesh axes for the two communication patterns. [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion#whats-new) also documents 2D Ring/USP support for Wan video inference since June 2026. These are independent implementations, not claims of YunChang code reuse.

Importantly, Google's reported 14-million-token training experiment uses pure Ring, and its blog reports Ring doing better in its tested TPU 7 setting. That is consistent with the design principle of USP: compare the available decompositions rather than assuming the mixed configuration always wins.

## 4. How to use USP as a research baseline

When designing distributed attention for a new LLM, VLM, or video diffusion transformer, consider evaluating pure Ulysses, pure Ring, and the feasible mixed layouts at a fixed total parallel degree. Report the hardware topology, Q/KV head counts, tensor-parallel degree, model and sequence shape, dtype, local kernel, masks, and direction (forward only or forward and backward). Measure end-to-end latency or training throughput alongside peak memory and attention communication time; include correctness against a reference implementation.

For diffusion inference, report the number of actual model evaluations as well as scheduler steps, resolution, frames, and whether CFG or caching is used. For training, distinguish the attention microbenchmark from the model-wide throughput and loss/convergence tests.

The [YunChang repository](https://github.com/feifeibear/long-context-attention) provides PyTorch implementations, test scripts, and benchmark examples. Other frameworks have their own paths. If your work uses, compares against, or builds upon the unified Ulysses/Ring decomposition, please cite the original USP paper along with the relevant underlying methods and implementation-specific work. This makes the research lineage clear and helps readers reproduce and compare the results.

## Citation

```bibtex
@article{fang2024unified,
  title = {USP: A Unified Sequence Parallelism Approach for Long Context Generative AI},
  author = {Fang, Jiarui and Zhao, Shangchun},
  journal = {arXiv preprint arXiv:2405.07719},
  year = {2024},
  doi = {10.48550/arXiv.2405.07719},
  url = {https://arxiv.org/abs/2405.07719}
}
```

For a living list of documented implementations and citations, see the [USP ecosystem table in the project README](../README.md#usp-in-the-ecosystem). Contributions and corrections with primary-source links are welcome.
