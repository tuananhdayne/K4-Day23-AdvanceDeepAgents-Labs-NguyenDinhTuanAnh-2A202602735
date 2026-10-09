# Efficient Inference and Small Language Models: A Comprehensive Survey

## TL;DR
- Post-training weight and activation compression methods like AWQ, SmoothQuant, and Wanda resolve systematic activation outliers, reducing model memory footprint by 30% to 70% and accelerating inference without expensive retraining [1][2][3][4].
- Memory-centric serving architectures—notably PagedAttention in vLLM—and IO-aware attention kernels (FlashAttention) eliminate up to 60%–80% of KV cache memory fragmentation while shifting decode steps from memory-bandwidth bounds to compute bounds [5][6].
- Speculative decoding paradigms, including feature-level draft extrapolation (EAGLE), multi-head drafting (Medusa), and multi-token prediction (MTP), deliver 2.2× to 3.0× end-to-end acceleration by parallelizing autoregressive verification [7][8][9].
- Modern Small Language Models (SLMs) spanning 0.1B to 4B parameters achieve frontier-grade reasoning density through extreme token overtraining (up to 18T tokens), synthetic data curation, and structured distillation, enabling sub-2GB memory footprints on edge devices [10][11][12][13][14].

## Background
The exponential scaling of Large Language Model (LLM) parameters has unlocked unprecedented reasoning capabilities but created severe computational and memory bottlenecks during deployment. Standard autoregressive generation is fundamentally memory-bandwidth bound during token decoding: each generated token requires reading every parameter weight and the accumulated Key-Value (KV) cache from High-Bandwidth Memory (HBM) into fast on-chip SRAM [5][6]. In resource-constrained edge environments and large-scale cloud serving systems, these requirements translate directly into high latency, prohibitive memory consumption, and exorbitant operational costs.

To address these limitations, the research community has pursued a dual trajectory. First, efficient inference frameworks have emerged across multiple abstraction layers, encompassing algorithmic kernel tiling, memory-efficient KV cache management, post-training quantization, structural pruning, and speculative draft-verification pipelines [1][5][7][6]. Second, the paradigm of Small Language Models (SLMs)—compact architectures typically ranging from 100M to 4B parameters—has rapidly matured. By combining massive token overtraining, highly curated synthetic data, and distillation from frontier models, SLMs achieve remarkable task performance while fitting into consumer hardware, mobile phones, and embedded edge devices [10][12][13][14].

## Post-Training Quantization and Weight Compression
Quantization compresses model weights and activations into lower-bit representations (such as INT8, INT4, FP8, or sub-8-bit microscaling formats), directly reducing memory footprint and memory-bus traffic [1][2][15]. However, naive uniform quantization suffers from severe perplexity degradation caused by systematic activation outliers in LLMs [1][2].

Activation-aware Weight Quantization (AWQ) addresses this challenge by observing that not all weights are equally important; protecting approximately the top 1% of salient weight channels corresponding to large activation magnitudes is sufficient to maintain model accuracy [1]. To avoid mixed-precision hardware overhead, AWQ applies per-channel scaling to protect salient weights and absorbs the scale factors into upstream layer parameters [1]. Complementing weight-only schemes, SmoothQuant introduces mathematically equivalent per-channel transformations that migrate quantization difficulty from dynamic activations to static weights offline, unlocking full W8A8 (8-bit weight and 8-bit activation) matrix multiplications across standard tensor cores with up to 1.56× speedup and 2× memory reduction [2].

At higher compression ratios, second-order optimization techniques such as GPTQ use Hessian error compensation to sequentially quantize weights to 3-bit or 4-bit precision, compressing 175B-parameter models in under 4 GPU-hours [4]. Furthermore, modern post-training pipelines extend these methods to microscaling (MX) floating-point and integer formats (such as MXINT4/MXFP4 with MX8 activations), leveraging specialized hardware execution to mitigate dynamic range limitations without sacrificing baseline perplexity [15]. Together, AWQ and GPTQ workflows reduce GPU memory consumption by 30% to 70% in production serving engines [4].

```
Raw Model Weights & Activations
       │
       ├─► Outlier Migration (SmoothQuant: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)) ──► W8A8 Matrix Mult
       │
       ├─► Salient Channel Protection (AWQ: Protect top 1% salient channels) ──────► W4A16 GEMM
       │
       └─► Second-Order Error Compensation (GPTQ: Inverse Hessian Cholesky update) ─► W3/W4 Storage
```

## Pruning: Structured and Unstructured Sparsity
Pruning eliminates redundant parameters to compress models and reduce arithmetic operations [3][16]. Classical magnitude-based pruning often degrades LLM reasoning capabilities because weight magnitude alone ignores activation dynamics [3].

Wanda (Pruning by Weights and activations) overcomes this limitation by evaluating the product of weight magnitudes and input activation norms ($|W_{ij}| \cdot ||X_j||_2$) on a per-output basis in a single forward pass without requiring second-order weight updates or retraining [3]. Wanda operates up to 300× faster than SparseGPT and reliably supports 50% unstructured sparsity as well as hardware-friendly 2:4 and 4:8 structured sparsity patterns (e.g., LLaMA-7B achieves 7.26 perplexity at 50% sparsity under Wanda versus 17.29 under standard magnitude pruning) [3].

For standard dense execution backends that do not feature sparse tensor cores, structured pruning removes entire architectural components [16]. Fluctuation-based Adaptive Structured Pruning (FLAP) removes channels, attention heads, or entire layers based on feature fluctuation metrics [16]. By applying bias compensation to recover accuracy without backpropagation, FLAP yields compact, dense sub-matrices natively executable on standard GEMM runtimes without custom kernel overhead [16].

## Serving Optimizations, Attention Kernels, and KV Cache Compression
At the serving and systems level, the primary bottleneck during multi-turn generation is the dynamic management and memory footprint of the KV cache [5][17]. Standard serving systems allocate contiguous memory buffers based on maximum possible sequence lengths, resulting in 60% to 80% memory waste due to fragmentation and pre-allocation over-reservation [5].

PagedAttention resolves this fundamental bottleneck by introducing virtual memory paging to KV cache management, partitioning continuous KV tensors into fixed-size physical pages allocated dynamically upon request [5]. Implemented within the vLLM serving engine, PagedAttention facilitates continuous (iteration-level) batching and zero-copy prefix sharing across parallel requests, achieving 2× to 4× higher serving throughput than legacy frameworks [4][5].

```
Sequence KV Cache Request
       │
       ▼
┌───────────────────────── PagedAttention Virtual Paging ─────────────────────────┐
│ Page Table: Logical Page 0 ──► Physical Block #12 (Allocated in GPU HBM)       │
│             Logical Page 1 ──► Physical Block #47 (Allocated on-demand)         │
│             Prefix Block   ──► Shared Physical Block #3 (Zero-Copy Multi-branch)│
└─────────────────────────────────────────────────────────────────────────────────┘
```

At the hardware execution level, FlashAttention eliminates intermediate quadratic memory transfers ($O(N^2)$ reads/writes) between GPU SRAM and High-Bandwidth Memory (HBM) through IO-aware tiling and online softmax scaling, reducing memory traffic to sub-quadratic $O(N)$ and yielding 2× to 4× speedups [6]. FlashAttention-3 extends this paradigm with warp specialization, asynchronous Tensor Memory Accelerator (TMA) operations on Hopper architectures, and FP8 tensor core integration [6].

Beyond standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA) represents a fundamental architectural advancement for KV cache efficiency [17]. MLA compresses Keys and Values into a compact low-rank latent representation, achieving 8× to 14×+ KV cache reduction compared to MHA [17]. Crucially, during inference decoding, matrix absorption allows query projection weights to absorb key decompression matrices ($W_{\text{absorb}} = W_{up}^Q W_{up}^{K,T}$), enabling direct dot-product attention against compressed latent vectors in memory and shifting decoding from memory-bandwidth-bound execution toward compute-bound operational intensity [17].

## Speculative Decoding and Fast Generation
Autoregressive decoding generates tokens sequentially, requiring an entire forward pass over model parameters for each generated token [7][8]. Speculative decoding breaks this sequential dependency by dividing inference into a lightweight drafting phase and a parallelized verification phase [7][8][18].

Rather than relying on separate draft models that introduce synchronization complexity and additional memory footprint, modern frameworks employ single-model or feature-level drafting [7][8][9]:
- **EAGLE**: Observes that autoregressive drafting at the second-to-top hidden feature level exhibits greater regularity than discrete token generation [7]. By combining feature extrapolation with lightweight draft heads and tree-based verification, EAGLE achieves draft acceptance rates of ~0.80 and up to 2.8× speedup with only 2–4B tokens of draft head training [7].
- **Medusa**: Appends multiple one-layer MLP decoding heads directly on top of the base model trunk to predict multiple subsequent token positions simultaneously [8]. Using tree-structured attention verification masks, Medusa-1 and Medusa-2 achieve 2.2× to 2.8× lossless acceleration in a single target forward pass [8].
- **Multi-Token Prediction (MTP)**: Trains models during pre-training with $n$ independent prediction heads to forecast $n$ future tokens simultaneously [9]. MTP not only improves sample efficiency and downstream coding/reasoning performance (e.g., +12% on HumanEval) but also enables self-speculative decoding without separate auxiliary draft networks, accelerating inference by 2.7× to 3.0× [9].
- **SpecFold**: Extends speculative verification efficiency to multi-branch decoding and diffusion language models by folding multi-branch attention and sharing common prefix representations, removing redundant tensor computations during parallel tree verification [18].

## Modern Small Language Model Landscape and On-Device Deployment
Recent breakthroughs in data curation and model architectures have demonstrated that compact models can match the reasoning capabilities of models an order of magnitude larger [10][11][12][13][14].

| Model Family | Key Architecture Features | Training Data & Recipe Highlights | Deployment & Efficiency Profile |
| :--- | :--- | :--- | :--- |
| **Phi-3 / Phi-4** [10] | Dense transformer, 3.8B params | 3.3T tokens; 2-phase heavily filtered educational web + curated synthetic data | Quantized INT4 fits in ~1.8 GB memory; runs locally on iPhone 14 at >12 tok/s [10] |
| **Gemma 2** [11] | 2B, 9B, 27B; GQA (num_groups=2), logit soft-capping | Distillation from large teacher models; trained >50× compute-optimal token volume | Interleaved sliding window attention (4096/8192) saves KV memory during decoding [11] |
| **SmolLM2** [12] | 135M, 360M, 1.7B params | 2T to 11T tokens; Warmup-Stable-Decay schedule; FineMath, Cosmopedia-v2, SmolTalk | Outperforms Qwen2.5-1.5B/Llama-3.2-1B; lightweight edge execution via llama.cpp/transformers.js [12] |
| **Qwen 2.5** [13] | Dense 0.5B to 3B edge sizes; tied embeddings, GQA | 18T tokens; granular multi-stage post-training (SFT, offline DPO, online GRPO) | 32K context support; 0.5B base outperforms earlier 2.6B baselines on math and coding [13] |
| **Llama 3.2** [14] | 1B, 3B edge variants; GQA | Structured pruning + logit distillation from 8B/70B models over 9T tokens | QAT + LoRA and SpinQuant with ExecuTorch/Kleidi AI: 56% size cut, 2.5×–4.2× speedup on mobile [14] |

On-device deployment pipelines bridge the gap between compressed model weights and edge hardware constraints [10][14]. Frameworks like PyTorch ExecuTorch and Arm Kleidi AI leverage Quantization-Aware Training (QAT) with low-rank adapters and post-training rotation (SpinQuant) to maintain FP16 accuracy while executing low-bit kernels directly on mobile CPUs and Neural Processing Units (NPUs), achieving 2.5× decode latency speedups and sub-2GB memory consumption [14].

## Trends and Open Problems
The convergence of efficient serving architectures, advanced post-training quantization, and high-density small language models points toward several prominent trends and remaining challenges:
1. **Convergence of Architecture and Quantization-Friendliness**: Traditional post-training quantization was retrofitted onto architectures designed purely for dense FP16/BF16 training [1][2]. Future architectures are increasingly co-designed with quantization-friendly primitives, such as logit soft-capping, tied embeddings, and low-rank latent KV projections (MLA) that natively prevent activation outlier explosion [17][11][13].
2. **Dynamic and Adaptive Speculative Verification**: While tree-attention speculative decoding provides substantial speedups, draft acceptance rates fluctuate based on task complexity and sampling temperature [7][8][9]. Developing dynamic draft head routing and prefix-folded multi-branch verification (such as SpecFold) remains an active area of optimization [18].
3. **Extreme Sub-4-Bit and Microscaling Quantization**: Moving below INT4 into 2-bit and 3-bit regimes without catastrophic perplexity loss requires integrating Quantization-Aware Training with second-order error compensation and microscaling floating-point formats (MXFP4/MXFP6) [4][15][14].
4. **On-Device Continuous Learning and Edge Agentic Serving**: As SLMs are deployed for local on-device agents, maintaining low latency during long-context tool invocation and local prefix caching while operating within strict thermal and memory budgets (under 2GB RAM) will be a critical engineering frontier [10][12][14].

## References
[1] AWQ: Activation-aware Weight Quantization for LLM Compression, Acceleration. hf-search. https://huggingface.co/papers/2306.00978 (2023-06-01)
[2] SmoothQuant: Accurate, Efficient Post-Training Quantization for Large Language Models. hf-search. https://huggingface.co/papers/2211.10438 (2022-11-18)
[3] Wanda: Simple, Effective Pruning Approach for Large Language Models. hf-search. https://huggingface.co/papers/2306.11695 (2023-06-20)
[4] Accelerating LLM Inference with Post-Training Weight, Activation using AWQ, GPTQ on SageMaker AI. web. https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/ (2026-01-09)
[5] Post Training Quantization of Large Language Models with Microscaling Formats. web. https://proceedings.mlr.press/v262/sharify24a.html (2024-12-10)
[6] Fluctuation-based Adaptive Structured Pruning for Large Language Models (FLAP). hf-search. https://huggingface.co/papers/2312.11983 (2023-12-19)
[7] PagedAttention, vLLM: Efficient Memory Management for Large Language Model Serving. hf-search. https://huggingface.co/papers/2309.06180 (2023-09-12)
[8] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. hf-search. https://huggingface.co/papers/2401.15077 (2024-01-26)
[9] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. web. https://proceedings.mlr.press/v235/cai24b (2024-07-08)
[10] FlashAttention: Fast, Memory-Efficient Exact Attention with IO-Awareness. hf-search. https://huggingface.co/papers/2205.14135 (2022-05-27)
[11] TransMLA: Multi-Head Latent Attention Is All You Need, Hardware-Centric Analysis of MLA. hf-search. https://huggingface.co/papers/2502.07864 (2025-02-11)
[12] SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models. hf-daily. https://huggingface.co/papers/2610.04875 (2026-10-06)
[13] Phi-3 Technical Report: Highly Capable Language Model Locally on Your Phone. hf-daily. https://huggingface.co/papers/2404.14219 (2024-04-22)
[14] Gemma 2: Improving Open Language Models at Practical Size. hf-search. https://huggingface.co/papers/2408.00118 (2024-08-01)
[15] SmolLM2: When Smol Goes Big -- Data-Centric Training of Small Language Model. hf-search. https://huggingface.co/papers/2502.02737 (2025-02-04)
[16] Qwen2.5 Technical Report. hf-search. https://huggingface.co/papers/2412.15115 (2024-12-19)
[17] Introducing Quantized Llama Models with Increased Speed, Reduced Memory Footprint. web. https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/ (2024-10-24)
[18] Better, Faster Large Language Models via Multi-token Prediction. hf-daily. https://huggingface.co/papers/2404.19737 (2024-04-30)
