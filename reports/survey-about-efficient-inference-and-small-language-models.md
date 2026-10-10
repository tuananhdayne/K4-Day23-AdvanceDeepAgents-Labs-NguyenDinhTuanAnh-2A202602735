# Efficient Inference and Small Language Models: A Comprehensive Survey

## TL;DR
- Post-training compression methods like AWQ and SmoothQuant resolve activation outliers, cutting memory footprints by 30% to 70% without retraining [1][2].
- Serving engines eliminate up to 80% of KV cache waste via virtual paging, while kernel tiling reduces memory bandwidth bottlenecks [3][4].
- Speculative decoding parallelizes candidate verification with lightweight draft heads, speeding up generation by 2× to 3× without altering output distributions [5][6].
- Small language models trained on trillions of curated tokens match larger baselines and fit within 2GB of memory on edge hardware [7][8].

## Background
Autoregressive models generate text one token at a time. This sequential loop creates a severe memory bandwidth bottleneck: each token requires streaming model weights and Key-Value (KV) cache tensors from memory into on-chip SRAM [3][4]. Hardware compute units stall while waiting for these transfers. In data centers, memory traffic limits request concurrency and raises hosting costs [9][3]. On edge devices, strict memory limits prevent running uncompressed models [7][8].

Two complementary strategies address these constraints. First, inference engines optimize execution through quantization, virtual memory paging, and speculative decoding [1][3][5]. Second, developers train compact models under 4B parameters across trillions of tokens using synthetic data and distillation [7][10]. Together, these methods enable efficient deployment on cloud servers and consumer hardware [9][8].

## Post-Training Quantization and Sparsity
Uniform low-bit quantization causes severe accuracy loss because extreme activation outliers emerge in transformer layers [1][2]. Activation-aware Weight Quantization (AWQ) protects the top 1% of salient weight channels linked to these outliers, scaling channels without custom hardware overhead [1]. SmoothQuant shifts quantization difficulty offline from dynamic activations to static weights, enabling 8-bit weight and activation matrix operations with up to 1.56× speedup [2]. For heavier compression, GPTQ applies second-order error compensation to quantize weights to 3-bit or 4-bit precision, cutting memory usage by 30% to 70% [9]. Microscaling formats like MXFP4 preserve dynamic numerical range without retraining [11].

Pruning removes redundant parameters to reduce workloads [12][13]. Wanda scores parameter importance by multiplying weight magnitudes by input activation norms, reaching 50% sparsity up to 300× faster than second-order methods [12]. Fluctuation-based Adaptive Structured Pruning (FLAP) removes channels and layers based on feature fluctuation metrics [13]. FLAP uses bias compensation to recover accuracy, creating dense matrices that run directly on standard tensor hardware [13].

## Memory-Centric Serving and Attention Kernels
Key-Value (KV) cache allocation creates major memory bottlenecks during multi-turn serving [3][14]. Standard frameworks allocate contiguous buffers for maximum sequence lengths, wasting 60% to 80% of memory through fragmentation [3]. PagedAttention eliminates this waste by applying virtual memory paging to KV tensors, partitioning them into non-contiguous physical blocks [3]. In the vLLM engine, this layout enables continuous batching and zero-copy prefix sharing, boosting serving throughput by 2× to 4× [9][3].

Attention kernels and architectural designs further reduce data movement [4][14]. FlashAttention tiles attention computation across on-chip SRAM and applies online softmax normalization, converting quadratic memory traffic into linear transfers [4]. FlashAttention-3 refines this with warp specialization and low-precision tensor operations [4]. At the architectural level, Multi-Head Latent Attention (MLA) compresses keys and values into low-rank latent vectors [14]. During generation, matrix absorption folds projection matrices into queries, cutting KV cache memory by 8× to 14× compared to standard attention [14].

## Speculative Decoding and Parallel Verification
Standard autoregressive decoding requires a full forward pass for every generated token [5][6]. Speculative decoding breaks this serial bottleneck by generating draft candidates quickly and verifying them in parallel with the target model [5][6]. EAGLE predicts draft features at the second-to-top hidden layer rather than decoding discrete tokens, reaching acceptance rates near 0.80 and speeding up inference by up to 2.8× [5]. Medusa attaches multiple lightweight heads to the base model trunk, predicting several future tokens concurrently and validating candidates with tree-structured attention masks for 2.2× to 2.8× gains [6].

Recent architectures integrate speculative verification directly into pre-training and diffusion frameworks [15][16]. Multi-Token Prediction (MTP) trains the core architecture to forecast multiple future tokens simultaneously, improving reasoning and enabling self-speculative decoding without separate draft models for 2.7× to 3.0× speedups [15]. In diffusion models, SpecFold folds multi-branch attention and shares prefix computations to eliminate redundant calculations during parallel verification [16].

## Architecture and Training Strategies for Small Language Models
Small language models demonstrate that compact networks achieve strong capability when trained on massive token volumes [7][17][18]. Phi-3 trains a 3.8B model across 3.3T tokens of filtered educational and synthetic data, allowing an INT4 model to fit in 1.8GB of memory on smartphones [7]. Gemma 2 stabilizes training with logit soft-capping and distills knowledge from larger teacher models [17]. SmolLM2 uses curated synthetic datasets like Cosmopedia-v2 to outperform older 1.5B baselines in math and code [10]. Qwen 2.5 scales models between 0.5B and 3B parameters across 18T tokens using tied embeddings and grouped-query attention [18].

Deployment pipelines optimize these small architectures for edge hardware [7][8]. Llama 3.2 combines structured pruning, logit distillation, and quantization-aware training to build 1B and 3B parameter models tuned for edge devices [8]. Pairing these models with weight rotation like SpinQuant and runtimes such as ExecuTorch and Arm Kleidi AI yields 2.5× to 4.2× latency improvements on mobile processors within a 2GB memory budget [8].

## Trends and open problems
- Architecture-quantization co-design: Future networks integrate primitives like logit soft-capping, tied embeddings, and latent attention into base training, preventing outlier spikes during low-bit execution [17][18][14].
- Adaptive speculative routing: Draft acceptance rates fluctuate across complex tasks [5][6]. Dynamic head selection and prefix-folded multi-branch verification are essential for consistent speedups [16].
- Sub-4-bit compression: Pushing weights down to 2-bit or 3-bit precision without quality loss requires combining quantization-aware training with microscaling formats and second-order error compensation [9][8][11].
- On-device agentic execution: Running compact models as autonomous local agents requires low latency during tool use and efficient prefix caching under strict mobile power constraints [7][10][8].

## References
[1] AWQ: Activation-aware Weight Quantization for LLM Compression, Acceleration. hf-search. https://huggingface.co/papers/2306.00978 (2023-06-01)
[2] SmoothQuant: Accurate, Efficient Post-Training Quantization for Large Language Models. hf-search. https://huggingface.co/papers/2211.10438 (2022-11-18)
[3] PagedAttention, vLLM: Efficient Memory Management for Large Language Model Serving. hf-search. https://huggingface.co/papers/2309.06180 (2023-09-12)
[4] FlashAttention: Fast, Memory-Efficient Exact Attention with IO-Awareness. hf-search. https://huggingface.co/papers/2205.14135 (2022-05-27)
[5] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. hf-search. https://huggingface.co/papers/2401.15077 (2024-01-26)
[6] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. web. https://proceedings.mlr.press/v235/cai24b (2024-07-08)
[7] Phi-3 Technical Report: Highly Capable Language Model Locally on Your Phone. hf-daily. https://huggingface.co/papers/2404.14219 (2024-04-22)
[8] Introducing Quantized Llama Models with Increased Speed, Reduced Memory Footprint. web. https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/ (2024-10-24)
[9] Accelerating LLM Inference with Post-Training Weight, Activation using AWQ, GPTQ on SageMaker AI. web. https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/ (2026-01-09)
[10] SmolLM2: When Smol Goes Big -- Data-Centric Training of Small Language Model. hf-search. https://huggingface.co/papers/2502.02737 (2025-02-04)
[11] Post Training Quantization of Large Language Models with Microscaling Formats. web. https://proceedings.mlr.press/v262/sharify24a.html (2024-12-10)
[12] Wanda: Simple, Effective Pruning Approach for Large Language Models. hf-search. https://huggingface.co/papers/2306.11695 (2023-06-20)
[13] Fluctuation-based Adaptive Structured Pruning for Large Language Models (FLAP). hf-search. https://huggingface.co/papers/2312.11983 (2023-12-19)
[14] TransMLA: Multi-Head Latent Attention Is All You Need, Hardware-Centric Analysis of MLA. hf-search. https://huggingface.co/papers/2502.07864 (2025-02-11)
[15] Better, Faster Large Language Models via Multi-token Prediction. hf-daily. https://huggingface.co/papers/2404.19737 (2024-04-30)
[16] SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models. hf-daily. https://huggingface.co/papers/2610.04875 (2026-10-06)
[17] Gemma 2: Improving Open Language Models at Practical Size. hf-search. https://huggingface.co/papers/2408.00118 (2024-08-01)
[18] Qwen2.5 Technical Report. hf-search. https://huggingface.co/papers/2412.15115 (2024-12-19)
