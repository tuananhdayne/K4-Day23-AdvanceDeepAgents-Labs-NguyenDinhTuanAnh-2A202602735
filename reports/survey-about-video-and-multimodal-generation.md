# Survey of Video and Multimodal Generation: Architectures, Evaluation, Efficiency, and Frontier Trends

## TL;DR
- Modern video generation has broadly transitioned from raw pixel-space cascaded 2D/3D UNets toward a standard tripartite architecture combining Causal 3D Variational Autoencoders (VAEs), Flow-Matching Diffusion Transformers (DiTs), and unified multimodal conditioning [1][2][3].
- Discrete visual autoregression and next-scale prediction offer unified next-token representations across text, video, and audio modalities, though continuous DiTs remain dominant for high perceptual fidelity and scaling [4][5][6][7].
- Evaluation methodologies have shifted from distribution metrics (FVD, IS) toward multidimensional human-aligned suites (VBench, EvalCrafter) and intrinsic physical reasoning benchmarks (VBench-2.0, WorldModelBench), revealing that visual quality correlates poorly (r = 0.28) with true physical consistency [8][9][10][11].
- Inference acceleration leverages training-free timestep-aware caching (TeaCache) and post-training mixed-precision quantization (QuantCache) to deliver 2x–6x speedups with negligible quality loss [12][13].
- Multimodal safety alignment faces unique temporal dynamics, while frontier open research expands toward 3D/4D dynamic world models and synchronized audio-visual co-generation [14][15][16].

## Background
Early visual generative models relied heavily on Generative Adversarial Networks (GANs) and pixel-space cascaded diffusion pipelines, such as Make-A-Video and Imagen Video [2]. While pixel-based diffusion models directly denoise raw video volumes and maintain tight text alignment, they impose exorbitant compute and memory overheads that restrict scalability across high resolutions and long frame horizons [2].

To overcome these computational bottlenecks, the field transitioned to Latent Video Diffusion Models, decoupling spatial-temporal compression from generative sequence modeling [2]. By compressing high-dimensional video arrays into compact continuous latent spaces via 3D VAEs, foundation models can scale training compute efficiently [1][2][3]. Concurrently, the success of large language models inspired discrete visual tokenization and autoregressive generative modeling [4][5][7]. Today, video and multimodal generation serves as the primary backbone for creative media synthesis, autonomous agent simulation, and physics-grounded world modeling [3][11][15].

## Architectural Foundations: Causal 3D VAEs and Flow-Matching DiTs
The prevailing architecture across modern frontier video generators combines a high-compression Causal 3D VAE with a spatiotemporal Diffusion Transformer (DiT) operating under Flow Matching objectives [1][17][18][3].

### Spatiotemporal Causal 3D VAEs
Raw video tensors $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$ are compressed by 3D VAEs across spatial and temporal dimensions into latent representations $\mathbf{Z} \in \mathbb{R}^{t \times h \times w \times c}$ [1][3]. CogVideoX employs a customized 3D VAE compressing $4\times$ temporally and $8\times$ spatially [1]. Step-Video-T2V introduces a Deep Compression Video-VAE integrating Causal 3D Convolutional Modules and Dual-Path Latent Fusion to achieve $8\times$ temporal and $16\times 16$ spatial downsampling [17]. WAN-VAE in the Wan2.1 family utilizes causal temporal caching and block encoding, strictly restricting receptive fields to $\tau \le t$ to eliminate future temporal leakage while preserving memory efficiency during streaming decompression [18][3].

### Diffusion Transformers and Flow Matching
Latent representations are sliced into $p \times p \times q$ spacetime patch tokens and processed via full spatiotemporal attention or single-stream cross-attention architectures [1][18][3]. Wan adopts a single-stream DiT with cross-attention modules conditioned on umT5 and Qwen2-VL multimodal embeddings [18], whereas CogVideoX utilizes an Expert Transformer with Adaptive Layer Normalization (AdaLN) for bidirectional text-visual fusion [1]. Step-Video-T2V scales 3D full-attention DiT parameters up to 30 billion under the Flow Matching framework [17]. Flow Matching (rectified flow / optimal transport paths) replaces classical DDPM/DDIM schedules, establishing straighter ODE trajectories that dramatically reduce required denoising steps [17][18][3].

## Discrete Autoregressive and Next-Scale Modeling
In parallel with continuous diffusion transformers, autoregressive architectures treat visual and multimodal generation as sequential next-token or next-scale prediction [4][5][6][7].

### Multimodal Next-Token Prediction
VideoPoet demonstrates zero-shot video generation, editing, and video-to-audio synthesis using a decoder-only language model over discrete tokens discretized by MAGVIT-v2 with Lookup-Free Quantization (LFQ) [4]. HiTVideo implements hierarchical 3D Causal VAEs with LFQ to provide an additional $8\times$ spatiotemporal compression ratio, generating coarse semantic layouts at top codebook tiers before progressively synthesizing high-frequency details [7].

### Next-Scale Prediction and Hybrid Distillation
Visual Autoregressive Modeling (VAR) introduces next-scale prediction, where autoregressive models predict entire multi-scale feature maps from coarse to fine rather than raster-scan sequences, matching or exceeding DiT performance and bridging discrete token modeling with diffusion-like multi-scale refinement [5]. Furthermore, hybrid causal diffusion architectures (e.g., CausVid) leverage Distribution Matching Distillation (DMD) and KV caching to enable continuous streaming generation with autoregressive inference efficiency [19][2][6].

## Evaluation Methodologies: From Perceptual Quality to Physical Faithfulness
Standard video evaluation historically relied on Fréchet Video Distance (FVD) and Inception Score (IS), which evaluate distribution distances in frozen feature spaces but correlate weakly with human visual judgment and temporal dynamics [8][10].

### Multi-Dimensional Human-Aligned Benchmarks
VBench and VBench++ decompose video evaluation into 16 disentangled hierarchical dimensions covering Video Quality (temporal flickering, motion smoothness, dynamic degree, subject consistency) and Video-Condition Consistency (spatial relationship, human action, object class) [8]. EvalCrafter aggregates 17 objective metrics across visual, content, motion, and text-alignment axes using FlowNet/RAFT, DOVER, and BLIP-2/CLIP, incorporating human-calibrated regression coefficients [10].

### Physical Consistency and World Modeling Benchmarks
Recent benchmarks assess whether generative video models function as physically faithful world simulators [9][11]. VBench-2.0 establishes intrinsic faithfulness evaluations across mechanical, thermal, and phase-transition state changes, measuring 3D/4D geometric consistency through SIFT/FLANN keypoint matching, RANSAC filtering, and RAFT optical flow normalization [9]. WorldModelBench stress-tests models across 56 subdomains against 5 fundamental physical principles: Newton's First Law, solid mechanics, fluid mechanics, object impenetrability, and gravity [11]. Crucially, WorldModelBench demonstrates that visual aesthetic scores exhibit very low correlation ($r = 0.28$) with physical law adherence, showing that visually pleasing models frequently violate real-world dynamics [11].

## Inference Acceleration and Efficiency Optimization
Deploying large video foundation models requires substantial compute and memory optimizations for interactive generation [12][13].

### Timestep-Aware DiT Caching
Timestep Embedding Aware Cache (TeaCache) exploits temporal redundancy across DiT denoising timesteps [12]. By estimating output differences from timestep-embedding modulated inputs via polynomial curve fitting, TeaCache dynamically skips redundant transformer blocks when relative L1 distance is below a calibrated threshold [12]. TeaCache yields $4.41\times$ to $6.83\times$ acceleration on Open-Sora-Plan and $1.86\times$ to $3.28\times$ on Latte/Open-Sora with less than $0.1\%$ VBench degradation, seeing broad adoption across Wan2.1, HunyuanVideo, Cosmos, and CogVideoX [12].

### Quantization and Memory Optimization
QuantCache integrates post-training low-bit quantization with hierarchical latent and layer caching [13]. By applying importance-guided mixed-precision quantization (INT8/INT4/FP6) across 3D spatiotemporal attention maps, QuantCache mitigates activation outliers caused by temporal dynamics, enabling high-resolution video generation on edge and consumer hardware [13].

## Safety Alignment and Temporal Defense
Multimodal safety alignment in video models introduces challenges distinct from static text-to-image systems [14]. Research reveals that static text filters fail against temporal composition attacks and implicit prompt jailbreaks where harmful semantics emerge dynamically across sequential frames [14]. Addressing these vulnerabilities requires Latent Variable Defense (LVD), automated adversarial red-teaming pipelines (e.g., DREAM, ART), and runtime anomaly filtering to steer latent trajectories away from unsafe manifolds during iterative denoising [14].

## Trends and open problems
- **Interactive 3D and 4D World Models**: Transitioning from monocular video generation to interactive 4D world simulation (TeleWorld, SWiT-4D, 4DWorldBench) combining sliding-window transformers, 3D Gaussian Splatting, and action conditioning to maintain long-horizon spatial-temporal consistency [11][15].
- **Synchronized Audio-Video Co-Generation**: Joint generation of synchronized multi-modal audio and video streams (VABench) while overcoming cross-attention alignment bottlenecks, acoustic hallucination, and multi-shot narrative cut transitions [16].
- **Physics-Grounded Post-Training Alignment**: Integrating Direct Preference Optimization (Video-DPO) and physics-informed reinforcement learning to penalize dynamic law violations and enhance spatiotemporal coherence [17][6][11].
- **Real-Time Streaming Generation**: Bridging full-sequence diffusion with causal autoregressive distillation (DMD, KV caching) to achieve low-latency interactive video streaming without temporal error accumulation [19][6][7].

## References
[1] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[2] Survey of Video Diffusion Models: Foundations, Methodologies, Applications, Trends. web. https://arxiv.org/html/2504.16081v3 (2025-04-20)
[3] From Pixels to Worlds: Survey on High-fidelity Video Generation. web. https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/ei/38/7/IMAGE-266 (2025-01-15)
[4] VideoPoet: Large Language Model for Zero-Shot Video Generation. hf-search. https://huggingface.co/papers/2312.14125 (2023-12-21)
[5] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. hf-search. https://huggingface.co/papers/2404.02905 (2024-04-03)
[6] Video Generation Models: Survey of Post-Training, Alignment. web. https://arxiv.org/html/2610.00812 (2026-09-30)
[7] HiTVideo: Hierarchical Tokenizers for Enhancing Text-to-Video Generation with Autoregressive LLMs. web. https://arxiv.org/html/2503.11513v1 (2025-03-18)
[8] VBench++: Comprehensive Versatile Benchmark Suite for Video Generative Models. hf-daily. https://huggingface.co/papers/2411.13503 (2024-11-20)
[9] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. hf-search. https://huggingface.co/papers/2503.21755 (2025-03-27)
[10] EvalCrafter: Benchmarking Large Video Generation Models. hf-search. https://huggingface.co/papers/2310.11440 (2023-10-17)
[11] WorldModelBench: Judging Video Generation Models As World Models. web. https://worldmodelbench-team.github.io/ (2025-05-11)
[12] Timestep Embedding Tells: It is Time to Cache for Video Diffusion Model. web. https://huggingface.co/papers/2411.19108 (2024-11-04)
[13] QuantCache: Adaptive Importance-Guided Quantization with Hierarchical Latent Layer Caching. hf-search. https://huggingface.co/papers/2503.06545 (2025-03-09)
[14] Towards Understanding Unsafe Video Generation. hf-search. https://huggingface.co/papers/2407.12581 (2024-07-17)
[15] 4DWorldBench: Comprehensive Evaluation Framework for 3D/4D World Generation Models. hf-search. https://huggingface.co/papers/2511.19836 (2025-11-25)
[16] VABench: Comprehensive Benchmark for Audio-Video Generation. hf-search. https://huggingface.co/papers/2512.09299 (2025-12-10)
[17] Step-Video-T2V Technical Report: Practice, Challenges, Future of Video Foundation Model. hf-search. https://huggingface.co/papers/2502.10248 (2025-02-14)
[18] Wan: Open Advanced Large-Scale Video Generative Models. hf-search. https://huggingface.co/papers/2503.20314 (2025-03-26)
[19] Controllable Video Generation: A Survey. web. https://arxiv.org/html/2507.16869v1 (2025-07-22)
