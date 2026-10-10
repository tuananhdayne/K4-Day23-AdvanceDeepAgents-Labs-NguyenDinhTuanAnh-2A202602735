# Survey of Video and Multimodal Generation: Architectures, Evaluation, Efficiency, and Frontier Trends

## TL;DR
- Video models combine causal 3D autoencoders with flow-matching transformers to generate compressed spacetime tokens efficiently [1][2][3][4].
- Discrete autoregression unifies video with language models, while continuous diffusion transformers provide superior visual detail [5][6][7].
- Evaluation has shifted from distribution metrics to physical tests, showing that visual quality correlates weakly with physical accuracy [8][9][10][11].
- Layer-skipping caches, mixed-precision quantization, and latent monitoring reduce inference latency and block temporal safety exploits [12][13][14].

## Background
Early video synthesis relied on generative adversarial networks and cascaded pixel-space diffusion models [2]. Denoising raw pixel arrays preserved prompt details, but processing full spacetime volumes demanded excessive memory and compute [2]. These limits restricted early systems to low frame rates, small resolutions, and short clips [2].

Modern frameworks resolve this bottleneck by splitting spatiotemporal compression from sequence generation [2][3]. Causal 3D autoencoders compress raw video into compact continuous latents before transformer backbones perform generative denoising [1][3][4]. In parallel, language modeling inspired discrete visual tokenization and autoregressive next-token synthesis [5][6][7]. Video models now serve as engines for creative media, simulation, and world modeling [3][11][15].

## Latent Video Transformers and Flow Matching
Spatiotemporal compression relies on causal 3D autoencoders that compress video across space and time [1][3]. CogVideoX compresses latents fourfold temporally and eightfold spatially [1]. Step-Video-T2V achieves eightfold temporal and sixteenfold spatial compression using causal 3D convolutions [16]. To prevent future frames from corrupting earlier states during streaming, Wan-VAE restricts receptive fields to current and past frames [4][3]. This causal design limits memory use and enables smooth decompression [4].

Within latent space, generative modeling centers on diffusion transformers paired with flow matching [1][16][4]. Spatiotemporal latents are flattened into tokens and processed through attention layers [1][4]. Step-Video-T2V scales full attention to 30 billion parameters, while Wan pairs cross-attention with umT5 and Qwen2-VL embeddings [16][4]. CogVideoX uses an expert transformer with adaptive normalization to link text and visual tokens bidirectionally [1]. Rather than following curved diffusion schedules, flow matching builds straight trajectories that reach high fidelity in fewer steps [3][16][4].

## Discrete Autoregressive and Multi-Scale Generation
Discrete autoregressive architectures cast video generation as sequential token prediction [5][7]. VideoPoet trains a decoder-only language model on discrete tokens from MAGVIT-v2, supporting text-to-video, editing, and video-to-audio tasks in one system [5]. HiTVideo adds hierarchical causal tokenizers that compress spacetime volumes by an extra factor of eight, predicting broad layouts before filling in fine details [7]. These discrete setups bring video generation directly into multimodal language frameworks [5][7].

To bypass raster-scan latency in token prediction, visual autoregressive modeling generates feature maps scale by scale from coarse to fine [6]. This next-scale method matches continuous diffusion quality while keeping exact likelihood training [6]. In parallel, hybrid architectures like CausVid combine distribution matching distillation with key-value caching [17][18]. These hybrid pipelines deliver real-time streaming video with fast autoregressive sampling [17][18].

## Evaluation from Visual Quality to Physical Faithfulness
Video evaluation has moved away from distribution metrics like Fréchet Video Distance and Inception Score [8][10]. These older metrics measure global feature spreads in frozen spaces, missing motion dynamics, prompt adherence, and temporal stability [8][10]. Modern benchmark suites break video quality into isolated dimensions [8][10]. VBench and VBench++ evaluate 16 distinct dimensions including motion smoothness, temporal flickering, subject consistency, and text alignment [8]. EvalCrafter combines 17 objective metrics using optical flow tools, aesthetic scorers, and vision-language models grounded in human ratings [10].

Recent benchmarks also test whether video generators follow physical laws [9][11]. VBench-2.0 assesses mechanical, thermal, and phase-change consistency using optical flow tracking [9]. WorldModelBench tests systems across 56 physical domains, including gravity, fluid dynamics, and collision physics [11]. The benchmark revealed that aesthetic quality correlates weakly with physical realism, showing a correlation coefficient of r = 0.28 [11]. Video models often produce sharp scenes that violate basic physical principles [11].

## Inference Acceleration and Temporal Safety Alignment
Iterative denoising in large video transformers creates heavy computational costs at inference [12][13]. TeaCache addresses this by tracking output similarity across consecutive transformer passes [12]. By estimating feature changes from timestep embeddings, TeaCache skips redundant transformer layers when latent differences are minor [12]. It yields 1.86x to 6.83x speedups on models like Open-Sora, Latte, Wan2.1, and CogVideoX while keeping quality drops below 0.1% on VBench [12]. Alongside caching, QuantCache applies mixed-precision quantization across attention maps, suppressing activation spikes to fit high-resolution generation onto constrained hardware [13].

Safety alignment brings unique challenges in video because temporal combinations create risks unseen in static images [14]. Standard text filters often miss temporal composition attacks, where benign individual actions combine across frames to create unsafe outcomes [14]. Defenses like Latent Variable Defense and automated red-teaming detect harmful trajectories within diffusion [14]. Runtime anomaly checks steer latent states back to safe paths, preserving safety across dynamic video sequences [14].

## Trends and open problems
- **Interactive 4D World Modeling**: Advancing from 2D video generation into interactive 3D and 4D environments using action conditioning and 3D Gaussians for spatial consistency [11][15].
- **Synchronized Multimodal Co-Generation**: Generating paired audio and video streams with precise temporal alignment while eliminating cross-modal bottlenecks and audio hallucinations [19].
- **Physics-Informed Post-Training**: Applying Direct Preference Optimization and reinforcement learning to penalize physical errors and teach models accurate dynamics [17][11][16].
- **Real-Time Interactive Streaming**: Combining causal compression, key-value caching, and distribution distillation to generate continuous video streams with minimal latency [17][7][18].

## References
[1] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[2] Survey of Video Diffusion Models: Foundations, Methodologies, Applications, Trends. web. https://arxiv.org/html/2504.16081v3 (2025-04-20)
[3] From Pixels to Worlds: Survey on High-fidelity Video Generation. web. https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/ei/38/7/IMAGE-266 (2025-01-15)
[4] Wan: Open Advanced Large-Scale Video Generative Models. hf-search. https://huggingface.co/papers/2503.20314 (2025-03-26)
[5] VideoPoet: Large Language Model for Zero-Shot Video Generation. hf-search. https://huggingface.co/papers/2312.14125 (2023-12-21)
[6] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. hf-search. https://huggingface.co/papers/2404.02905 (2024-04-03)
[7] HiTVideo: Hierarchical Tokenizers for Enhancing Text-to-Video Generation with Autoregressive LLMs. web. https://arxiv.org/html/2503.11513v1 (2025-03-18)
[8] VBench++: Comprehensive Versatile Benchmark Suite for Video Generative Models. hf-daily. https://huggingface.co/papers/2411.13503 (2024-11-20)
[9] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. hf-search. https://huggingface.co/papers/2503.21755 (2025-03-27)
[10] EvalCrafter: Benchmarking Large Video Generation Models. hf-search. https://huggingface.co/papers/2310.11440 (2023-10-17)
[11] WorldModelBench: Judging Video Generation Models As World Models. web. https://worldmodelbench-team.github.io/ (2025-05-11)
[12] Timestep Embedding Tells: It is Time to Cache for Video Diffusion Model. web. https://huggingface.co/papers/2411.19108 (2024-11-04)
[13] QuantCache: Adaptive Importance-Guided Quantization with Hierarchical Latent Layer Caching. hf-search. https://huggingface.co/papers/2503.06545 (2025-03-09)
[14] Towards Understanding Unsafe Video Generation. hf-search. https://huggingface.co/papers/2407.12581 (2024-07-17)
[15] 4DWorldBench: Comprehensive Evaluation Framework for 3D/4D World Generation Models. hf-search. https://huggingface.co/papers/2511.19836 (2025-11-25)
[16] Step-Video-T2V Technical Report: Practice, Challenges, Future of Video Foundation Model. hf-search. https://huggingface.co/papers/2502.10248 (2025-02-14)
[17] Video Generation Models: Survey of Post-Training, Alignment. web. https://arxiv.org/html/2610.00812 (2026-09-30)
[18] Controllable Video Generation: A Survey. web. https://arxiv.org/html/2507.16869v1 (2025-07-22)
[19] VABench: Comprehensive Benchmark for Audio-Video Generation. hf-search. https://huggingface.co/papers/2512.09299 (2025-12-10)
