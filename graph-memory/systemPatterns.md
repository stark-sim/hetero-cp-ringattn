# System Patterns

架构概览、关键设计模式与架构决策。

### 异构分布式推理的数值验证策略

type: `belief` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `memory-bank/systemPatterns.md`

BF16 场景下，跨平台 BLAS 差异导致 logits 数值对比不是有意义的 correctness 指标。Correctness 应分层：L1 float32 数学正确性（cargo test synthetic weights）、L2 工程正确性（argmax 一致性/文本任务指标）、L3 端到端冒烟。强证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异，证明差异主要来自 BF16 online softmax block-wise 处理顺序，而非跨平台 bug。

_updated: 2026-06-29 05:34:19_
### 产品问题：异构设备协作支撑超长 context

type: `blueprint` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `memory-bank/productContext.md`

长上下文需求持续增长，但单卡显存和同构高端集群供给无法无限增长。现实资源通常是混合的（CUDA、Apple Silicon/MLX、其他加速器）。HCP 的问题是：能否通过增加异构 domain / 设备继续支撑任务，而不是受制于最强单卡。

_updated: 2026-06-29 05:34:19_
### 架构概览：Rust + C++ 为主、Python 原型为历史对照

type: `blueprint` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `memory-bank/systemPatterns.md`

C++ 部分定义 HCP Ring Attention 低边界 runtime 抽象和 libtorch bridge。Rust 部分负责 correctness model、report、可序列化协议 schema 和 P2P transport smoke。每个 domain 持有本地 Q chunk，ring 中持续传递 K/V block，每个 domain 更新 online softmax state。

_updated: 2026-06-29 05:34:19_
### 技术栈：Rust + C++ + Python 原型

type: `component` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/techContext.md`

Core: C++17, CMake 3.16+, Rust 2021, Python 3。
Libtorch/PyTorch 2.11.0, tch-rs 0.24.0（可选 tch-backend）。
QUIC: quinn 0.11 + rustls 0.23 + rcgen 0.13。
模型权重：safetensors, tokenizers, half。

_updated: 2026-06-29 05:34:19_
### 架构决策：采用原始论文 P2P 而非 PyTorch CP Collective

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `memory-bank/systemPatterns.md`

Ring Attention 原始论文（Liu et al. 2023）的通信本就是 P2P send/recv。PyTorch 2.7+ Context Parallel 改用 all-gather/all-to-all 是对同构 NVLink 集群的工程优化，不是数学必须。P2P 支持异构、非均分、任意拓扑，更符合 HCP 定位。

_updated: 2026-06-29 05:34:19_
### 容量感知非均等 CP 分片是异构长 context 的必需

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `memory-bank/systemPatterns.md`

2026-06-19 1M context 验证：24GB CUDA + 16GB HIP 无法通过 1:1 分片完成 1M。必须使用 capacity-aware 不均等分片（white 750K / pearl 250K，即 3:1）。均匀分片在异构显存下会因小显存设备 OOM 而失败；按可用显存比例分配 chunk 才能使 heterogeneous ring 达到可行性边界。

_updated: 2026-06-29 05:34:19_
### 产品决策：P2P、correctness 优先、结构化实验产物

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `memory-bank/productContext.md`

HCP 不是 HLPP 的细粒度版本，而是 intra-layer / low-boundary 路线。跨异构域坚持 P2P，不把 all-gather / reduce-scatter / all-to-all / all-reduce 作为主假设。correctness 和协议闭环优先于性能图。每个阶段输出结构化实验产物。

_updated: 2026-06-29 05:34:19_
### 部署铁律：1 GPU = 1 worker

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/techContext.md`

每个 worker 加载完整模型权重。3B bf16 × 2 workers 在 RTX 4090 loopback 上实测 OOM。--local-domain-ids 仅限 <1GB 小模型的本地协议验证；生产/大规模验证必须每平台一 worker。

_updated: 2026-06-29 05:34:19_
### Correctness-First 开发纪律

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `memory-bank/systemPatterns.md`

当前处于 correctness 验证阶段，尚未进入性能调优。在全部 target 设备上稳定通过前，禁止实施量化、近似 attention、非 deterministic kernel、投机/跳过层优化。每次提出优化前必须写 trade-off 分析。

_updated: 2026-06-29 05:34:19_
### Striped Attention HCP 适配计划

type: `decision` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `docs/STRIPE_ATTENTION_ADAPTATION_PLAN.md`

已将详细实现计划写入 docs/STRIPE_ATTENTION_ADAPTATION_PLAN.md。
核心思路：通过细粒度 scheduling unit 实现 capacity-aware 不均等 stripe；用原始位置 id 计算 causal mask；worker 输入/输出做 permutation / inverse-permutation；online softmax 与 KV transport 不变。
实施顺序：先在 correctness model 验证，再改 coordinator/worker，最后跑 uneven 分布式 smoke。

_updated: 2026-06-29 06:18:40_
### QUIC Transport 配置：512MB stream window / 1GB connection window / 300s idle timeout

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.8 · source: `memory-bank/techContext.md`

显式覆盖 quinn 默认值：max_concurrent_bidi/uni_streams=256, keep_alive_interval=1s, max_idle_timeout=300s, stream_receive_window=512MB, receive_window=1GB, send_window=1GB。历史上因 send_window 和 stream_receive_window 不足导致 16K/64K 死锁。

_updated: 2026-06-29 05:34:19_
### 可插拔域内后端架构

type: `decision` · status: `held` · confidence: 0.8 · importance: 0.8 · source: `memory-bank/systemPatterns.md`

HCP 的边界是跨域低层协议（P2P KV ring + online softmax），域内实现是黑盒。同构域内可通过接口实现替换为 vLLM、TensorRT-LLM、MLX 等社区框架。Python Worker SDK 和 Rust Worker SDK 提供标准接口。

_updated: 2026-06-29 05:34:19_
