# 系统模式

## 架构概览

本仓采用 Rust + C++ 为主、Python 原型为历史对照的结构。C++ 部分定义 HCP Ring Attention 的低边界 runtime 抽象和 libtorch bridge；Rust 部分负责 correctness model、report、可序列化协议 schema 和当前 P2P transport smoke。

HCP 的核心形态是：每个 domain 持有本地 `Q chunk`，ring 中持续传递 `K/V block`，每个 domain 对收到的 block 更新 online softmax state，直到完整遍历 ring。

## 项目结构

```text
project-root/
├── CMakeLists.txt
├── Cargo.toml / Cargo.lock / build.rs
├── README.md / AGENTS.md
├── include/hcp_ringattn/core/         # C++ 独立公共类型、协议、runtime 抽象
│   ├── status.h
│   ├── tensor_types.h
│   ├── ringattn_protocol.h
│   └── ringattn_runtime.h
├── src/                               # C++ NoOp runtime 与 coordinator smoke
│   ├── ringattn_runtime.cc
│   ├── ringattn_coordinator_smoke_main.cc
│   └── rust_bridge.cc
├── python/                            # controller、worker、online softmax 原型、HCP Worker SDK
│   ├── ringattn_controller.py
│   ├── ringattn_worker.py
│   ├── ringattn_kernel_stub.py
│   └── hcp_worker_sdk/                # Python 版 Worker SDK（vLLM/TensorRT-LLM/MLX 适配基础）
│       ├── __init__.py
│       ├── types.py
│       ├── backend.py
│       ├── transport.py
│       └── server.py
├── rust/                              # Rust correctness model、report、C++ bridge、模型实现
│   ├── src/
│   │   ├── main.rs                    # CLI + smoke 入口
│   │   ├── lib.rs
│   │   ├── protocol.rs                # RingAttnMessage、P2P transport、CP node runtime
│   │   ├── correctness.rs             # Rust correctness model（7 cases）
│   │   ├── report.rs                  # 结构化 report 生成
│   │   ├── tch_backend.rs             # tch-rs 桥接（6 个函数）
│   │   ├── compute_runtime.rs         # ComputeRuntime trait（Tch/NoOp）
│   │   ├── kv_transport.rs            # KvTransport trait + Mock/Tcp/Quic 实现
│   │   ├── quic_transport.rs          # QuicKvTransport（quinn-based）
│   │   ├── distributed_worker.rs      # 多进程分布式 worker（薄壳：解析参数 → 创建后端 → 运行 runtime）
│   │   ├── distributed_coordinator.rs # 多进程分布式 coordinator
│   │   ├── distributed_protocol.rs    # WorkerCommand/WorkerResponse 协议
│   │   ├── infer.rs                   # inference CLI
│   │   ├── worker_sdk/                # Rust Worker SDK：协议层 + 传输层 与 模型计算层 解耦
│   │   │   ├── backend.rs             # WorkerBackend trait：框架接入点
│   │   │   ├── runtime.rs             # WorkerRuntime<B>：QUIC 协议循环、handshake、command loop
│   │   │   ├── tch_backend.rs         # TchWorkerBackend：默认 tch-rs 实现
│   │   │   └── mod.rs                 # 模块导出
│   │   └── model/                     # 真实模型实现
│   │       ├── mod.rs
│   │       ├── model.rs               # LlamaModel、Generator、DecoderLayer
│   │       ├── backend.rs             # AttentionBackend（Local/HcpRingAttention）
│   │       ├── weights.rs             # ModelWeights、ModelConfig（safetensors 加载）
│   │       ├── attention.rs           # GqaAttention、RotaryEmbedding
│   │       └── mlp.rs                 # SwiGLU MLP
│   └── target/
├── config/                            # 最小 ring 配置
│   └── minimal_2domain_ring.json
├── scripts/                           # 本地/远程 smoke 入口
│   ├── run_local_ringattn_smoke.sh
│   ├── run_rust_ringattn_smoke.sh
│   ├── run_tch_ringattn_smoke.sh
│   ├── run_rust_remote_p2p_server.sh
│   ├── run_rust_remote_p2p_client.sh
│   ├── run_rust_remote_cp_node.sh
│   └── run_rust_remote_cp_3node_smoke.sh
├── docs/                              # 设计、验证、路线图、产品论证、部署指南
│   ├── DESIGN.md
│   ├── DEPLOYMENT_GUIDE.md            # 手动部署指南（单节点/双节点异构）
│   ├── PLUGIN_ARCHITECTURE.md         # 可插拔域内后端架构（vLLM 适配设计）
│   ├── VLLM_INTEGRATION.md            # vLLM Worker 适配器详细设计
│   ├── HISTORY_AND_LESSONS.md
│   ├── HLPP_VS_HCP.md
│   ├── PRODUCT_THESIS.md
│   ├── PROTOCOL_SMOKE.md
│   ├── RINGATTN_MODEL.md
│   ├── ROADMAP.md
│   ├── RUST_CPP_TORCH_PLAN.md
│   ├── TCH_RS_USAGE_PLAN.md
│   └── VALIDATION_PLAN.md
├── reports/                           # 实验报告输出目录
└── memory-bank/                       # 跨会话上下文
```

## 使用中的设计模式

- **低边界协议隔离**：`RingAttnProtocol` 只表达 attention 内部跨域数据流，不带 HLPP 的 layer plan / batch 语义。
- **域内黑盒 runtime**：`RingAttnRuntime` 只定义跨域合同，不规定 CUDA / MLX / NPU 内部实现。
- **最小可见 smoke**：C++ coordinator smoke 先验证 standalone build 和 runtime lifecycle；Rust smoke 同时验证 correctness、protocol、C++ bridge、可选 libtorch device bridge。
- **Context Parallel ring 消息流**：每个 source domain 的 K/V block 沿 ring 逐 hop 转发；每个跨域 hop 都是一个可序列化 `RingAttnMessage`。
- **Transport 可替换**：P2P 是 point-to-point message 语义；`local_p2p_queue` 用于本地协议闭环，`tcp_remote_pair` 用于双进程 / 双机器工程 smoke，后续可替换为 UCX/RDMA、NCCL send/recv 或 GPU-direct。
- **CP 节点双角色**：`cp_ring_node_runtime` 中每个 domain thread 同时具备 inbound receiver 和 outbound peer；`tcp_remote_cp_node` 已在 Mac/GPU 上验证每个进程同时 listener + outbound peer，并在 3-node remote ring 中验证多 hop forwarding。
- **CP payload 驱动 device compute**：`RingAttnMessage.payload` 携带 float32 K/V bytes；`cp_ring_node_runtime` 和 `tcp_remote_cp_node` 都会捕获每次 compute update 的 payload block，并通过 C ABI 驱动 C++ ATen/libtorch 在目标设备上执行 payload-backed attention block compute、online softmax state update 和小尺寸 Q chunk output。
- **Domain-local model state**：`DomainModelState` 是当前最小模型状态边界；每个 domain 持有自己的 Q chunk 和 K/V storage。source domain 从自己的 K/V storage 切出 block，target domain 捕获 compute update 时携带自己的 Q payload。
- **Layer activation state**：`LayerActivationState` 将 domain-local hidden states、Q chunk、K cache、V cache 与 output slot 作为同一个 layer 生命周期的所有权边界；CP compute capture 会带出 output slot 元数据，Rust report 校验设备侧 output value 数与本地 output slot 匹配。
- **Projection-first Q/K/V**：`ModelLayerWeights` 表达当前最小 Q/K/V projection weights；`DomainModelState` 不再直接公式生成 Q/K/V，而是先构造 domain-local hidden states，再通过 `hidden @ Wq/Wk/Wv + bias` 生成 Q chunk 与 K/V cache。当前 weights 仍是 deterministic 初始化，用于可复现 smoke；后续可替换为真实权重加载。
- **Rust/domain-side Q payload**：`torch_query_chunk_bridge` 使用 captured block 上的 target-domain Q payload，并按 `compute_domain` 分组调用 C++ ATen bridge；C++ 只负责 tensor parse / device compute / CPU reference 对比，不在该路径内部生成 Q。
- **统一 remote CP launcher**：3-node remote CP smoke 通过 `scripts/run_rust_remote_cp_3node_smoke.sh` 收敛 Mac 地址发现、GPU git 同步、cargo preflight build、三节点统一启动和 launcher 日志，避免手工时序导致 connect / accept retry 窗口失败。
- **reference-first correctness**：Python kernel stub 保留 reference attention，用于与 block-wise online softmax 对照。
- **报告纪律**：实验产物应落在 `reports/<RUN_ID>/` 下，便于回溯。
- **分级 tolerance policy**：数值验证不采用单一阈值，而是按测试层级分为三级：`Strict`（同设备算法等价，`rtol=1e-5`）、`Relaxed`（异构设备交叉验证，`rtol=1e-4`）、`EndToEnd`（多层模型累积误差，`rtol=1e-3`）。阈值基于 float32 机器精度（~1.2e-7）和业界框架默认值推导，给 100~1000 倍安全余量。`--tolerance-tier` CLI 参数可在运行时切换，correctness JSON report 包含 `tolerance_tier` 字段以明确当前验证标准。
- **异构 runtime 配置**：`ComputeRuntime` trait 解耦计算实现与协议逻辑；`TchComputeRuntime` 通过 `HCP_TCH_DEVICE` / `HCP_TORCH_DEVICE` 环境变量选择目标设备（`cpu` / `mps` / `cuda` / `cuda:N`），默认 fallback 到 `cpu`；`NoOpComputeRuntime` 作为无 `tch-backend` feature 时的编译兼容 stub。device 选择不依赖交互 shell 的偶然环境，而是通过统一的环境变量或构造函数参数传入。
- **长序列 Chunking 策略**：单节点 prefill 时，当 `seq_len > 8192` 时对 token-independent 运算做 chunking，避免 cuBLAS 大 M 维度执行失败和显存峰值爆炸：
  - **MLP (`layers.rs`)**：`gate_proj` / `up_proj` / `down_proj` 逐 chunk 计算后 `cat`，峰值从 ~3.6GB 降到 ~225MB。
  - **Attention projection (`layers.rs`)**：`q_proj` / `k_proj` / `v_proj` / `o_proj` 逐 chunk 计算后 `cat`，避免 `[1,131071,896]×[896,896]` 触发 `CUBLAS_STATUS_EXECUTION_FAILED`。
  - **LM head (`model.rs`)**：长 prefill 只计算最后一个 token 的 logits，避免预分配 `[batch, seq_len, vocab_size]` 大 buffer（32K 时 ~20GB）。
  - **Causal mask (`model.rs`)**：`seq_len > 8192` 时跳过 dense `[seq_len, seq_len]` mask 分配（64K 时 ~16GB），改用 `[1,1,1,1]` dummy zero tensor 作 causal 标志。
  - **Attention scores (`backend.rs`)**：`HcpRingAttentionBackend` 的 `ring_attention` 本身按 `q_chunk_size=2048` 和 `kv_chunk_size=2048` 循环，不 materialize 完整的 `[seq_len, seq_seq]` scores。

## 可插拔域内后端架构（新增）

HCP 的边界是**跨域低层协议**（P2P KV ring + online softmax），**域内实现是黑盒**。在同构计算域内，可以通过接口实现的形式，将默认 Rust/tch-rs Worker 替换为 vLLM、TensorRT-LLM、MLX 等社区成熟框架。

### 核心接口契约

```
Coordinator (Rust) ──QUIC──► Worker Interface ──► 框架实现（vLLM / TensorRT-LLM / MLX）
                              ├─ 控制面: WorkerCommand / WorkerResponse (bincode)
                              ├─ 数据面: KvTransport::exchange_kv_block()
                              └─ 模型面: HcpWorkerBackend (load/prefill/decode/get_kv/apply_peer)
```

### 适配器分层

```
Layer 3: 框架原生层
  ├─ vLLM: LLMEngine, Worker, CacheEngine, GPUModelRunner
  ├─ TensorRT-LLM: GptSession, KVCacheManager
  └─ MLX: nn.Module, KV cache array

Layer 2: HCP 适配层（必须自己实现）
  ├─ CommandHandler: WorkerCommand → 框架 API
  ├─ KvTransportBridge: 框架 KV cache ↔ KvBlock 序列化
  ├─ OnlineSoftmaxAggregator: peer KV 增量合并
  └─ HandshakeReporter: capacity_mb / device 上报

Layer 1: HCP 协议层（可复用 SDK）
  ├─ QUIC endpoint 管理
  ├─ Frame I/O（length-prefixed bincode）
  ├─ WorkerCommand / WorkerResponse 序列化
  └─ KvTransport trait 实现（QUIC/TCP）
```

### Python Worker SDK

已创建 `python/hcp_worker_sdk/` 目录，提供 Python 版 Worker SDK：
- `types.py`: `KvBlock`, `WorkerCommand`, `WorkerResponse`, `WorkerHandshake`
- `backend.py`: `HcpWorkerBackend` 抽象接口（框架适配器必须实现）
- `transport.py`: `KvTransport` + `TcpKvTransport` 实现
- `server.py`: `HcpWorkerServer` 通用事件循环

详见 `docs/PLUGIN_ARCHITECTURE.md` 和 `docs/VLLM_INTEGRATION.md`。

### Rust Worker SDK（新增）

`rust/src/worker_sdk/` 将 `distributed_worker.rs` 中原有的**协议循环 + 模型计算**紧耦合代码彻底解耦为三层：

```
┌─────────────────────────────────────────────────────────────┐
│  distributed_worker.rs（薄壳）                               │
│  解析 CLI 参数 → 选择 device → 加载 TchWorkerBackend          │
│  → 创建 WorkerRuntime → run()                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  WorkerRuntime<B: WorkerBackend>（协议层，完全可复用）         │
│  - QUIC endpoint 管理、handshake、command loop                │
│  - WorkerCommand 分发：Prefill / Decode / SyncGlobalSeqLen    │
│  - KvTransport 生命周期管理（每层一个 QuicKvTransport）        │
│  - 与 B 完全无关，换 backend 不需要改 runtime 代码            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  WorkerBackend trait（模型面接口）                            │
│  load() / prefill() / decode() / sync_global_seq_len()       │
│  / capacity_mb() / setup_kv_transports()                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ TchWorkerBackend│ │ VllmWorkerBackend│ │ TensorRTBackend │
│ （默认 tch-rs）  │ │ （未来）          │ │ （未来）          │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

**解耦目的**：
1. **关注点分离**：协议循环（QUIC 连接管理、handshake、command 解析、超时处理）与模型计算（tensor forward、KV cache 管理、RoPE 应用）完全独立。协议工程师不需要理解 LlamaModel 内部，模型工程师不需要理解 QUIC 帧格式。
2. **框架接入零侵入**：外部框架（vLLM、TensorRT-LLM、MLX）只需实现 `WorkerBackend` trait，即可接入完整的 HCP 分布式网络，复用 `WorkerRuntime` 的协议层和 `KvTransport` 的传输层。无需 fork 或修改 HCP 核心代码。
3. **零行为变更**：`TchWorkerBackend` 内部仍调用 `LlamaModel::forward`、`setup_distributed_domain`、`capacity::query_device_capacity_mb`，与重构前 `domain_worker_loop` 的行为完全一致。`cargo test` 42/42 通过验证无回归。
4. **单进程多 domain 不变**：`distributed_worker.rs` 仍通过 `Arc<ModelWeights>` + `shallow_clone` 在多个 domain thread 间共享权重，每个 domain 独立 `LlamaModel` + `KvCaches` + `WorkerRuntime`。
5. **性能无损**：`WorkerRuntime` 内部继续使用 `QuicKvTransport::exchange_kv_block`（`tokio::join!` 并发 send+recv + 512MB stream window），没有增加额外的网络往返、tensor 拷贝或 trait object dispatch 开销。prefill/decode 的耗时与重构前在同一数量级。

**验证矩阵**：
- `test_distributed_llama_model_prefill` — 2-layer GQA prefill，diff=2.79e-6 ✅
- `test_distributed_llama_model_decode` — 4-step 连续 decode，每步 diff ~2e-6 ✅
- `test_distributed_generator_tokens_match_reference` — 4-step greedy decode，domain0/domain1 token 一致，logits diff ~1e-5 ✅

## 组件关系

- `ringattn_protocol.h` 定义 `RingAttnBlock`、`RingAttnSoftmaxState`、`RingAttnMessage`、domain/global config。
- `ringattn_runtime.h` 定义 domain runtime 接口与 factory。
- `ringattn_runtime.cc` 当前提供 `NoOpRingAttnRuntime`。
- `rust/src/protocol.rs` 定义 Rust 侧可序列化 message schema、本地 P2P queue transport、并发 `cp_ring_node_runtime`、remote `tcp_remote_pair` / `tcp_remote_cp_node` transport smoke 和 protocol reports。
- `ringattn_coordinator_smoke_main.cc` 读取/构造配置并驱动 runtime 生命周期 smoke。
- `ringattn_kernel_stub.py` 提供 reference attention 与 online softmax update 原型。
- `ringattn_controller.py` / `ringattn_worker.py` 是后续 P2P / protocol smoke 的 Python 占位。

## 状态管理

- C++ 侧通过 `RingAttnSoftmaxState` 表达 per-domain online softmax 运行态。
- Python 侧通过 NumPy array 表达 `Q/K/V`、running max、running sum、output。
- 当前没有持久化应用状态；实验状态应通过 report 文件记录。

## 数据流

1. 每个 domain 初始化本地配置：`domain_id`、host、port、`seq_chunk_len`、`block_size`、device。
2. domain 保留本地 `Q chunk`。
3. ring 中传递 `K/V block`。
4. 收到 block 后计算局部 score 和 `P @ V` 贡献。
5. 使用 online softmax 更新 `running_max`、`running_sum`、`output`。
6. block 转发给下一个 domain，直到 ring 遍历完成。
7. 当前 Rust protocol smoke 以本地 P2P queue transport 验证完整 ring 转发路径，以 `cp_ring_node_runtime` 验证每节点双角色并发收发、payload-backed device compute、online state update 和 Rust/domain-side Q payload chunk output，以 `tcp_remote_pair` 验证双进程 / 双机器 send/recv frame，以 `tcp_remote_cp_node` 验证 remote 多节点 forwarding、payload-backed device compute、online state update 和 chunk output。

## 两种 Context Parallel 路径的对比

本项目在 Context Parallel 的通信层选择了 **P2P (point-to-point)** 设计，与 PyTorch 2.7+ 官方 Context Parallel 的 **Collective** 设计形成对照。以下是关键差异。

### 1. 原始 Ring Attention（Liu et al., 2023）— P2P 模式

**通信方式**：每个 host 只与 ring 中的邻居通信——**发送 K/V block 到 next host，同时从 previous host 接收 K/V block**。

**数学基础**：online softmax / blockwise attention 允许 KV block 以**任意顺序**处理，只要正确合并 running max / running sum / output。

**论文原话**：
> *"The self-attention between a query block and a group of key-value blocks can be computed in any order, as long as the statistics of each block are combined correctly for rescaling."*
> *"Each host efficiently coordinates by concurrently sending key-value blocks to the next host while receiving key-value blocks from the preceding host."*

**适用场景**：任意互联拓扑（TCP、PCIe、不同子网）、异构设备、非均分 block size。

### 2. PyTorch Context Parallel（2.7+）— Collective 模式

**通信方式**：用 `all-gather` 或 `all-to-all` **collective** 替换 `F.scaled_dot_product_attention`，通过 Python context manager 自动 hook。

**工程动机**：
- NCCL 的 collective 在 NVIDIA GPU 集群（NVLink / InfiniBand）上有拓扑感知路由优化
- PyTorch distributed 的 process group 抽象天然面向 collective
- 目标场景是 Llama3 训练（32-128 张同构 H100）

**限制**：
- 依赖 NCCL / process group，**不支持异构设备**（如 Mac MPS + Linux CUDA）
- 通常假设**均分 sequence**（`Sequential sharder` 或 `Round Robin`）
- 只在 **Python 层**实现，libtorch C++ API 没有暴露

### 3. 对照表

| 维度 | 原始 Ring Attention (P2P) | PyTorch Context Parallel (Collective) | HCP 本项目 |
|------|--------------------------|--------------------------------------|-----------|
| **来源** | Liu et al., 2023 | PyTorch 2.7+ `torch.distributed._tensor.experimental.context_parallel` | 独立实现 |
| **通信语义** | `send` / `recv` P2P | `all-gather` / `all-to-all` collective | `send_kv_block` / `recv_kv_block` P2P |
| **通信库** | 任意 socket / MPI | NCCL | 自定义 TCP / QUIC / 本地 queue |
| **设备假设** | 任意（论文用 TPU/GPU） | 同构 NVIDIA GPU | **异构**（MPS + CUDA） |
| **分块策略** | 支持非均分 | 均分（Sequential / Round Robin） | **支持非均分**（uneven blocks） |
| **实现层** | JAX / Python | Python dispatch hook | **Rust + tch-rs**（C++ libtorch） |
| **Correctness** | online softmax 增量更新 | online softmax 增量更新 | online softmax 增量更新，18 个单元测试验证 |
| **性能优化** | 计算-通信重叠 | NCCL 拓扑优化 + 计算-通信重叠 | 当前阶段先保证 correctness，性能优化后置 |
| **与官方关系** | 原始定义 | PyTorch 官方实现 | **不依赖 PyTorch CP**，从头基于底层 tch-rs 算子实现 |
| **域内后端** | 论文未涉及 | PyTorch / Transformers | **可插拔**（vLLM / TensorRT-LLM / MLX） |

### 4. 为什么 HCP 选择 P2P

1. **异构是刚需**：HCP 的目标场景是跨平台（Apple Silicon + NVIDIA），collective 需要所有设备加入同一个 NCCL process group，这不可能。
2. **非均分对异构必要**：MPS 和 CUDA 的算力/显存/带宽不同，均分 sequence 会导致负载失衡；P2P 允许每个 domain 根据自己的 capacity 持有不同大小的 block。
3. **P2P 是论文的原始定义**：PyTorch 的 collective 实现是一种"同构集群特化版"，不是 Ring Attention 的数学必须。
4. **Rust 层实现**：脱离 Python GIL 和 PyTorch distributed 的 runtime 假设，更适合长期运行的分布式推理服务。
5. **域内可插拔**：HCP 不绑定域内实现，允许同构域内使用 vLLM/TensorRT-LLM/MLX 等最强社区框架，最大化复用社区轮子。

## Correctness-First 开发纪律

本项目当前处于** correctness 验证阶段**，尚未进入性能调优阶段。在此阶段，所有代码变更和架构决策必须服从以下纪律：

### 1. 禁止实施的优化（在当前阶段）

在 correctness 流程完全走完、多级 tolerance 策略在全部 target 设备上稳定通过之前，**不得实施**任何可能引入数值误差或行为偏差的优化：

- **量化**：FP8/INT8 KV cache、BF16 权重、INT8/INT4 weight-only 量化、GPTQ/AWQ 等。这些方案在推理社区有成熟实现，但都会引入 ~1e-3 ~1e-2 级别的数值误差，足以在决策边界翻转 argmax。
- **近似 Attention**：FlashAttention 以外的稀疏 attention、local attention window 缩小、KV cache 压缩（H2O、Scissorhands 等）。这些改变 attention 的数学定义。
- **非 deterministic kernel**：使用 `CUBLAS_WORKSPACE_CONFIG` 以外的环境、TF32 模式、混合精度 training 的残余设置。
- **投机/跳过层优化**：speculative decoding、layer skipping、early exit。这些改变 layer stack 的语义，使 hidden states 不再等价于完整模型。
- **有损 padding/short-cut**：为了支持不同长度 prompts 的 batching 而简化 attention mask（如忽略 padding 位置的影响），或为了节省内存而截断 KV cache。

### 2. 提出任何优化前的强制 trade-off 分析

如果任何人（包括 AI coding agent）提出优化建议，必须先回答四个问题：

1. **为什么默认存在**：当前（非优化）方法解决的是什么问题？
2. **牺牲了什么**：该优化 discard 了哪些 correctness 保证、灵活性或通用性？
3. **被牺牲的东西在一般情况下的作用**：这些 guarantee 在训练、评估、高级解码（beam search、contrastive search、perplexity）中起什么作用？
4. **对本项目的具体影响**：为什么这些 sacrifice 在当前阶段可以接受（或不可接受）？

**示例 — "last token only" LM head 优化**：
- 默认计算完整 logits 是因为 `LlamaModel::forward` 的 contract 是返回 `[batch, seq, vocab]`，支持 per-position loss、perplexity、contrastive search、speculative decoding verification。
- 牺牲：模型从"全序列 logits"变为"仅最后一个位置"，破坏了通用 transformer 输出 contract。
- 被牺牲的东西的作用：训练（cross-entropy over all positions）、评估（perplexity）、高级解码（contrastive search 比较多个位置分数）、speculative decoding（draft model 需要评分所有 draft tokens）。
- 对本项目的影响：HCP 是 inference-only，greedy/temperature sampling 只关心最后一个 token。但 correctness tests 目前比较所有位置的 logits，"last token only" 会 break 这些 tests。增加 `return_full_logits: bool` flag 会增加 API 复杂度。Prefill 是一次性成本，decode 是循环，收益仅限于 prefill 阶段。**结论：skip。**

### 3. 什么情况下可以放宽

以下条件下，上述纪律自动解除：

- `test_distributed_llama_model_prefill` / `test_distributed_llama_model_decode` / `test_distributed_generator_tokens_match_reference` 在 **全部 target 设备**（CPU / MPS / CUDA）上稳定通过至少 5 个不同 seed 的随机验证。
- 新增至少 1 个真实权重端到端验证（如 Qwen2-0.5B greedy decode 与 HuggingFace transformers 输出逐 token 一致）。
- 用户明确书面批准某一特定优化，并附带接受其 correctness 风险的声明。

### 4. 当前已验证的 correctness 基线

| 测试 | 场景 | 误差 | 状态 |
|------|------|------|------|
| `test_batch_forward_correctness` | batch=2 vs 独立 batch=1，prefill + 4-step decode | logits diff ~1e-6，token 完全一致 | ✅ [2026-05-09] |
| `test_batch_generator_correctness` | `BatchGenerator` batch=2 vs 两个独立 `Generator` | token 序列完全一致 | ✅ [2026-05-09] |
| `test_distributed_llama_model_prefill` | 2-domain 分布式 prefill vs 单节点参考 | diff=2.79e-6 | ✅ |
| `test_distributed_llama_model_decode` | 4-step 连续分布式 decode | diff~2e-6 | ✅ |
| `test_distributed_generator_tokens_match_reference` | 4-step greedy decode domain0/domain1 | logits diff~1e-5 | ✅ |

## 异构分布式推理的数值验证策略

### 核心认知

在**异构平台**（如 CUDA + HIP）分布式推理中，**logits 数值对比不是有意义的 correctness 指标**。

### 根因

每个 worker 的 KV cache 由两部分组成：
- **本地 KV**：本域设备计算（cuBLAS / rocBLAS / MPS）
- **Peer KV**：对端设备通过 ring 交换而来

不同平台的 BLAS 库在 BF16 matmul 的累加顺序和舍入行为上存在差异，导致：

```
单节点:     Q(CUDA) × K(纯 CUDA) × V(纯 CUDA)     → logits_ref
分布式 w0:  Q(CUDA) × K(CUDA+HIP混合) × V(CUDA+HIP混合) → logits_dist
```

`logits_dist` 与 `logits_ref` 在数值上不同（实测 ~0.1-0.5 logits 差异，3B 模型最大可达 ~38），但**top-1 argmax 通常一致**，因此生成的 token 序列相同。

### 验证策略分层

| 层级 | 验证方法 | 适用场景 | 状态 |
|------|---------|---------|------|
| **L1: 协议数学正确性** | `cargo test` float32 synthetic weights + 同构分布式 BF16 argmax 一致性 | 验证 ring attention 算法本身 | ✅ 已完成 |
| **L2: 工程正确性** | 文本/任务级指标对比 | 异构分布式实际部署 | ✅ 已完成（LongBench 20 examples，90% 匹配） |
| **L3: 端到端冒烟** | 生成文本连贯性、无 crash | 快速回归测试 | ✅ 已完成 |

### 关键结论

- **L1 Float32 金标准**：`test_distributed_llama_model_prefill`（synthetic weights, float32）diff=2.79e-6 ✅。这是算法正确性的不可辩驳证据。
- **L1 BF16 同构验证**：White CUDA loopback 双 domain（3B: max_diff=0.406, 0.5B: max_diff=0.344），argmax=10/10 ✅。证明即使同构平台、相同 BLAS，BF16 下也有 ~0.3-0.4 的 logits 差异。
- **BLAS 根因已排除**：同构分布式（0.34-0.41）≈ 跨平台单节点（0.438）≈ 异构分布式（0.484）。三者同量级，证明跨平台 BLAS 仅贡献 ~0.1 的额外差异，**不是 logits 差异的主导因素**。
- **真正根因**：BF16 的 7-bit 尾数精度（步长 ~0.06@logit=12）下，online softmax 的 block-wise processing order（单轮 vs 多轮累加/rescaling）导致 ~0.3-0.4 的固有差异。
- **工程验证准则**：BF16 场景下 correctness 以 argmax 一致性和文本/任务级指标为准；严格 logits 数值对比仅在 float32 下有意义。
- **证据门槛**：任何未来声称"分布式 logits 差异是 ring attention 实现 bug"的假设，必须首先解释为什么同构分布式（纯 cuBLAS）也有 ~0.34-0.41 的差异。

### 代价与风险

- **延迟**：在 NVLink 全互联集群上，P2P 的逐 hop 延迟可能略高于 NCCL collective 的拓扑优化路由。
- **生态**：无法直接复用 PyTorch FSDP / TP / PP 的组合式并行框架，需要自己处理 multi-dimensional parallelism 的交互。
- **验证责任**： correctness 完全由我们自己保证，没有 PyTorch 官方背书。当前已通过多重证据链验证：
  - `test_ring_attention_matches_local_full`（diff=2.9e-8）和 `test_ring_attention_with_mock_transport`（diff=3.6e-8）验证数学等价性
  - `test_distributed_llama_model_prefill`（diff=2.79e-6）验证 float32 分布式算法正确性
  - BF16 同构/异构 argmax 一致性（10/10）验证工程正确性
  - LongBench 20 examples 任务级准确率一致性（95%）验证端到端正确性

## 架构决策

| 决策 | 理由 | 日期 |
|------|------|------|
| HCP 独立于 HLPP | 两者分别处理 high-boundary layer-wise 和 low-boundary intra-layer 问题，不能混用语义 | [2026-04-24] |
| 跨异构域只采用 P2P 假设 | 异构设备算力、显存、延迟、带宽不对称，collective 的对称同步假设不适合作为主线 | [2026-04-24] |
| 先 correctness，再 protocol / transport，再 remote smoke | 当前阶段目标是证明可行性，不是先追求性能最优 | [2026-04-24] |
| 保留 standalone repo 边界 | 本仓不依赖 `phase2_native/` / `phase3_layerwise/` 源码，降低历史包袱 | [2026-04-24] |
| 先固定 message schema，再扩展 transport | P2P 表示 point-to-point 非 collective，不应过早绑定到 IP/TCP；先做本地可诊断闭环 | [2026-04-25] |
| TCP 只作为 remote smoke transport | 双机阶段需要一个最小可诊断传输；`tcp_remote_pair` 不改变 HCP protocol 的 P2P/非 collective 语义 | [2026-04-25] |
| 每个 CP 节点最终应同时 server/client | Ring 中每个 domain 都既接收上游消息也向下游转发，本阶段 server/client 分离只用于最小连通性验证 | [2026-04-25] |
| 先在单进程 thread runtime 固定 CP node 语义 | 并发收发、持续转发、统计验证应先和远端网络问题解耦，再映射到 TCP / RDMA transport | [2026-04-25] |
| 双机 remote CP node 先验证 2-domain 双角色 | 现有 Mac/GPU 两台机器可以先证明每个进程同时 listen/connect 和多 block 双向流动；remote 多 hop 需要 3+ node 扩展 | [2026-04-25] |
| 先用本地 CP runtime 接入 payload-backed device compute | 这样可以先验证 Rust message payload -> C ABI -> C++ ATen 的执行闭环和 MPS/CUDA 设备路径，再把同一能力接入 remote CP node | [2026-04-25] |
| accepted TCP stream 必须恢复 blocking mode | macOS 上非阻塞 listener accept 出来的 stream 可能继承 nonblocking；大 payload frame 用 `read_exact` 时会触发 `WouldBlock` | [2026-04-25] |
| 3-node remote smoke 可先用两台物理机器三进程 | 当前目标是验证 remote process-level multi-hop forwarding；node0/node2 可同在 Mac，node1 在 GPU，后续再扩展到三台物理机器 | [2026-04-25] |
| Q payload 先从 Rust/domain-side 显式传入 C++ | 这样先切断 C++ bridge 内部 synthetic Q 的隐式依赖，再逐步升级到真实 model activation / state lifecycle | [2026-04-26] |
| 用 `DomainModelState` 收敛 Q/K/V ownership | 先在 Rust runtime 中明确每个 domain 拥有哪些 Q/K/V state，再接真实模型 activation / weight lifecycle | [2026-04-26] |
| 3-node remote CP smoke 使用统一 launcher | 三节点需要同时覆盖本机 MPS、远端 CUDA、git 同步、当前 Mac 子网地址和启动时序；用脚本统一执行比手工命令更可复现 | [2026-04-26] |
| 用 `LayerActivationState` 表达真实模型层生命周期 | 真实模型推进前必须先明确 Q/K/V cache 与 output buffer ownership，否则设备侧 output digest 无法对应到 domain-local 输出槽 | [2026-04-27] |
| 先接 projection 数据流，再接权重文件 | Q/K/V 的来源应变成 hidden states + projection weights；权重加载格式可后置，避免把 protocol/runtime 验证与外部模型文件解析耦合 | [2026-04-29] |
| HCP 采用原始论文 P2P 而非 PyTorch CP Collective | Ring Attention 原始论文（Liu et al. 2023）的通信本就是 P2P send/recv；PyTorch 2.7+ Context Parallel 改用 all-gather/all-to-all 是对同构 NVLink 集群的工程优化，不是数学必须。P2P 支持异构、非均分、任意拓扑，更符合 HCP 定位 | [2026-04-30] |
| QUIC 替代每层独立 TCP connection | 24 层 = 24 条 TCP connection 的 connection 建立开销在高延迟网络下显著。QUIC 单 connection + per-layer bidirectional stream 复用，跨机器 ~150ms RTT 场景下快 29%（76.4s vs 107.3s） | [2026-04-30] |
| QUIC 2-domain 对称连接死锁解决：domain 0 只 dial，domain 1 只 accept | quinn 在 loopback 上同时 dial/accept 同一地址时可能将两个方向合并为同一个 connection，导致互相等待。domain 0 负责 dial、domain 1 只 accept，共享同一个 connection handle | [2026-04-30] |
| QUIC stream 建立 workaround：sender 先写 1-byte dummy | quinn 的 `open_bi()` 在数据写入前不发送 STREAM 帧，导致对端 `accept_bi()` 永远挂起。sender 先写 1-byte dummy，receiver 首次 `recv_kv_block` 跳过该 byte | [2026-04-30] |
| 分布式 prefill 用 dummy mask 替代密集 causal mask | `ring_attention` 已通过 `global_seq_start` + position 比较实现 causal，从不读取 mask 张量数据。分布式场景传 `[1,1,1,1]` dummy zero tensor 替代 `[seq_len, seq_len]` 密集 mask，消除 O(seq²) 内存分配 | [2026-04-30] |
| `is_prefill_done` 标志替代 `seq_len > 1` 区分 prefill/decode | `seq_len > 1` 在 1-token prefill chunk（如 `--chunk-sizes 10,1`）时失效，误判为 decode。`is_prefill_done` 确保第一次 `forward()` 无论 `seq_len` 都走 prefill 路径 | [2026-04-30] |
| 动态不均等分片 Phase 1：手动 `--chunk-sizes` | Coordinator CLI 新增逗号分隔的 `--chunk-sizes`，显式指定每个 domain 的 chunk 长度。自动校验 `len == num_domains` 且 `sum == prompt_len`。为后续 Phase 2（capacity 感知自动分配）铺垫 | [2026-04-30] |
| `prefill_kv_len` 字段解决多步 decode 重复发送 KV | 多步 decode 时 `history_len = k.size()[2] - 1` 会包含之前 decode append 的 token，导致 peer 收到重复 KV。`prefill_kv_len` 记录 prefill 阶段 KV 长度，decode 只发送 prefill 分区 | [2026-05-01] |
| Handshake 扩展 capacity 上报 | Worker 加载模型后查询 device 可用显存/内存，通过 16-byte handshake 上报 coordinator。Coordinator 使用 largest-remainder method 按比例分配 chunk sizes，替代手动 `--chunk-sizes` | [2026-05-02] |
| 统一 QUIC 控制面 | Worker-coordinator 控制面从 TCP 迁移到 QUIC，与 KV ring 共用同一链路。 Coordinator 创建 QUIC endpoint 接受 worker 连接，worker 通过 QUIC bidirectional stream 发送/接收 WorkerCommand/WorkerResponse。消除 TCP 在高延迟 VPN 下的 EAGAIN/write timeout 问题，统一享受 256MB stream window | [2026-05-03] |
| Prefill 命令携带动态 seq_offset | Capacity-aware 分配下每个 worker 的 chunk start 可能与其 CLI `--seq-offset` 不一致。Coordinator 在 `WorkerCommand::Prefill` 中携带实际 `seq_offset`，worker 收到后同步更新 `LlamaModel` 和所有 layer backend 的 `seq_offset`，确保 causal mask 使用正确的全局位置 | [2026-05-02] |
| `set_distributed` 保留现有 transport | 当 `transport` 参数为 `None` 时，`HcpRingAttentionBackend::set_distributed` 只更新 `seq_offset` 和 `domain_id`，不覆盖已有的 `kv_transport`。避免 worker 在更新 seq_offset 时意外丢失 QUIC stream | [2026-05-02] |
| 单进程多 domain worker | 单个 OS 进程内通过 `std::thread::spawn` 运行多个 domain，共享 `Arc<ModelWeights>`（权重 shallow_clone），每 domain 独立 LlamaModel/KV cache/coordinator/QUIC。减少 per-card 权重内存占用，但 KV cache 和 LM head output 仍按 domain 数倍增 | [2026-05-02] |
| 移除 `forward_lock`，改用 `no_grad` + 自然 stream 串行化 | `forward_lock` 在 ring attention 中导致死锁：domain0 在 `recv_kv_block` 阻塞等待 domain1，但 domain1 被 lock 挡在外面。推理时 `no_grad_guard()` 已禁用梯度图，MPS/CUDA 天然按 stream 顺序执行，无需额外互斥 | [2026-05-02] |
| **🚫 铁律：1 GPU = 1 worker，禁止单卡多 worker** | 每个 worker 加载**完整模型权重**。3B bf16 (~6GB) × 2 workers = ~12GB，RTX 4090 (24GB) 本地 loopback 实测 OOM。即使 0.5B 模型可行，也不推广到 3B+ 场景。`--local-domain-ids` 仅限 <1GB 小模型的本地协议验证。生产/大规模验证必须每平台一 worker。 | [2026-05-02] → [2026-06-02] 强化为铁律 |
| MPS 后端 tensor op workaround：`arange_start` CPU fallback、`add+mul` 替代 `masked_fill`、`amax` 替代 `max_dim` | MPS 后端多个 tch-rs/libtorch op 存在 bug：`arange_start` 在 MPS 设备上产生错误结果；`masked_fill` 行为不正确；`max_dim` 返回 argmax 时异常。统一策略：可疑 op 先在 CPU 创建再 `to_device`，或用数学等价 op 替代（如 `add+mul` 等价于 `masked_fill`） | [2026-05-04] |
| QUIC `max_idle_timeout` 适配长计算阶段 | prefill/decode 阶段 worker 可能在数分钟内只进行本地计算而不发送任何 QUIC 帧，quinn 默认 idle timeout（~30s）会断开 connection。显式设置 `max_idle_timeout=300s` 覆盖长计算阶段的静默期 | [2026-05-04] |
| 并发 `exchange_kv_block` 替代串行 send→recv 防止 ring 死锁 | `ring_attention` 中原先 `send_kv_block(&block)` 后 `recv_kv_block()` 是串行阻塞的；当双方同时发送的 KV block 超过 stream receive window 时，send 阻塞且无人 recv → 死锁。`KvTransport::exchange_kv_block()` 在 `QuicKvTransport` 中用 `tokio::join!` 并发执行 send 和 recv，确保 recv 侧始终运行，不受 send 阻塞影响；同时提取独立 async 函数以拆分 `&mut self.send` 和 `&mut self.recv` 借用 | [2026-05-05] |
| 跨节点异构验证：每个平台必须有 worker | coordinator 只加载 tokenizer+config，不做模型计算。若 Platform A 只跑 coordinator、Platform B 只跑 worker，则 Platform A 的异构计算能力未被验证。正确架构：Mac 跑 `coordinator + worker 0 (MPS)`，GPU 跑 `worker 1 (CUDA)`，双方均执行 forward 和 KV ring 交换 | [2026-05-05] |
| **可插拔域内后端架构** | 在同构计算域内，通过接口实现的形式将默认 Rust/tch-rs Worker 替换为 vLLM、TensorRT-LLM、MLX 等社区框架。HCP 只定义跨域 P2P KV 交换和 online softmax 协议，域内实现是黑盒。Python Worker SDK (`python/hcp_worker_sdk/`) 提供标准接口降低适配门槛。详见 `docs/PLUGIN_ARCHITECTURE.md` | [2026-04-30] |
| **vLLM 适配器设计** | ~~优先用非侵入式 Wrapper 方案（后处理模式）~~ → **修正为 Block-Aware Ring**：vLLM 的 PagedAttention block 是 Ring Attention 的天然粒度单位，抛弃 block = 抛弃 vLLM 全部价值。正确路径是让 ring 交换粒度对齐 vLLM block，改 CacheEngine 支持远程 block，而非提取连续 tensor。详见 `docs/BLOCK_RING_FUSION.md` | [2026-05-09] |
| **Transformers backend 限制** | Python transformers `model.forward` 无法在层间注入 peer KV 并重算 attention。`recalculate_logits()` 只能修复 prefill 阶段最后一个 token 的 logits，decode 阶段 layer 1+ 的 hidden states 仍基于不完整 KV。数学 correctness 由 Rust 层保证，Python 层负责 control-plane + transport 验证 | [2026-05-09] |
| **冻结 Python 层，聚焦 Rust + C++ + libtorch** | Python 层的存在理由只有 vLLM 适配。一旦 vLLM 走 Block-Aware Ring（长期）、transformers  correctness 已由 Rust 层覆盖，Python 就成了无根之木。继续投入 Python = 维护两套协议 + 两套 SDK + 版本兼容性包袱。后续以 Rust 为主干，Python 代码进入维护模式不再扩展 | [2026-05-09] |
| **Static Batching：等长 prompts + 0-token 填充** | `BatchGenerator` 支持 batch > 1，但要求所有 prompts 等长（避免 padding mask 的 correctness 风险）。已完成的 request 喂 0 token 保持 KV cache 形状一致。不实现连续 batching 或动态 padding，直到 correctness 基线更稳固 | [2026-05-09] |
| **Transport 序列化必须携带 dtype 元数据** | TCP/QUIC transport 的 `tensor_to_bytes`/`bytes_to_tensor` 曾硬编码 `f32`，导致 BF16 KV block 跨节点传输后被重建为 Float32 tensor → `matmul` dtype 不匹配 panic。修复：序列化时记录 dtype（float32/float16/bfloat16/float64），反序列化后 `.to_kind(kind)` 还原原始 dtype。此教训适用于任何非 f32 精度的分布式 tensor 传输场景 | [2026-06-04] |
| **Correctness-First 纪律：优化必须证明无害** | 在 correctness 流程完全走完之前，不实施任何可能损害服务质量的优化。包括但不限于：量化（FP8/INT8/BF16 KV cache）、近似 attention、非 deterministic kernel、跳过层/投机解码。每次提出优化前必须写 trade-off 分析：为什么默认存在、牺牲了什么、牺牲的东西在一般情况下的作用、对本项目的具体影响。见下方「Correctness-First 开发纪律」 | [2026-05-09] |
