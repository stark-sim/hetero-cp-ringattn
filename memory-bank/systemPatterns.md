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
├── python/                            # controller、worker、online softmax 原型
│   ├── ringattn_controller.py
│   ├── ringattn_worker.py
│   └── ringattn_kernel_stub.py
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
│   │   ├── distributed_worker.rs      # 多进程分布式 worker
│   │   ├── distributed_coordinator.rs # 多进程分布式 coordinator
│   │   ├── distributed_protocol.rs    # WorkerCommand/WorkerResponse 协议
│   │   ├── infer.rs                   # inference CLI
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
├── docs/                              # 设计、验证、路线图、产品论证
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
| **通信库** | 任意 socket / MPI | NCCL | 自定义 TCP / 本地 queue |
| **设备假设** | 任意（论文用 TPU/GPU） | 同构 NVIDIA GPU | **异构**（MPS + CUDA） |
| **分块策略** | 支持非均分 | 均分（Sequential / Round Robin） | **支持非均分**（uneven blocks） |
| **实现层** | JAX / Python | Python dispatch hook | **Rust + tch-rs**（C++ libtorch） |
| **Correctness** | online softmax 增量更新 | online softmax 增量更新 | online softmax 增量更新，18 个单元测试验证 |
| **性能优化** | 计算-通信重叠 | NCCL 拓扑优化 + 计算-通信重叠 | 当前阶段先保证 correctness，性能优化后置 |
| **与官方关系** | 原始定义 | PyTorch 官方实现 | **不依赖 PyTorch CP**，从头基于底层 tch-rs 算子实现 |

### 4. 为什么 HCP 选择 P2P

1. **异构是刚需**：HCP 的目标场景是跨平台（Apple Silicon + NVIDIA），collective 需要所有设备加入同一个 NCCL process group，这不可能。
2. **非均分对异构必要**：MPS 和 CUDA 的算力/显存/带宽不同，均分 sequence 会导致负载失衡；P2P 允许每个 domain 根据自己的 capacity 持有不同大小的 block。
3. **P2P 是论文的原始定义**：PyTorch 的 collective 实现是一种"同构集群特化版"，不是 Ring Attention 的数学必须。
4. **Rust 层实现**：脱离 Python GIL 和 PyTorch distributed 的 runtime 假设，更适合长期运行的分布式推理服务。

### 5. 代价与风险

- **延迟**：在 NVLink 全互联集群上，P2P 的逐 hop 延迟可能略高于 NCCL collective 的拓扑优化路由。
- **生态**：无法直接复用 PyTorch FSDP / TP / PP 的组合式并行框架，需要自己处理 multi-dimensional parallelism 的交互。
- **验证责任**： correctness 完全由我们自己保证，没有 PyTorch 官方背书。当前已通过 `test_ring_attention_matches_local_full`（diff=2.9e-8）和 `test_ring_attention_with_mock_transport`（diff=3.6e-8）验证数学等价性。

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
| 默认 1 worker / 1 GPU，开发可尝试 2 worker / 1 GPU | 生产环境单 worker 最稳定、显存最可控；开发环境双 worker 可验证权重共享逻辑，但不能突破单卡显存上限（LM head + KV cache 仍倍增） | [2026-05-02] |
