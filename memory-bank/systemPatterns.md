# 系统模式

## 架构概览

本仓采用 Rust + C++ 为主、Python 原型为历史对照的结构。C++ 部分定义 HCP Ring Attention 的低边界 runtime 抽象和 libtorch bridge；Rust 部分负责 correctness model、report、可序列化协议 schema 和当前 P2P transport smoke。

HCP 的核心形态是：每个 domain 持有本地 `Q chunk`，ring 中持续传递 `K/V block`，每个 domain 对收到的 block 更新 online softmax state，直到完整遍历 ring。

## 项目结构

```text
project-root/
├── CMakeLists.txt
├── README.md
├── include/hcp_ringattn/core/
│   ├── status.h
│   ├── tensor_types.h
│   ├── ringattn_protocol.h
│   └── ringattn_runtime.h
├── src/
│   ├── ringattn_runtime.cc
│   └── ringattn_coordinator_smoke_main.cc
├── python/
│   ├── ringattn_controller.py
│   ├── ringattn_worker.py
│   └── ringattn_kernel_stub.py
├── config/
│   └── minimal_2domain_ring.json
├── scripts/
│   └── run_local_ringattn_smoke.sh
├── docs/
└── reports/
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
