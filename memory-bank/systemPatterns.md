# 系统模式

## 架构概览

本仓采用 standalone C++ core + Python 原型验证的结构。C++ 部分定义 HCP Ring Attention 的低边界协议与 runtime 抽象，Python 部分用于快速验证 online softmax、controller / worker、以及后续最小 P2P 协议。

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
- **reference-first correctness**：Python kernel stub 保留 reference attention，用于与 block-wise online softmax 对照。
- **报告纪律**：实验产物应落在 `reports/<RUN_ID>/` 下，便于回溯。

## 组件关系

- `ringattn_protocol.h` 定义 `RingAttnBlock`、`RingAttnSoftmaxState`、`RingAttnMessage`、domain/global config。
- `ringattn_runtime.h` 定义 domain runtime 接口与 factory。
- `ringattn_runtime.cc` 当前提供 `NoOpRingAttnRuntime`。
- `rust/src/protocol.rs` 定义 Rust 侧可序列化 message schema、本地 ring transport 和 protocol smoke。
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
7. 当前 Rust protocol smoke 以本地 in-memory transport 验证上述转发路径，后续替换为 P2P transport。

## 架构决策

| 决策 | 理由 | 日期 |
|------|------|------|
| HCP 独立于 HLPP | 两者分别处理 high-boundary layer-wise 和 low-boundary intra-layer 问题，不能混用语义 | [2026-04-24] |
| 跨异构域只采用 P2P 假设 | 异构设备算力、显存、延迟、带宽不对称，collective 的对称同步假设不适合作为主线 | [2026-04-24] |
| 先 correctness，再 protocol / transport，再 remote smoke | 当前阶段目标是证明可行性，不是先追求性能最优 | [2026-04-24] |
| 保留 standalone repo 边界 | 本仓不依赖 `phase2_native/` / `phase3_layerwise/` 源码，降低历史包袱 | [2026-04-24] |
| 先固定 message schema，再替换 transport | 避免把协议语义和 TCP / remote 细节耦合，便于先做本地可诊断闭环 | [2026-04-25] |
