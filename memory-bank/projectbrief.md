# 项目简报

## 项目名称

HCP Ring Attention Repo (`hetero-cp-ringattn`)

## 描述

这是从 `honolulu` 主仓中抽离出来的独立研究仓，只聚焦 HCP（Heterogeneous Context Parallelism）在超长 context 场景下的 intra-layer / low-boundary Ring Attention 路线。

## 愿景

当 context 从 `200k` 走向 `1M`、再继续走向 `10M` 时，系统不应只能依赖单张更强显卡。HCP 的目标是允许多个异构 domain 以不均分方式共同承担同一个 attention layer，通过增加 domain / 设备继续外推可支持的 context 长度。

## 核心需求

- 跨异构 domain 只允许 P2P，不把 collective 作为主通信假设。
- 多个 domain 共同完成同一个 attention layer，而不是按 layer range 分工。
- 支持不均分 `seq_chunk_len` 和 `block_size`。
- 每个 domain 维护本地 `Q chunk`，ring 中持续流动 `K/V block` 与 online softmax state。
- 先证明 correctness、protocol、remote heterogeneous smoke，再讨论性能。
- 仓库不依赖 `phase2_native/` 或 `phase3_layerwise/` 的源码边界。

## 目标

- 建立独立、可构建、可 smoke 的 HCP Ring Attention 研究骨架。
- 形成 online softmax correctness report。
- 让 `RingAttnMessage` 成为可序列化、可传输、可检查的真实协议单元。
- 完成最小 P2P send/recv transport。
- 推进到 2-domain remote heterogeneous smoke。

## 范围

### 范围内

- `include/hcp_ringattn/core/`：独立公共类型、protocol、runtime 抽象。
- `src/`：最小 `NoOp` runtime 与 C++ coordinator smoke。
- `python/`：controller / worker / kernel stub，用于 correctness 和本地协议原型。
- `config/`：最小 2-domain ring 配置。
- `scripts/`：本地 smoke 入口。
- `docs/`：产品论证、边界、历史经验、设计、验证计划、路线图。

### 范围外

- HLPP 的 layer-wise 编排。
- 域内 TP / collective / kernel 优化。
- `honolulu` 主仓源码级依赖。
- 当前阶段的性能最优。

## 关键利益相关者

- 长上下文推理系统研究与工程实现者。
- 需要利用混合算力完成超长 context attention 的部署方。
- 后续维护本仓的 AI coding agents。
