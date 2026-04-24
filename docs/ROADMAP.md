# Roadmap

## 目标

这份路线图只回答一个问题：

`hcp_ringattn_repo` 要怎样从一个独立骨架，走到一个足以建立远端仓并继续独立推进的研究仓。

## M0: 独立化完成

### 状态

- 已完成

### 标志

- 最小公共类型已内化
- 不再依赖 `phase2_native/` / `phase3_layerwise/` 头文件
- standalone build / smoke 已通过

## M1: 问题定义固定

### 目标

把仓库叙事从“phase4 抽离物”升级为“独立研究仓”。

### 交付物

- 首页 README
- 产品论证文档
- `HLPP vs HCP` 定位文档
- 历史经验迁移文档

### 退出条件

- 新读者只看本仓文档，也能理解：
  - HCP 是什么
  - 它为什么不是 HLPP
  - 它解决什么产品问题

## M2: 数学闭环

### 目标

优先把 online softmax correctness 变成正式实验资产。

### 交付物

- correctness script
- correctness report
- tolerance policy

### 退出条件

- 不均分 `seq_chunk` / `block_size` 下，结果仍能与 reference 对齐

## M3: 协议闭环

### 目标

让 ring 中流动的 message 成为可传输、可检查、可调试的真实协议单元。

### 交付物

- message schema
- serialization
- local send/recv smoke

### 退出条件

- `RingAttnMessage` 可以稳定编码、传输、解码

## M4: 异构 runtime 闭环

### 目标

让至少两类 runtime stub 能接入同一套低边界协议。

### 交付物

- heterogeneous runtime stubs
- config discipline
- environment discipline

### 退出条件

- 不同设备类型可在同一 ring 中参与最小实验

## M5: 远端闭环

### 目标

完成 2-domain remote heterogeneous smoke。

### 交付物

- runbook
- deployment config
- known-good logs / reports

### 退出条件

- 新环境可按 runbook 复现实验

## M6: 扩展性论证

### 目标

形成“为什么 HCP 值得继续推进到更长 context”的系统级论证。

### 交付物

- memory / bandwidth scaling notes
- context-length growth argument
- failure envelope / operating envelope

### 退出条件

- 可以清楚说明：
  - 为什么 `200k -> 1M -> 10M` 的演进会继续逼近 HCP
  - 为什么“通过增加卡继续支撑 context”是一个值得追求的方向

## 什么时候适合建立远端仓

严格说，本地目录已经是独立 git 工作区，但还不适合马上把它当作“叙事完整的新仓”推出去。

更稳妥的时机是同时满足下面三件事：

1. 文档叙事已经完整
2. online softmax correctness 已形成正式报告
3. 至少一次 heterogeneous remote smoke 已成立

在这之前，可以先把它视为：

- standalone candidate repo
- 而不是已经成熟的新远端主仓
