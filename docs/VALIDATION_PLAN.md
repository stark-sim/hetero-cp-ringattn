# Validation Plan

## 目标

HCP 不能只靠“方向看起来合理”成立。这个仓库需要一套从数学正确性到远端异构闭环的验证路径。

验证顺序应该始终是：

1. correctness
2. protocol
3. transport
4. remote heterogeneous deployment
5. scaling argument

而不是一上来追求性能图。

## M0: Standalone Skeleton

### 目标

- 独立 repo 可单独 configure / build
- 最小 coordinator smoke 能通过
- 不依赖 `honolulu` 主仓头文件

### 当前状态

- 已完成

## M1: Online Softmax Correctness

### 目标

证明逐 block 聚合的 online softmax 与参考 attention 在数学上保持等价或在约定误差内一致。

### 需要产物

- reference implementation
- block-wise implementation
- correctness report
- 明确阈值：
  - `max_abs_err`
  - `mean_abs_err`
  - 必要时 `max_rel_err`

### 通过条件

- 不同 `seq_chunk_len`
- 不同 `block_size`
- 不均分 domain

在这些组合下，结果都能与 reference 保持在约定误差范围内。

## M2: Ring Message / Serialization

### 目标

让 `RingAttnMessage` 不再只是内存内抽象，而是具有最小可传输格式。

### 需要产物

- message schema
- serialization / deserialization
- 本地双端点 send / recv smoke

### 通过条件

- `K/V block`
- softmax state
- metadata

都可以正确往返传输并恢复。

## M3: Minimal P2P Transport

### 目标

建立一个最小、可观测、易调试的 P2P transport，用于 ring 中的 send/recv。

### 当前原则

- 优先最小可见性，不优先最优性能
- 优先让错误可定位，不优先抽象漂亮

### 需要产物

- transport stub
- 基本 metrics：
  - bytes sent / received
  - transport latency
  - failure classification

### 通过条件

- ring 中至少 2 个参与者可稳定收发 message
- transport failure 能写入结构化报告

## M4: Heterogeneous Runtime Stub

### 目标

不是马上接入完整真实内核，而是先证明不同类型设备的 runtime stub 能通过统一低边界协议参与同一个 attention 过程。

### 需要产物

- 至少两种 runtime stub
  - 例如 CUDA side / MLX side
- 明确各自本机配置方式
- 本机路径与解释器解析规则

### 通过条件

- 两类 runtime 都能通过同一套协议参与 ring
- 不依赖交互 shell 的偶然环境

## M5: Remote End-to-End Smoke

### 目标

完成至少一次真正的 2-domain remote heterogeneous smoke。

### 需要产物

- runbook
- known-good config
- known-good report
- compare script / compare report

### 通过条件

- 两侧服务稳定启动
- coordinator / controller 能驱动完整流程
- 最终报告中可见：
  - correctness pass
  - transport metrics
  - failure info
  - artifacts location

## M6: Scaling Argument

### 目标

即使暂时还不能做到 `10M` 真实生产级运行，也要形成一套可信的 scaling argument，说明 HCP 为什么有资格被继续投资。

### 需要回答的问题

- `seq_chunk_len` 和 `block_size` 如何决定单卡显存压力
- 增加 domain 后理论上如何外推可支持 context
- 带宽瓶颈会如何主导性能
- 在什么条件下，继续增加设备仍然有产品意义

## 报告纪律

每个阶段都应输出结构化实验产物，而不是只保留口头结论。

最少应包含：

- config
- log
- correctness summary
- transport summary
- failure summary

只要这套纪律成立，HCP 才能从“研究想法”逐步变成“可复现工程路线”。
