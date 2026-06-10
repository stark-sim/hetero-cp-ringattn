# Validation Plan

## 目标

HCP 不能只靠"方向看起来合理"成立。这个仓库需要一套从数学正确性到远端异构闭环的验证路径。

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

## M2: Protocol Serialization

### 目标

`RingAttnMessage` 能在不同 runtime 间无损传输。

### 需要产物

- 序列化/反序列化 roundtrip 测试
- 至少覆盖 `KvBlock`、`SoftmaxState`、`Terminate`
- 跨语言/跨 runtime 的兼容性验证

### 通过条件

- 100% message kind 覆盖
- 大 payload（如 1MB+ KV block）不丢 byte

## M3: Transport Layer

### 目标

P2P send/recv 能在真实网络环境下工作。

### 需要产物

- TCP transport 实现
- 本地 loopback 测试
- 跨机器测试

### 通过条件

- 2-domain 本地通过
- 2-domain 跨机器通过
- 无数据损坏

## M4: Runtime Stub Integration

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

## 分布式 logits 数值对比的已知限制

### 背景

`--export-logits` 在单节点和分布式模式下均可导出每步 logits，但**分布式导出的 logits 数值与单节点参考有预期差异**，这不代表 correctness 问题。

### 根因

在 BF16 异构分布式推理中，每个 worker 的 KV cache 由两部分组成：
- **本地 KV**：本域设备计算（如 CUDA 的 cuBLAS）
- **Peer KV**：对端设备通过 ring 交换而来（如 HIP 的 rocBLAS）

cuBLAS 与 rocBLAS 的 BF16 matmul 在累加顺序和舍入行为上有微小差异（~0.1-0.5 logits 范围）。这导致：

```
单节点: Q(CUDA) × K(纯 CUDA) × V(纯 CUDA) → logits_ref
分布式 worker 0: Q(CUDA) × K(CUDA+HIP混合) × V(CUDA+HIP混合) → logits_dist
```

`logits_dist` 与 `logits_ref` 在数值上不同，但**top-1 argmax 通常一致**，因此生成的 token 序列相同。

### 验证结论（2026-06-11 更新）

- **文本一致性**：LongBench 2wikimqa 4/20 examples，分布式与单节点文本输出 90-100% 匹配
- **准确率一致性**：任务级准确率一致（19/20），无系统性错误
- **Logits 数值差异根因已定位**：
  - **同构分布式 BF16**（White CUDA loopback）：max_diff=0.34-0.41，argmax=10/10
  - **异构分布式 BF16**（White CUDA + Pearl HIP）：max_diff=0.48，argmax=10/10
  - **跨平台单节点 BF16**（White vs Pearl）：max_diff=0.44
  - 三者同量级 → **BLAS 不是根因**，主要差异来自 BF16 online softmax block-wise processing order
- **Float32 数学金标准**：`test_distributed_llama_model_prefill` diff=2.79e-6 ✅

### 建议

- **Correctness 验证应以文本/任务级指标为准**，BF16 下 logits 数值对比不作为 correctness 依据
- `compare_logits.py` 在分布式模式下仅用于调试，atol 需放宽至 0.5+ 或跳过数值对比
- **同构分布式 BF16 也有 ~0.3-0.4 差异**，这是 BF16 精度限制下的预期数学行为，不是实现 bug
- 如需严格 logits 对比，必须使用 **float32 权重**（而非仅修改 config `torch_dtype`），因为 `weights.rs` 的 `keep_original_dtype=true` 会覆盖 config 设置

## 报告纪律

每个阶段都应输出结构化实验产物，而不是只保留口头结论。

最少应包含：

- config
- log
- correctness summary
- transport summary
- failure summary

只要这套纪律成立，HCP 才能从"研究想法"逐步变成"可复现工程路线"。
