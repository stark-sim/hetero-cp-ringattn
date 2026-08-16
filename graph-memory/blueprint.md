# HCP Ring Attention 项目蓝图

## 一句话描述

HCP（Heterogeneous Context Parallelism）让多个异构计算 domain 以**不均分**方式共同完成同一个 attention layer，从而把超长 context（200k → 1M → 10M）从“单卡显存墙”变成“可调度问题”。

## 核心信念

1. **异构是常态**：真实部署中 GPU 代际、显存、互联带宽 rarely 一致。
2. **非均等 CP 是可行性前提**：均匀分片在异构显存下会让小显存设备先 OOM；分片应匹配设备能力边界。
3. **P2P 是数学必须，collective 是同构优化**：Ring Attention 原始论文本就是 P2P send/recv；PyTorch CP 的 collective 是对同构 NVLink 集群的工程优化。
4. **Correctness 优先于性能**：在数值正确性未跨平台稳定通过前，不引入量化、近似 attention、非 deterministic kernel 等优化。

## 架构边界

```
Coordinator (Rust) ──QUIC──► Worker / Domain ──► 域内后端（tch-rs / vLLM / TensorRT-LLM / MLX）
                              ├─ 控制面：WorkerCommand / WorkerResponse (bincode)
                              ├─ 数据面：P2P KV ring + online softmax
                              └─ 模型面：可插拔 WorkerBackend
```

- **跨域协议**：P2P `send_kv_block` / `recv_kv_block`，支持任意 `seq_chunk_len` 和 `block_size`。
- **域内黑盒**：HCP 只关心跨域数据流，不规定 CUDA / MPS / NPU 内部实现。
- **调度面**：coordinator 根据设备 capacity 动态分配 chunk sizes，worker 上报可用显存/内存。

## 当前状态（2026-08-16）

- 🎯 **主赛道（用户裁决 decision-prefill-cp-pivot-20260816）**：进入 PD 分离生态的 P 侧——**异构请求级 CP prefill**。第一证明目标：混合卡舰队（4090 CUDA + 9060XT ROCm + 4060 CUDA + Mac MPS）以 capacity-aware 不均分 CP 完成长 context prefill，TTFT/聚合能力优于任一单节点，且同构生态（vLLM CP / dynamo / llm-d）无法复制。形态参照 Moreh SLOPE（PD 模式下的 dedicated prefill worker）。
- ✅ 生态空档已外部验证：PD 分离主流化（DistServe/SGLang/llm-d/vLLM）；prefill CP 近线性扩展且中低带宽可行（MLSys 2025）；业界异构仅做到池级调度，请求级跨厂商 CP 无人做（evidence-ecosystem-pd-cp-survey-20260816）。
- ✅ decode 自驱动环已合并为主线默认（474e9cb），N=3 WiFi 复测双优（延迟 -6~-11%、带宽 -33.4%）。
- ⏸️ **decode 工作 depress（非删除）**：代码与证据保留，不再投入新工作；K2（decode 适用性）/K3（PD 退路）已由用户裁决收口——PD 不是退路而是主赛道。
- ⏸️ NIXL block-direct 传输线挂起（三机环 transfer bug 延后，非关键路径）。
- ⏸️ 已挂起：Striped Attention 适配（非均等切分兼容性问题未解）。

### Prefill 线当前 debt（推进顺序）
1. workspace-aware admission：30k mc=32 OOM 闸门（准入按并发计入 attention 瞬态工作区）
2. fused attention kernel（SDPA/flash 桥接）：消除 per-block scores ~134MB 中间体
3. 长 context 异构 prefill 对外证明实验

## 关键约束

- **1 GPU = 1 worker**：禁止单卡多 worker 加载多份完整权重。
- **BF16 数值验证以 argmax/文本指标为准**：跨平台 logits 数值差异主要由 BF16 online softmax 处理顺序导致，不是实现 bug。
- **不引入有损优化**：量化、稀疏 attention、投机解码等在当前 correctness 阶段被禁止。

## 重要文档

- `docs/DESIGN.md`：设计总览
- `docs/HLPP_VS_HCP.md`：与 HLPP 的边界
- `docs/SCALING_ARGUMENT.md`：context 长度与显存/网络/域数的 scaling 分析
- `docs/PLUGIN_ARCHITECTURE.md`：可插拔域内后端架构
