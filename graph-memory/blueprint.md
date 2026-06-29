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

## 当前状态（2026-06-29）

- ✅ 1M context 本地异构分布式推理成功（RTX 4090 CUDA + RX 9060 XT HIP，3:1 分片）
- ✅ 昇腾 910B NPU 控制面 E2E 打通
- ✅ Rust correctness model、QUIC transport、capacity-aware 分片均已验证
- ✅ Striped Attention 原型已验证：在 white/pearl 单进程 3:1 场景下未改善负载均衡，已挂起
- 🔄 核心目标：论证 CXL / 类 RDMA 高速互联对异构推理服务上主流舞台的必要性
- ⏸️ 已挂起：更大模型 / 更多 domain / 更长 seq 验证（受限于当前硬件环境）
- ⏸️ 已挂起：Striped Attention / Stripe Ring Attention 适配（非均等切分兼容性问题未解）

## 关键约束

- **1 GPU = 1 worker**：禁止单卡多 worker 加载多份完整权重。
- **BF16 数值验证以 argmax/文本指标为准**：跨平台 logits 数值差异主要由 BF16 online softmax 处理顺序导致，不是实现 bug。
- **不引入有损优化**：量化、稀疏 attention、投机解码等在当前 correctness 阶段被禁止。

## 重要文档

- `docs/DESIGN.md`：设计总览
- `docs/HLPP_VS_HCP.md`：与 HLPP 的边界
- `docs/SCALING_ARGUMENT.md`：context 长度与显存/网络/域数的 scaling 分析
- `docs/PLUGIN_ARCHITECTURE.md`：可插拔域内后端架构
- `reports/1m-white-pearl-20260619/`：1M 里程碑报告
