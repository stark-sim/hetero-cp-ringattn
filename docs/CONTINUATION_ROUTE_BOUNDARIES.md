# Continuation 路线边界定义(路线 B 一期合 main 时确立)

本文定义 continuation prefill/decode 各实验路线在 main 上的资产边界,服务两个目的:

1. 各路线可以**复用路线中性基建并组合彼此优势**;
2. 无法组合时,可以在**同一测量锚点**上公平比较当前阶段谁更符合项目总要求(异构、带宽受限、neighbor-only P2P ring、capacity-weighted 永久 KV)。

路线 B 一期已按三期里程碑门禁过关(synthetic → 真实权重 → 跨设备 → 三平台三节点真 ring),其资产以纯增量方式合入 main,**main 上所有生产调用路径行为不变**。

## A. 路线中性基建(已在 main,任何路线可复用)

| 组件 | 位置 | 说明 |
|---|---|---|
| `ReservedPositionedKvShard` | `rust/src/model/self_driving.rs` | 显式绝对位置 + 预分配 append 的 KV 驻留合同;capacity 逐层预约 |
| `FrozenKvAssigneeSchedule` | `rust/src/model/self_driving.rs` | capacity tickets + request_id 派生的确定性 position ownership;多进程各自推导同一份日历 |
| `KvTransport` trait + wire codec | `rust/src/model/transport/`、`rust/src/distributed/transport/` | TCP/QUIC 全双工;`KvBlock` / `RingPacket` / `SelfDrivingPacket` codec 均 shape-generic(m≥1、任意 position 子集) |
| `KvCache::committed_position_ids` | `rust/src/model/cache.rs` | P_Q/P_KV 双位置合同 |
| phase 显式 causal 语义 | `rust/src/model/model.rs`、`rust/src/model/attention/ring.rs` | causal 由 phase 而非 local seq_len 决定(3aa7282 起) |
| positioned online-softmax oracle | `rust/src/model/attention/` | `positioned_local_compact_partial` / `positioned_merge_compact_partial`,m≥1、非连续 P_KV |

## B. 路线 B 执行原语(KV-stationary packet 族,一期合入)

| 组件 | 位置 | 说明 |
|---|---|---|
| `process_layer_packet_with_reserved_history` | `rust/src/model/self_driving.rs` | 单 assignee 的整层 packet 处理(m≥1) |
| `process_layer_packet_with_reserved_history_for_positions` | 同上 | 按 frozen plan 的 position offset 子集投影/append 新 KV,历史 KV 不进 packet |
| `run_model_ring_with_reserved_history_for_positions` + `PositionedLayerRingStats` / `PositionedModelRingResult` | 同上 | 24 层递推 runner 与统计 |
| `project_final_logits` | 同上 | finisher 侧 final norm + LM head |

语义边界:

- continuation/decode **不移动历史 KV**;packet 携带 residual/normalized/Q/O/LSE + position_ids,payload 与历史长度 T 无关;
- position ownership 由 frozen request plan 本地推导,**不进 packet payload**;
- attention 按节点逐跳串行,Norm/MLP/LM head 集中在 finisher;
- decode 是 m=1 的特例。

## C. 验证资产(实验态,不改变生产行为)

- 单测:`self_driving.rs` 与 `tch_backend.rs` 的 `#[cfg(test)]`,含 `#[ignore]` 真实 Qwen 权重测试;
- 实验 harness:`rust/src/bin/route_b_cross_node_smoke.rs`(local / server / client / node 四模式,N 节点真 ring)、`rust/examples/dense_forward_probe.rs`、`scripts/compare_route_b_dumps.py`(tie-aware);
- 实验 API 面:`lib.rs` 的 pub 重导出与若干 `pub(crate)→pub` 提升均带 `experimental: raised for route_b_cross_node_smoke` 注释,二期工程化时可能收窄,不作为稳定合同。

## D. 合并不改变什么

- KV-ring baseline(initial/continuation prefill)与 Q-ring decode 的生产路径零改动;main 行为不变;
- placement/ledger WIP 保持 main 工作区未提交 DEFER 状态(二期 admission 素材);
- 路线分支 `codex/route-b-continuation-stationary-packet` 保留继续探索。

## E. 组合接缝(供路线间结合)

- **路线 A/C(KV-ring 变体)**:共享 A 类全部基建;continuation 可按 segment 大小静态选择 ring 或 packet(大 m/T 时 KV-ring 的 query 并行与紧凑 payload 仍可能占优;不引入动态 planner);
- **路线 F(可组合 kernel 技术)**:可作为 `process_layer_packet_*` 内部 attention 计算的替代实现插入,packet 状态机不变;
- **路线 G(显式阶段组合)**:以阶段(prefill / decode / continuation)为粒度组合 KV-ring 与 stationary packet;两族已在同一 shard/schedule 合同上共存(97ca355 的 prefill-ring + decode/continuation-packet 场景即实例)。

## F. 对比锚点与验证矩阵(供公平比较)

- 公共测量锚点:本次合并后的 main tip;任何路线间比较必须对齐同一 commit、模型、T/m/N/dtype、warmup/repetition、payload/hop 定义(route-experimentation 纪律);
- 一期验证矩阵(证据边界):N∈{2,3}、m∈{1,4}、24 层、Qwen2-0.5B BF16;平台 Mac MPS / white CUDA(RTX 4090)/ pearl HIP(RX 9060 XT);拓扑为 neighbor-only 逐跳 ring(N=3 实证无直连捷径);
- 数值判据:tie-aware argmax(1 bf16 ulp = 0.0625 平局豁免)+ mean diff < 0.1 + max diff < 0.75;
- 一期全部证据为 correctness,**不含性能声明**;性能比较属三期公平 benchmark 出口标准。

## 关键证据索引

| 证据 | graph 节点 | 提交 |
|---|---|---|
| m>1 stationary packet 单层合同 | (progress) | 5777d51 |
| position owner-local KV 分散 | (progress) | 73cd0e8 |
| 24 层 mixed-history continuation | (progress) | a7a583d |
| continuation 后 decode 闭环 | (progress) | b523bc7 |
| 真实 Qwen 权重 continuation | (progress) | 97ca355 |
| 跨设备数值验证(MPS+CUDA) | evidence-route-b-cross-device-mac-mps-white-cuda-20260809 | d9c1d35/ed6e658/9490c6f |
| 三平台三节点真 ring | evidence-route-b-three-node-ring-mac-white-pearl-20260809 | 246aa8c/0a67cce/9ab909b/8a5b04e |
| bf16 argmax 平局翻转判据 | belief-bf16-argmax-near-tie-flip-20260809 | 9ab909b |
