# Rust Single-Layer Self-Driving Ring Experiment Plan

> **For Claude:** Use the `tdd` skill and execute only this single experimental checkpoint.

**Goal:** 用单进程真实 tensor 证明一个 decoder layer 可以在任意 `N` 的逻辑 P2P ring 上以单 packet、`N-1` hops 完成 attention，并由唯一 finisher 继续 residual、norm 和 MLP。

**Architecture:** 实验复用现有 Rust/tch 权重、RoPE、online-softmax、O projection、RMSNorm 和 MLP。调用方提供互斥的本地历史 KV shards；starter 生成唯一 Q，packet 按 successor 顺序访问全部 shards，assignee 在 packet 抵达时计算并持久保存唯一一份 current K/V，最后一个节点完成 layer continuation。第一步不接 QUIC、worker runtime、admission 或生产级 planner。

**Tech Stack:** Rust 2021、tch-rs/libtorch、现有 `HcpRingAttentionBackend`、CPU synthetic tensor correctness test。

---

## 本节点要回答的问题

在模型权重各 worker 已复制、历史 KV 已互斥切分的前提下，真实 decoder layer 是否能满足：

- Q 在 starter 投影一次，current K/V 在 assignee 投影一次；
- 每个 worker 只计算自己的历史 KV partial；
- current K/V 只由一个 assignee 贡献一次；
- packet 走 `N-1` hops 后得到完整 attention；
- 只有 finisher 执行 O projection、attention residual、post-attention norm、MLP residual；
- 输出与把相同 KV 合并后做单节点 attention 的参考结果一致。

## 改动、原因和计划贡献

| 改动 | 原因 | 对总计划的贡献 |
|---|---|---|
| 从 ring backend 提取 decode Q/K/V、partial merge 和 O projection 原语 | 现有 `forward()` 把投影、ring 和 O projection 绑在一个调用栈里，无法交接 layer continuation | 建立自驱动数据面的最小模型接口，不改变现有 forward |
| 新增单层 in-process ring experiment | 先隔离验证模型数学，避免网络/runtime 同时干扰 | 直接验证 attention 与 MLP/norm 可以组成一个单逻辑 forward |
| 使用 caller 提供的 uneven shards | 第一节点不需要 planner，但必须证明不均匀 KV 分片不破坏数学 | 保留 capacity-weighted 的核心兼容性 |
| 记录 exact-once 与 hop 统计 | 仅比较输出无法证明没有冗余 | 为后续真实 transport 提供同一验收口径 |

## TDD 顺序

1. RED：`N=3`、不均匀 shards、assignee 与 starter 不同，调用实验 API；当前模块/API 不存在，测试必须失败。
2. GREEN：实现最小 projection/merge/continuation，使 ring 输出与单节点参考一致，并断言 `hops=2`、`partials=3`、Q/KV/finisher 各一次。
3. RED -> GREEN：补 `N=1/2/4`，验证 `hops=N-1`、任意 starter/assignee 路由和输出一致。
4. 回归：运行 focused test，再运行现有 ring attention 测试，确认旧 forward 未改变。

## 明确不做

- 不连接物理 laptop、CUDA、ROCm 或 MPS 节点；本节点不依赖远程可达性。
- 不增加 QUIC packet schema、worker command、mode negotiation 或 scheduler。
- 不实现 admission、memory ledger、二维 assignee calendar 或动态迁移。
- 不宣称全模型、跨节点或生产可用；本节点只证明单层 tensor 可行性。
