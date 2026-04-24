# 产品上下文

## 问题定义

长上下文需求持续增长，但单卡显存和同构高端集群供给无法无限增长。现实资源通常是混合的：CUDA 卡、Apple Silicon / MLX、其他加速器。若这些设备不能协作，超长 context 任务只能截断、强摘要降级或直接放弃。

HCP 的产品问题是：当 context 长度继续增长时，系统能否通过增加异构 domain / 设备继续支撑任务，而不是受制于最强单卡。

## 目标用户

- 研究超长 context attention、context parallelism、异构推理系统的工程师。
- 需要在混合设备环境中运行长上下文任务的系统开发者。
- 需要可复现实验报告、transport metrics、correctness 对照的研究人员。

## 核心使用流程

1. 阅读 `README.md` 和 `docs/`，理解 HCP 与 HLPP 的边界。
2. 运行本地 C++ coordinator smoke，确认 standalone skeleton 可构建。
3. 扩展 Python online softmax 原型，生成 correctness report。
4. 为 `RingAttnMessage` 增加 serialization / deserialization。
5. 实现最小 P2P send/recv smoke。
6. 引入异构 runtime stub，推进 remote heterogeneous smoke。

## 产品决策

- HCP 不是 HLPP 的细粒度版本，而是 intra-layer / low-boundary 路线。
- 跨异构域坚持 P2P，不把 all-gather、reduce-scatter、all-to-all、all-reduce 作为主假设。
- correctness 和协议闭环优先于性能图。
- 每个阶段都要输出结构化实验产物，而不是只保留口头结论。

## 关键能力

- 不要求单张卡持有完整长上下文 attention 工作集。
- 允许不同设备以不同 `seq_chunk_len` / `block_size` 参与。
- 用 Ring Attention 形态传递 `K/V block` 并维护 online softmax state。
- 保留单机 reference / baseline 对照。
- 为后续 remote heterogeneous deployment 保留 path、interpreter、report discipline。

## 成功标准

- online softmax 在不均分 `seq_chunk_len` / `block_size` 下与 reference attention 对齐。
- `RingAttnMessage` 可以稳定编码、传输、解码。
- 2-domain remote heterogeneous smoke 可以复现，并产出 correctness、transport、failure summary。
