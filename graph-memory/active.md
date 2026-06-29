# Active Context

当前活跃的任务、决策、风险和假设。

### 当前焦点：1M 异构分布式推理已闭环

type: `task` · status: `superseded` · confidence: 0.95 · importance: 0.95 · source: `memory-bank/activeContext.md`

1M v9（3:1 split）成功，prefill 24/24 + decode 5/5，exit=0。文档已同步：1M_CONTEXT_THUNDERBOLT_PLAN.md、SCALING_ARGUMENT.md、systemPatterns.md。当前无未完成的 1M 攻坚任务；下一步决定是否需要更大模型 / 更多 domain 验证。

_updated: 2026-06-29 06:01:28_
### 下一阶段：从 1M 可行性验证走向多条扩展线探索

type: `task` · status: `ongoing` · confidence: 0.8 · importance: 0.95 · source: `user-direction`

1M 只是众多验证线中的一条。接下来需要并行探索：
1. 网络速度对异构 CP 收益的影响（CXL / 类 RDMA 方向）。
2. Stripe Ring Attention 等算法升级在 HCP 框架中的适用性。
3. Block KV cache + vLLM 集成：插件解耦 vs HCP 内联 PageAttention 两条路线。

_updated: 2026-06-29 06:01:28_
### Block KV cache + vLLM 集成：插件解耦 vs HCP 内联 PageAttention 双路线

type: `hypothesis` · status: `open` · confidence: 0.5 · importance: 0.9 · source: `user-direction`

当前 HCP 主要关注整段 KV cache 的 P2P 传输。下一步探索与 vLLM 生态结合：
路线 A（插件解耦）：HCP 作为 vLLM 外部的 context-parallel 插件，通过标准接口交换 block-level KV，保持 vLLM 内部完整。
路线 B（HCP 为主 + 内联 PageAttention）：HCP 自身管理 page/block 粒度的 KV，内联 PageAttention 的 scheduling/block 机制，深度整合以获得最佳性能。
需要并行验证两条路线的工程可行性、correctness 风险和对 vLLM 版本升级的耦合度。

_updated: 2026-06-29 06:06:09_
### [论文] Striped Attention: Faster Ring Attention for Causal Transformers

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `https://arxiv.org/abs/2311.09431`

作者：William Brandon 等 (MIT)，arXiv:2311.09431，2023。
核心发现：因果 attention 的三角结构导致 Ring Attention 工作负载不均。
方案：每个 device 持有均匀分布在整个序列上的 token 子集（striped permutation），而非连续 chunk。
效果：A100 256K 序列上端到端吞吐提升最高 1.45×；16×TPUv4 786K 序列上 1.65×。
实现复杂度：只需在 forward 开始前对输入序列做一次 permutation，并调整 attention mask 结构。
与 HCP 相关性：直接相关，可能缓解 pearl 等小/慢 domain 在 Phase 2 成为瓶颈的问题。

_updated: 2026-06-29 06:06:09_
### 先做：Stripe Attention 调研与 HCP 适配性分析

type: `task` · status: `ongoing` · confidence: 0.75 · importance: 0.9 · source: `user-direction`

具体步骤：
1. 精读 Striped Attention 论文（arXiv:2311.09431），提取 permutation + mask 调整细节。
2. 在 HCP Rust correctness model 中实现 striped permutation 原型，验证与同构/异构 uneven chunk 的兼容性。
3. 对比 vanilla ring vs striped 在因果 attention 下的 per-domain 计算量（理论 + 模拟）。
4. 输出设计文档：如何在 capacity-aware 不均等分片下应用 stripe（或不适用）。

_updated: 2026-06-29 06:06:09_
### 异构 CP 对网络速度敏感，CXL / 类 RDMA 互联可显著突破网线局限

type: `hypothesis` · status: `open` · confidence: 0.6 · importance: 0.85 · source: `user-direction`

当前 2.5G 有线以太网在 1M context 下不是带宽瓶颈（prefill 受显存与 memory-bound compute 主导），但随着 chunk 缩小、domain 增多或模型变大，KV ring 的通信量会快速上升。需要系统测试不同 RTT/带宽（WiFi、2.5G、10G、RDMA、CXL）对 prefill/decode 的边际收益，论证在异构节点上投资高速互联（CXL / GPU Direct / RDMA）能否取得与增加显存同量级的回报。

_updated: 2026-06-29 06:06:09_
### [论文] Ring Attention with Blockwise Transformers for Near-Infinite Context

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `https://arxiv.org/abs/2310.01889`

作者：Hao Liu, Matei Zaharia, Pieter Abbeel (UC Berkeley)，arXiv:2310.01889，ICLR 2024。
提出 blockwise attention + online softmax，使 self-attention 计算可分布到多个设备；KV block 沿 ring 传递。
HCP 的数学基础即来源于此。

_updated: 2026-06-29 06:06:09_
### Ring Attention 主流衍生方案综述

type: `claim` · status: `held` · confidence: 0.75 · importance: 0.85 · source: `web-search-survey`

除原始 Ring Attention 与 Striped Attention 外，当前主流/相关方案包括：
- Ring Flash Attention（zhuzilin 等开源）：将 FlashAttention kernel 与 Ring 通信重叠。
- ZigZag Ring Attention（ring-flash-attention issue #2 / Megatron-Core）：通过折叠 query 维度并在 worker 间镜像 block 平衡负载。
- DeepSpeed Ulysses（Microsoft, arXiv:2309.14509）：用 all-to-all 替代 all-gather/reduce-scatter 的序列并行，聚焦同构集群通信效率。
- USP（Tencent, arXiv:2405.07719）：统一 Ulysses + Ring Attention 的序列并行框架。
- Context Parallelism for Scalable Million-Token Inference（arXiv:2411.01783）：面向推理的 context parallelism。
- MoBA (Mixture of Block Attention, arXiv:2502.13189)：块级稀疏 attention，可与 ring 结合。
- XAttention (arXiv:2502.xxxxx)：block sparse attention with antidiagonal scoring。
- MTraining (arXiv:2510.18830)：基于 Striped Ring Attention 的动态稀疏 attention 训练系统。
- Mnemosyne (arXiv:2409.17264)：多百万 token 推理服务系统，讨论 Ring/Striped 在推理中的 head-of-line blocking 与 batching 局限。

_updated: 2026-06-29 06:06:09_
### 1M white+pearl 是可行性里程碑，而非生产实用配置

type: `belief` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `user-reflection`

在 16GB + 24GB 两台消费级机器上跑通 1M context 证明了 HCP 异构不均等 CP 的可行性边界，但 decode 每 token ~3 分钟、white 显存几乎满载，距离实际生产部署仍有显著差距。其价值在于验证架构路径，而非直接作为产品配置。

_updated: 2026-06-29 06:01:28_
### Stripe Ring Attention 可适配 HCP 并改善异构负载均衡

type: `hypothesis` · status: `open` · confidence: 0.55 · importance: 0.8 · source: `user-direction`

传统 Ring Attention 按 chunk 顺序遍历，小显存 domain 在 Phase 2 接收更多 remote block，容易成为瓶颈。Stripe Ring Attention 通过更细粒度或交错式的 KV block 调度，把负载分配得更均匀。需要评估其是否兼容 HCP 的 P2P / online-softmax / 非均等 chunk 设计，以及能否缓解 pearl 类慢节点的瓶颈。

_updated: 2026-06-29 06:01:28_
### 下一步决策：更大模型 / 更多 domain？

type: `uncertainty` · status: `open` · confidence: 0.5 · importance: 0.8 · source: `memory-bank/activeContext.md`

1M 里程碑已达成，需决定后续方向：1) 引入第三台设备做 3-domain 1M 降低单设备压力；2) 7B 1M context 可行性评估；3) KV cache 量化/压缩以缩短 decode 时间和降低显存占用。

_updated: 2026-06-29 05:34:19_
