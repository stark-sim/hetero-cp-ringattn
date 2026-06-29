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
### 先做：Stripe Attention 调研与 HCP 适配性分析

type: `task` · status: `ongoing` · confidence: 0.75 · importance: 0.9 · source: `user-direction`

精读已完成。下一步：
1. 在 HCP Rust correctness model 中实现 striped permutation（含 RoPE position id permutation）。
2. 修改 online softmax mask 逻辑以匹配 Striped GetMask。
3. 用同构/异构 uneven chunk 配置跑 correctness test，测量 per-domain 计算量。
4. 输出设计文档：capacity-aware 不均等分片下如何应用 stripe，以及是否需同时调整 chunk 分配策略。

_updated: 2026-06-29 06:16:16_
### 精读：Striped Attention 机制与 HCP 适配点

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `https://ar5iv.org/html/2311.09431`

来源：William Brandon 等，MIT，arXiv:2311.09431。

核心机制：
- 输入序列按 token 下标对 N（device 数）取模做 permutation，device i 持有下标满足 i mod N 的 token。
- 因此每个 device 的 Q/K/V block 包含均匀散布在整个原始序列中的 token，而非连续 chunk。
- 在每层 attention 开始前，Q/K/V 已经按此 layout 分好，不需要额外的 per-layer 通信。
- Mask 调整：因果 mask 仍基于原始序列顺序；Striped 的 GetMask 保证每个 device 每轮遇到的上三角 mask 比例大致相同，从而负载均衡。
- 对每轮 (Q_j, K_k, V_k)，若 j<k 则 mask 为下三角（含对角线以上全 -inf）；若 j≥k 则 mask 为上三角（含对角线以下全 -inf）。
- Workload：i≥j 时约 c(c+1)/2，i<j 时约 c(c-1)/2；最大 workload 从 Ring 的 c² 降到接近 c²/2，理论极限 speedup 2×。

实验结果：
- 8×A100 80GB，256K 序列，最高端到端吞吐提升 1.45×；16×TPUv4，786K 序列，1.65×。
- 序列越长、device 越多、block 越大，收益越明显。
- 实现基于 JAX，使用 bfloat16 + float32 attention，tile-based skipping。

HCP 适配关键点：
- P2P-only 友好：仍然保持 Q 固定、KV 沿 ring P2P 传递，通信原语不需要 all-to-all / all-gather。
- 非均等 chunk 兼容性：Striped 原始假设均分 block，但 permutation 本身可以推广到不均等 block（只要每个 device 的 token 在原始序列中均匀散布）。
- RoPE/位置编码：必须对 position ids 同步 permutation；HCP 的 distributed RoPE 需要知道原始全局位置。
- Online softmax：与 Ring Attention 完全一致，可直接复用 HCP 的 online softmax state 更新逻辑。
- 当前 HCP 中 pearl（小/慢 domain）在 Phase 2 接收更多 remote block 的瓶颈，有望通过 striped 缓解。

_updated: 2026-06-29 06:16:16_
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
### Stripe Ring Attention 可适配 HCP 并改善异构负载均衡

type: `hypothesis` · status: `open` · confidence: 0.75 · importance: 0.85 · source: `user-direction`

将 Striped Attention 的 striped permutation 引入 HCP，以缓解因果 attention 下 Ring Attention 的负载不均。具体需验证：1) 不均等 chunk size 下的 permutation 定义；2) RoPE position ids 的同步 permutation；3) 对 pearl 类慢节点的实际加速效果。预计对长序列、多 domain 场景收益最大。

_updated: 2026-06-29 06:16:16_
### P2P-only 异构场景下的 Ring Attention 衍生方案筛选

type: `decision` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `web-survey + paper analysis`

筛选标准：HCP 跨异构 domain 只支持 P2P send/recv，不支持 all-to-all / all-gather / reduce-scatter 等 collective。因此只保留可在纯 P2P ring 上实现的算法，排除依赖 NCCL/process-group 的方案。

✅ 适合 P2P-only / HCP：
- 原始 Ring Attention（Liu et al. 2023）：Q 固定，KV 沿 ring P2P 传递，online softmax。
- Striped Attention（Brandon et al. 2023）：在 Ring 基础上只做输入 permutation + mask 调整，通信原语不变。
- ZigZag Ring Attention（ring-flash-attention issue #2）：通过折叠 query 维度平衡负载，仍只需 P2P KV 传递。
- Ring Flash Attention（zhuzilin 等开源）：将 FlashAttention kernel 与 Ring P2P 重叠，支持 ring/zigzag/stripe 模式。

❌ 不适合 P2P-only（需要 collective 或与 HCP 假设冲突）：
- DeepSpeed Ulysses：依赖 all-to-all 交换 Q/K/V，需要同构 NCCL process group。
- USP（Tencent）：混合 Ulysses + Ring，Ulysses 段仍需 all-to-all，无法纯 P2P。
- Llama3 flash_attn_varlen_func（ring-flash-attention）：技术上不是 ring attention，使用不同 CP 机制。
- MoBA / XAttention / MTraining：稀疏/动态 attention 改变 attention 数学定义，HCP correctness-first 阶段不引入近似；且 MTraining 基于 Striped 但加入动态稀疏，需先验证基础 Striped。
- LightSeq：优化 sequence-parallel 的 all-to-all / reduce-scatter 通信，非 P2P。
- Mnemosyne：服务调度系统，非算法本身。

_updated: 2026-06-29 06:16:16_
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
### 下一步决策：更大模型 / 更多 domain？

type: `uncertainty` · status: `open` · confidence: 0.5 · importance: 0.8 · source: `memory-bank/activeContext.md`

1M 里程碑已达成，需决定后续方向：1) 引入第三台设备做 3-domain 1M 降低单设备压力；2) 7B 1M context 可行性评估；3) KV cache 量化/压缩以缩短 decode 时间和降低显存占用。

_updated: 2026-06-29 05:34:19_
### 训练场景评估：Striped Attention 训练收益对 HCP 当前目标意义有限

type: `claim` · status: `held` · confidence: 0.75 · importance: 0.7 · source: `paper-analysis + user-direction`

Striped Attention 论文主要面向训练（forward + backward）。HCP 当前聚焦推理，且目标硬件是异构消费级设备（CUDA + HIP/MPS），互联带宽/延迟远低于训练集群。若扩展到训练，需要：
- backward 阶段沿反方向传递梯度，并维护 ring 中的 activation/gradient buffer。
- 跨 domain 的梯度同步（all-reduce 或类似机制），这与 P2P-only 假设冲突。
- 消费级设备的 PCIe/Ethernet 互联难以支撑训练所需的高吞吐参数/梯度通信。
结论：训练在理论上可行，但不是 HCP 当前阶段的高优先级方向；先把推理 + Striped 走通。

_updated: 2026-06-29 06:16:16_
