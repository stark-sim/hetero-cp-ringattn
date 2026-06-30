# Active Context

当前活跃的任务、决策、风险和假设。

### 下一阶段：从 1M 可行性验证走向多条扩展线探索

type: `task` · status: `ongoing` · confidence: 0.8 · importance: 0.95 · source: `user-direction`

当前核心方向：以 Ring Attention 为策略基础，推进与 vLLM 的 Block KV cache 集成。\n\n已完成/持有：\n1. hyp-net-speed：white-pearl 带宽矩阵与稳定性复测证明网络是首要瓶颈。\n2. claim-ring-derivatives：在 HCP 上实现并对比 Vanilla/Striped/ZigZag；Ring Flash 挂起。\n3. decision-ring-attn-chosen：用户确认以 Ring Attention 为模型策略继续推进。\n\n下一步开放工程线：\n- hyp-block-kv-vllm：Block KV cache + vLLM 集成。

_updated: 2026-06-30 09:00:34_
### 异构 CP 对网络速度敏感，CXL / 类 RDMA 互联可显著突破网线局限

type: `hypothesis` · status: `held` · confidence: 0.85 · importance: 0.95 · source: `user-direction`

HCP 跨节点推理性能对网络带宽极度敏感。\n\n证据（正常规模工作负载）：\n1. Qwen2.5-3B/1K 单节点 CUDA 0.14s，分布式 ~12s（~85× 慢）。\n2. Qwen2.5-3B/4K 单节点 CUDA 0.27s，分布式 ~40s（~148× 慢）。\n3. 分布式 3B 甚至慢于单节点 CPU（3B/1K 12s vs 7.8s；3B/4K 40s vs 29s）。\n4. 策略差异仅在 3B/1K 可见（ZigZag ~5%），4K 时被网络完全掩盖。\n5. 7B bf16 无法装入 pearl 16GB HIP，分布式 7B 在当前无量化路径下不可行。\n\n结论：对正常规模的 3B/7B 模型和 1K/4K seq，跨节点网络仍是首要瓶颈；CXL/类 RDMA 高速互联是 HCP 实用的必要前提。

_updated: 2026-06-30 06:27:31_
### HCP P2P KV ring 在 ≤1 Gbps 跨节点以太网下会成为端到端瓶颈

type: `belief` · status: `held` · confidence: 0.85 · importance: 0.95 · source: `ev-net-speed-matrix-20260629`

基于 white-pearl 限速矩阵：\n- 2.35 Gbps 基线 20.5 s\n- 1 Gbps 29.5 s（1.44x）\n- 500 Mbps 50 s（2.44x）\n- 100 Mbps 445 s（21.7x）\n\n在 Qwen2-0.5B-1M、seq=4096、max_tokens=5 的异构推理任务中，端到端 latency 随跨节点带宽下降呈非线性增长。低于 1 Gbps 时，P2P KV ring 的通信时间显著超过计算时间；100 Mbps 时通信完全主导总时间。\n\n推论：若要在生产环境中部署异构 CP 推理，需要 CXL / RDMA / 高速 NVLink 等级别的互联带宽，否则网络将把多卡聚合的显存优势抵消为极高的延迟惩罚。

_updated: 2026-06-29 14:32:15_
### 当前焦点：1M 异构分布式推理已闭环

type: `task` · status: `superseded` · confidence: 0.95 · importance: 0.95 · source: `memory-bank/activeContext.md`

1M v9（3:1 split）成功，prefill 24/24 + decode 5/5，exit=0。文档已同步：1M_CONTEXT_THUNDERBOLT_PLAN.md、SCALING_ARGUMENT.md、systemPatterns.md。当前无未完成的 1M 攻坚任务；下一步决定是否需要更大模型 / 更多 domain 验证。

_updated: 2026-06-29 06:01:28_
### 决策：以 Ring Attention 为 HCP 模型策略继续推进

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `user direction`

用户确认：ZigZag/Striped/Vanilla 的策略差异已理解，继续以 Ring Attention 作为 HCP 的跨节点上下文并行策略。\n\n当前选择：\n- 调度策略保留 Vanilla 为默认，ZigZag 在中小长度/计算敏感场景可作为备选。\n- Ring Flash Attention 因当前网络瓶颈挂起。\n\n下一步：推进 hyp-block-kv-vllm（Block KV cache + vLLM 集成）。

_updated: 2026-06-30 09:00:34_
### 任务：实现并对比两种 HCP 调度策略

type: `task` · status: `suspended` · confidence: 0.75 · importance: 0.9 · source: `user-direction`

对比已暂停。当前证据（CPU/CUDA/HIP 单进程 3:1 4096）不支持 Striped，但结论尚未定论。根本开放问题是 Striped 与非均等切分的兼容性，需要逻辑解构层面的新设计，而非简单实测。

_updated: 2026-06-29 13:05:20_
### Striped Attention 与非均等 capacity-aware 切分的兼容性

type: `uncertainty` · status: `open` · confidence: 0.5 · importance: 0.9

原始 Striped Attention 解决的是"均分设备 + 因果 mask"导致的每轮负载不均。\n\nHCP 面临的则是"非均等 capacity-aware 切分 + 异构算力"导致的负载不均。两者不完全等价：\n- 均分 Stripe：每 device token 数相同，每轮有效 pair 比例 ≈ 50%。\n- 不均分 Stripe：每 device token 数不同，但 token 仍散布。此时每个 device 处理的有效 pair 数不仅取决于自己的 token 数，还取决于它作为 Q 和作为 KV 被其他 domain 访问的方式。\n\n开放问题：\n1. 能否定义一种"capacity-aware striped scheduling"，使得 domain i 处理的总有效 pair 数 ∝ capacity_i，且每轮仍有良好 mask 比例？\n2. 在线 softmax 的 block 处理顺序是否可优化（例如先处理对自己最有利的 KV block）？\n3. 是否需要打破"Q 固定、KV 轮转"的约束，允许根据 capacity 动态调整 KV 传输顺序或数量？\n\n在这些问题有答案之前，不能判定 Striped 对 HCP 无用。

_updated: 2026-06-29 13:05:20_
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
### 决策：将 claim-ring-derivatives 降级为文献引用背景

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `user-direction`

用户指出：claim-ring-derivatives 如果只是综述而没有真实实现和硬件对比，缺乏说服力和工作量。\n\n评估结果：\n- Ring Flash Attention 实现成本高（kernel 层）。\n- ZigZag 实现成本中等但可能重蹈 Striped + uneven 兼容覆辙。\n- hyp-net-speed 已有直接带宽证据，足以支撑 CXL/RDMA 必要性。\n\n因此将该线从“需实现的支撑线”降级为“文献引用背景”，资源继续集中在 hyp-net-speed 深化。

_updated: 2026-06-29 15:48:58_
### 1M white+pearl 是可行性里程碑，而非生产实用配置

type: `belief` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `user-reflection`

1M white+pearl 是可行性里程碑，证明异构不均等 CP 在极端长 context 下可以跑通。但它不是生产实用配置，也不是论证 CXL/RDMA 必要性的核心证据。当前 CXL/RDMA 论证应基于网络带宽对 P2P KV ring 吞吐的直接影响，而非 1M 端到端结果。

_updated: 2026-06-29 13:27:24_
### Stripe Ring Attention 可适配 HCP 并改善异构负载均衡

type: `hypothesis` · status: `suspended` · confidence: 0.75 · importance: 0.85 · source: `user-direction`

挂起：与 Striped Attention 一并挂起。在 Striped + 非均等切分的兼容性问题解决前，不再推进 Stripe Ring Attention 适配。

_updated: 2026-06-29 13:27:24_
### HCP 调度策略对比：capacity-aware 连续分片 vs 加权 Striped

type: `claim` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `user-direction + design-reasoning`

capacity-aware 连续分片与加权 Striped 是 HCP 的两种候选调度策略。\n\n当前状态：\n- CPU/CUDA/HIP 单进程 3:1 实测均显示 vanilla 更优，但结论**尚未定论**。\n- 核心瓶颈是 Striped 与非均等切分的兼容性问题：当前加权 round-robin 只是简单扩展，没有从"有效 attention pair 数 ∝ capacity"的角度重新设计 work distribution。\n- 默认策略仍保留连续分片；Striped 代码保留但挂起，等待更深入的理论分析或长序列 multi-node 证据。

_updated: 2026-06-29 13:05:20_
### 决策：挂起 Striped，转向其他扩展方向

type: `decision` · status: `held` · confidence: 0.8 · importance: 0.85

基于当前分析，Striped Attention 在 HCP 中的验证暂时挂起。团队资源转向其他方向，包括：\n1. 网络速度对异构 CP 收益的影响（CXL / 类 RDMA 方向）。\n2. Block KV cache + vLLM 集成：插件解耦 vs HCP 内联 PageAttention 双路线。\n\nStriped 问题保持开放，未来若有人提出与非均等切分兼容的理论设计，再重启。

_updated: 2026-06-29 13:05:20_
### Striped Attention 可以推广到 capacity-aware 不均等分片

type: `claim` · status: `held` · confidence: 0.75 · importance: 0.85 · source: `paper-analysis + design-reasoning`

原始 Striped Attention 论文假设每个 device 持有 L/N 个 token（均分），但其核心思想——让各 device 的 token 均匀散布在原始序列中——可以推广到任意比例。

推广方式：加权循环调度（weighted round-robin scheduling unit）
- 对 3:1 的 2-domain 场景，调度周期为 4，模式为 [0,0,0,1]。
- device 0 持有所有满足 p mod 4 ∈ {0,1,2} 的位置，占 75%。
- device 1 持有所有满足 p mod 4 = 3 的位置，占 25%。
- 当 scheduling unit 足够小（如 1 token 或几十 tokens）时，每个 device 的位置在原始序列中近似均匀散布。

为什么这能保留 Striped 的好处：
1. early-return 不对称性被消除：domain 0 的 Q 会“看到”domain 1 的部分历史 KV，   不再像连续 chunk 那样整 block 被跳过。
2. 负载按容量比例分配：domain 0 处理约 75% 的有效 attention pair，domain 1 处理约 25%，   与它们的 chunk 比例一致，符合 capacity-aware 的初衷。
3. 不需要改变通信原语：仍然是 Q 固定、KV 沿 ring P2P 传递。

需要放弃原始论文的简单 block-triangular mask：
- 加权 stripe 的 residue 关系不再是简单的 j<k 或 j>k。
- 必须改用原始位置 id 比较来构造 causal mask（已在适配计划中提出）。

限制：
- 论文中的理论 2× speedup 上界仅在均分且 N 较大时严格成立；  不均等场景下收益是启发式的，取决于 scheduling unit 大小和具体比例。
- scheduling unit 过大时，device 的 token 会局部聚集，early-return 会重新出现。

_updated: 2026-06-29 07:49:48_
### 基线测量：vanilla Ring Attention 在 3:1 不均等分片下的 compute 失衡

type: `evidence` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `HCP_PERF_LOG /tmp/ring_perf_8192.jsonl`

测试配置：seq_len=4096，2 domain，chunk0=3072 (75%)，chunk1=1024 (25%)，num_heads=8，head_dim=128，float32 CPU。
命令：DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib HCP_PERF_LOG=/tmp/ring_perf_8192.jsonl cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture

测量结果（单次 layer，mock transport）：
- domain 0 (大 domain): total 150.3 ms，local_compute 148.0 ms，peer_compute 0.001 ms
- domain 1 (小 domain): total 41.4 ms，local_compute 14.7 ms，peer_compute 26.2 ms
- domain 0 总耗时约为 domain 1 的 3.6 倍

解读：
- 由于 chunk 连续且因果 mask，domain 0 的 peer KV（来自 domain 1，全局位置 3072-4096）全部位于 Q0 的“未来”，触发 early-return，几乎不耗计算。
- domain 1 的 Q 需要 attend 到 domain 0 的全部 3072 个位置，因此 peer_compute 占其总时间 63%。
- 在相同算力设备上，大 domain 成为瓶颈；在异构设备上，若小 domain 算力更慢，瓶颈会进一步恶化。

_updated: 2026-06-29 07:44:40_
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
### [论文] Ring Attention with Blockwise Transformers for Near-Infinite Context

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `https://arxiv.org/abs/2310.01889`

作者：Hao Liu, Matei Zaharia, Pieter Abbeel (UC Berkeley)，arXiv:2310.01889，ICLR 2024。
提出 blockwise attention + online softmax，使 self-attention 计算可分布到多个设备；KV block 沿 ring 传递。
HCP 的数学基础即来源于此。

_updated: 2026-06-29 06:06:09_
### Ring Attention 衍生方案综述仅作为文献背景，不单独实现

type: `claim` · status: `held` · confidence: 0.8 · importance: 0.8 · source: `user-direction + cost-benefit review`

原始 Ring Attention、Striped Attention、ZigZag Ring Attention 等方案都基于 P2P KV ring，天然对跨节点带宽敏感。\n\n已完成：\n- Phase 1：在 Rust 中抽象出 RingSchedulingStrategy，实现 Vanilla / Striped / ZigZag 的 assignment 与 CPU mock 正确性验证。\n- Phase 2a：3:1 容量感知切分下完成三种策略的真实硬件对比。\n- Phase 2b：1:1 等分切分下完成三种策略的真实硬件对比。\n- Phase 4：撰写 docs/RING_DERIVATIVES_BENCHMARK.md 并更新 SCALING_ARGUMENT.md。\n\n关键结论：\n1. HCP 的异构设计能承载 Vanilla/Striped/ZigZag 三种调度策略。\n2. 无论是 3:1 还是 1:1 切分，策略差异都 <6%，网络 recv 是绝对瓶颈。\n3. Ring Flash Attention 是 kernel 层优化，在当前网络瓶颈下无法改善端到端性能，已挂起。

_updated: 2026-06-30 04:41:51_
### 决策：Ring Flash Attention 实现线挂起

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `real-hardware measurements`

Ring Flash Attention 的核心收益是减少 local attention tile 的 HBM 访问，从而加速计算。\n\n评估：\n- 当前 white+pearl 4K 任务中，local compute 只占总时间 <12%，网络 recv 占 >88%。\n- 实现 Ring Flash 需要自定义 CUDA/HIP kernel 或 PyO3 SDPA 桥接，工程量大。\n- 即使完美实现，也只能压缩那 <12% 的时间，无法改善跨节点带宽瓶颈。\n\n结论：在当前阶段不投入 Ring Flash 实现资源，优先用现有 Vanilla/Striped/ZigZag 证据完成 CXL/RDMA 必要性论证。未来网络升级后可重启。

_updated: 2026-06-30 03:34:13_
### 下一步决策：更大模型 / 更多 domain？

type: `uncertainty` · status: `suspended` · confidence: 0.5 · importance: 0.8 · source: `memory-bank/activeContext.md`

挂起：当前基础实验环境只有 white + pearl 两台机器，且 1M 实验已证明可行性边界。更大模型 / 更多 domain 的验证需要额外硬件资源，与当前核心目标（论证 CXL/RDMA 对异构推理服务的重要性）不直接相关。

_updated: 2026-06-29 13:27:24_
### Striped 预计能将 3:1 分片下的 domain 总耗时差距从 ~3.6× 降到 ~1.2× 以内

type: `hypothesis` · status: `rejected` · confidence: 0.1 · importance: 0.8 · source: `theoretical projection`

原假设"在 3:1 分片下 Striped 能将 domain 总耗时差距从 ~3.6x 降到 ~1.2x"已被 white CUDA 和 pearl HIP 真实硬件证据否定。在两种加速卡上单进程 3:1 4096 场景下，Striped 均使瓶颈 domain 0 更慢。该假设仅对 homogeneous CPU mock 成立的可能性已被排除。

_updated: 2026-06-29 12:44:16_
### Vanilla Ring Attention 的 early-return 在不均等分片下加剧负载不均

type: `claim` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `code-inspection + baseline measurement`

process_kv_block 在因果路径下会跳过 kv_global_start >= q_global_end 的 block。连续 chunk 场景下，持有靠前 token 的大 domain 会跳过来自后续小 domain 的 peer block，导致其 peer_compute 接近零；而小 domain 必须处理来自大 domain 的全部历史 KV。这是 vanilla ring 在 capacity-aware 不均等分片下出现 3.6× 耗时差距的根本原因。

_updated: 2026-06-29 07:44:40_
### 100 Mbps 重复实验方差极大的根因未明

type: `uncertainty` · status: `open` · confidence: 0.6 · importance: 0.75 · source: `ev-net-speed-matrix-20260629`

完整矩阵中 100 Mbps 两次重复分别为 206 s 和 684 s，差距超过 3x。可能原因包括：\n1. pearl RX 9060 XT 热节流或功耗状态变化。\n2. QUIC / tch-rs 在低速链路上的拥塞控制或重传行为。\n3. 操作系统 / 网络栈的 bufferbloat 或 tc burst 参数导致偶发排队。\n4. 模型 / runtime 内部某个 warmup / cache / 分配路径在第二次运行时触发不同路径。\n\n在把 100 Mbps 数字作为核心论据前，需要复现并解释该方差。

_updated: 2026-06-29 14:32:15_
### 训练场景评估：Striped Attention 训练收益对 HCP 当前目标意义有限

type: `claim` · status: `held` · confidence: 0.75 · importance: 0.7 · source: `paper-analysis + user-direction`

Striped Attention 论文主要面向训练（forward + backward）。HCP 当前聚焦推理，且目标硬件是异构消费级设备（CUDA + HIP/MPS），互联带宽/延迟远低于训练集群。若扩展到训练，需要：
- backward 阶段沿反方向传递梯度，并维护 ring 中的 activation/gradient buffer。
- 跨 domain 的梯度同步（all-reduce 或类似机制），这与 P2P-only 假设冲突。
- 消费级设备的 PCIe/Ethernet 互联难以支撑训练所需的高吞吐参数/梯度通信。
结论：训练在理论上可行，但不是 HCP 当前阶段的高优先级方向；先把推理 + Striped 走通。

_updated: 2026-06-29 06:16:16_
### [挂起] Striped + 非均等切分兼容性问题

type: `task` · status: `suspended` · confidence: 0.75 · importance: 0.5

状态：挂起（on hold）。\n\n原因：当前 CPU/CUDA/HIP 单进程 3:1 实测均显示 Striped 使瓶颈 domain 0 更慢，但测试覆盖的是最简单实现（加权 round-robin + 原始位置 mask）。从逻辑解构上看，尚不能下定论：\n1. Striped 的核心收益来自"每轮都有有效计算"，而非简单把 peer compute 从小 domain 转到大 domain。\n2. 当前实现把 domain 1 的 peer compute 推给 domain 0，是因为 3:1 不均等下 domain 0 本已持有 75% token，天然会承担更多跨域 pair。\n3. 真正的兼容性问题：如何设计 scheduling + work distribution，使得在非均等容量下，各 domain 的"有效 attention pair 数量"与"其算力/容量"匹配，而不是与"token 数量"线性匹配。\n\n关键开放问题：\n- 在非均等切分下，是否存在一种 Striped 变体，使得每个 domain 处理的有效 pair 数 ∝ 其 capacity，同时仍保持每轮 mask 比例均衡？\n- 是否需要引入 dynamic load balancing、sub-block tiling、或 redistribute KV blocks？\n- 当前 early-return 和 online softmax 是否在不均等场景下隐藏着额外的调度空间？\n\n重启条件：有人能对上述问题给出形式化分析或可行的算法设计，或在真实 multi-node 长序列（≥128k）不均等场景下获得 wall-time 收益证据。

_updated: 2026-06-29 13:05:20_
