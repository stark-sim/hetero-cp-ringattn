# System Patterns

架构概览、关键设计模式与架构决策。

### 异构分布式推理的数值验证策略

type: `belief` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `memory-bank/systemPatterns.md`

BF16 场景下，跨平台 BLAS 差异导致 logits 数值对比不是有意义的 correctness 指标。Correctness 应分层：L1 float32 数学正确性（cargo test synthetic weights）、L2 工程正确性（argmax 一致性/文本任务指标）、L3 端到端冒烟。强证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异，证明差异主要来自 BF16 online softmax block-wise 处理顺序，而非跨平台 bug。

_updated: 2026-06-29 05:34:19_
### P2P 约束下 ring 是数据面最小充分拓扑:每 worker 只需 2 个 peer 连接

type: `belief` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `user-direction + validated runs`

不考虑 controller/scheduler 控制面时,HCP 数据面(prefill KV ring + decode Q/LSE 累积器环)的全部通信都是与环上相邻两节点的 send/recv;无需 all-to-all/broadcast/reduce/gather。这使异构消费级硬件(以太网/tailscale/未来 CXL)即可组成 CP 系统,是"异构是常态"核心信念在拓扑层的直接推论。证据:三机真环 ringc-160010、decode dsplit4/dsplit6 均只使用邻接连接。

_updated: 2026-07-27 09:10:04_
### 产品问题：异构设备协作支撑超长 context

type: `blueprint` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `memory-bank/productContext.md`

长上下文需求持续增长，但单卡显存和同构高端集群供给无法无限增长。现实资源通常是混合的（CUDA、Apple Silicon/MLX、其他加速器）。HCP 的问题是：能否通过增加异构 domain / 设备继续支撑任务，而不是受制于最强单卡。

_updated: 2026-06-29 05:34:19_
### 架构概览：Rust + C++ 为主、Python 原型为历史对照

type: `blueprint` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `memory-bank/systemPatterns.md`

C++ 部分定义 HCP Ring Attention 低边界 runtime 抽象和 libtorch bridge。Rust 部分负责 correctness model、report、可序列化协议 schema 和 P2P transport smoke。每个 domain 持有本地 Q chunk，ring 中持续传递 K/V block，每个 domain 更新 online softmax state。

_updated: 2026-06-29 05:34:19_
### 技术栈：Rust + C++ + Python 原型

type: `component` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/techContext.md`

Core: C++17, CMake 3.16+, Rust 2021, Python 3。
Libtorch/PyTorch 2.11.0, tch-rs 0.24.0（可选 tch-backend）。
QUIC: quinn 0.11 + rustls 0.23 + rcgen 0.13。
模型权重：safetensors, tokenizers, half。

_updated: 2026-06-29 05:34:19_
### 决策:自驱动环(self-driving ring)——decode forward 整体骑在环上,无 owner 无冗余,每层 N-1 跳

type: `decision` · status: `held` · confidence: 0.9 · importance: 1.0 · source: `user-direction`

用户方向(2026-07-28,ISSUE-006 收口方案):自驱动环是关键。子环弹性已否决(环外 KV 必然集中,违反切分);Rust 线 N× 冗余 forward 同样不可接受。

【核心机制】包不回家:层 L 的包从 S 出发(S 以本地 partial 做种子),经 N-1 跳到达 S 的前驱时 N 份 partial 已齐——前驱就地做 finisher(W_o+MLP+LayerNorm→h_{L+1})并以发起者身份发出层 L+1 的包;末层 finisher 算 logits、采样、embed 新 token,直接发起下一步第一圈。decode 全程无中心驱动,coordinator 只做准入/释放。

【五个目标】
1. 无 owner:starter/finisher/sampler 全部逐层逐 token 轮转,任何节点可在任意层接替;
2. 无冗余:单 token 单 forward,non-attention 计算(MLP/投影/logits/采样)随包逐层轮转分布——同时消除 Rust 线 N× 冗余和 plugin 线 owner 集中;
3. 显存切分保持:每节点持久持有=本 chunk+growth/N;当前 token 在种子侧瞬时;
4. 拓扑不变:每节点恰 2 连接,线性成本,部分可达(N>3 非全连通)网络可用;
5. 跳数最优:每层 N-1 跳(非 N 跳,无需回程),每 token 24×(N-1)——层间依赖决定这是 ring decode 的下限,单 token 压不成一圈。

【关键点】
1. 增长分片不变:包经过全节点,携带 h_L(或当前 token KV),assignee(p%N)自算自留,零额外传输;当前 token 永远只在种子侧瞬时参与本步(causal 尾);
2. 正确性不变量:每位置 KV 恰好一个节点持有;online softmax 归并可交换可结合,归并顺序无关;
3. 与 attention 的 online softmax 环上归并是同一数学——forward 整体骑环是 attention 环的自然延伸(用户原话:两者契合);
4. 与 Rust 线现状的差异:现状每节点发起自己的包(N 包×(N-1) 跳,N× 冗余);自驱动环单包 (N-1) 跳,零冗余;
5. 与 plugin 线现状的差异:现状 owner 种子+N 跳回程;改 successor-seeded 即同样 N-1 跳(见下)。

【vLLM plugin 线:不抛弃,受限形态最优】
全自驱动需在层间接管 forward(MLP/norm/logits 在 vLLM model 代码里,attention backend 与 KV connector 两个扩展点都不够;自写 model runner=fork,违背插件跟随上游原则)。改为两步:
(a) successor-seeded 优化:包从 owner 后继出发、owner 最后归并,同样达到每层 N-1 跳(省 1/3 跳,今天即可做);
(b) driver 角色分散:请求级 owner 轮转(ringc 已验证的轮转放置)使 driver 负载在多请求间均摊。
边界声明:plugin 线 driver 钉死是 vLLM 嵌入架构税,非语义缺陷;若未来 vLLM 提供层间扩展点,再评估全自驱动。

_updated: 2026-07-28 16:36:22_
### HCP 两阶段统一 ring 架构:prefill 传 KV、decode 传 Q+LSE,全程 P2P 逐跳,每节点只需 2 个连接

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `user-direction`

用户阐述的框架设计与成因链(2026-07-27,落实自其口述;decode 部分由 Decode_Ring_P2P_Transport_task-d39.md 补充):

【成因 1:P2P 网络约束决定拓扑】异构节点间无 collective 原语可用(无 NCCL/RCCL/NVLink),P2P send/recv 是唯一通信模型。ring 拓扑下每个 worker 节点只需连接 2 个 peer(predecessor + successor)即满足全部数据面需求(不含 controller/scheduler 控制面)。

【成因 2:prefill 学 Ring Attention 原论文】prefill 是 compute-bound:ring 逐跳传 KV block + 本地 Q 计算 + online softmax 累积,通信被计算隐藏,且天然切分显存压力。HCP 在论文均分假设上扩展了 capacity-aware 不均等切分(按设备显存/算力分配 chunk)。

【成因 3:显存可控性要求 decode 也必须切分】若 decode 退化为"全量 KV 复制到单 worker"(vLLM P/D 式)或 TP decode(按头分片但需同构互联+collective),显存切分在 decode 阶段失效,显存压力不可计算,capacity-aware 不成立。两条业界主流路线都被排除。

【成因 4:decode 学 LoongServe——传 Q+LSE 不传 KV】decode 是 memory-bound:每步只有 1 个新 Q,计算极小但必须遍历全部历史 KV;继续传 KV 则每 hop 每层 O(seq_len×d)(128K 下 ~64MB),通信完全主导。改传 Q + 累积器 (O, LSE) 做 online-softmax 归并,每 hop 每层仅 O(d)(~4KB,128K 下差 ~1000x),同时 Q 仍与全量 KV 分布式计算,数学精确。

【成因 5:LoongServe 依赖 collective,HCP 改 P2P 绕环与 prefill 统一】LoongServe 是同构集群,用 broadcast Q + reduce 结果(collective 原语);HCP 把同一数学改为沿 ring 逐跳传递累积器——与 prefill 的 KV ring 共享同一网络架构(每节点仍只有 2 个连接),simple and effective。

落地状态:dsplit4(前缀切分)+dsplit6(增长分片)在星形 HTTP 上验证了数学语义;ce70afc(+41cdcd1/8696639 修复)把 decode 传输改为真 P2P TCP 环(task-d39 计划),prefill HTTP KV store 保留不变。

[2026-07-27 验证闭环] p2p3:真 P2P TCP 环 decode 全项 PASS(ev-decode-p2p-ring-p2p3-20260727),Reviewer APPROVE。架构端到端成立。
[2026-07-27 三机闭环] p2p3n-175719:laptop(4060 CUDA)+white(4090 CUDA)+pearl(9060XT ROCm) 三机真异构 P2P decode Q-ring PASS(ev-decode-p2p-3node-p2p3n-20260727),Reviewer APPROVE。

_updated: 2026-07-27 10:19:39_
### 架构决策：采用原始论文 P2P 而非 PyTorch CP Collective

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `memory-bank/systemPatterns.md`

Ring Attention 原始论文（Liu et al. 2023）的通信本就是 P2P send/recv。PyTorch 2.7+ Context Parallel 改用 all-gather/all-to-all 是对同构 NVLink 集群的工程优化，不是数学必须。P2P 支持异构、非均分、任意拓扑，更符合 HCP 定位。

_updated: 2026-06-29 05:34:19_
### 容量感知非均等 CP 分片是异构长 context 的必需

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `memory-bank/systemPatterns.md`

2026-06-19 1M context 验证：24GB CUDA + 16GB HIP 无法通过 1:1 分片完成 1M。必须使用 capacity-aware 不均等分片（white 750K / pearl 250K，即 3:1）。均匀分片在异构显存下会因小显存设备 OOM 而失败；按可用显存比例分配 chunk 才能使 heterogeneous ring 达到可行性边界。

_updated: 2026-06-29 05:34:19_
### 产品决策：P2P、correctness 优先、结构化实验产物

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `memory-bank/productContext.md`

HCP 不是 HLPP 的细粒度版本，而是 intra-layer / low-boundary 路线。跨异构域坚持 P2P，不把 all-gather / reduce-scatter / all-to-all / all-reduce 作为主假设。correctness 和协议闭环优先于性能图。每个阶段输出结构化实验产物。

_updated: 2026-06-29 05:34:19_
### 部署铁律：1 GPU = 1 worker

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/techContext.md`

每个 worker 加载完整模型权重。3B bf16 × 2 workers 在 RTX 4090 loopback 上实测 OOM。--local-domain-ids 仅限 <1GB 小模型的本地协议验证；生产/大规模验证必须每平台一 worker。

_updated: 2026-06-29 05:34:19_
### Correctness-First 开发纪律

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `memory-bank/systemPatterns.md`

当前处于 correctness 验证阶段，尚未进入性能调优。在全部 target 设备上稳定通过前，禁止实施量化、近似 attention、非 deterministic kernel、投机/跳过层优化。每次提出优化前必须写 trade-off 分析。

_updated: 2026-06-29 05:34:19_
### Striped Attention HCP 适配计划

type: `decision` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `docs/STRIPE_ATTENTION_ADAPTATION_PLAN.md`

已将详细实现计划写入 docs/STRIPE_ATTENTION_ADAPTATION_PLAN.md。
核心思路：通过细粒度 scheduling unit 实现 capacity-aware 不均等 stripe；用原始位置 id 计算 causal mask；worker 输入/输出做 permutation / inverse-permutation；online softmax 与 KV transport 不变。
实施顺序：先在 correctness model 验证，再改 coordinator/worker，最后跑 uneven 分布式 smoke。

_updated: 2026-06-29 06:18:40_
### HCP 设计原则：简洁性 / Occam's Razor

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `~/.agents/AGENTS.md`

全局 AGENTS.md 已加入简洁性原则：如果更复杂的设计没有可验证的明显收益，就选择更简单的方案。
对 HCP 当前工作的影响：
- Striped Attention 的引入必须通过与 capacity-aware 连续分片的实际对比来证明其价值。
- 如果 Striped 在 wall-time、代码复杂度、decode 复杂度、kernel 兼容性上没有明显优势，  则保留更简单的 capacity-aware 连续分片。
- 决策必须基于同一测试配置下的 HCP_PERF_LOG 数据，而不是论文理论 speedup。
- 最终结论需记录到 graph-memory 和 commit message。

_updated: 2026-06-29 07:58:41_
### 默认保留 capacity-aware 连续分片，Striped 暂不启用

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.85

基于 CPU mock、white CUDA、pearl HIP 三重证据：在 2-domain 3:1 seq_len=4096 场景下，Striped 均未改善负载均衡，反而增加瓶颈 domain 0 的 wall-time。\n\n决策：\n1. HCP 默认调度策略继续采用 capacity-aware 连续分片。\n2. Striped Attention 代码保留在仓库中（作为可选项和对比基准），但不作为默认路径。\n3. 若未来在真实 multi-node 大 context（如 1M）实验中出现新的反证，可重新评估。

_updated: 2026-06-29 12:44:16_
### QUIC Transport 配置：512MB stream window / 1GB connection window / 300s idle timeout

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.8 · source: `memory-bank/techContext.md`

显式覆盖 quinn 默认值：max_concurrent_bidi/uni_streams=256, keep_alive_interval=1s, max_idle_timeout=300s, stream_receive_window=512MB, receive_window=1GB, send_window=1GB。历史上因 send_window 和 stream_receive_window 不足导致 16K/64K 死锁。

_updated: 2026-06-29 05:34:19_
### 可插拔域内后端架构

type: `decision` · status: `held` · confidence: 0.8 · importance: 0.8 · source: `memory-bank/systemPatterns.md`

HCP 的边界是跨域低层协议（P2P KV ring + online softmax），域内实现是黑盒。同构域内可通过接口实现替换为 vLLM、TensorRT-LLM、MLX 等社区框架。Python Worker SDK 和 Rust Worker SDK 提供标准接口。

_updated: 2026-06-29 05:34:19_
