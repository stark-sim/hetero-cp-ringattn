# Product Context

产品问题、目标用户、成功标准与产品决策。

### 固定 owner-return 的单向 ring decode 至少需要 N 条网络边

type: `belief` · status: `held` · confidence: 0.98 · importance: 0.95 · source: `docs/plans/2026-07-29-self-driving-ring-decode-revision.md`

前提:Q 只在 owner 产生;仅允许 successor 链路;全局 attention 结果必须回到同一个 owner forward 栈。在这些前提下,访问其余 N-1 节点并回到 owner 的唯一环路含 N 条边。若允许结果停在 finisher,下限才降为 N-1。

_updated: 2026-07-28 17:11:06_
### 异构分布式推理的数值验证策略

type: `belief` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `memory-bank/systemPatterns.md`

BF16 场景下，跨平台 BLAS 差异导致 logits 数值对比不是有意义的 correctness 指标。Correctness 应分层：L1 float32 数学正确性（cargo test synthetic weights）、L2 工程正确性（argmax 一致性/文本任务指标）、L3 端到端冒烟。强证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异，证明差异主要来自 BF16 online softmax block-wise 处理顺序，而非跨平台 bug。

_updated: 2026-06-29 05:34:19_
### P2P 约束下 ring 是数据面最小充分拓扑:每 worker 只需 2 个 peer 连接

type: `belief` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `user-direction + validated runs`

不考虑 controller/scheduler 控制面时,HCP 数据面(prefill KV ring + decode Q/LSE 累积器环)的全部通信都是与环上相邻两节点的 send/recv;无需 all-to-all/broadcast/reduce/gather。这使异构消费级硬件(以太网/tailscale/未来 CXL)即可组成 CP 系统,是"异构是常态"核心信念在拓扑层的直接推论。证据:三机真环 ringc-160010、decode dsplit4/dsplit6 均只使用邻接连接。

_updated: 2026-07-27 09:10:04_
### 旧决策(已修订):自驱动环 decode 无 owner/无冗余/N-1 hops

type: `decision` · status: `revised` · confidence: 0.9 · importance: 1.0 · source: `user-direction`

[2026-07-29 状态] 本节点的自驱动 Rust 核心方向保留,但 plugin successor-seeded、sampler 必然逐 token 轮转等子结论已被新理论裁定修订。当前规范见 decision-self-driving-ring-theory-20260729。
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

【动机剖析六问(规范版,2026-07-28 计划冻结前)】
1. 问题:decode 期三类非对等/浪费并存——plugin 线 owner 钉死(driver+全 forward+采样集中)、Rust 线 N× 冗余 forward(每节点全量重算,仅 worker 0 被采用)、每层 N 跳(含一次纯回程)。
2. 现状:任务C 的 Q-ring 已验证(传 Q+LSE 不传 KV),但 Rust 线每节点发起 N 包且全节点冗余,plugin 线 owner 种子+N 跳回程;增长分片、显存切分、ring-only 拓扑已闭环。
3. 目标态:单包轮转;角色(starter/finisher/sampler)逐层逐 token 轮转;每层 N-1 跳;零冗余(单 token 单 forward);token 与单节点参考一致;hop 计数、轮转计数、零冗余证明可机器验证。
4. 别人怎么做:LoongServe 弹性 DoP+KV 迁移(已否决:需全连通+破坏切分);TP 按头分片(需 NVLink/collective);流水线并行(切层不切 KV,不解决 KV 显存);Ring Attention 原论文不管 decode serving。无现成轮子——自驱动环=attention KV 环归并(既有)×层间流水轮转(新)的复合。
5. 我们怎么做:包不回家(第 N-1 跳 N 份 partial 已齐,前驱就地 finisher 续层);末层 finisher 采样+embed 续发;增长分片照旧(assignee 自算自留)。关键架构领悟:decode 期 worker 无 forward 调用栈,是纯事件循环(收包→partial→完整则 MLP 续发/否则转发),比现状更简单而非更复杂。
6. 为什么:P2P-only+部分可达+显存切分三重约束下,这是唯一同时满足无 owner/无冗余/拓扑线性的形态;且与 attention online softmax 环归并同一数学(用户判定:两者契合)。

【牺牲四问】
1. 默认为什么存在:Rust 冗余=复用 prefill 环控制流最省事+全节点状态一致随便挑节点采样;N 跳回程=发起者收回结果的直觉写法;plugin owner=vLLM 单引擎语义最自然。
2. 牺牲什么:(a) 放弃"每节点每步都持全局结果"——任一时刻只有 finisher 链上的节点持有中间态;(b) 放弃 N 包并发——单包串行,带宽利用降为一路;(c) plugin successor-seeded 把 owner 本地计算从"与传输重叠"变为"发包后与绕环重叠"(严格分析:现=本地计算+N跳;新=发包+本地计算‖(N-2)跳+收包,仍省 1 跳,本地重叠性不损失)。
3. 被牺牲者用途:冗余提供简单性与状态一致性;并发包在带宽饱和时利用多链路。
4. 对本项目意义:包为 KB 级,链路远未饱和,RTT 主导,(b) 无实际损失;(a) 状态一致性改由包不变量保证(每位置 KV 恰一节点持有);宕机语义不变(环断即死,与现状同,PoC 接受)。结论:implement。

【实施排序(DEPENDS_ON 见边)】
E(plugin successor-seeded,小步先行)→ D1(Rust 单包轮转 attention,ring.rs)→ D2(finisher 就地续层,model.rs 事件循环化,最大工程风险点)→ D3(采样轮转+coordinator 退位)→ D4(验证阶梯:mock→MPS 双节点→跨节点 CUDA+HIP,判据=token 一致+轮转计数+零冗余证明+hop 计数)。

【风险登记】
risk-1(中):D2 把 model.rs 从一次调用栈改为事件驱动,侵入面大——缓解:decode 期与 prefill 期代码路径分离,prefill 不动;
risk-2(低):单包在大 head_dim 模型下带宽饱和场景损失并发——KB 级包下不构成;
risk-3(低):首个 finisher 的等待编排冷启动(prefill 后第一包从 owner 发出)——从既有 decode-ring 注册状态推导。
[2026-07-29 可行性修订] 总体 Rust 自驱动方向保持;plugin successor-seeded N-1 子结论被 topology proof 推翻,task E rejected。Rust D1 不能作为独立生产 checkpoint,必须与最小 model continuation 合并。见 revision-self-driving-ring-plan-20260729。
[2026-07-29 理论深化] 在抽象掉既有 plugin/owner 实现后,自驱动 ring 与 HCP 总设计适配:每层 N-1 attention hops、KV 互斥完备分区、单逻辑 forward、P2P-only 线性拓扑。修正:若 L mod N=0,sampler 不会自然跨 token 轮转;需 +1 token-ID phase-shift hop。详见 decision-self-driving-ring-theory-20260729。

_updated: 2026-07-28 17:41:07_
### 理论裁定:自驱动 ring 适合 HCP decode,严格 KV 分区优先于 sampler 形式轮转

type: `decision` · status: `superseded` · confidence: 0.95 · importance: 1.0 · source: `docs/plans/2026-07-29-self-driving-ring-theory.md`

【结论】适合。它在不依赖 collective/全连通网络的条件下同时实现:每节点仅 predecessor+successor 两条 peer 关系;每层 N-1 attention packet 边;durable KV 按位置互斥完备分区;单个逻辑 distributed forward;非 attention continuation 唯一执行。
【核心不变量】对任意 request/layer,各节点 durable KV 位置集合两两不交且并集为全部历史位置;round-robin 下每节点位置数属于 floor(T/N) 或 ceil(T/N),有限 token 下不可能严格等于 T/N,最大只差 1 token。环上 packet 为 O(model_width),不随 context 增长。所有 N 节点各算一次本地 attention partial;Q 只由 starter 算一次;当前 token K/V 只由 assignee 算/存/参与一次;W_o+residual+MLP 只由 finisher 算一次;最终 logits/sample 只算一次。
【packet 状态机】starter 持 h_L 并只算 norm+Q。若 starter!=kv_assignee,seed 只含本地 durable history partial;若 starter==kv_assignee,starter 唯一计算 current K/V,直接形成 history+current 的单一 seed,partial 后 durable append,首跳前置 kv_committed=true,不得再次计算 history。普通 relay 只合并 durable history;assignee relay(含 finisher 重合)唯一计算 current K/V,形成 history+current 的单一 partial,随后 append 并置 commit。第 N-1 跳抵达 finisher时 N 份 partial 齐全;finisher 完成本地 assignee 分支后必须断言 kv_committed,再做 W_o/MLP 得 h_(L+1)。R0 必测 assignee==starter、middle relay、finisher 三种重合。online-softmax merge 可交换结合,所以 decode 单 query 不要求 KV token 顺序与物理 ring 顺序一致。
【动机六问】1问题:在异构、P2P-only、每节点仅两个 peer、KV 严格切分下消除 owner 与全 forward 冗余。2现状:owner-return 多 1 跳且集中;全节点 full forward 有 N 倍非 attention 冗余。3目标:N-1 attention hops/layer,ceil(T/N) KV 上界,单 continuation/单 logits producer,机器可验证 exact-once。4他者机制:TP/collective 要同构高速互联;pipeline parallel 切层不切 KV;经典 Ring Attention 未提供 serving decode 的层间 ownership 状态机,无法直接复用。5本方案:Q+O+LSE+h 小包环行,finisher 续层,KV assignee exact-once,control plane 仅准入释放。6为什么:它是同时满足部分可达拓扑、KV 容量聚合、无 collective 与线性网络成本的最小形态。
【牺牲四问】默认冗余/owner-return提供简单同步、任意节点可取全局结果和多包并发;本方案放弃全节点结果复制、单请求多包并发与节点故障容忍。它们在通用系统中服务简单性、吞吐和容灾;对当前 inference PoC,严格 KV 容量和拓扑约束优先,环断即请求失败可接受。结论:implement。
【token 边界策略】角色并非必然跨 token 轮转。零 handoff 时 s(t+1,0)=s(t,0)-L mod N;L mod N=0 时 sampler 对单请求固定,但不增加 durable KV。默认用 request 初始 phase 做请求间均衡,保留 L*(N-1) attention hops;异构时 sampler 可偏向更快节点。若实测出现 LM-head/sampling queue 瓶颈,可选 k-hop token phase shift,跨 token 位移为 k-L mod N;遍历所有节点需 gcd(k-L,N)=1。目标 L=24,N=3 时 k=1 有效,但不是通用固定规则。
【异构边界】严格 1/N KV 是 memory correctness policy,但总容量受最小显存节点约束且单 token 延迟包含所有节点。capacity/speed weighted placement 可提高利用率,却必然让部分节点超过 1/N,必须作为显式替代策略,不能暗中弱化默认保证。多请求 packet pipeline 可提高吞吐,但需 bounded queue/backpressure/request ordering,不得复制 durable KV。
[2026-07-29 用户裁定] 不为形式上的无 owner 强制 sampler 逐 token 轮转。固定 sampler 不增加 durable KV;默认用 request phase 做请求间均衡并保留 L*(N-1) attention hops。+1 token-ID phase shift 降为实测 sampler 计算/队列瓶颈后的可选策略。

_updated: 2026-07-29 05:05:41_
### 理论裁定 v2:自驱动 decode ring 采用显存硬上界内 compute-balanced KV placement

type: `decision` · status: `held` · confidence: 0.98 · importance: 1.0 · source: `docs/plans/2026-07-29-self-driving-ring-decode-review.md`

【结论】自驱动 ring 适合 HCP decode。数据面保持单 packet、每层 N-1 hops、每 worker 仅 predecessor/successor、无 collective；durable KV 互斥完备无复制，Q/current K/V/W_o+MLP/LM-head 各有唯一执行者。1/N 只表示无 owner-collapse、无 context-sized 远端临时 KV 和显存压力可计算，不是异构设备的 equal-placement 合同。
【placement 裁定】admission 先扣除模型/runtime/safety/active-request reservation，得到每节点 free KV bytes；按该节点实际 KV dtype/layout 的 per-layer bytes 计算请求份额上界 u_i。若 sum(u_i)<1 拒绝。可行域内解 min max_i(x_i/attention_rate_i)，约束 sum(x)=1 且 x_i<=u_i；解为 x_i=min(u_i,lambda*s_i)。容量富余时降低慢节点份额；请求逼近聚合容量墙时 sum(u)->1，唯一可行解 x->u，退化为纯 capacity。任一参与节点 throughput 缺失时保守回退 x_i=u_i/sum(u) 的 capacity-only 目标，不混用实测 rate 和猜测值。
【冻结与 memory ledger】versioned worker profile 显式提供 per-layer/position 的 B_i^K 与 B_i^V、allocator granularity G_i、非零份额整笔计费的 request overhead H_i，以及 W_i(0)=0 的本地 shard attention workspace bound。C_i 取模型加载后可靠 device-free telemetry 与显式 KV budget 的较小值，再扣 static/packet reserve；不能从 coarse capacity_mb 直接换算。K/V 两个物理 slab 分别按 G_i round。单线程 executor 下 ledger 约束=sum(active persistent slab+H)+max(active W)<=C_i；u_i 由该单调 hard bound 求出，再做 bounded water-filling。prompt contiguous chunk 与 decode calendar tickets 都由同一 bounded target x_i 量化；整数计划逐节点复核。self-driving v1 使用预分配 reserved KV slab + narrow view，禁止现有 Tensor::cat 全 shard 临时副本。kv_phase 与 starter/sampler phase 分离；已 admission plan 冻结。
【协议协商】只有 coordinator 选择集群 mode；worker 加载模型后先只连 control plane，hello 上报 version/capability/profile。全员 ack 同一 mode/version/ring_epoch 后才建立 predecessor/successor data-plane streams；收齐 DataPlaneReady 后才可 admission/prefill，mixed-mode 提前拒绝。
【多请求与 sampler】每个 request 是独立异步 packet 流，不要求同 token index、等速或 lockstep；stable request_id 分散 starter/sampler 初始 phase。L mod N=0 时允许单请求 sampler 固定，因为它不新增 durable KV；logits 必须在同一 compute quantum 内采样并释放，不进入 packet/ready queue/backlog。只有 LM-head/sample queue 成为实测瓶颈才启用 token handoff。
【六问】1问题:严格 equal 1/N 浪费异构大显存，纯 capacity 又可能让大显存慢设备成为每层瓶颈。2现状:旧理论节点把 strict 1/N 当默认，代码仍用 capacity_mb 比例且无 active-request reservation/throughput 维度。3目标:显存绝不越界，在可行域内最小化 attention 串行关键路径，并用精确 reservation、ownership、hop/exact-once 和硬件 trace 验证。4他者:TP/collective 依赖同构互联；vLLM paged scheduler 做 per-request 容量管理但不提供 HCP 序列维跨异构 ring placement；可复用 reservation/continuous scheduling 思想，不能直接复用数据面。5本方案:bounded water-filling + 冻结二维 weighted calendar + packet pipeline。6为什么:同时保留聚合 KV 容量、两 peer 部分可达拓扑和异构算力利用，是 HCP 约束下的最小可计算策略。
【牺牲四问】默认 equal 便于证明和均匀计数，纯 capacity 最大化可装 context；新策略牺牲 equal 的简单性，并在有容量余量时暂不使用慢节点的全部可存空间。equal 在同构系统提供简单确定性，纯 capacity 在极限 context 提供最大可行性；对 HCP，两者分别作为同构特例和容量墙退化路径保留。结论:implement。

_updated: 2026-07-29 06:26:48_
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
### 成功标准：online softmax 对齐 + RingAttnMessage 可传输 + remote heterogeneous smoke

type: `fact` · status: `held` · confidence: 0.8 · importance: 0.8 · source: `memory-bank/productContext.md`

online softmax 在不均分 seq_chunk_len / block_size 下与 reference attention 对齐。RingAttnMessage 可以稳定编码、传输、解码。2-domain remote heterogeneous smoke 可复现，并产出 correctness、transport、failure summary。

_updated: 2026-06-29 05:34:19_
