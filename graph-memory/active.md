# Active Context

当前活跃的任务、决策、风险和假设。

### 将 schedule 显存保证限定为完整 horizon reservation

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：旧测试在一个 [1,3,2]、24 units、单 phase 样例上检查前缀比例误差小于等于 1，容易继续暗示该结论对任意 phase 成立；反例已证明这不是普遍定理。2.现状：largest remainder 产生完整 horizon 精确 counts，phase 只循环旋转同一 sequence；exact slab 已证明已知 horizon 可以按 layer×domain 精确预留并原地 append，但 schedule 测试尚未把 counts 明确验证为每域 reservation 上界。3.目标：对多组 tickets/horizon 和所有 phase，任意 prefix 的消费计数都不超过 counts，完整 horizon 后精确等于 counts；现有确定性、容量份额、唯一 assignee和零容量语义不变。4.他者：vLLM 等 serving engine 依赖 admission reservation、block quota 或预分配 arena 保证显存，调度顺序负责平滑吞吐而不是充当显存硬界。5.本方案：不改算法或 API，只用纯单元测试把 counts 解释并验证为完整 horizon reservation，删除旧样例中的 scaled prefix-error 断言。6.为什么：这是把已修订数学结论落实到代码合同的最小方案；无需发明 cyclic discrepancy 算法，也不引入生产 allocator。【牺牲四问】旧前缀检查的目的，是约束单请求短期 event 分布和平滑计算；本节点放弃把小于等于 1 当作普遍保证，但不删除 phase 轮转或 smooth sequence；短期平滑本质上服务并发负载均衡，而不是物理显存安全；本项目现阶段优先保证可证明的 capacity hard bound，多请求效果以后单独实验。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-08-01 17:31:51_
### 用冻结计划精确 slab 验证无 Tensor::cat KV append

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：Tensor::cat 每次 append 都分配并复制完整本地历史，语义复用虽已证明，但峰值会暂时同时持有旧 shard 和新 shard。2.现状：test-only PositionedKvShard 的 prefill append 直接 cat；decode adapter 也借用现有 cat-based runner。3.目标：分布式 prefill/decode 都以精确 reservation + cursor 原地写；每层每域 capacity 与最终 usage 相等；overflow 写入前拒绝；原 24 层四阶段数值、position union、144 个 continuation 投影和 [56,168,112] 总量全部保持。4.他者：vLLM 等用 paged KV/block table 或 reserved arena 将逻辑增长与物理分配分离。5.本方案：test-only ReservedPositionedKvShard，容量由固定四阶段计划精确推导，append 用 narrow().copy_()，attention 只看 committed prefix；decode 使用同样的 test-only positioned runner，不经过 cat-based production cache。6.为什么：固定 14-position horizon 允许最小而严格的预留证明，不需要引入生产 page allocator、admission 或 runtime。【牺牲四问】默认 cat 为未知长度提供简单动态增长与拥有型连续 tensor；精确 slab 牺牲超出冻结 horizon 的增长和运行期重分配；这些能力服务开放式生成、动态 batch 与 allocator 调度；当前固定 correctness 实验不需要，但因此结果不能外推为生产 allocator。备选：各域统一最大 slab 会浪费小节点容量，拒绝；paged allocator 当前过宽，延后。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-08-01 08:02:53_
### 实施 24 层 positioned KV 四阶段复用实验

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：capacity-weighted CP prefill KV 与按 token×layer event 分配的 decode growth tensor 形状相容，但当前持久 cache 不保存 slot 的全局 position，也没有证明混合历史能被 continuation prefill 正确复用；若这一点不成立，后续 schedule 与物理 append 优化没有意义。2.现状：decode 单 token 对历史顺序不敏感，现有两 token实验只需 K/V tensor；KvBlock 与 Ring Attention 已支持显式 position 和按 q_pos>=k_pos causal，但 self_driving local history 只有 K/V，模型生命周期也未覆盖 prefill-decode-prefill。3.目标：N=3、L=24、tickets=[1,3,2]；prefill_1=6 tokens、decode_1=1、prefill_2=6、decode_2=1；24 个真实 DecoderLayer；第二次 prefill 只投影六个新位置并读取已有 distributed positioned KV；hidden/logits/argmax 对齐完整有序参考；每层 position union 完整唯一。4.他者：PagedAttention/block-table 系统用逻辑位置到物理 slot 的映射复用历史 KV；Ring/Striped Attention 用显式 q/k positions 保持任意 shard layout 的 causal correctness。可复用其 position-aware 数据合同，但本节点不引入生产 block allocator。5.本方案：在实验边界加入 PositionedKvShard 和多 query positioned online-softmax 原语；先用 Tensor::cat 形成语义证据；初始与 continuation prefill 每层按 [1,3,2] token split 持久化，新 decode KV 沿现有 self-driving path 唯一落点。6.为什么：24 层覆盖真实逐层 hidden 依赖，6-token prefill 与 24-layer decode 都能无舍入地表达 [1,3,2]；它是验证能否 append 的最小完整里程碑，同时把怎样无副本 append 延后。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-08-01 05:54:01_
### 术语约束：decode KV 按 token×layer event 分配，不称为 pipeline parallel

type: `preference` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-01`

用户明确要求后续不再用 pipeline parallel 描述 decode KV 分配。规范术语为 layer-striped KV growth、decode KV event assignment 或按 token×layer 事件分配；pipeline parallel 仅保留给固定模型 stage 与 activation stage handoff 的标准含义。

_updated: 2026-08-01 05:54:01_
### 分层裁定：单请求核心闭环，系统闭环仍待后续小节点

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `code-audit-2026-08-01`

【动机六问】1.问题：连续实验已走到两 token TCP，但需要确认它证明的是完整核心数据流，还是被同步测试结构掩盖了缺口。2.现状：LayerPacket 显式携带 residual、normalized、position、Q、O/LSE；assignee 唯一追加 current K/V；finisher 唯一执行 W_o、residual、post norm、MLP；末层本地 logits、greedy sampling、embedding 后继续下一 token。真实 TCP 证据仅为单请求 N=3、L=2、两 token。3.目标：分别判断单请求数学/数据流、HCP 异构显存目标和可运行系统三层是否闭环，并用代码位置、反例和新鲜测试支撑。4.他者：Ring Attention 显式传递 Q 与 online-softmax accumulator，pipeline parallel 显式传 activation；vLLM 类运行时另外用 request-keyed KV/page state、调度队列和 admission 处理多请求与显存硬界。Ring Attention 本身不提供这些生命周期能力。5.本方案：保留当前最小实验设计；把已证明的 exact-once、N-1 hops、context-independent packet 与未证明的物理峰值、多请求 demux、runtime 集成严格分层，不把生产能力前置。6.为什么：这既回答核心方案是否自洽，又遵守核心优先、小步验证约束；现阶段没有证据要求重写数据面，也没有证据允许宣称系统完成。VERDICT: DEFER 后续实现；接受单请求核心设计，下一节点须另行确认。

_updated: 2026-07-31 19:13:51_
### 当前唯一实施任务：自驱动 decode ring 最小核心切片

type: `task` · status: `ongoing` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-07-30`

范围只包含：1. 简单、冻结且可复现的 capacity-weighted KV assignee schedule（按 token×layer append event），并用 stable request_id 分散初始 phase；第一版不引入 layer 维二维 calendar 或运行期动态迁移。2. 单请求、最小两-token 的真实 tensor decode 路径：starter 生成 Q，packet 用 N-1 hops 合并所有本地 KV partial，唯一 assignee 计算并持久保存 current K/V，finisher 唯一执行 W_o + residual + norm + MLP，末层唯一产生 logits。3. 最小机器验证：与单节点参考一致；hop、Q、KV project/append、partial、MLP、logits exact-once；无完整远端历史 KV 临时副本。每完成一个小节点即汇报结果、下一节点和方向判断，未经用户确认不跨入生产化能力。当前工作树中的大型 placement/ledger 草稿不自动视为本任务成果，需先单独审查取舍。

[2026-07-30 checkpoint 1] 任意 N 的单层真实 tensor ring 已验证 attention + residual/norm/MLP 数学与 exact-once 角色。

[2026-07-30 checkpoint 2] 显式 LayerPacket 已验证：runner 只编排 packet+local shard step；历史长度 2 与 47 的首跳 payload 元素数相同。

[2026-07-30 checkpoint 3] 固定两层 N=3 handoff 已验证：角色 1->0->2，总 hops=4；每层 Q/KV/partial/finisher exact-once，输出与两层单节点参考一致。

[2026-07-30 checkpoint 4] 末层 finisher 已唯一执行 final RMSNorm + 独立或 tied LM head；logits projection=1，N=3 两层总 hops 仍为 4，完整 Rust 测试 85/85 通过。

[2026-07-30 checkpoint 5] 任意 L 的单 token 全模型 runner 已验证：N=3 下 L=3 回到 starter、L=4 轮转 producer，满足 producer=(starter-L) mod N、总 hops=L*(N-1)、逐层 exact-once，完整 Rust 测试 87/87 通过。

[2026-07-31 checkpoint 6] N=3 localhost 单层真实 TCP self-driving ring 已验证：独立 packet 经 0->1->2 两个 hop，三个 worker 各处理一次 local partial，只有 capacity map 指定的 assignee shard 增长，finisher 唯一完成 W_o+residual+norm+MLP；输出对齐参考，实际 wire bytes 对历史长度 2/47 恒定，完整 Rust 测试 90/90 通过。实现 71c8698 已推送。

下一候选仅为任意 N localhost 网络证据：用 N=2/3/4 与非零 starter 覆盖 successor wrap-around；未经用户确认不实施。sampling/token continuation、任意 L 网络循环、QUIC、远端硬件与 runtime 仍未开始。

[2026-07-31 checkpoint 7] localhost 单层 TCP ring 的任意 N 与闭环边已验证：N=2/3/4 使用 starter=N-1，实际 route 为 1->0、2->0->1、3->0->1->2，均经过 wrap-around；每例 N-1 sends、local partial/assignee KV/finisher exact-once 与参考输出断言通过。没有新增生产路由代码，仅参数化试验。完整 Rust 测试 91/91 通过，实现 2150d7a 已推送。

下一候选为 N=3 固定两层 localhost TCP handoff：让 layer 0 finisher 原地成为 layer 1 starter，验证跨层无需 coordinator return；未经用户确认不实施。

[2026-07-31 checkpoint 8] N=3 固定两层 localhost TCP handoff 已验证：layer 0 route=1->2->0，domain 0 finisher 不经 coordinator 回传，直接用本地输出 hidden 启动 layer 1 route=0->1->2；两层总 sends=4，每层 partial/assignee KV/finisher exact-once，最终 hidden 对齐两层参考。完整 Rust 测试 92/92 通过，实现 c5751f1 已推送。

下一候选尚未实施：先对最小后续证据节点做动机剖析，仍不进入生产化。

[候选 checkpoint 9] 末层 TCP finisher 本地唯一产生 final logits：保持 N=3 两层 localhost，证明 final head 不增加 hop；待用户确认，不进入 sampling、token continuation、任意 L 网络化或 placement planner。

[2026-07-31 checkpoint 9] 末层 TCP finisher 本地唯一 final logits 已验证：N=3 两层 route 仍为 1->2->0 与 0->1->2；仅 domain 2 本地执行 final RMSNorm+LM head，logits producer=1、数值对齐参考，总 sends 保持 4。完整 Rust 测试 92/92 通过，实现 6ef5a18 已推送。

下一候选尚未实施：重新评估核心剩余项，优先保持实验性与小步。

[候选 checkpoint 10] 一维冻结 capacity-weighted KV owner map：capacity tickets + stable request_id phase，纯函数/纯数据结构；明确不采用当前 1079 行 production placement/ledger 草稿，不含 layer calendar、throughput、动态迁移或 admission。待用户确认。

[2026-07-31 checkpoint 10 revision] 取消 owner 命名及 token-wide owner 粒度，改为按 append ordinal=(token_offset*num_layers)+layer_idx 的 FrozenKvAssigneeSchedule；待实现。

[2026-07-31 checkpoint 10] 冻结 capacity-weighted KV assignee schedule 已验证并推送（cfe25d9）：分配粒度为 (token_offset, layer_idx) append event，不是固定 owner；[1,3,2] 在 24 units 上精确为 [4,12,8]，smooth 序列前缀偏差不超过 1 unit，request_id 只分散 phase，零容量节点无分配，N=1/2/4 通用。完整 Rust 测试 94/94 通过。当前仍未接入 TCP runner。

[2026-08-01 checkpoint 11] 冻结 KV assignee schedule 已接入两层 localhost TCP 实验并推送（271de7f）：[1,3,2] tickets、request phase=1 在两层 horizon 生成 [2,1]，每层仅指定 domain KV +1；两层 route、4 sends、finisher handoff、唯一 final logits 与参考数值不变。完整 Rust 测试 94/94 通过。边界仍为单 token 实验，尚未进入 sampling/下一 token continuation。

[2026-08-01 checkpoint 12] 两 token localhost TCP 自驱动 continuation 已验证并推送（b237266）：token 0 末层 finisher domain 2 原地 greedy sampling+embedding，零边界消息启动 token 1；sampler/finisher 轮转到 domain 0。四个 append assignee=[[2,1],[1,0]]，每项 exact-once；position 连续，参考侧累积 token 0 K/V 后两步 hidden/logits/token 均对齐；总 sends=8=2*2*(N-1)。边界仍为固定 L=2、greedy、单请求实验。

_updated: 2026-07-31 17:55:51_
### 末层 finisher 原地 sampling/embedding 并启动下一 token

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：当前 TCP 证据在末层 logits 处终止，尚未证明自驱动 packet 能跨 token 边界形成最小 decode loop，也未让 schedule 的 token_offset=1 进入真实数据路径。2.现状：两层 finisher 已唯一产生 logits，所有 worker 都复制 embedding/层权重；packet 支持 finisher-to-starter 层间 handoff，但测试仅运行 token 0。3.目标：N=3、L=2、两个连续 forward；token 0/1 各有唯一 logits+greedy sample；token 0 finisher 原地 embedding sampled token 并启动 token 1；position 从 history_len 增至 history_len+1；schedule 覆盖 4 append events；每层每 token 仅指定 KV shard +1；总 sends=2 tokens*2 layers*(N-1)=8；两步 hidden/logits/sample 对齐显式累积 KV 的未切分参考。4.他者：标准 autoregressive decode 在末层执行 LM head/sampling，再把 token embedding 作为下一 forward 输入；pipeline 系统一般由最后 stage 采样后广播 token，或在权重复制时由持 token 的 stage 继续。Ring Attention 本身不规定 token 边界。5.本方案：仅扩现有 localhost 测试为 token×layer 双循环；末层 finisher argmax 后直接 Tensor::embedding，并保留 hidden 作为下个 token layer 0 的本地 starter 输入；其余节点进入 predecessor recv；参考侧用相同 K/V projection 显式追加每层历史。6.为什么：全节点已有 embedding 权重且 sampled token 已在 finisher，本地 continuation 不需新消息，保持每层 N-1 hop 路线；两 token 足以验证边界而不引入生成器生命周期。VERDICT: IMPLEMENT EXPERIMENT ONLY。边界：只验证 greedy、固定两层、两个 token、localhost CPU；不处理 EOS、随机采样状态、用户可见 token 回传、多请求 fairness、错误恢复或性能。

_updated: 2026-07-31 17:22:35_
### 将冻结 KV assignee schedule 接入既有两层 TCP 实验

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：纯 FrozenKvAssigneeSchedule 已验证，但真实 TCP 两层实验仍用手写 [2,1]，schedule 与数据路径之间缺少组合证据。2.现状：SelfDrivingPacket 已携带 assignee，LayerPacket::start 和 process_layer_packet 已按该值实现唯一 K/V projection+append；缺口仅在测试入口没有消费 schedule。3.目标：capacity_tickets+request_id 生成 token 0 的 layer 0/1 assignee；每层只有指定 domain 的 KV 长度 +1；其余既有断言继续成立，包括 route 1->2->0、0->1->2，总 sends=4、唯一 final logits、hidden/logits 对齐参考。4.他者：推理系统通常让执行层消费 admission/allocator 产生的冻结 placement 元数据；vLLM block allocator 可预先分配物理 block，但不能直接复用为 HCP 的 P2P layer packet assignee。5.本方案：只在现有 two_layer_tcp_ring_finisher_produces_final_logits 测试中构造 FrozenKvAssigneeSchedule(total_kv_units=2)，用 assignee_for(0,layer,2) 形成两层数组，并让既有 packet 与 KV growth 断言消费它。6.为什么：这是检验 schedule/API 与真实 ring 数据流是否相容的最小节点；不需要为两个值新增 runner API、协议字段或生产 planner。VERDICT: IMPLEMENT EXPERIMENT ONLY。边界：capacity tickets 仍被视为已计算输入；不证明 byte-level capacity、并发 admission、物理显存 reservation、远端网络或吞吐收益。

_updated: 2026-07-31 16:19:17_
### 候选下一小节点：一维冻结 capacity-weighted KV owner map

type: `hypothesis` · status: `superseded` · confidence: 1.0 · importance: 1.0 · source: `analysis-after-6ef5a18`

【动机六问】1.问题：TCP 单 token 垂直路径已闭到 logits，但 assignees 仍由测试手写数组，尚未落实异构容量加权与按 request_id 分散初始 phase。2.现状：现有 self_driving runner 接收显式 assignee；未提交 production placement 草稿超出当前范围。3.目标：在 self_driving 实验边界新增纯 FrozenKvOwnerMap：输入 capacity tickets、request_id、已知 total_positions，输出固定的一维 token->owner vector；每个 token 仅一个 owner，计数按容量比例做确定性整数分配，stable request_id 只旋转初始 phase，不改变各节点总份额；同输入跨运行一致，任意 N 通用，零容量节点不分配。4.他者：weighted round-robin/smooth weighted scheduling 用固定权重生成近似均匀序列；生产引擎另有 admission ledger，但不能直接复用为本实验最小 owner map。5.本方案：在 self_driving.rs 内实现小型纯数据结构；先按 largest remainder 得到请求长度内的精确整数份额，再平滑交织 owner 序列并按稳定 request hash 旋转；不含 layer 维度、throughput、动态迁移、ledger 或协议。6.为什么：它直接满足用户当前 capacity-weighted+request phase 目标，能被纯单测完整验证，且不触碰用户未提交草稿；后续单独节点再把生成的 assignee 接入现有 TCP 路径。【牺牲四问】生产 placement 默认复杂是为显存硬界与并发吞吐；最小 map 不提供 byte admission、active-request accounting、throughput balance 或 OOM 保证；这些能力一般用于生产安全与效率；当前只把 capacity tickets 当已给定实验输入，因此只能证明确定性 ownership 语义，不能声称生产显存安全。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-31 13:53:28_
### 修订：从 KV owner map 改为 KV assignee schedule

type: `revision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-31`

用户确认：owner 不是固定控制角色，但该命名混淆了 request/control owner 与唯一 KV 落点；原 token->owner 粒度还会把同一 token 的所有 layer K/V 集中到一个节点。修订为 FrozenKvAssigneeSchedule：对每个 append ordinal=(token_offset*num_layers)+layer_idx 产生唯一 KV assignee；starter、finisher、sampler 与 assignee 彼此独立且可轮转。

_updated: 2026-07-31 13:53:28_
### 实施一维冻结 KV assignee schedule 实验

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-31`

【动机六问】1.问题：capacity-weighted ownership 尚未落实，现有 TCP runner 的每层 assignee 仍是手写数组；同时 owner 术语会暗示固定控制节点。2.现状：self_driving 已验证每层唯一 assignee 的 KV project/append；未提交 production placement draft 已明确延后；旧候选 token->owner 被本 revision 修正。3.目标：输入 capacity tickets、request_id、total_kv_units；生成固定、可复现的一维 assignee schedule。append ordinal=(token_offset*num_layers)+layer_idx；每个位置一个 assignee；计数按容量比例确定；request_id 只旋转 phase；零容量节点无分配；任意 N。4.他者：weighted round-robin/smooth scheduling 生成比例化工作序列；生产 KV allocators 还需 admission/ledger，但不属于本节点。5.本方案：在 self_driving.rs 内加入纯 FrozenKvAssigneeSchedule，小规模 largest-remainder 计数 + smooth 交织 + stable request hash phase；只做单元测试，不接 TCP、不引入 layer calendar、throughput、迁移或 ledger。6.为什么：直接实现用户修订后的唯一 KV 落点语义，保留环数据面不变量并避免 owner 概念和生产 planner。VERDICT: IMPLEMENT EXPERIMENT ONLY。【牺牲四问】不实现 production byte admission、active-request accounting、throughput balancing、OOM 保证；这些由后续生产 allocator 负责，当前 capacity tickets 被视为已计算输入。

_updated: 2026-07-31 13:53:28_
### 当前阶段延后未提交的生产级 placement/ledger 草稿

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `working-tree-audit-2026-07-31`

【动机六问】1.问题：核心还缺 capacity-weighted KV owner map，但工作树存在一份未接纳的大型 placement 草稿，若直接使用会把已收缩的实验重新扩成生产 admission 系统。2.现状：placement.rs 共 1079 行，包含精确 byte profile、active-request ledger、workspace max、throughput bounded allocation、prompt+decode calendar、逐层计费、整数修复与 hash；capacity.rs/mod.rs/Cargo.toml 还有配套未提交修改。3.目标：保留这些用户修改不回退，但当前核心实验不依赖、不提交；先实现可单独证明的简单一维冻结 owner map。4.他者：vLLM 等生产引擎需要 paged KV admission、reservation 和 continuous scheduling；这些机制解决真实并发 OOM 与利用率。5.本方案：把生产 draft 明确延后，下一节点只在 self_driving 实验边界实现 capacity tickets->固定 token owner vector，并用 stable request_id 旋转 phase。6.为什么：实验当前只有 synthetic shard 与单请求 CPU correctness，没有可靠 allocator/profile/active-request 输入，无法诚实验证 1079 行生产合同。【牺牲四问】默认复杂 draft 存在是为了精确显存硬界、并发 reservation 与异构 throughput 优化；延后会牺牲 byte-level admission、workspace accounting、compute-balanced water-filling、运行期 ledger 与协议 hash；这些能力的一般用途是防 OOM 和提高生产吞吐；对当前单请求单 token/localhost 实验，它们既不可被真实验证，也不是 ring 数学成立的前提。VERDICT: DEFER production draft, preserve working-tree changes, do not integrate now。

_updated: 2026-07-31 09:52:06_
### 候选下一小节点：末层 TCP finisher 本地唯一产生 final logits

type: `hypothesis` · status: `superseded` · confidence: 1.0 · importance: 1.0 · source: `analysis-after-c5751f1`

【动机六问】1.问题：固定两层真实 TCP 已把 activation 从 layer 0 finisher 原地续到 layer 1，但网络垂直路径仍停在末层 hidden；核心目标要求末层唯一产生 logits，尚未证明 final RMSNorm/LM head 不经 coordinator 回传即可在 TCP finisher 本地完成。2.现状：in-process 两层与任意 L runner 已验证唯一 final logits 和 tied/独立 head；TCP 两层只返回 final_hidden。任意 L 网络化此时主要重复已证的跨层归纳步骤，新增信息少。3.目标：保持 N=3、两层、单 token、localhost CPU；末层 domain 2 finisher 本地执行 final RMSNorm+独立 LM head，其他 worker 不产生 logits；logits 与未切分两层参考 max diff<既有阈值，logits projection=1，总 sends 仍为 4，无额外 activation/logits hop。4.他者：pipeline parallel 通常由最后 stage 执行 final norm/LM head；Ring Attention 只定义 attention accumulator，不负责模型尾部。vLLM 的同步 model forward 可产生 logits，但无法直接复用为 P2P finisher 事件。5.本方案：只扩展现有固定两层 TCP 实验，让每个 worker 保留自己的模型 final norm/head；仅在最后一层 Finished 分支调用既有 project_final_logits，并返回 Option logits 供测试断言。6.为什么：这把真实网络单 token 垂直路径闭到核心目标的最后输出，且不引入 sampling/下一 token 终止协议；比任意 L 网络循环提供更直接的新证据，也避免触碰未接纳的大型 placement/planner 草稿。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-31 08:51:21_
### 实施末层 TCP finisher 本地唯一 final logits 实验

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-31`

【动机六问】1.问题：两层 TCP 路径停在末层 hidden，核心目标的末层唯一 logits 尚无网络证据。2.现状：in-process runner 已证明 final norm/head 数学与唯一性；TCP 已证明两层 finisher 原地续层，但未运行 final head。3.目标：N=3 两层 localhost；末层 domain 2 唯一产生 logits并对齐参考，发送仍为 4，无额外回传。4.他者：pipeline parallel 由最后 stage 执行 final norm/head；Ring Attention 不处理模型尾部；vLLM 同步 forward 合同不能直接复用到 P2P finisher 事件。5.本方案：只扩现有 TCP 测试，每个 worker 保留本地模型 final norm/head，仅末层 Finished 分支调用既有 project_final_logits。6.为什么：它直接闭合单 token 网络垂直路径，新增风险只在末层边界；任意 L 是已证归纳的重复，sampling 与 placement 会扩大范围。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-31 08:51:21_
### 候选下一小节点：N=3 固定两层 localhost TCP finisher-to-starter handoff

type: `hypothesis` · status: `superseded` · confidence: 1.0 · importance: 1.0 · source: `analysis-after-2150d7a`

【动机六问】1.问题：单层真实 TCP 已覆盖任意 N 与 wrap-around，但网络 worker 仍是单包单层后退出；尚未证明 layer 0 finisher 能在本节点用输出 hidden 原地启动 layer 1，而不把 activation 返回 coordinator。2.现状：in-process 两层和任意 L runner 已证明 s(l+1)=s(l)-1 mod N 与数值正确；TCP 试验只证明 LayerPacket 在一层内流动。3.目标：N=3、两层、单 token、localhost CPU；layer 0 finisher 直接创建并先处理 layer 1 packet，然后沿已有 predecessor/successor 连接继续；断言 layer1 starter==layer0 finisher、总 sends=2*(N-1)=4、每层每节点 partial exact-once、每层唯一 assignee KV 增长、末层唯一 finisher hidden 对齐两层参考。4.他者：pipeline parallel 在 stage 间传 activation，Ring Attention 每层独立传 accumulator；常见事件循环根据 layer/step 元数据继续，但没有现成实现表达 HCP 同层 KV 分片与 finisher 续层组合。5.本方案：只把 localhost test worker 从单层 one-shot 改为固定两层事件循环，复用 SelfDrivingPacket.layer_idx 与同一 TCP ring；不泛化任意 L，不接 logits/sampling、QUIC、remote 或 runtime。6.为什么：这是非零 starter 在多层中的第一个真实应用，同时仍把失败面限制在一次 layer boundary；直接任意 L 或 token continuation 会混入终止协议与 sampling。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-31 07:11:19_
### 实施固定两层 TCP finisher 原地续层实验

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-31`

【动机六问】1.问题：单层 TCP 已证明任意 N 与 wrap-around，但 worker 仍单包单层后退出，尚未证明 layer 0 activation 无需返回 coordinator 即可继续 layer 1。2.现状：in-process 两层与任意 L 已证明角色递推和数值；TCP 只闭合单层。3.目标：N=3、两层、单 token；layer 0 route=1->2->0，domain 0 finisher 原地启动并先处理 layer 1，route=0->1->2；总 sends=4、逐层 partial/KV assignee/finisher exact-once，最终 hidden 对齐参考。4.他者：pipeline parallel 传 activation，Ring Attention 逐层传 accumulator；事件循环依 layer_idx 继续，但没有现成 HCP 同层 KV 分片加 finisher 续层组合。5.本方案：把 localhost test worker 扩为固定两层事件循环，复用 SelfDrivingPacket.layer_idx 和已有 TCP ring；不改 transport trait。6.为什么：它隔离唯一未证的 layer boundary；直接任意 L 或 token continuation 会混入终止协议、final head 与 sampling。执行环境为 mac-local-shell + libtorch CPU，只声明 localhost correctness。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-31 07:11:19_
### 候选下一小节点：任意 N 与 wrap-around 的 localhost TCP 单层 ring

type: `hypothesis` · status: `superseded` · confidence: 1.0 · importance: 1.0 · source: `analysis-after-71c8698`

【动机六问】1.问题：N=3 starter=0 已证明两个真实 TCP hop，但用户要求任意 N 通用；当前机器证据没有覆盖 N=2/N=4，也没有让请求跨过最后节点->0 的 wrap-around 边。2.现状：LayerPacket/SelfDrivingPacket 的 domains/current_domain/visited 与 localhost listener 拓扑构造均按 N 参数化；in-process 测试已覆盖 N=1/2/4，但网络测试把 N=3、starter=0、assignee=1 固定。3.目标：保持单层、单 token、localhost CPU，最小泛化测试覆盖 N=2/3/4 与非零 starter；每例断言 route 沿 successor 且包含需要的 wrap-around、send=N-1、每 worker local partial exact-once、唯一 assignee KV 增长、唯一 finisher 输出对齐参考、wire 不携带历史 KV。4.他者：标准 ring send/recv 以 rank、world_size、successor=(rank+1)%N 参数化，正确性通常用多 world-size 和 wrap-around case 验证；无需 collective 或动态 planner。5.本方案：提取现有 localhost 测试的参数化 helper，只扩测试和必要的最小观测字段；不修改 wire 合同、transport trait、任意 L 循环、sampling、QUIC、远端脚本或 runtime。6.为什么：直接进入任意 L 网络循环会把 domain-count 路由问题与 layer handoff 混在一起；先补 N 与 wrap-around 是更小、可归因的证据节点，且 laptop 不可达也不阻塞。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-31 05:33:39_
### 实施任意 N 与非零 starter 的 localhost TCP ring 路由验证

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-31`

【动机六问】1.问题：现有 N=3 starter=0 的真实 TCP 路径 0->1->2 没有经过尾节点->0，只证明了链式两跳；不能排除 modulo 闭环边或非零 phase 有错。2.现状：LayerPacket/SelfDrivingPacket 的 domains/current_domain/visited 与 listener 拓扑均按 N 参数化，in-process 已覆盖 N=1/2/4，但真实 TCP 测试固定 N=3、starter=0、assignee=1。非零 starter 不只来自多层：finisher-to-starter handoff、跨 token continuation、以及 stable request_id 分散初始 phase 都会产生。3.目标：保持单层单 token，覆盖 N=2/3/4，并令至少 N=3/N=4 使用 starter=N-1；机器断言实际 route 经过 wrap-around、send=N-1、每节点一次 partial、唯一 assignee 增长、唯一 finisher 输出对齐参考。4.他者：标准 P2P ring 用 successor=(rank+1)%world_size，正确性验证必须覆盖多 world size 和 wrap-around；否则只能证明 linear pipeline。5.本方案：把既有 N=3 localhost 测试提取为参数化 helper，增加最小 route 观测；不修改独立 SelfDrivingPacket、TcpFrame、KvTransport trait 或运行时。6.为什么：直接做多层网络循环会把 modulo 路由与 layer handoff 混在一起；单层非零 starter 能独立证明物理环闭合，随后多层只需复用该不变量。执行环境仍为 mac-local-shell + libtorch CPU，只声明 localhost correctness。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-31 05:33:39_
### 候选下一小节点：N=3 localhost 单层真实 P2P self-driving ring

type: `hypothesis` · status: `superseded` · confidence: 0.97 · importance: 1.0 · source: `analysis-after-c2a0483`

【动机六问】1.问题：任意 L 的模型控制循环已成立，但所有 domain step 仍在同一进程直接调用；尚未证明 LayerPacket 能沿只有前驱/后继的真实字节流传递，当前最核心的 P2P 拓扑主张仍缺机器证据。2.现状：self_driving::LayerPacket 需要 residual、normalized hidden、position_ids、Q、O/LSE、assignee/current_domain/domains/visited；现有 model::transport::RingPacket 和 TCP/QUIC codec 只承载 layer_idx、Q、O、LSE、scale，无法直接恢复 finisher 的 residual/norm/MLP continuation。3.目标：做一个单层、单 token、N=3 localhost 的真实 TCP P2P 垂直切片；三个 worker 各只持自己的 local KV shard，只连接 predecessor/successor，packet 经过 N-1=2 个网络 hop 后由 finisher 完成 W_o+residual+norm+MLP；输出与单节点参考一致，只有 assignee shard 增长，wire payload 不随历史 KV 长度增长。4.他者：Ring Attention 通过 P2P send/recv 传 online-softmax accumulator，pipeline parallel 通过 stage link 传 activation；项目现有 TCP/QUIC RingPacket codec 已实现 Tensor 字节化，可复用 framed transport 与 dtype/shape roundtrip，但其 payload 合同不足以承载两者组合。5.本方案：先扩展一个最小 self-driving wire packet，并用 localhost N=3 单层线程/连接测试贯通 encode-send-recv-decode-domain-step；复用现有 tensor codec，不接任意 L 网络循环、sampling、多请求、QUIC、远端硬件、重试、版本协商或 runtime。6.为什么：codec-only 测试不能证明两 peer ring 数据流，直接网络化任意 L 又会同时扩大协议与层循环失败面；单层 N=3 TCP 垂直切片是能证明真实 P2P 核心主张的最小可归因步骤。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-31 03:55:20_
### 实施独立 SelfDrivingPacket 的 N=3 localhost TCP 垂直切片

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-31`

【动机六问】1.问题：任意 L 的模型控制循环已成立，但所有 domain step 仍在同一进程直接调用；尚未证明 LayerPacket 能沿只有前驱/后继的真实字节流传递，当前最核心的 P2P 拓扑主张仍缺机器证据。2.现状：self_driving::LayerPacket 需要 residual、normalized hidden、position_ids、Q、O/LSE、assignee/current_domain/domains/visited；现有 model::transport::RingPacket 和 TCP/QUIC codec 只承载 layer_idx、Q、O、LSE、scale，无法直接恢复 finisher 的 residual/norm/MLP continuation。3.目标：单层、单 token、N=3 localhost TCP；三个 worker 各只持自己的 local KV shard，只连接 predecessor/successor，packet 经过 N-1=2 个网络 hop 后由 finisher 完成 W_o+residual+norm+MLP；输出与单节点参考一致，只有 assignee shard 增长，wire payload 不随历史 KV 长度增长。4.他者：Ring Attention 通过 P2P send/recv 传 online-softmax accumulator，pipeline parallel 通过 stage link 传 activation；项目现有 TCP RingPacket codec 已实现 Tensor 字节化，可复用 framing 与 dtype/shape roundtrip，但 payload 合同不足以承载两者组合。5.本方案：新增独立 SelfDrivingPacket wire struct；LayerPacket 只在发送前/接收后转换。TcpKvTransport 使用私有 TcpFrame 分派和实验性固有 send/recv 方法，复用 tensor codec；不修改 KvTransport trait、旧 RingPacket 或 QUIC。N=3 loopback 测试建立完整定向 ring，每 worker 各有 incoming predecessor 与 outgoing successor 连接，但单请求只使用 starter→successor→finisher 两条边。6.为什么：直接扩旧 RingPacket 会污染已验证 Q-ring 合同；测试内自写 socket codec 会形成一次性重复；独立 packet + TCP 私有帧是最小可复用边界。codec-only 测试不能证明两 peer ring 数据流，直接网络化任意 L 又扩大失败面，因此选择单层 N=3 垂直切片。执行环境按 infrastructure-inventory 选择 mac-local-shell + local libtorch CPU，只声明 correctness 与 loopback 字节流，不声明硬件性能。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-31 03:55:20_
### 候选下一小节点：任意 L 的单 token 全模型 ring

type: `hypothesis` · status: `superseded` · confidence: 0.98 · importance: 1.0 · source: `analysis-after-e2c6cd6`

【动机六问】1.问题：当前 final logits 已闭合，但 runner 固定为两层，尚不能直接运行真实模型层数，也不能机器验证用户关心的 L%N=0 时末层 producer 固定现象。2.现状：run_two_layer_ring 使用 [usize; 2] assignee，并正确完成两次 finisher-to-starter handoff 与唯一 final head；任意 N 的单层数学已验证，因此推广的未知点主要是全模型控制循环和角色递推，不是 attention 数学。3.目标：新增仅供实验的任意 L 单 token runner，接收每层 local KV shards 与冻结 assignee 序列；逐层令上一层 finisher 成为下一层 starter，末层只执行一次 final norm/head；至少验证 N=3 下 L=3 与非整倍数 L 的 logits 对齐、总 hops=L*(N-1)、producer=(starter-L) mod N、每层 exact-once。4.他者：普通 transformer forward 以循环串联 decoder layers，pipeline parallel 以 activation 串联 stages；可复用这种顺序 composition，但现成实现不表达 HCP 的同层 ring accumulator、capacity-owned KV 与每层轮转角色。5.本方案：把已验证两层逻辑泛化成一个小循环，保留现有单层 primitive 和 in-process 边界；不加入 sampling、跨 token 状态、serde、QUIC、runtime、动态 planner 或生产治理。6.为什么：这是进入真实 P2P 前最小的去人工边界步骤；先网络化固定两层会把测试专用限制写入线协议，先做 sampling 又无法证明真实层数下 sampler 所在节点。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-30 14:46:39_
### 实施任意 L 的单 token 全模型 self-driving ring

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-30`

【动机六问】1.问题：当前 final logits 已闭合，但 runner 固定为两层，尚不能直接运行真实模型层数，也不能机器验证用户关心的 L%N=0 时末层 producer 固定现象。2.现状：run_two_layer_ring 使用 [usize; 2] assignee，并正确完成两次 finisher-to-starter handoff 与唯一 final head；任意 N 的单层数学已验证，因此推广的未知点主要是全模型控制循环和角色递推，不是 attention 数学。3.目标：新增仅供实验的任意 L 单 token runner，接收每层 local KV shards 与冻结 assignee 序列；逐层令上一层 finisher 成为下一层 starter，末层只执行一次 final norm/head；至少验证 N=3 下 L=3 与非整倍数 L 的 logits 对齐、总 hops=L*(N-1)、producer=(starter-L) mod N、每层 exact-once。4.他者：普通 transformer forward 以循环串联 decoder layers，pipeline parallel 以 activation 串联 stages；可复用这种顺序 composition，但现成实现不表达 HCP 的同层 ring accumulator、capacity-owned KV 与每层轮转角色。5.本方案：把已验证两层逻辑泛化成一个小循环，保留现有单层 primitive 和 in-process 边界；不加入 sampling、跨 token 状态、serde、QUIC、runtime、动态 planner 或生产治理。6.为什么：这是进入真实 P2P 前最小的去人工边界步骤；先网络化固定两层会把测试专用限制写入线协议，先做 sampling 又无法证明真实层数下 sampler 所在节点。执行环境按 infrastructure-inventory 选择 mac-local-shell + local libtorch CPU，因为本节点只声明 correctness。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-30 14:46:39_
### 候选下一小节点：末层 finisher 唯一产生 logits

type: `hypothesis` · status: `superseded` · confidence: 0.98 · importance: 1.0 · source: `analysis-after-dc1aeb5`

【动机六问】1.问题：两层 handoff 已闭合，但末层 finisher 的 hidden 仍作为实验结果返回；尚未证明完整 decode 模型输出能在该节点就地产生且只产生一次。2.现状：LlamaModel::forward 在所有层后集中执行 final RMSNorm，再用独立 lm_head 或 tied embedding 计算 logits；self-driving 两层 runner 尚未接这段尾部。3.目标：末层 finisher 用自己的 hidden 执行 final norm + LM head，唯一生成单 token logits；记录 logits producer domain=末层 finisher、次数=1，并与标准两层模型参考 logits 一致；不增加 ring hop。4.他者：pipeline parallel 通常由最后 stage 持有/执行 output head；vLLM 在 model hidden 后由 logits processor 生成 logits。可复用 final-stage ownership，但现成实现不能直接表达 HCP 中随层轮转的 finisher。5.本方案：在固定两层 in-process 实验上增加最小 final-head helper/result，复用 LlamaModel 的 norm、lm_head 与 tied embedding fallback；不接 sampling、token handoff、serde、网络或 runtime。6.为什么：这是核心 tensor 垂直路径的最后一个独立模型边界；sampling 是控制/RNG 合同，分开验证更容易归因。【牺牲四问】1.当前每 worker 返回 logits 的对称合同简化 coordinator 汇总和故障诊断。2.唯一 producer 放弃冗余 logits 副本与任意 worker 可替代返回。3.这些副本在一般分布式系统中可用于一致性对照或失败接管。4.当前 PoC 不做容错，目标正是消除 N 倍冗余 forward；保留标准 reference 路径用于对照即可。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-30 11:51:39_
### 实施末层 finisher 唯一 final norm + LM head

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-30`

【动机六问】1.问题：两层 handoff 已闭合，但末层 finisher 的 hidden 仍作为实验结果返回；尚未证明完整 decode 模型输出能在该节点就地产生且只产生一次。2.现状：LlamaModel::forward 在所有层后集中执行 final RMSNorm，再用独立 lm_head 或 tied embedding 计算 logits；self-driving 两层 runner 尚未接这段尾部。3.目标：末层 finisher 用自己的 hidden 执行 final norm + LM head，唯一生成单 token logits；记录 logits producer domain=末层 finisher、次数=1，并与标准两层模型参考 logits 一致；不增加 ring hop。4.他者：pipeline parallel 通常由最后 stage 持有/执行 output head；vLLM 在 model hidden 后由 logits processor 生成 logits。可复用 final-stage ownership，但现成实现不能直接表达 HCP 中随层轮转的 finisher。5.本方案：在固定两层 in-process 实验上增加最小 final-head helper/result，复用 LlamaModel 的 norm、lm_head 与 tied embedding fallback；不接 sampling、token handoff、serde、网络或 runtime。6.为什么：这是核心 tensor 垂直路径的最后一个独立模型边界；sampling 是控制/RNG 合同，分开验证更容易归因。【牺牲四问】1.当前每 worker 返回 logits 的对称合同简化 coordinator 汇总和故障诊断。2.唯一 producer 放弃冗余 logits 副本与任意 worker 可替代返回。3.这些副本在一般分布式系统中可用于一致性对照或失败接管。4.当前 PoC 不做容错，目标正是消除 N 倍冗余 forward；保留标准 reference 路径用于对照即可。执行环境按 infrastructure-inventory 选择 mac-local-shell + local libtorch CPU，因为本节点只声明 correctness。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-30 11:51:39_
### 候选下一小节点：两层 packet handoff，不含 logits

type: `hypothesis` · status: `superseded` · confidence: 0.98 · importance: 1.0 · source: `analysis-after-76be3b6`

【动机六问】1.问题：单层 packet 已自足，但 finisher 产出的 hidden 仍作为函数结果返回，尚未证明它能在同一逻辑 domain 直接成为下一层 starter 并保持 N-1 hops/layer。2.现状：一层内 attention+residual+post-attention norm+MLP 已闭合；多层角色递推与层边界 packet 初始化未进入真实 tensor 测试。3.目标：仅用两个真实 DecoderLayer，在 layer 0 finisher 上以其 hidden 创建 layer 1 packet；验证 layer 1 starter==layer 0 finisher、两层输出与单节点参考一致、总 hops=2*(N-1)、每层 exact-once 与 capacity-owned local KV 不变。4.他者：pipeline parallel 在 stage 边界传 activation，Ring Attention 每层独立传 accumulator；可复用 activation continuation，但没有 HCP 同层序列分片与下一层 starter 递推的组合测试。5.本方案：扩展 in-process 实验为固定两层 handoff，复用现有 LayerPacket/process_layer_packet，不加通用全模型 driver、LM head、sampling、serde 或网络。6.为什么：它只隔离验证层边界递推；把末层 logits 同时加入会让失败无法区分是 handoff 还是 head/sampling 边界。【牺牲四问】默认完整 model.forward 一次遍历所有层并最终做 norm/head，控制流最简单；本节点暂时牺牲全模型与可生成 token 的完整性；完整 forward 的价值是提供最终模型合同；当前只验证自驱动 ring 的下一项最小数学依赖，因此两层固定实验足够。VERDICT: PROPOSE IMPLEMENT EXPERIMENT ONLY；待用户确认。

_updated: 2026-07-30 10:52:57_
### 实施固定两层 packet handoff，隔离验证层间 continuation

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-30`

【动机六问】1.问题：单层 packet 已自足，但 finisher 产出的 hidden 仍作为函数结果返回，尚未证明它能在同一逻辑 domain 直接成为下一层 starter 并保持 N-1 hops/layer。2.现状：一层内 attention+residual+post-attention norm+MLP 已闭合；多层角色递推与层边界 packet 初始化未进入真实 tensor 测试。3.目标：仅用两个真实 DecoderLayer，在 layer 0 finisher 上以其 hidden 创建 layer 1 packet；验证 layer 1 starter==layer 0 finisher、两层输出与单节点参考一致、总 hops=2*(N-1)、每层 exact-once 与 capacity-owned local KV 不变。4.他者：pipeline parallel 在 stage 边界传 activation，Ring Attention 每层独立传 accumulator；可复用 activation continuation，但没有 HCP 同层序列分片与下一层 starter 递推的组合测试。5.本方案：扩展 in-process 实验为固定两层 handoff，复用现有 LayerPacket/process_layer_packet，不加通用全模型 driver、LM head、sampling、serde 或网络。6.为什么：它只隔离验证层边界递推；把末层 logits 同时加入会让失败无法区分是 handoff 还是 head/sampling 边界。【牺牲四问】默认完整 model.forward 一次遍历所有层并最终做 norm/head，控制流最简单；本节点暂时牺牲全模型与可生成 token 的完整性；完整 forward 的价值是提供最终模型合同；当前只验证自驱动 ring 的下一项最小数学依赖，因此两层固定实验足够。执行环境按 infrastructure-inventory 选择 mac-local-shell + local libtorch CPU，因为本节点只声明 correctness，不声明硬件性能。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-30 10:52:57_
### 候选下一小节点：显式化单层自驱动 packet 数据边界

type: `hypothesis` · status: `superseded` · confidence: 0.95 · importance: 1.0 · source: `analysis-2026-07-30-next-node-motivation`

【动机六问】1.问题：当前单层 runner 的数值与角色正确，但共享作用域让 assignee/finisher 读取 normalized hidden 与 residual，尚未证明真实 P2P packet 自足，也无法直接审计通信 payload 是否与 context 长度无关。2.现状：starter 投影 Q，assignee 唯一投影并 commit current K/V，finisher 唯一完成 O projection+residual+norm+MLP；然而 domain step 没有只依赖 packet+local KV 的接口。3.目标：定义仅供实验使用的显式 LayerPacket，至少携带 residual h、normalized h、Q、O/LSE 与最小路由元数据；每个 domain step 只能访问 packet 和本地 shard；验证 N=1/2/3/4 数学不变、N-1 hops、exact-once 不变、packet tensor 元素数不随历史 KV 长度增长。4.他者：Ring Attention 显式传 online-softmax accumulator；pipeline parallel 显式传 activation。可复用“状态载体必须自足”的思想，但二者没有 HCP 同层 KV 分片下 assignee 自算 K/V 与 finisher continuation 的组合 packet。5.本方案：先做纯 in-process struct 与 domain-step 边界，不加 serde、QUIC、worker command、重试或 planner；把现有闭包隐式读取改成 packet 字段读取。6.为什么：这是从单层数学证明走向真实两-peer P2P 的最小依赖；先做多层会复制当前隐藏边界，先做 QUIC则同时引入网络失败面。【牺牲四问】1.默认共享借用用于减少类型和状态搬运，最适合普通单进程 forward。2.牺牲：增加一个实验 packet 类型，并显式保留 residual h 与 normalized h 两个 O(d) 瞬时 tensor。3.这些共享借用在一般单进程代码中提供简单控制流和较少 bookkeeping。4.对 HCP，显式 O(d) packet 是验证 P2P 自足性和线性通信的必要成本，且不随 context 增长，不破坏 KV capacity-weighted 切分。结论：建议 implement this experiment only；多层 handoff+唯一 logits 延后一节点。

_updated: 2026-07-30 09:30:41_
### 实施显式 LayerPacket，先证明跨节点所需状态边界

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-07-30`

【动机六问】1.问题：单层数值与角色已正确，但 assignee/finisher 仍从共享作用域读取 normalized hidden 与 residual，无法证明真实 P2P packet 自足。2.现状：starter 唯一投影 Q、assignee 唯一投影并 commit 当前 K/V、全节点各算一个 partial、finisher 唯一做 O projection+residual+norm+MLP；隐藏共享借用掩盖了线上必须传输的 O(d) 状态。3.目标：显式 LayerPacket 使 domain step 只能访问 packet+本地 shard，并以测试证明数学与 exact-once 不变、payload 与历史 context 长度无关。4.他者：Ring Attention 显式传 online-softmax accumulator，pipeline parallel 显式传 activation；可复用自足状态载体思想。5.本方案：仅在现有 in-process runner 中增加实验 struct 和 step 边界，不接线协议与 runtime。6.为什么：这是进入两-peer P2P 前最小且必要的边界证明，先多层会复制隐藏共享状态，先网络化会同时扩大失败面。【牺牲四问】默认共享借用让普通单进程 forward 更简单且少 bookkeeping；本次牺牲这种简洁性，显式保留 residual h 与 normalized h 两个 O(d) 瞬时 tensor；它们在普通 forward 中本可由调用栈隐含持有；对 HCP 而言这是 packet 自足的必要成本，且不随 context 增长、不形成远端历史 KV 副本。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-07-30 09:30:41_
### Rust 首个实验切片：任意 N 的单层真实 tensor 自驱动 ring

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `docs/plans/2026-07-30-rust-single-layer-self-driving-experiment.md`

【动机六问】1.问题：需要先验证 attention accumulator 能否和 residual/norm/MLP 组成一个沿 ring 交接的真实 decoder layer，而不是继续堆控制面设计。2.现状：Rust Q-ring 已验证真实 Q+O+LSE 数学，但每个 worker 仍运行完整 forward；现有 DecoderLayer.forward 与 attention.forward 都是整段同步调用。3.目标：单进程真实 tensor、单层、任意 N；uneven 历史 KV shards 互斥；单 Q/current K/V、N 个 local partial、N-1 hops、单 finisher continuation；输出与单节点合并 KV 参考一致。主例 N=3，通用性覆盖 N=1/2/4。4.他者做法：Ring Attention 用 online-softmax 合并局部 attention；pipeline parallel 在 stage 边界传 hidden state。可复用前者的 accumulator 数学和后者的 continuation 思路，但现成系统没有 HCP 这种同层序列分片加层间角色轮转接口。5.本方案：只从现有 Rust ring backend 提取 decode projection/partial/O-projection 原语，新增不接网络的单层实验 runner，由 caller 提供 uneven shards 和 starter/assignee。6.为什么：这是能使用真实模型算子验证核心假设的最小切片；比纯 mock 证据强，又避免 QUIC/runtime/planner 同时进入失败面。【牺牲四问】1.真实分布式默认还需要 transport、lifecycle 和故障处理。2.本节点暂不证明跨进程通信、全模型多层和物理显存 reservation。3.这些能力用于最终部署和完整系统正确性。4.当前只判断核心模型数据流是否成立，明确实验边界后可接受。结论：implement this experiment only。

_updated: 2026-07-30 05:59:08_
### 用户约束：核心优先，小步验证，未经明确要求不做生产级扩张

type: `preference` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-07-30`

用户明确要求：现阶段只实现自驱动 decode ring 的核心和为核心闭环不可缺少的能力；每完成一个小步先验证正确性与实际效果，再讨论下一步。凡用户没有主动提出的生产级 admission、精确资源 ledger、完整版本协商、容错重试、调度泛化、性能策略泛化等，不得自动前置或扩入当前任务。复杂性必须由当前核心验证的具体阻塞证据触发。

_updated: 2026-07-29 17:24:29_
### 修订自驱动 decode 实施范围：先做最小真实核心闭环

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-07-30`

【动机六问】1.问题：14 Task 计划把 ring 核心验证与生产级资源管理、协议治理和通用调度绑定，步幅过大，核心效果迟迟不可见。2.现状：核心理论已明确，但真实 tensor 路径尚未证明 attention + norm/MLP continuation + N-1 hops；工作树却先出现约千行 placement/ledger 实现。3.目标：以最小真实切片证明单 packet、每层 N-1 hops、每 worker 仅两 peer、KV 唯一持久归属、attention partial 全员参与、finisher 唯一执行 W_o/residual/norm/MLP、末层唯一 logits；完成一个节点即验证并停下审查。4.他者做法：算法原型通常先固定简单 ownership 和单请求路径证明数据流/数学正确，再依据测量补 admission、continuous batching、容错与生产治理。5.本方案：保留 starter/finisher、唯一 KV assignee、简单冻结 capacity-weighted owner map 和 request phase；暂停二维 layer calendar、throughput water-filling、完整 memory ledger、versioned cluster negotiation、通用 backpressure/counter RNG 等非核心能力。6.为什么：这些保留项是 ring 正确性与异构 KV 切分的直接依赖；暂停项不影响验证核心数据流，提前实现只会扩大失败面。【牺牲四问】1.生产级默认用于并发资源安全、兼容升级和故障治理。2.当前牺牲的是尚未实装这些保证，不把原型称为生产可用。3.这些能力在真实多租户长期运行中不可替代。4.本项目当前先判断核心机制效果，明确标注原型边界即可；出现实测阻塞或用户明确要求后再逐项引入。结论：implement minimal core only；defer production-grade expansion。

_updated: 2026-07-29 17:24:29_
### 修订自驱动 ring 计划:拒绝插件 E,合并 Rust D1 与最小 D2 垂直切片

type: `revision` · status: `held` · confidence: 0.98 · importance: 1.0 · source: `docs/plans/2026-07-29-self-driving-ring-decode-revision.md`

【冲突】冻结计划声称 plugin successor-seeded 可把 N 跳降为 N-1,并把 Rust 单包 attention D1 作为不动 model 层间的独立 checkpoint。代码与拓扑审计反驳两点。
【修订】1) task E rejected:owner-return 单向环物理下限仍是 N;现 plugin Q-ring 已在其合同内 hop-minimal。2) D1 不能独立切生产路径:只有 finisher 获得 layer output,其余同步 full forward 会阻塞。3) 第一个可运行切片合并 D1+最小 D2:coordinator 暂时广播 decode 命令使所有 worker 进入 collective decode;环内单 starter/单包,N-1 peer hops;finisher 做 W_o/MLP 并成为下一层 starter;末层只一个 logits producer。4) 后续再做 batch 隔离、finisher 采样和 coordinator 退位。
【六问】问题:避免实施拓扑不可能的 E 和不可运行的 D1 中间态。现状:WorkerRuntime 按 coordinator command 调 backend.decode_request;WorkerBackend 要求每 worker 返回 logits;RingPacket 仅有 layer/Q/O/LSE/scale;LlamaModel.forward 是全层同步调用栈。目标:第一个生产切片本身可运行且机器证明 N-1 hops、单 continuation、单 logits producer、exact-once growth。别人做法:collective/pipeline 系统一般先定义可独立运行的 stage/state-machine contract,再迁移 ownership;无可直接复用的 vLLM 插件扩展点。我们做法:R0 协议+mock,R1 coordinator-triggered collective vertical slice,R2 batch,R3 autonomous sampling,R4 hardware ladder。为什么:复用现有 coordinator 作为临时同步屏障,把最大风险限制在模型 continuation/packet contract,避免一次同时重写 admission、sampling 和 lifecycle。
【牺牲复核】仍放弃全节点结果复制与多包并发,PoC 可接受;不再牺牲 checkpoint 可运行性。结论:implement revised Rust plan;reject plugin E。
[2026-07-29 exact-once 细化] current K/V assignee 与 starter/relay/finisher 可能重合。starter==assignee 时必须直接生成唯一 history+current seed 并 commit,不能先 seed history 后再合并;R0 覆盖三种角色重合。

_updated: 2026-07-28 18:04:28_
### 用户裁定:当前显存切分是工程欺骗,必须彻底修复后才继续;vLLM/Rust 两线同查同修

type: `decision` · status: `held` · confidence: 0.95 · importance: 1.0 · source: `user-direction`

用户方向(2026-07-27,最高优先级):语义边界现状被定性为"工程欺骗"——一直强调真正的显存切分,现状不达标:
1. 尾节点 prefill 前拉取全量前缀 KV(瞬时),仍承担全部显存压力,违背 CP 核心目标(ISSUE-003 critical);
2. 并发 decode 增长落回 owner 池,显存压力无 CP 分担(ISSUE-004 critical);PoC 最小要求=2 请求并发;
3. 从未做真 ring 模式 overlap(Ring Attention 优势域)(ISSUE-005 high);
4. owner/peer 角色不对称,设计范式要求所有 worker 同等地位(ISSUE-006 high;LoongServe 转化的对等化设计另行探讨,不在本轮修复范围);
5. vLLM 与 Rust 是同一设计控制层的两个数据实现层,问题要两线同查同修。
Rust 线审查结论(explore 复核,file:line 证据):
- prefill 流式逐块+overlap+worker 对称:Rust 线领先,无 003/005/006 对应问题(ring.rs:665-733 流式、636-663 overlap、worker 对称+coordinator 集中采样);
- 但 decode 同病不同机制:增长 KV 全节点复制(ring.rs:457-466+cache.rs:55-69,ISSUE-007 critical)、decode 仍逐 token 重发 prefill KV 分区而非 Q+LSE 环(ring.rs:399,ISSUE-008 high)、全 worker 冗余算 logits(coordinator.rs:344)。
修复范围(LoongServe 对等化探讨除外):
A. vLLM prefill 真 ring 化:按层/按块流式 staging+compute/comm overlap,单节点任意时刻持有量有界(修 003+005,Rust 线为参照实现);
B. vLLM decode ≥2 并发+增长分片保持(修 004,PoC 最小要求);
C. Rust decode 移植 Q+LSE 累积器环+增长分片(修 007+008)。

_updated: 2026-07-27 15:10:25_
### 冻结 assignee schedule 的可靠保证是完整 horizon 精确份额，不是任意旋转前缀误差小于等于 1

type: `belief` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `code-audit-2026-08-01`

FrozenKvAssigneeSchedule 先用 largest remainder 得到 total_kv_units 内的精确整数 counts，再生成 smooth sequence；任意 request phase 对完整 horizon 的遍历仍精确保持 counts。现有 request_id 旋转后，所有前缀偏差小于等于 1 unit 并非普遍定理。该修订不破坏唯一 assignee、完整 horizon capacity-weighted 份额或单请求数据流；若未来需要物理显存 hard bound，应按完整 horizon 预留 counts，或另行证明/实现更强的 cyclic prefix bound。

_updated: 2026-07-31 19:13:51_
### 修订冻结 schedule 的任意 phase 前缀误差结论

type: `revision` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `code-audit-2026-08-01`

旧 evidence ev-rust-frozen-kv-assignee-schedule-20260731 中“任意前缀比例误差不超过一个 KV unit”只由 [1,3,2]、24 units、request_id=41 的单一实例覆盖，不能推广到任意 capacity、horizon 和 phase。保留旧 evidence 对实现、完整 counts、稳定 phase、零容量排除及测试通过的证明；仅撤回其普遍 prefix-bound 子结论，由 belief-frozen-schedule-guarantee-revised-20260801 取代。

_updated: 2026-07-31 19:13:51_
### 任务D:Rust 线实现自驱动环 decode(单包 N-1 跳/层,角色全轮转,零冗余)

type: `task` · status: `ongoing` · confidence: 0.9 · importance: 0.95 · source: `docs/plans/2026-07-29-self-driving-ring-decode-implementation.md`

decision-self-driving-ring-20260728 的 Rust 落地。改动面:ring.rs decode 路径从"每节点发起 N 包"改为"单包轮转+finisher 就地续层";model.rs 层间允许 finisher 节点就地应用 W_o/MLP/norm 并续发;采样/logits 移到末层 finisher;coordinator 退为准入/释放。验证:既有 68 测试回归+新正确性测试(token 对单节点参考、角色轮转计数、零冗余证明=每节点每 token 恰好 1 次 forward 份额)+MPS 双节点+跨节点 CUDA+HIP 冒烟。前置:无(任务C已闭环)。
[2026-07-28 细化] 子步分解:D1 单包轮转 attention(ring.rs,不动 model 层间)→ D2 finisher 就地续层(model.rs decode 期事件循环化,最大风险点)→ D3 采样轮转+coordinator 退位 → D4 验证阶梯(mock→MPS→跨节点 CUDA+HIP)。前置:任务E(plugin successor-seeded)先行预演 owner-最后归并。
[2026-07-29 计划修订] task E 已因 owner-return N-hop 下限被拒绝,不再作为前置。D1 单包 attention 与最小 D2 层间 continuation 合并为首个可运行垂直切片:coordinator 暂保留 token 广播,所有 worker 进入 collective decode;环内单包 N-1 hops,finisher 续层,末层唯一 logits producer。之后再做 batch 隔离、采样自治和 coordinator 退位。
[2026-07-29 最终策略与实施计划] KV placement 改为显存 hard bound 内 bounded compute balance，容量墙处退化 pure capacity；二维 assignee=(request phase+position+layer)；每 request 独立异步 packet pipeline；详细 TDD 计划见 docs/plans/2026-07-29-self-driving-ring-decode-implementation.md，按 Task 1-14 和 R1/R2/R3/R4 checkpoint 执行。
[2026-07-30 范围收缩] 当前只按 task-self-driving-minimal-core-20260730 实施最小真实核心切片；14 Task 生产化路线已 superseded。未经用户明确要求，不前置生产级能力。

_updated: 2026-07-29 17:24:29_
### 不为形式上的无 owner 强制轮转 sampler;以 KV/计算瓶颈为判据

type: `preference` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `user-direction-2026-07-29`

用户确认:若 L mod N=0 导致末层 sampler 对单请求固定,只要它不造成显存或关键计算瓶颈即可接受。设计含义:kv_assignee 必须与 sampler 解耦;默认保持零 token-boundary handoff 和 L*(N-1) attention hops。多请求用 request phase 分散 sampler;异构时可把 sampler 偏向更快节点。仅在实测 LM-head/sampling queue 成为吞吐瓶颈时启用 +1 token-ID phase shift。

_updated: 2026-07-28 17:26:22_
### 决策:禁止星形传输——decode 只走 ring;prefill 只走邻接累积转发;拓扑成本必须线性

type: `decision` · status: `held` · confidence: 0.95 · importance: 0.95 · source: `user-direction`

用户方向(2026-07-27):N>3 时设备间可能互相不可直达,星形传输(owner 直连全部 peer)要求全连通,拓扑成本 O(N) 连接/节点且依赖直达性;ring 模式每节点只连 2 个 peer,拓扑成本线性,不可直达的节点经环上中继可达。
执行:
1. decode 删除星形 HTTP 路径(PartialAttentionService/merge_remote_partial/transport=http),P2P TCP 环成为唯一 decode 传输;validate_ring_decode_split.py(星形验证器)退役删除,p2p 验证器覆盖全部判据;
2. prefill 保持并强化邻接累积转发(ringc 机制):consumer 只从物理前驱拉取累积前缀(url 列表可全指向前驱),不直连远端 producer——三机驱动改为 C 仅从 white 拉 c0+c1(laptop→white→pearl 中继);
3. belief-two-peers-topology-20260727 从"最小充分"升级为"硬约束":不只省连接,更是部分可达网络下的唯一可行拓扑。

_updated: 2026-07-27 15:43:05_
### 决策:vLLM prefill 改逐层流式 staging(窗口 W=2)+decode-ring 在有前缀时成为默认;legacy owner-collapse 须显式开启

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `user-direction`

动机剖析六问(任务A,修 ISSUE-003+005):
1. 问题:尾节点 prefill 前一次性 stage 全量前缀(24层×全部 chunk),瞬态显存峰值=本 chunk+全量前缀,显存切分瞬态失效,被用户定性为工程欺骗。
2. 现状:connector start_load_kv 在 forward 前 bulk 循环拉取所有层入 GPU staging,wait_for_layer_load 空实现;staging 按请求生命周期常驻(legacy)或 decode 开始才释放(decode-ring)。vLLM 调用时序已核实:start_load_kv 在 forward 前调一次(kv_connector_model_runner_mixin.py:102),wait_for_layer_load(layer) 在每层 attention 入口(kv_transfer_utils.py:51)——逐层流水线是 API 设计意图。
3. 目标态:任一时刻 GPU 瞬态 staging ≤ W×num_chunks 层(W=2);拉层 L+W 与算层 L 重叠(connector 后台 fetch 线程+按层 event);STAGING_STATS.max_staged_layers ≤4(2层×2chunk)作为有界探针;token 与参考一致;dsplit6/p2p3/p2p3n 回归全过。
4. 别人怎么做:vLLM connector API 注释明示 "useful for layer-by-layer pipelining"(base.py:317);Rust 线 micro-block 流式(ring.rs:665-733)是同族参照;OffloadingConnector 等官方 connector 也用异步分层加载。
5. 我们怎么做:connector 后台 fetch 线程按层序拉取(层内多 chunk 同层同批),wait_for_layer_load(L) 等 event[L] 并释放所有 <L 的 GPU staging(磁盘 store 不动,re-serve/partial 服务不受影响);decode-ring 在有前缀 chunk 时默认开启(env 默认改 1);HCP_RING_DECODE_RING=0 显式选择时保留 bulk staging+警告(对照用,明确标注非显存切分)。
6. 为什么:语义完整性——瞬态也有界才是真显存切分;vLLM API 的逐层钩子本就是为此设计,顺 API 意图而非对抗。
牺牲四问:
1. 默认为什么存在(bulk staging):实现最简,一次拉取后任何时刻可用;decode 复用 staging 零重取。
2. 牺牲什么:legacy owner-collapse decode 成为显式选项(默认路径改变);流式引入 fetch 线程/事件同步复杂度;若网络慢于单层计算,prefill 会 stall 在 wait(暴露真实网络下限,这正是论据)。
3. 被牺牲者用途:bulk 模式在快网络下延迟最优、代码最少。
4. 对本项目意义:正确性语义 > 实现简洁;stall 暴露正是 CXL 论据;legacy 保留为显式对照。结论:implement。

_updated: 2026-07-27 15:26:00_
### 决策:decode 增长按全局位置轮转分片(p mod N),RPC 捎带保序;策略函数作为计划层可换缝

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `user-direction`

动机剖析六问:
1. 问题:decode 增长 KV 全部落在 owner(dsplit4 的 528=c2 512+增长 8),请求内增长非均衡;长 decode 下 owner 显存先撞墙,"显存压力可计算"只覆盖 prefill 不覆盖增长,全局显存切分语义不完整。
2. 现状:dsplit4 已验证 decode 期前缀显存切分(staging 于 decode 开始释放+累积器绕环 RPC,168+168 跳);但 engine 无条件把每个新 token 的 KV 写进 owner 池,其他节点增长份额为 0。
3. 目标态:decode 增长也按环分片——每节点持久 KV=自己 chunk+自己增长份额(token 按全局位置轮转指派 p mod N);owner 池只写 c2,增长走 backend 紧凑 buffer,非自有 token 仅本步瞬时参与(causal 要求)后捎带给指派节点;token 与单节点参考一致;可验证:slots_written≈c2 级、A/B growth 计数符合轮转、token 一致。
4. 别人怎么做:Ring Attention 论文/Striped/ZigZag 面向静态序列无增长概念;TP 按头分片天然均衡(代价每层 all-reduce,绑 NVLink);vLLM P/D 全量搬移无切分语义;推理 CP 多在 NVLink 域内绕开。慢互联+序列维切分的 decode 增长放置是主流空白,无可直接复用机制。
5. 我们怎么做:复用 decode-ring 累积器 RPC 通道捎带上一步增长 KV(append-then-serve 一次 RPC 保序,边际跳数为 0);owner 本地 partial=池内 c2+紧凑 growth buffer+当前 token(瞬时);指派策略默认 p mod N 轮转,HCP_RING_DECODE_GROWTH=rr|owner 策略函数独立作为计划层可换缝(owner=退化为旧行为,供对照);HCP_RING_DECODE_RING=1 仍为总开关(0=legacy owner-collapse 保留)。
6. 为什么:语义统一优先——prefill/decode/增长全程每节点只持自有份额,显存压力全程可计算(chunk+growth/N);捎带保序使额外跳数为零;慢 decode 本身即 CXL 论据,局部性不是本项目目标。
牺牲四问:
1. 默认为什么存在(增长留 owner):engine 无条件写池最自然,decode 局部性最好、零额外通信、实现最简。
2. 牺牲什么:owner 的 decode 局部性——非自有 token 的 KV 需外流(捎带在已有 RPC 上,边际跳数≈0),owner 每步多一次小 cat。
3. 被牺牲者的用途:局部性省通信、省代码,是 latency 优先系统的直觉解。
4. 对本项目意义:实验级研究项目,语义完整性(全局显存切分)优先于 decode 局部性;捎带设计使代价近似为零;保留 owner 策略开关可随时退化对照。结论:implement。

_updated: 2026-07-26 05:10:16_
### 决策:decode 也必须显存切分(累积器绕环),decode 慢作为 CXL 论据接受

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.95 · source: `user-direction`

用户方向(2026-07-25):decode 若有一个 worker 常驻全部 KV,ring-attn CP 的根本目的就被破坏;decode 要和 prefill 一样显存切分,显存压力才可计算、capacity-aware 分配才成立;带宽压力(慢 decode)正好作为异构高带宽(CXL)需求论据,实验阶段可接受。
动机剖析六问:
1. 问题:decode 阶段 owner 常驻全量前缀 KV(staging 按请求生命周期持有),显存切分失效,显存压力不可计算,capacity-aware 无法成立。
2. 现状:prefill 显存切分(relay 链+瞬时 staging)已三机验证;decode 塌缩为 owner 单节点(0 跳/全量常驻),语义不对称。
3. 目标态:decode 全程显存切分——owner 只持自己 chunk+decode 增长;各节点从 store 加载本 chunk 快照,服务部分 attention;online-softmax 累积器沿环归并(L×(P-1) 跳/token);token 与单节点参考一致;decode 慢可接受。
4. 别人怎么做:真实 CP 系统压小 P、NVLink 域内环、或 decode 降级并行度(重分片/切回单卡)——多数是绕开而非保持切分;vLLM P/D 分离是全量搬移,无此语义。
5. 我们怎么做:prefill 不变;owner prefill 后释放 staging;decode 各节点用 store 快照(免 pool 生命周期问题)起 partial-attention 服务;owner 每层先算本地部分(含 decode 增长,causal)再沿环 RPC peer 归并(ring_attn_with_lse/merge_attn_states 原语复用);HCP_RING_DECODE_RING=1 开关,默认关保持兼容。
6. 为什么:语义完整性优先——显存切分在 prefill/decode 对称,显存压力才可计算;慢是论据不是缺陷。
牺牲四问:
1. 默认为什么存在(owner 汇聚):decode 0 跳最快、省网络,是 latency-bound 阶段的工程直觉解。
2. 牺牲什么:decode 延迟(L×(P-1) RTT/token,比单机慢几个数量级)。
3. 被牺牲者的用途:低 decode 延迟是生产服务质量指标(TTFT/TBT)。
4. 对本项目意义:HCP 处于研究/论据阶段,语义完整性与 capacity-aware 可计算性优先;decode 慢本身即 CXL 必要性证据。结论:implement。

_updated: 2026-07-25 18:47:17_
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
### 层数与节点数模数共振可能形成 sampler 计算热点,但不破坏冻结 KV quota

type: `risk` · status: `open` · confidence: 0.98 · importance: 0.9 · source: `docs/plans/2026-07-29-self-driving-ring-theory.md`

当 L mod N=0,finisher->next starter 的零额外跳策略使单请求 token sampler 固定。它不改变 kv_assignee,因此不增加 durable KV;只固定承担 final norm/LM-head/sampling 与 O(batch*vocab) 瞬时 logits。默认缓解:request 初始 phase 在节点间分散;异构时可偏向更快节点。只有实测 sampler queue/利用率成为吞吐瓶颈时才启用 k-hop token phase shift;要求全节点轮转时选择 gcd(k-L,N)=1 的 k。L=24,N=3 可用 k=1。验证覆盖 L mod N=0,统计每请求与全局 sampler 分布、LM-head 时间和 queue wait。

_updated: 2026-07-29 05:05:41_
### 决策:Rust decode 改 Q+LSE 累积器环;增长按 p mod N 分片且零传输(全节点同算 forward,非自有即弃)

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `user-direction`

动机剖析六问(任务C,修 ISSUE-007+008):
1. 问题:Rust 线 decode 每 token 每层把整个 prefill KV 分区重发一遍(O(seq×d)/层/token,ring.rs:399/457-466),且增长 KV 全节点复制(cache.rs:55-69)——显存切分对增长失效,通信模式正是 Q-ring 要消除的。
2. 现状:coordinator 广播 token,所有 worker 对称跑完整 1-token forward;ring.rs 复用 prefill 的 KV-block 环(seq_len==1 时只发 prefill 分区,2048/块);增长 append 到每节点本地 cache;logits 全节点冗余,coordinator 取 worker 0。
3. 目标态:decode 每 token 每层每节点只传 (Q,O,LSE)(O(d));增长 KV 按全局位置 p mod N 指派,因所有节点本就同算 forward,新 K/V 各节点自算自留(自有即留,非自有即弃)——增长零传输;token 与单节点参考一致;1M 式显存切分对增长成立(每节点 prefill/N + growth/N)。
4. 别人怎么做:与 plugin 线同源(LoongServe 式传 Q 不传 KV);Rust 线协议 protocol/node.rs 已预留 SoftmaxState 消息语义(未使用);plugin 线 p2p3n/conc2b 已三机+并发验证该数学。
5. 我们怎么做:ring.rs 增加 seq_len==1 的 Q-ring 路径:本地 partial(本 chunk+本增长+当前 token,因果尾)后,按既有环收发 N-1 轮"收(Q,acc)→本地 partial 归并→转发"(与 KV-block 环同构的控制流,payload 换 (Q,O,LSE));tch_backend/cache 按 p mod N 决定 append/丢弃;新消息类型(3 张量+scale)。
6. 为什么:与 plugin 线共享同一设计控制层语义;Rust 线全节点同算 forward 使增长分片零传输(比 plugin 线的捎带更简),这是该架构的独有红利。
牺牲四问:1) 默认为什么存在(decode 重发 KV):复用 prefill 环控制流,改动最小;2) 牺牲什么:无正确性牺牲;冗余 logits 计算保留(另行评估);3) 被牺牲者用途:省一次消息类型设计;4) 对本项目:decode 通信是 CXL 论据核心变量,必须最小化。结论:implement。

_updated: 2026-07-27 18:36:34_
### 决策:decode 并发按请求全隔离——跳池门走 forward_context 元数据,growth packet 带 req id,peer growth buffer 按 (req,layer) 键

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `user-direction`

动机剖析六问(任务B,修 ISSUE-004):
1. 问题:decode 并发(≥2 请求)时增长分片失效——跳池门是 n==1 启发式(批量 decode 步 n>1 时增长写回 owner 池);更隐蔽的是 peer 的 RingDecodeNode._growth 只按层键,两请求的增长 KV 混在同一 buffer,partial attention 会吃进别的请求的增长 token(正确性 bug,不只是显存问题)。
2. 现状:DECODE_RING_MAP/dr/growth/pending 已按 first_block_id 隔离(forward 逐行查 dr);但跳池门无请求身份;ring packet 无请求标识,peer 无法区分增长归属。
3. 目标态:2 请求并发 decode(同 chunk 集,不同 prompt)token 与各自单节点参考一致;skipped=2×168(每请求每步每层都跳);BATCH_STATS.max_reqs=2 证明真同批;peer growth 按 req 统计隔离;staging 窗口上界随流数线性(2 流≤12);streaming prefill 双流并行。
4. 别人怎么做:vLLM 原生 per-request KV 天然按请求隔离(block table);LoongServe/TP 的 decode 批处理同样按请求组织 KV;无直接可复用的"按请求跳池"先例(这是 HCP 特有的显存切分记账)。
5. 我们怎么做:do_kv_cache_update 经 vllm.forward_context.get_forward_context() 取 attn_metadata(query_start_loc/seq_lens/block_table),逐请求判定 tq==1 且 first_block 已注册 → 按 qsl 索引精确跳过该请求的槽位(废除 n==1 启发式);ring packet 增加 req 字段(first_block_id 字符串),RingDecodeNode._growth 改 (req, layer) 键,统计按 req 输出;dr/growth/pending 维持 per-request 不变;ring 遍历天然串行化(backend 逐行处理,ring_decode_step 阻塞),wire 上无交错。
6. 为什么:按请求隔离是并发 CP 的唯一正确语义;用框架已有的 forward_context 元数据而非自创启发式(教训:n==1 这类启发式在批量场景必破)。
牺牲四问:1) 默认为什么存在(n==1 门):单请求 PoC 时最简;2) 牺牲什么:do_kv_cache_update 每次多一次 metadata 解析(小);3) 被牺牲者用途:启发式省一次上下文查找;4) 对本项目:并发是 PoC 最小要求,精确性必须。结论:implement。

_updated: 2026-07-27 17:24:11_
### 后续路线:N>2 真 ring(三机:white CUDA + pearl ROCm + laptop 4060 CUDA)

type: `task` · status: `ongoing` · confidence: 0.8 · importance: 0.9 · source: `user-direction`

基于三步收官后的现状,N>2 ring 的技术改动已定位为三处小改:
1. connector 加 ring_role=relay:中间节点同时标前序 external(get_num_new_matched_tokens)并存自己 chunk(build_connector_meta 的 store/load 两条路径本来就分开),就绪状态级联;
2. backend 多 peer 合并:N-1 个 peer chunk 是全局连续前缀,cat 成一段连续 KV 做一次 peer pass (线性拷贝,数学等价),merge_attn_states 调用不变;
3. hcp_ring 每请求参数加 chunk_ids(复数,向后兼容追加);staging 已按 chunk 键,请求→chunk 映射从单值变列表。
验证拓扑两步走:
(a) white 当 relay(吃 c0 产 c1) + pearl 当 consumer(2 前缀 chunk)——证明 N>2 机制,不依赖 Mac;
(b) 三机真异构:laptop(Mac)需自建 vLLM CPU(无 macOS wheel,VLLM_TARGET_DEVICE=cpu),担任 chunk0 producer(不吃前缀、计算量最小)——满足"每平台必须跑 worker"纪律的最小可行角色。
前置:pearl 恢复可达,完成 task-gfx1200-repo-extraction。
[2026-07-24 更新] 前置已清(gfx1200 repo 完成,双机迁移复验 PASS)。用户确认下一步:laptop 节点并入 HCP ring,进入真 ring 阶段;三节点 worker 同级(peer),coordinator 默认 white 但位置解耦(见 decision-coordinator-placement-20260724)。实施顺序仍按 (a) 双机 relay 机制验证 → (b) 三机真异构。laptop 侧工作:自建 vLLM CPU(VLLM_TARGET_DEVICE=cpu) + 安装 hcp-vllm-plugin,担任 chunk0 producer。
[2026-07-24 修订] 用户更正 laptop 硬件:RTX 4060 Laptop 真 CUDA,非 Mac/CPU-only,"自建 vLLM CPU 担任 chunk0 producer"的最小可行角色方案作废(见 decision-n3-direct-20260724)。两步走 (a)→(b) 同时作废:直接三机真 ring,relay/多 peer 机制按通用 N 实现,N=2 为退化兼容。三处插件改动定位不变。
[2026-07-25 更新] laptop 环境就绪(ev-laptop-env-ready-20260725):vllm 0.23.1rc1 源码编译 + 插件 compat 6/6 + GPU smoke 通过,模型已同步。三节点环境条件齐备,可开始插件侧通用 N 实现(relay role/多 peer 前缀合并/chunk_ids 复数)。
[2026-07-25 里程碑] 三机真 ring 验证通过(ring3-033045):laptop A + white B relay + pearl C,token 与单节点参考一致。本任务的核心目标达成。后续开放项:capacity-aware 不均等三档分片(4060 8GB/9060XT 16GB/4090 24GB)、更长 seq(16k+)、多并发 CP 在 N=3 下的复验、coordinator 与插件 ring 的关系统一。
[2026-07-25 环闭合里程碑] 统一 ring 角色 + 邻接累积转发 + 轮转放置三机验证通过(ringc-160010):(N+1)%N 消费关系字面成立,真环成型。后续开放项:capacity-aware 不均等分片、更长 seq、compute/comm overlap(邻接传输已铺路)、N>3 扩展。

_updated: 2026-07-25 08:04:18_
### 决策:laptop 实为 RTX 4060 Laptop 真 CUDA 节点;直接做三机真 ring,框架按通用 N 实现

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `user-direction`

用户更正与方向(2026-07-24):
1. 硬件更正:laptop 节点具备 RTX 4060 Laptop 显卡,可运行官方 vLLM+CUDA。此前"laptop=Mac、无 macOS wheel、只能自建 vLLM CPU"的假设作废;任何设备都不再需要 vLLM CPU 路径。
2. 路线修正:跳过原 (a) 双机 relay 先行验证步骤,直接做三机真 ring(white CUDA + pearl ROCm + laptop CUDA)。
3. 通用性要求:N>2 框架按通用 N 实现,双机(N=2)只是退化情形——现有双机配置(white producer + pearl consumer)在通用框架下应保持可用,不另开专用代码路径。
不变的部分:插件三处改动定位不变(connector ring_role=relay、backend 多 peer 连续前缀合并做一次 peer pass、每请求 chunk_ids 复数),但 relay/多 peer 路径直接按通用 N 写。chunk 分配按 capacity-aware 三档(white 4090 24GB / pearl 9060XT 16GB / laptop 4060 8GB),具体比例实施时定。

_updated: 2026-07-24 08:43:31_
### 决策:三节点 worker 同级;coordinator 默认放 white,但位置与 worker 拓扑解耦

type: `decision` · status: `held` · confidence: 0.9 · importance: 0.9 · source: `user-direction`

用户方向(2026-07-24):laptop(Mac)节点并入 HCP ring 后即进入"真正的 ring"阶段。拓扑原则:三个节点(white CUDA / pearl ROCm / laptop CPU)上的 worker 是同级别 peer,无主次之分,各自持有自己 chunk 并参与 ring 交换。coordinator(控制面:tokenizer 分片、token 广播、chunk 分配)默认部署在 white——考虑其 CPU 与内存资源最丰富;但 coordinator 只做控制面、不做模型计算,架构上可放任意节点(包括 laptop),位置选择是部署问题而非拓扑约束。与既有纪律一致:每个平台必须跑至少一个 worker,coordinator 不计入异构计算能力验证。

_updated: 2026-07-24 08:34:57_
### 决策：两个产品级产出解耦为独立 GitHub repo(private),主仓保留研究/驱动/知识库

type: `decision` · status: `held` · confidence: 0.95 · importance: 0.9 · source: `user-direction`

用户方向(2026-07-22):HCP vLLM 插件与 gfx1200 适配是两个不同生命周期的产出,独立成 repo 比保留子文件夹更清晰。
执行:
1. hcp_vllm_plugin/ 经 git subtree split 带全部 23 个 commit 历史切出 => github.com/stark-sim/hcp-vllm-plugin(private, main);clone 验证(文件/历史/语法)通过;
2. 主仓删除 hcp_vllm_plugin/ 避免双源漂移,新增根 README.md 仓库地图;跨节点驱动脚本改为读 *_PLUGIN_REPO(默认 /home/stark/hcp-vllm-plugin);
3. white 已迁移:/home/stark/hcp-vllm-plugin clone + pip install -e 重装,import 验证通过;
4. 第二 repo(gfx1200 适配)待 pearl 恢复后从 /home/stark/vllm 源码树整理补丁。
主仓定位:Rust/Python 调度核心、transformers 线、跨节点驱动、graph-memory、docs/reports。

_updated: 2026-07-22 07:10:03_
### 工作方式规则：任何工作开始前先做动机剖析六问

type: `preference` · status: `held` · confidence: 0.95 · importance: 0.9 · source: `user-direction`

用户确立的通用工作方式(2026-07-21，适用于优化工作与普通工作)：开始任何一项工作前，必须先能回答六个问题，并把答案写进对应 decision/task 节点的 content(或 commit message)：
1. 面对什么问题——要解决的问题/缺口是什么；
2. 现状是什么——当前代码/系统处于什么状态，为什么不够用；
3. 做完能怎样——完成后的目标态与可验证标准；
4. 其他人怎么做——生态/同行(特别是 vLLM)遇到同样或类似问题时的解法，能否直接复用；
5. 我们怎么做——本项目采用的具体方案；
6. 为什么我们要这么做——相对第 4 问的现成方案，我们的方案差异在哪、为什么差异是必要的。
扩展规则：若工作属于优化/做减法类(丢弃现有行为换速度/显存/简洁)，在六问之外追加牺牲四问(为什么默认存在/牺牲了什么/被牺牲者的用途/对本项目的意义)，并给出 implement/defer/reject 结论；reject 也要记录，避免同一想法被重复提出。
全局沉淀：该方法论已融入 graph-memory skill 的 "Pre-Action Motivation Analysis" 一节(含六问→节点/边的映射：DEPENDS_ON 记顺序、belief+证据记外部做法、GOVERNS 关联规则与应用)。原 optimization-trade-off skill 已按用户决策退役(移入 _removed)，其牺牲四问作为扩展条款并入；项目 AGENTS.md 对应章节已同步改为动机剖析六问+牺牲扩展。

_updated: 2026-07-21 13:48:11_
### 下一步顺序：1) 双平台 flash_attn 2) decode 充分验证(continuous batch+多步) 3) 异构跨节点切分 CP

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `user-direction`

用户确定的 vLLM 线后续三步顺序：\n【第1步】flash_attn 在 CUDA 和 ROCm 都接通。理由：flash_attn2 算法本身不绑定特定硬件，CUDA 有官方实现，ROCm 有 ROCm/flash-attention fork。目标：white(CUDA) 与 pearl(ROCm gfx1200) 都能用 flash_attn（及其 LSE 输出），让 ring backend 的 attention 从 plain-PyTorch 升级到 flash_attn。\n【第2步】decode 阶段更充分验证：continuous batching（多并发请求）+ 多步 decode，证明在接入 ring backend / 插件后，vLLM 的常规基础能力（连续批处理、多步解码）仍正常。\n【第3步】异构跨节点切分 CP：white(CUDA producer) + pearl(ROCm consumer) 跑通显存切分 context-passing CP，这是整个 vLLM 线可行性的关键收尾点，必须做到异构跨节点。
[2026-07-21 更新] 三步全部完成：1) flash_attn 双平台(white vendored FA 含 LSE / pearl TRITON_ATTN+CUSTOM)；2) decode 充分验证（连续批 6 请求、多步 decode=8/16 全过）；3) 异构跨节点切分 CP（ringx-210415 PASS，见 ev-ring-cross-node-split-cp-20260721）。

_updated: 2026-07-21 13:08:24_
### 决策：ring backend 接 KV connector 时必须区分“全量搬移”与“切分瞬时”

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `user-direction`

用户明确指出：KV connector 默认语义是 disaggregated prefill 的整段 KV 搬移（把某请求的完整 KV 从一处全量复制到另一处），而 HCP ring attention 的场景是切分后的 KV——每个 worker 只永久持有自己 chunk 的 KV，peer chunk KV 只是 attention 时瞬时借用、用完即弃。因此接线原则：1) connector 调度侧仅用 get_num_new_matched_tokens 把前序 chunk 标记为 external，从而给本 chunk 提供全局 RoPE 位置（并阻止本 worker 重复计算前序 chunk）；2) connector worker 侧 start_load_kv/wait_for_layer_load 把 peer chunk KV 拉取后写入 ring backend 的 PEER_KV_STAGING（瞬时），绝不写入常驻 paged pool；3) ring backend 用 online softmax 合并 local（本 chunk，causal）+ peer（前序 chunk，transient，non-causal）。这样 worker 常驻 KV 只有自己 chunk，peer KV 瞬时，实现显存切分而非全量复制。

_updated: 2026-07-17_
### Block KV cache + vLLM 集成：插件解耦 vs HCP 内联 PageAttention 双路线

type: `hypothesis` · status: `ongoing` · confidence: 0.65 · importance: 0.9 · source: `user-direction`

vLLM 与 HCP Ring Attention 的融合路线已确定为：不改 vLLM attention kernel，以 vLLM physical block 为粒度做跨节点 KV 交换（plugin 路线）。\n\n已完成：\n1. 分析 vLLM 0.6.4 CacheEngine 结构。\n2. PoC 验证 KV block 提取与重新写入可行。\n3. 撰写 docs/VLLM_BLOCK_RING_PLUGIN.md 设计文档。\n4. 搜索确认无现成 vLLM gfx1200 wheel；正在 pearl 上用 TheRock gfx120X-all nightly + 源码编译 vLLM 0.6.4。\n5. 实现 VllmBlockRingPlugin 骨架：prefill/decode 直接调用 model_executor，block 提取/插入，combined block table。\n6. 修复 PoC decode 语义：使用最后 prompt token 作为 decode 输入，同步全局 tokens。\n7. 修复跨层 block id 一致性：为 peer KV 在所有层复用同一组物理 block。\n8. 增加 RoPE 位置校正：对 local-position 预fill 的 peer key 做 delta 旋转，使合并后 decode 的 RoPE 位置对齐。\n\n进行中：\n- pearl 上 vLLM 源码编译（当前在下载 rocm_sdk_libraries-gfx120X-all）。\n\n下一步：\n1. 等待编译完成，验证 `python -c "import vllm"`。\n2. 在 pearl 上运行单进程 PoC，确认 distributed decode token 与 reference 一致。\n3. 跨节点 2-worker PoC：white vLLM CUDA + pearl vLLM ROCm。

[2026-07-17 更新] pearl 上 vLLM 0.23.1rc1 源码编译成功并通过 gfx1200 prefill；V1 引擎版插件 hcp_vllm_block_ring_plugin_v1.py 已实现并在 pearl 单进程 PoC 验证（prefill/decode 与单节点参考一致）。社区 lemonade portable 已确认 ABI 不兼容弃用。

_updated: 2026-07-02 14:58:04_
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
### [论文] Striped Attention: Faster Ring Attention for Causal Transformers

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `https://arxiv.org/abs/2311.09431`

作者：William Brandon 等 (MIT)，arXiv:2311.09431，2023。
核心发现：因果 attention 的三角结构导致 Ring Attention 工作负载不均。
方案：每个 device 持有均匀分布在整个序列上的 token 子集（striped permutation），而非连续 chunk。
效果：A100 256K 序列上端到端吞吐提升最高 1.45×；16×TPUv4 786K 序列上 1.65×。
实现复杂度：只需在 forward 开始前对输入序列做一次 permutation，并调整 attention mask 结构。
与 HCP 相关性：直接相关，可能缓解 pearl 等小/慢 domain 在 Phase 2 成为瓶颈的问题。

_updated: 2026-06-29 06:06:09_
### 任务E:plugin 线 successor-seeded 优化(owner 最后归并,每层 N 跳→N-1 跳)

type: `task` · status: `rejected` · confidence: 0.99 · importance: 0.85 · source: `user-direction`

decision-self-driving-ring-20260728 的 plugin 受限落地。改动:ring_decode_step 种子改为 q-only(中性累积器或首跳播种标志),owner 在收包后本地 partial 最后归并;hop 数每层 3→2(N=3),每 token 72→48。验证:p2p 单机+三机回归(判据不变,token 一致,hop 计数变为 N-1)。小改动,不动 vLLM 行为。
[2026-07-29 可行性审计] REJECTED:固定 owner-return 的单向环必须走 owner->successor->...->predecessor->owner,物理下限为 N 条边。q-only seed/owner 最后归并不减少边数。详见 revision-self-driving-ring-plan-20260729。

_updated: 2026-07-28 17:19:59_
### decode 通信策略分析:当前 owner 汇聚(0 跳/全量 KV 常驻) vs 累积器绕环(L×P 跳/语义保持),选择由互联延迟决定

type: `belief` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `user-direction + analysis`

用户提出的 decode 纯 P2P 方案(累积器 (o,m,l) 绕环归并,双向减半,树形 log P)分析确认:
1. 插件现状:前缀 KV prefill 时一次 stage 后按请求生命周期常驻,decode 每 token 0 网络跳——属"decode 降级回单卡"家族的隐式版本,延迟最优;代价是 owner 在 decode 期持有全量前缀 KV,显存切分在 decode 阶段失效(PoC 规模无碍,1M 规模撞墙)。
2. 累积器绕环方案语义保持(每节点只持自己 chunk),原语已齐(ring_attn_with_lse 的 LSE + merge_attn_states 两路归并),缺 per-token per-layer 绕环编排。但跳数是 L×P 而非 P:第 L 层 Q 依赖 L-1 层全局归并完成,归并无法跨层流水(24层×3节点=72跳/token;双向 36;树形 ~48)。
3. 延迟测算:tailscale 125ms RTT → ~9s/token(不可用);2.5G 有线 ~0.2ms → ~14ms/token≈70tok/s(小模型可用);CXL/RDMA μs 级 → 可忽略。
4. 结论:prefill 带宽敏感、decode 延迟敏感,两角度同证 CXL/类 RDMA 是异构 CP 的前置条件(主线论据+1)。当前 staging 策略不变;accumulator-ring decode 列为开放实现项,触发条件=decode 期 owner 显存不足 + 互联足够快。

_updated: 2026-07-25 18:35:58_
### kernel-hardening backlog(性能/规模,非正确性;128K+ 启动时按序做)

type: `task` · status: `ongoing` · confidence: 0.85 · importance: 0.85 · source: `analysis`

第 2 步 kernel 化完成后,正确性层面无遗留;以下均为性能/规模项,128K+/1M 规模测试启动时按此顺序做:
1. local 段 paged 直读(中-高):现 forward 每请求每层先把本地 KV 从分页池 gather 成连续张量再进 kernel,128K/1M 规模下每层每步多出上百 MB 线性拷贝流量;应让 ring_triton_attn 直接吃 block_table 对 local 段直读(peer 段连续 staging 不变)。这是 PagedAttention 整合未完成的一半。
2. 长上下文 decode split-KV(中-高):decode(Tq=1) 现走 prefill 风格 kernel 沿 KV 串行扫;参照 vllm unified_attention 的 softmax_segm split-K 分段并行。1M 教训中 decode 本是瓶颈。
3. 批量 kernel 启动(中):per-request Python 循环,每请求每层 2 次 launch;local 段按 batch 合并 (cu_seqlens + block table) 一次 launch。纯性能。
4. N>2 真 ring 多路合并(将来):merge_attn_states 两路迭代可行;connector 每请求限一个 peer chunk (PoC 限制),配置面可向后兼容地追加 chunk_ids。v0.1 插件明确声明不支持。
不做这些的理由(当下):均为 kernel 内部实现或配置追加项,不影响 v0.1 插件对外配置面冻结;正确性已由三件套 + 16k + 跨节点验证覆盖。

_updated: 2026-07-22 06:05:13_
### TRITON_ATTN 是 ROCm/RDNA 上 flash attention 算法的原生路径(非降级替代)

type: `fact` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `code-reading`

[2026-07-21 源码+外部资料核实]
1. TRITON_ATTN(vllm/v1/attention/backends/triton_attn.py)是 vLLM 一等后端:prefill 用 context_attention_fwd、decode 用 unified_attention 两个 Triton kernel,直读 block_table paged KV,分块 tiling + online softmax——与 flash_attn 同算法类,kernel 语言不同(Triton vs CK/CUDA)。
2. RDNA 不走 flash_attn 包的根因是硬件矩阵指令集分裂:ROCm 的 flash_attn 包实体是 Composable Kernel tile kernel,专门为 Instinct/CDNA(gfx9, MFMA/matrix core, wave64)写(vllm#4514 原话);RDNA(gfx11/gfx12 消费卡)是 WMMA(AI acceleration, wave32),rocWMMA 文档支持矩阵分列两类指令集。CK kernel 不以 RDNA 为目标。
3. Triton 从高层 IR 编译,ROCm 官方 Triton 后端原生支持 gfx11/gfx12(pearl 为 triton 3.7.0+rocm7.13);vLLM ROCm 安装文档历来要求装 ROCm Triton flash attention。
4. 因此 rocm.py 的分层(gfx9→flash_attn 包/AITER,gfx1x→Triton)是硬件现实的直接映射;pearl(gfx1200)走 TRITON_ATTN 是设计意图。
对第 2 步的含义:pearl 的"原生 kernel"即这套 Triton kernel,kernel 化=复用它并取 LSE。

_updated: 2026-07-21 16:52:33_
### vLLM cascade/LSE 机制存在但平台分层：CUDA 有 vendored FA(含 LSE),ROCm/RDNA 走 Triton + merge_attn_states 算子

type: `belief` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `code-reading`

vLLM 的 cascade attention(共享前缀与各请求私有后缀分别算 attention 再按 LSE 合并)与 HCP ring backend 的 local(chunk B) + peer(chunk A) LSE merge 数学同构。但[2026-07-21 源码核实]平台能力分层：
1. "vLLM 内置 flash_attn" 仅覆盖 CUDA(vllm.vllm_flash_attn vendored kernel)与 XPU;ROCm 在 fa_utils.py 里是 try: from flash_attn import ...(依赖用户自装上游包),pearl 的 vllm-rocm env 无 flash_attn/aiter 包 => FLASH_ATTN 后端不可用;
2. vLLM 官方对 ROCm 分层(rocm.py):gfx9(CDNA) 预期 AITER FA / 上游 flash_attn 包;RDNA(gfx11xx/gfx12xx, pearl 9060 XT 为 gfx1200) 官方预期路径即 Triton 实现(注释原文);有 kv_connector 时 ROCM_ATTN 因 KV layout 不兼容被排除 => pearl connector 场景后端只有 TRITON_ATTN/CUSTOM;
3. TRITON_ATTN 后端 assert attn_metadata.use_cascade is False(不接 cascade),但 vllm.v1.attention.ops.merge_attn_states(含 triton 版)输入正是 (prefix_out, prefix_lse, suffix_out, suffix_lse),形状与 HCP merge 一致;
4. 推论(第 2 步平台策略):white 复用 vendored FA 的 LSE;pearl 用 triton kernel 算两段 + merge_attn_states——但该算子在 gfx1200 上须先做数值稳定性验证(HCP 团队曾在 ROCm 见过 inf),不可靠则保留已验证的 plain-PyTorch merge(3e-7)兜底。

_updated: 2026-07-21 16:42:06_
### 动机剖析六问能在行动前暴露顺序错误与现成轮子，值得作为默认动作

type: `belief` · status: `held` · confidence: 0.8 · importance: 0.85 · source: `experiment`

首次完整应用(2026-07-21，vLLM 线三个下一步)即产生两类实质收益：
(a) 暴露顺序错误——原记录顺序 1→2→3(插件化→kernel→staging)，剖析依赖后修正为 3→2→1(staging 是数据结构地基，kernel 化需按请求取 staging，插件配置面最后冻结)，避免返工；
(b) 暴露现成轮子——"别人怎么做"一问发现 vLLM cascade attention 与 HCP local+peer LSE merge 数学同构、AttentionMetadata/connector metadata 本就按请求组织，两步工作都可直接复用框架机制 而非自造。
代价：每项工作启动前增加约一次剖析的固定开销。对多步骤、跨系统的工作收益大于开销；对单行修复类琐碎工作可从简。

_updated: 2026-07-21 13:48:11_
### vLLM 官方长上下文分布路线是 disaggregated prefill(全量 KV 搬移)

type: `belief` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `code-reading`

vLLM 官方对"长上下文分布式"的答案是 P/D 分离：prefill 节点算完全量 KV，整体搬给 decode 节点。该路线每个节点都必须容纳全量 KV；HCP 切分 CP 不需要——各节点只持有自己 chunk 的 KV，peer chunk 仅以瞬时 staging 参与计算。这是 HCP 相对 vLLM 官方路线的差异化价值，也是三步工程化值得做的原因：把差异化的正确性证明变成差异化的可用能力。

_updated: 2026-07-21 13:28:03_
### 决策：flash_attn 用 vLLM 内置实现，不编独立 ROCm 包

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.85 · source: `user-direction + white/pearl flash_attn probe`

用户澄清：flash_attn 用 vLLM 内置实现即可。确认两端内置 flash attention 均可用：white（CUDA，vLLM 0.23）用 vendored vllm_flash_attn（_vllm_fa2_C），is_flash_attn_varlen_func_available()=True，flash_attn_varlen_func(..., return_softmax_lse=True) 返回 (out,lse)；pearl（ROCm gfx1200，vLLM 0.23）用内置 TRITON_ATTN（Triton 版 flash_attn2，架构无关），此前所有 vLLM PoC 在其上正常运行。放弃在 pearl 源码构建独立的 ROCm/flash-attention + aiter 包（CK 路径 arch 列表不含 gfx1200 只有 Triton/aiter 路径可行，aiter 嵌套子模块下载慢、github TLS 不稳、构建重且脆，投入产出不成比例）。结论：flash_attn 双平台已可用（vLLM 内置），ring backend 在 CUDA 侧可直接用 vendored flash_attn 的 LSE，ROCm 侧用 TRITON_ATTN/手动 LSE。

_updated: 2026-07-21_
### 偏好：多节点多库环境下的环境变量卫生规则

type: `preference` · status: `held` · confidence: 0.95 · importance: 0.85 · source: `user direction`

用户明确的环境变量治理规则，适用于 white(CUDA)/pearl(HIP) 异构环境：\n\n1. 永不全局 export LD_PRELOAD；仅在命令作用域使用（hiprun 交互，或脚本里 LD_PRELOAD=... ./binary 单行前缀）。\n2. 每个环境变量只在一个文件里设置：机器/设备级变量放 ~/.bashrc；函数/别名也放 ~/.bashrc（不继承，不能放 profile）。\n3. 改 LD_LIBRARY_PATH 用幂等追加，避免每次 source 叠层；可用 _ld_prepend 或临时命令前缀。\n4. 不要把大杂烩 lib 目录（miniconda3/lib、多套 torch）常驻全局 LD_LIBRARY_PATH，避免 libstdc++/libzstd/libtorch 互相顶。\n5. 可选：用 direnv(.envrc) 做更强隔离。\n\n已同步到 AGENTS.md。

_updated: 2026-07-01 04:49:12_
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
### KVConnectorBase_V1 是 experimental API，插件边界收敛才能跟进 vLLM 升级

type: `belief` · status: `held` · confidence: 0.9 · importance: 0.8 · source: `experiment`

vLLM 运行日志明示 "KVConnectorBase_V1. This API is experimental and subject to change"。HCP 对 vLLM 的依赖面 = attention backend 注册表(CUSTOM) + KV connector 接口两个扩展点。不收敛成干净插件边界，vLLM 升级可能悄悄破坏兼容性且无人发现；收敛后每次升级跑一遍兼容性验证即可。

_updated: 2026-07-21 13:28:03_
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
