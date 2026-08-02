# Active Context

当前活跃的任务、决策、风险和假设。

### 先补齐 Positioned KV 的 TCP/QUIC wire 合同

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmation-2026-08-03`

【动机六问】
1. 问题：continuation prefill 需要重用 decode 后按 layer assignee 分散、位置可能非连续的 KV；网络边界若丢失 position_ids，peer causal mask 会把这些 KV 错当成连续区间。
2. 现状：KvBlock 已有 optional position_ids，clone、LinkedMockKvTransport 和 HcpRingAttentionBackend 都会保留或消费它；但 TCP/QUIC 的 KV codec 只编码 K/V 与区间 metadata，接收端固定构造 position_ids=None。
3. 目标：通过 KvTransport trait 发送带非连续 Int64 positions 的 KvBlock 后，TCP 和 QUIC 接收端都获得相同 shape、dtype和值；测试包含 16,777,217 以证明未经过 f32；不带 position metadata 的旧 frame 仍解析为 None。
4. 他者：vLLM 等 serving runtime 通过 block table、slot mapping 和显式 token position 维护 paged KV 的逻辑位置，通常不把独立 KvBlock 绕 P2P ring 传输，因此其 runtime metadata 机制只能借鉴“位置必须显式”的原则，不能直接复用 codec。
5. 本方案：在现有 JSON metadata + raw tensor payload 中追加 optional position tensor 描述与原始 Int64 bytes；payload 顺序固定为 K、V、positions（若存在），TCP/QUIC 使用各自已有 tensor codec，接收时按字段存在性选择 Some 或 None。
6. 为什么：这是让既有 positioned attention 语义跨进程成立的最小改动，不改变 KV 归属、ring hop、调度或 backend API；直接改 runtime 会把 wire 丢失与 continuation 计算混在一起，无法独立定位正确性。
【兼容边界】保留缺失 position metadata 的旧 KV frame 为 None，不删除连续区间 fallback；本节点不承诺跨版本协议协商或生产级 schema versioning。
VERDICT: IMPLEMENT。

_updated: 2026-08-02 19:28:05_
### 真实 Qwen 两阶段请求接入 WorkerRuntime/coordinator

type: `task` · status: `planning` · confidence: 1.0 · importance: 1.0 · source: `user-selection-2026-08-03`

以小节点把真实 self-driving decode 与 positioned continuation prefill 接入 WorkerRuntime/coordinator。最终验收是同一 request 在 Mac MPS worker 与 white CUDA worker 上完成 initial prompt、两次 decode、continuation prompt、再两次 decode；coordinator 不做模型计算，新主路径不调用 legacy Decode。当前阶段限定单请求、固定两段 prompt，不接多请求 pipeline、HTTP 或生产治理。

_updated: 2026-08-02 19:01:54_
### 拆分推进完整 runtime 两阶段 self-driving 请求

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `code-audit+user-selection-2026-08-03`

【动机六问】
1.问题：Task 3c 只证明 SelfDrivingPacket 可跨主机 QUIC；真实 WorkerRuntime 仍走 legacy Decode，且同一请求第二段 prompt 会重置 KV，尚不能形成真实对话式请求。
2.现状：WorkerRuntime::Decode 调用 decode_request/LlamaModel::forward；prefill_request_with_reservation 每次创建新 cache；self-driving real-Qwen 只存在 test helper。更关键的是 decode 后 positioned KV 在各 worker 上非连续，现有 prefill ring 仍按 seq_offset 连续区间解释 peer KV，QUIC KvBlock 不传 position_ids。
3.目标：Mac coordinator + Mac MPS worker0 + white CUDA worker1 对同一 request 完成 prompt A initial prefill -> 两个 self-driving decode step -> prompt B continuation prefill -> 两个 self-driving decode step。两平台都执行真实 Qwen forward；KV history 不重建、不丢失、不重复；position union 完整；最终 token 与独立 contiguous reference 对齐。
4.他者：vLLM/主流 serving runtime 用显式 request state、prefill/decode phase、block table/slot mapping 和 scheduler command 驱动 worker；这些机制说明 phase 与物理 KV 位置必须显式，但其同构 collective 和 paged-attention worker loop不能直接承载 HCP 的异构 P2P ring 与 SelfDrivingPacket。
5.本方案：选择完整 runtime/coordinator 路径，但拆成小节点。先让 positioned KV position IDs 成为 prefill ring wire 合同并建立 continuation backend；再把 self-driving 单步执行提升为 WorkerBackend 能力；随后增加明确的 ContinuationPrefill/SelfDrivingDecode 控制协议与 worker phase loop；最后让 coordinator 生成容量/assignee plan 并做真实 Mac-white 两阶段请求验证。
6.为什么：专用 tracer 会再次把已证明算法留在服务边界之外；直接一次性改完则混合 KV 语义、peer event loop、控制协议和设备部署。分层推进既不让 legacy forward 决定新主路径，也能让每个 correctness 缺口单独 RED/GREEN。
【legacy 边界】本阶段不以删除旧命令为目标，避免把兼容性清理混入核心闭环；新两阶段请求不得调用 legacy Decode。旧路径保留为非主路径，待新路径稳定后另做带牺牲分析的删除决策。
VERDICT: IMPLEMENT。

_updated: 2026-08-02 19:01:54_
### Continuation prefill 必须先获得 positioned KV wire 语义

type: `risk` · status: `open` · confidence: 1.0 · importance: 1.0 · source: `code-audit-2026-08-03`

代码审计确认 continuation prefill 不能直接复用现有 forward：
1. TchWorkerBackend::do_prefill 会重建所有 KV cache，并把 model/global layer prefill state 清零。
2. LlamaModel::forward 在 is_prefill_done=true 时，无论输入 seq_len，都只构造一个 global_seq_len position，不能表达多-token continuation segment。
3. self-driving decode 按 layer assignee append 后，每个 worker 的 ReservedPositionedKvShard positions 不再是单一连续区间。
4. HcpRingAttentionBackend prefill 仍以 seq_offset/global_seq_start 解释整个 local K/V；KvBlock 虽有 position_ids 字段，但当前 QUIC/TCP KV codec 没有传递它。
因此直接把第二段 prompt 送进现有 Prefill/Decode 会重置或错误掩码。先建立 positioned KV wire + continuation segment forward 才能继续 runtime 接线。

_updated: 2026-08-02 19:01:54_
### 先建立传输无关的 SelfDrivingPacket 数据面合同

type: `decision` · status: `superseded` · confidence: 1.0 · importance: 1.0 · source: `user-confirmation+code-audit-2026-08-02`

【动机六问】
1.问题：真实模型 self-driving decode 与 TCP wire roundtrip 已成立，但独立 worker runtime 尚无法通过它实际使用的 transport 抽象收发 SelfDrivingPacket。
2.现状：WorkerRuntime 为每层持有 Box<dyn KvTransport>，部署数据面使用 QuicKvTransport；KvTransport 和 RingMessage 目前只支持 KvBlock/RingPacket。SelfDrivingPacket 的 send/recv 是 TcpKvTransport 的固有方法，QUIC 没有对应 variant、codec 或暂存队列；coordinator 的 Decode/DecodeBatch 仍逐 token 广播并调用 legacy decode_request。
3.目标：TCP 和 QUIC 都能经同一个 KvTransport trait object 无损传递 SelfDrivingPacket，保持 BF16 tensor dtype/值、Int64 position 和全部 route metadata；现有 KV/RingPacket 行为回归不变。定向 TCP/QUIC loopback 与相关 Rust 回归测试通过即完成。
4.他者：分布式推理通常先定义 rank/worker 间稳定的数据面消息合同，再让 scheduler 或 coordinator 启动请求；vLLM 的 worker/rank 生命周期可参考，但其同构 collective/内部消息不能直接承载 HCP 的 activation、Q、online-softmax accumulator 与逐层路由状态。
5.本方案：最小扩展 RingMessage、KvTransport、TcpKvTransport 和 QuicKvTransport，使 SelfDrivingPacket 成为第三类一等消息；复用现有 tensor wire 编码语义，只补齐 QUIC codec、分派、交叉暂存和 trait-level 收发测试。
6.为什么：这是从 Task 3b 到独立 worker 进程循环之间最小且真实的依赖。它不重做 attention/Norm/MLP/KV ownership 数学，也不提前引入 coordinator 协议、请求状态机、多请求或生产治理；失败时只会暴露 transport 合同问题。
【部署边界】coordinator 与 worker 正交共存：coordinator 负责 tokenize、capacity/reservation、请求启动/停止和结果收集；每个异构节点一个 worker 进程，worker 持有 backend、完整权重和本地 KV shard；decode 的逐层 hop 只经过相邻 worker，不回 coordinator。
VERDICT: IMPLEMENT。

_updated: 2026-08-02 12:05:44_
### Task 3c 从 loopback 完成条件修订为真实跨主机 QUIC

type: `revision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-correction-2026-08-02`

用户确认：HCP 小体量服务框架必须在现有异构节点上推进，不能把本地线程或 loopback 当作部署完成。因此 Task 3c 保持 transport-only 的小范围，但证据阶梯末端改为 Mac 与 white 两个独立进程的真实 QUIC 收发。本地测试仍保留，因为它能隔离 codec/trait 错误；它只是前置验证，不是最终结论。后续最小服务节点才接真实 Qwen prefill 与两个 self-driving decode token。

_updated: 2026-08-02 12:05:44_
### 以真实跨主机 QUIC 闭合 SelfDrivingPacket transport 合同

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmation+inventory-verification-2026-08-02`

【动机六问】
1.问题：Task 3c 需要建立独立 worker 可用的数据面合同；如果只做 loopback，仍不能证明该合同经过真实跨主机 QUIC stream。
2.现状：Task 3b 已证明真实 Qwen packet 的 TCP wire 语义，但 SelfDrivingPacket 尚不是 KvTransport/RingMessage 的一等消息，QuicKvTransport 没有 codec、分派或暂存；Mac 与 white 当前均可用，但还没有通过该合同交换 self-driving packet。
3.目标：RingMessage、KvTransport、TcpKvTransport、QuicKvTransport 统一支持 SelfDrivingPacket；本地 trait-object TCP/QUIC 测试保持 BF16 dtype/值、Int64 position 与全部 route metadata；随后 Mac 与 white 两个独立进程通过真实 QUIC roundtrip 同样通过。现有 KvBlock/RingPacket 回归不变。
4.他者：分布式 runtime 通常以稳定消息抽象屏蔽 TCP、QUIC、RDMA 等 transport，并用本地 codec 测试加跨主机 smoke 分层验证；成熟框架的 control plane 生命周期可参考，但其同构 collective 消息不能直接代替 HCP packet。
5.本方案：先用一个公开 trait 行为测试驱动最小接口，再为 TCP 与 QUIC 分别补齐发送、接收和交叉暂存；新增最薄的 transport smoke CLI 或复用现有远程 smoke 入口，在 Mac/white 间只发送一个确定性 packet 并回传，不加载模型。
6.为什么：这一步直接服务真实 worker 进程，又把失败面限制在 transport；若同时接 Qwen、coordinator 和请求循环，会把 codec、网络、模型状态和控制面四类故障混在一起。真实模型最小服务保持为紧随其后的独立节点。
【边界】不新增 coordinator command，不实现 worker request/event loop，不接 HTTP、多请求、重试、版本协商或生产治理；跨主机 smoke 不声明推理服务已经完成。
VERDICT: IMPLEMENT。

_updated: 2026-08-02 12:05:44_
### 远程节点连接方式以 infrastructure inventory 为唯一现行来源

type: `preference` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-correction-2026-08-02`

用户明确要求：远程节点的 SSH host、user 和可用平台从 ~/.agents/inventory.yaml 查询，不再把历史 LAN/Wi-Fi 地址写成项目规则或运行脚本默认值。文档可保留带日期的历史实验端点作为证据，但必须明确其为历史信息，不能指导当前连接。

_updated: 2026-08-02 11:12:40_
### 双线程 backend 不是独立 worker 部署闭环

type: `decision` · status: `rejected` · confidence: 1.0 · importance: 1.0 · source: `user-correction+code-audit-2026-08-02`

【动机六问】
1.问题：Task 3b 之后需要向真实 worker 进程接线，但候选方案把两个 backend 放进同一进程的两个线程，容易把线程隔离误认为部署隔离。
2.现状：Task 3a 已证明两个独立 backend 的真实 Qwen correctness，Task 3b 已证明每层 SelfDrivingPacket 真实经过 TCP；重复做线程版只新增共享进程内的调度，不补齐 runtime 使用的 KvTransport/QUIC 合同。
3.目标：下一节点必须直接消除独立 worker 接线的最前置缺口，并且仍然保持小步、可独立验证。
4.他者：成熟分布式推理系统通常由 coordinator/scheduler 负责请求控制，各 rank/worker 进程持有模型和本地状态；线程可用于本地测试，但不是跨节点数据面合同。
5.本方案：拒绝把 backend 移入双线程作为独立节点；保留已有线程测试只作为 synthetic 并发证据，不扩大其结论。
6.为什么：HCP 的部署约束是每个异构节点一个 worker，coordinator 只做请求级控制；下一步应服务这个边界，而不是再证明同进程组合。
VERDICT: REJECT。

_updated: 2026-08-02 10:39:00_
### 以持久 loopback TCP roundtrip 隔离真实 decode packet 的 wire compatibility

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmation-2026-08-02`

【动机六问】1.问题：Task 3a 的真实 decode correctness 已成立，但 LayerPacket 仍在同一进程中直接从一个 backend 传给另一个，尚未证明真实 BF16 activation/Q/O/LSE/position 与 route metadata 可经过 wire format。2.现状：现有 TcpKvTransport 已支持 SelfDrivingPacket，synthetic N=3 两-token线程测试也证明邻接 send/recv；真实 Qwen oracle 则覆盖24层、reserved cache和1:3 ownership，但未使用 decode transport。3.目标：保持 Task 3a 所有输入与不变量，每个 Forward packet 必须执行 into_self_driving_packet、send_self_driving_packet、recv_self_driving_packet、from_self_driving_packet；两 token 共48次 hop且每帧非零，数值和KV结果不变。4.他者：分布式推理 rank 通常以阻塞/异步 point-to-point send/recv 在独立进程传 activation 或 attention accumulator；本项目现有 synthetic threaded TCP 是同类最小先例，QUIC和runtime负责更高层生命周期。5.本方案：在 ignored真实 oracle helper 中建立一条持久全双工 loopback TCP连接，为两个 backend 各持一个 TcpKvTransport；仅在 Forward 分支做真实 wire roundtrip并累计hop/bytes，其余计算与断言复用Task 3a。6.为什么：该方案只增加真实模型 wire compatibility 这一项新证据，能把序列化错误与线程调度、coordinator和服务状态错误隔离；改用QUIC或独立worker loop会同时引入协议/并发未知量，新runtime adapter则是未被当前证据要求的抽象。VERDICT: IMPLEMENT。

_updated: 2026-08-02 06:51:53_
### 先以两个独立 backend 闭合真实 Qwen self-driving decode correctness

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `code-audit+user-confirmation-2026-08-02`

【动机六问】1.问题：真实 Qwen worker-local reserved prefill 与 self-driving decode 分别成立，但尚未证明真实 BF16 权重、24 层 prefill cache 和 packet continuation 能组合。2.现状：两个独立 backend 已按 1:3 持有真实 prompt KV；synthetic 模型已证明多层、多 token、reserved append 和 TCP ring；coordinator 仍走 legacy decode，直接接服务会混合多个未知量。3.目标：同一 request 从真实两-worker prefill 状态执行两个 decode token；每 token 每层访问两个 backend、仅 assignee append、每层一跳；24 层位置并集完整唯一，reserved capacity 不越界，logits 包络和 greedy token 对齐 contiguous reference。4.他者：分布式 inference 通常先用每-rank真实模型与本地 cache加集中 reference oracle验证数值，再接网络/runtime；vLLM 的 paged decode可作为生命周期参考，但不能直接复用其同构 collective与 block table来证明 HCP packet。5.本方案：新增 ignored integration oracle，复用 LayerPacket 与 process_layer_packet_with_reserved_history，backend 各自持有本地 model和request cache；只增加必要的 crate 内 final-logits helper可见性，不新增 runtime all-shard API。6.为什么：这是隔离真实模型/缓存兼容性的最小 tracer bullet；若先接 coordinator或transport，失败无法区分权重数值、ownership还是协议问题。VERDICT: IMPLEMENT。

_updated: 2026-08-02 04:00:07_
### 以两个独立 backend 验证真实 worker-local reserved prefill

type: `decision` · status: `held` · confidence: 0.99 · importance: 1.0 · source: `hetero-cp-ringattn@c465244`

【动机六问】1.问题：单 worker 真实 prefill 和 reservation 消费端已成立，但还没有证据证明不均匀 CP prefill 经 P2P ring 后，每个 worker 只持自己的真实 24 层 reserved shard。2.现状：LlamaModel 的 legacy distributed prefill 已支持 local chunk KV ring；Tch backend 现可消费显式逐层 capacity；直接让 coordinator 发送 Some 会因后续 legacy decode 与 reserved cache 不兼容而提前失败。3.目标：真实 Qwen 1:3 initial prefill 在两个独立 backend 上完成；last-position logits/argmax 与 contiguous reference 对齐；24 层每个位置只在一个 worker durable，capacity/dtype/positions 正确。4.他者：分布式 CP correctness 通常用多 rank local state加集中 reference oracle；serving runtime 自身不暴露全 rank cache。5.本方案：新增 ignored integration test，逐层 LinkedMock 只模拟 P2P，先跑早位置 worker再跑晚位置 worker；测试结束后只作为 oracle 汇总两个 context。6.为什么：复用现有真实 prefill 网络数学与新 reservation 合同，不新增全局 runtime helper，也不在 self-driving decode 尚未接线时破坏现有服务。VERDICT: IMPLEMENT。

[2026-08-02 实现] c465244 完成 test-only oracle；runtime API 仍只持本 worker context，没有提升 all-shard helper。VERDICT: IMPLEMENTED。

_updated: 2026-08-01 21:22:10_
### reserved prefill 不能在 self-driving decode 接线前启用 coordinator

type: `risk` · status: `open` · confidence: 1.0 · importance: 1.0 · source: `code-audit-2026-08-02`

当前 coordinator 的 Decode/DecodeBatch 仍调用 TchWorkerBackend::decode_request -> LlamaModel::forward。ReservedPositionedKvShard 对 legacy update_sharded(keep=false) 明确拒绝，以保护 self-driving ownership；因此 coordinator 若现在发送 layer_kv_capacities，initial prefill 可成功但后续 decode 会失败。处理顺序：先用 backend-level oracle 完成多 worker prefill证据，再接 self-driving decode runtime，之后才在 coordinator 启用 Some capacities。该风险是接线顺序，不是核心数学冲突。

_updated: 2026-08-01 21:12:57_
### prefill 协议缺少逐层 finite-horizon KV reservation 合同

type: `risk` · status: `resolved` · confidence: 1.0 · importance: 1.0 · source: `hetero-cp-ringattn@b4db994`

代码审计确认 WorkerCommand::Prefill 只携带 chunk、seq_offset、position_ids，TchWorkerBackend::do_prefill 固定创建可增长 contiguous cache。改用 reserved cache 后，worker 若只按 prompt chunk 长度预留会在首次 decode assignee append 时越界；若按 max_position_embeddings 猜测则破坏 capacity hard bound。由于 decode ownership 由 token×layer 决定，所需容量是每 worker 的逐层向量。该缺口不否定 ring 数学，但阻断 worker-local reserved prefill 进入后续 decode。

[2026-08-02 解决] worker 现可从 Prefill 命令接收逐层 finite-horizon capacities，不再需要根据 chunk 或模型上限猜测。coordinator 生成真实矩阵仍是下一任务。

_updated: 2026-08-01 21:10:25_
### Prefill 由 coordinator 显式提供可选逐层 local KV capacities

type: `decision` · status: `held` · confidence: 0.99 · importance: 1.0 · source: `hetero-cp-ringattn@b4db994`

【动机六问】1.问题：worker-local reserved prefill 必须在 forward 前知道每层 finite-horizon 容量，否则无法同时保证后续 decode 可 append 与显存硬界。2.现状：协议只有 chunk/position；Tch backend 创建 contiguous cache，worker 无法区分 prompt ownership 与未来 decode ownership。3.目标：runtime 可把逐层 capacity 向量交给 backend；Tch 在任何 tensor 写入前校验 num_layers 和 capacity>=local prompt tokens，并创建对应 BF16/运行 dtype reserved shards；缺失向量时 legacy 行为不变。4.他者：vLLM 等 serving engine 在 prefill 前由 scheduler/admission 分配 block table，worker 消费明确物理配额而非根据模型上限猜测。5.本方案：WorkerCommand::Prefill 增加 optional layer_kv_capacities；WorkerBackend 新增带默认回退的 prefill_request_with_reservation；Tch override 使用 reserved caches，vLLM/旧实现无需改；本节点只消费显式计划，不计算计划。6.为什么：逐层向量与 token×layer ownership 精确同构，避免单标量过度预留；optional/default 保留现有实验路径并把 planner 与执行器分开。【边界】bincode schema 会要求 coordinator/worker 同版本；当前非生产实验服务不提供滚动升级兼容，version negotiation 风险已另有记录。VERDICT: IMPLEMENT。

[2026-08-02 实现] b4db994 完成 optional 协议字段、默认 backend 回退、runtime 透传和 Tch reserved cache 消费端；coordinator 暂发 None，容量生成保持为后续独立节点。VERDICT: IMPLEMENTED。

_updated: 2026-08-01 21:10:25_
### 通过 KvCacheImpl adapter 接入真实 prefill

type: `decision` · status: `held` · confidence: 0.98 · importance: 1.0 · source: `hetero-cp-ringattn@d86ac47`

【动机六问】1.问题：真实 LlamaModel::forward 只能消费 KvCaches，reserved shard 虽已支持 BF16但仍无法进入真实 prefill。2.现状：test-only all-domain helper 能证明数学，却同时看见所有 worker shard；直接提升会违反每 worker 只持本地 KV。另写 model forward 会复制 Norm/Attention/MLP 主流程。3.目标：真实 Qwen initial prefill 通过唯一 LlamaModel::forward 直接写入每层 reserved positioned cache；positions/dtype/committed prefix 正确，logits 与 contiguous reference 一致。4.他者：serving engine 通过统一 cache adapter/block table 让模型 forward 写入不同物理 cache；位置与 cache metadata 由 request state 提供。5.本方案：扩展 KvCache trait 一个默认 no-op 的 position preparation hook；ReservedPositionedKvShard 实现该 trait；KvCacheImpl 增加 reserved variant。LlamaModel 在每层 forward 前把本轮 position_ids 交给 cache。reserved variant 的 legacy update_sharded 明确拒绝，decode 继续走 self-driving core。6.为什么：该 adapter 复用现有真实模型和 ring attention forward，不复制计算主流程，也不把全局 shard 可见性带入 runtime。默认 hook 保持 contiguous/block 行为不变。VERDICT: IMPLEMENT。

[2026-08-02 实现] d86ac47 完成 position preparation hook、reserved KvCache adapter 与真实 Qwen 24 层验证。legacy contiguous/block 行为保持；reserved update_sharded(keep=false) 继续拒绝并要求 self-driving decode。VERDICT: IMPLEMENTED。

_updated: 2026-08-01 20:58:52_
### Rust 线优先完成完整推理服务框架

type: `preference` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-02`

用户确认：保持已经验证成立的 self-driving ring 核心方案稳定，Rust 线持续纵向完成 prefill、decode、continuation prefill、后续 decode、多请求隔离与稳定服务循环。服务接线过程中若出现与核心的接口张力或局部矛盾，先记录为 risk/uncertainty 并继续推进；只有数值错误、KV 容量越界、请求串扰或无法运行才升级为 blocker。当前阶段只实现核心和必要能力，不自动追求生产级治理、性能优化或完整生态兼容。

_updated: 2026-08-01 20:27:19_
### Rust 完整推理服务框架主线

type: `task` · status: `ongoing` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-02`

持续以小型纵向节点完成真实模型 initial prefill -> self-driving decode -> continuation prefill -> decode -> 多请求隔离 -> 稳定请求生命周期与网络服务循环。每个节点独立验证和提交；已验证核心默认不重开。非 correctness 的接口张力记录后继续，关键节点执行动机剖析并绑定 Graph Memory evidence。

_updated: 2026-08-01 20:27:19_
### 以纵向闭环持续完成 Rust 推理服务框架

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-02`

【动机六问】1.问题：当前 ring 数学、packet、reserved positioned KV 与多层/多 token correctness 已分别成立，但仍主要存在于实验 helper/test，尚无一个真实模型请求能贯穿 prefill、decode、continuation 和多请求服务生命周期。2.现状：LlamaModel/WorkerBackend/transport 已有若干独立能力，self-driving core 也已抽出；缺口是服务状态、真实模型 KV 数据边界和阶段组合，而不是重新设计 attention 数学。3.目标：真实权重请求可在 Rust 服务中完成 initial prefill -> self-driving decode -> continuation prefill -> decode；随后支持多个 request_id 的独立 KV、交错推进、释放与重复服务，结果与单请求参考一致。4.他者：vLLM 等成熟 engine 使用 paged KV、block table、scheduler、request state 与 continuous batching 形成一体化服务；这些机制可作为生命周期和隔离参考，但其同构设备假设和内部通信路径不能直接表达 HCP 的 capacity-weighted 异构 P2P ring ownership。5.本方案：保持已验证核心冻结，以逐段可运行的 tracer bullet 接线真实模型和服务状态；先打通最薄单请求真实模型流程，再加 continuation、多请求隔离、网络循环与稳定性。发现非致命张力写入 risk/uncertainty 并继续；correctness、显存硬界和隔离失败才阻断。6.为什么：重做核心会重复已经获得的证据；一次性复制成熟 serving engine 又会引入不需要的 production 复杂度。纵向小节点能持续暴露真实接口缺口，同时让每个新增能力直接服务最终闭环。VERDICT: IMPLEMENT。

_updated: 2026-08-01 20:27:19_
### 候选下一节点：只提升已验证的 reserved positioned KV 数据边界

type: `decision` · status: `held` · confidence: 0.98 · importance: 1.0 · source: `hetero-cp-ringattn@91df8c4`

【动机六问】1.问题：capacity hard bound 目前只存在于 cfg(test) slab，框架编译产物中的公开 self-driving packet 路径仍通过 Tensor::cat 复制完整本地历史，因而无法被后续核心代码复用。2.现状：ReservedPositionedKvShard 已在 24-layer mixed-history 和 two-token TCP 中证明 exact reservation、stable pointer、overflow rejection 与数值正确；但其 struct、append/view API 和 reserved adapter 全在 tests 模块。通用 KvCache trait 既不携带 global positions，也以 update 返回完整 tensor 为合同；直接改造会扩大范围。3.目标：让 reserved positioned shard 和 packet adapter 在 tch-backend 正常编译并由现有实验直接使用；验证行为不变、无新增 Tensor::cat、legacy API 不变。4.他者：成熟 serving engine 使用 reserved arena 或 paged cache，把 committed view 与物理 capacity 分离；完整方案还包含 allocator、block table 和 lifecycle。5.本方案：只做代码抽取和最薄公开 experimental API，不实现成熟 allocator，不替换通用 KvCache。6.为什么：它是把已证明显存性质从测试夹具提升为框架能力的最小改动；比继续扩大集成测试增加更多可复用价值，也避免重新进入 production planner。【边界/牺牲】保留 legacy cat 意味着默认旧路径仍无硬界；新 API 需要调用者预先知道 capacity，不能开放式增长。这个限制符合当前 finite-horizon 实验合同。VERDICT: PROPOSE IMPLEMENTATION, PENDING USER CONFIRMATION。

[2026-08-02 路线约束] 本节点是走向真实模型完整 prefill/decode/continuation 流程的前置能力，不是终点；后续节点仍需逐步把真实模型数据流接到该 cache contract。

[2026-08-02 用户确认与实现] 用户确认该节点符合真实模型完整推理路线；实现已由 hetero-cp-ringattn@91df8c4 和 ev-promote-reserved-positioned-kv-core-20260802 验证。VERDICT: IMPLEMENTED。

_updated: 2026-08-01 20:19:50_
### 框架路线转向真实模型的完整推理流程

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-02`

后续 Rust 框架节点以真实模型完成 initial prefill -> self-driving decode -> continuation prefill -> decode 的完整流程为目标。仍采用小步推进：当前先让 reserved positioned KV 成为正常核心 API；随后再选择最薄的真实模型组合入口。模块数学、重复规模测试、production planner、多请求 scheduler 都不能取代该完整流程目标。

_updated: 2026-08-01 20:06:16_
### 整理 HCP 论文的完整核心推理框架

type: `task` · status: `paused` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-02`

从整个请求生命周期整理 HCP 核心方案：非均等 Ring Attention prefill、自驱动 decode ring、两种 KVCache 切分规则的兼容、capacity-weighted 显存合同、线性 P2P 数据流与阶段转换不变量。论文系统定位为互联无关的异构全生命周期 Context Parallelism：每个 worker 只需 predecessor/successor P2P，CXL、RDMA/RoCE、InfiniBand、UALink、PCIe peer access 或未来高速 fabric 均可承载；当前聚合 KV/context capacity，模型权重仍由每个 worker 完整持有。先形成可审查的方案决策和论文骨架，再写正文。\n\n[2026-08-02 写作边界] 先完成问题定义、系统模型、方法、正确性/复杂度、局限和评测设计；真实实现细节、实验结果、成本结论、摘要与最终结论等待当前框架和新证据。

[2026-08-02 暂缓] 英文和中文方法草稿已形成检查点。论文线暂不继续扩写，等待框架实现和新实证后再恢复。

_updated: 2026-08-01 19:58:15_
### 双语方法稿完成后暂缓论文线并恢复 Rust 核心框架

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-02`

用户已确认英文草稿可接受，并要求中文版完成后暂时停止论文推进。现阶段不写 Abstract、Implementation Results、性能结果、最终 Conclusion 或新相关工作。下一主线回到 Rust 实验性核心框架；仍遵守核心和必要能力优先，不进入多请求 pipeline、真实 runtime 接入或生产级 allocator，除非用户后续明确恢复这些范围。

_updated: 2026-08-01 19:58:15_
### 先做只读缺口审计，再选择一个实验性核心节点

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `motivation-analysis-2026-08-02`

【动机六问】1.问题：论文方法已经固定，但需要回到框架实现；若直接照旧生产计划推进，会重新引入用户已拒绝的 planner、runtime 和大步改造。2.现状：Rust 已有 self-driving tensor 数学、wire packet、localhost TCP、frozen capacity schedule、positioned exact slab 和 24-layer mixed-history 等模块证据；公开路径仍可能保留 test-only 边界和 Tensor::cat，且工作区存在用户未提交的 capacity/placement 改动，不能凭旧记忆决定下一步。3.目标：只读核对当前代码、测试与 Graph 节点，选择一个必要、可独立验证、不会触碰生产级范围的最小实现节点；若核心已经闭合，则明确指出并选择最薄的组合缺口。4.他者：成熟 serving framework 通常下一步会接 paged cache、admission、runtime queue 和 backpressure，但这些机制解决生产生命周期，不是当前实验性算法可行性的必要条件。5.本方案：优先检查当前 modular proof 是否仍有一个未组合的算法/数据边界；避开多请求、runtime、动态 allocator 和硬件性能。6.为什么：这能让实现继续为 HCP 核心论点增加独立证据，而不是为尚未验证收益的生产框架增加复杂度。VERDICT: IMPLEMENT READ-ONLY AUDIT, THEN PROPOSE ONE SMALL NODE。

_updated: 2026-08-01 19:58:15_
### 中文版只做等价翻译，完成后暂缓论文线

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `motivation-analysis-2026-08-02`

【动机六问】1.问题：英文方法稿已通过用户审查，但中文审阅和后续框架开发需要一份术语一致、可直接对照的中文版；同时论文线若继续保持 ongoing，会误导后续会话继续扩写。2.现状：英文稿已固定问题定义、系统模型、capacity-weighted prefill、自驱动 decode、continuation、正确性/复杂度、证据边界与评测设计；尚无中文版，论文总任务仍为 ongoing。3.目标：新增 docs/paper/HCP_METHOD_DRAFT_ZH.md，章节、公式、代码块、claim 标签和边界与英文稿对应；结构/禁用论断审计通过；提交后把论文总任务设为 paused，并明确框架线恢复。4.他者：双语技术文档通常保持稳定源稿和逐节对应译稿，术语与公式不在翻译中重新设计，避免两个版本形成不同方法。5.本方案：直接翻译已批准英文稿，保留英文术语作为必要括注，公式原样复制；不加入摘要、结果、最终结论或新引用。6.为什么：这是满足中文审阅需求且不重新开启论文设计的最小方案；把论文状态同步为 paused 可防止后续 agent 误把写作当当前主线。VERDICT: IMPLEMENT EQUIVALENT TRANSLATION AND PAUSE PAPER LINE。

_updated: 2026-08-01 19:52:13_
### 方法草稿只固化当前算法与证据边界

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `motivation-analysis-2026-08-02`

【动机六问】1.问题：HCP 的核心定义、两阶段 KV 分片和自驱动 decode 证据已分散在计划、代码实验与 Graph Memory 中，若不统一成论文级方法稿，后续实现和评测容易偏离项目目标；若直接写完整论文，又会把尚未完成的硬件与性能验证误写成结果。2.现状：已有 transport-agnostic 异构 CP 定位、position-indexed KV 模型、capacity-weighted ownership/reservation、Ring Attention prefill、自驱动 decode、continuation prefill、online-softmax 合并、任意 N/L 与 N-1 hops 的数学及 Rust 模块证据；缺少当前方案的真实跨后端硬件 E2E、大 context、物理显存、性能/TCO、多请求 runtime 和系统基线。3.目标：新增一份可由用户逐节审查的英文方法草稿，统一符号、完整数据流、正确性不变量和复杂度；所有陈述按 Method claim、Proved invariant、Prototype evidence、Open empirical question 分级；搜索确认没有旧 1M 结果、性能数字或 vLLM 工程适配。4.他者：Ring Attention 论文以分块 attention、online softmax 和环通信描述 context parallelism；系统论文通常先稳定 problem/system model、algorithm、proof obligations 与 evaluation questions，再在实现闭合后补 implementation/results/abstract/conclusion。其现成结构可复用，但同构均分、collective 或阶段级异构调度不能直接表达 HCP 的不均等同请求 context ownership。5.本方案：只创建 docs/paper/HCP_METHOD_DRAFT.md，覆盖问题定义、系统模型、capacity-weighted placement、prefill、self-driving decode、continuation、正确性、复杂度、证据边界和 Evaluation Design；不修改实现，不补文献编号，不产生经验结论。6.为什么：这是让理论、实验合同和论文目标对齐的最小产物；它保留未来替换 runtime 与硬件的自由，同时使后续实验可以直接对应明确的 research questions 和不变量。VERDICT: IMPLEMENT METHOD DRAFT；DEFER ABSTRACT、FINAL CONCLUSION、IMPLEMENTATION RESULTS 与 PERFORMANCE CLAIMS。

_updated: 2026-08-01 19:37:46_
### 当前先整理 HCP 论文的方法主体，推迟实现结果与最终结论

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `paper-readiness-audit-2026-08-02`

【动机六问】1.问题：框架尚未接入真实跨后端 runtime 和新硬件实验，需要判断现在写论文能推进多少，避免等待全部工程完成，也避免提前固化未经验证的结果。2.现状：HCP 的研究问题、transport-agnostic 异构 CP 定位、position-indexed KV 模型、prefill/decode/continuation 数据流、online-softmax 正确性、capacity-weighted ownership/reservation、任意 N/L 路由与通信复杂度已有数学和 Rust 模块证据；缺少当前实现的真实跨后端硬件 E2E、大规模 context、性能/TCO、多请求 runtime 和系统基线。相关工作定位已确认但引用审计未完成。3.目标：先形成稳定的论文骨架和方法正文，所有 empirical claim 使用明确 placeholder；当新实现完成后只补 implementation/evaluation/results，而不重写主方法。4.他者：系统论文通常先固定 problem model、design、algorithm 和 evaluation questions，结果、abstract 与 conclusion 在实现和实验闭合后完成；workshop/position paper可以用较轻实证，完整 systems paper 需要端到端与量化对比。5.本方案：现在可定稿 system/problem model、method、correctness/complexity 和 limitations；可起草 introduction、related-work taxonomy 与 evaluation design；推迟 implementation details、results、economic claims、abstract 和 final conclusion。按 8-10 页完整系统论文估算，可整理 65-70% 的结构化正文，当前可真正定稿约 35-40%；若改投 position/workshop paper，可完成约 80-90%。6.为什么：方法不依赖尚未稳定的 runtime API，提前整理能暴露数学和论证缺口；结果章节直接依赖最终实现与硬件，提前写只会制造伪证据和返工。VERDICT: IMPLEMENT METHOD-FIRST PAPER SCAFFOLD；DEFER EMPIRICAL COMPLETION。

_updated: 2026-08-01 18:51:11_
### HCP 系统定位：面向 CXL-class 互联的异构全生命周期 Context Parallelism

type: `decision` · status: `superseded` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：论文需要把 Ring Attention 显存分担思想、异构 capacity weighting、线性 P2P ring 和未来硬件目标统一成一个准确的系统命题，避免只像 decode 优化，也避免把 HCP 误写成通用模型并行。2.现状：核心算法已经定义完整推理生命周期的异构 Context Parallelism，现有数学与模块实验给出了异构 KV 容量聚合的核心可行性，带宽矩阵说明传统低带宽网络会成为瓶颈；但现有表述尚未明确目标互联与参数内存边界。3.目标：论文将 HCP 定位为基于 Ring Attention 的 context/KV 显存分担思想，在每 worker 仅连接 predecessor/successor 的线性 P2P ring 上，对 prefill、decode 和 continuation prefill 的同一逻辑 context 做 capacity-weighted 分片；系统愿景是借助 CXL-class、memory-semantic、高带宽低时延 P2P fabric，使低成本异构加速卡池能够承载单设备 KV 显存不足的超长上下文推理。4.他者：Ring Attention 主要以 sequence/context shard 和环传 KV 扩展长序列；主流并行推理常依赖同构高速互联与 collective；CXL/类内存语义互联提供设备内存共享或低开销访问的硬件方向，但并不自动给出异构 attention 的 placement、精确归并和全生命周期 KV 所有权。5.本方案：HCP 保持统一的 position-indexed KV context，prefill 环传 KV block，decode 环传 Q、online-softmax accumulator 与 activation，新 KV 由 capacity-weighted 唯一 assignee 原地保留；通信合同只要求邻居 P2P，不要求 collective 或全连接。6.为什么：它直接聚合异构设备最稀缺且随 context 增长的 KV 容量，同时使连接数与单请求流量随节点数线性；相比照搬同构 CP 或把 CXL 当透明共享内存，它保留设备本地计算和明确所有权，更符合不均等显存与算力。边界：当前每个 worker 仍复制完整模型权重，因此已支持的是超长上下文和 KV-heavy inference，不是参数量超过单节点权重容量的模型；CXL-class 是目标硬件类别与待验证条件，不是现有实验已经证明的充分条件。VERDICT: IMPLEMENT AS PAPER SYSTEM POSITIONING。

_updated: 2026-08-01 18:40:49_
### HCP 系统定位：互联无关的细粒度异构 Context Parallelism

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-correction-2026-08-02`

【动机六问】1.问题：上一版把系统愿景集中在 CXL-class，容易让 HCP 看起来依赖一种特定硬件，也没有充分表达真正的系统缺口：现有分布式推理通常把同构设备作为同一请求内的细粒度协作单元，异构资源更多用于请求级、模型级或 prefill/decode 阶段级分配，难以把不同代际、不同品牌和不同容量的加速卡聚合到同一个 attention context。2.现状：当前 HCP 已由数学推导、Rust tensor correctness、localhost P2P、capacity-weighted reservation 以及 24 层 prefill-decode-continuation mixed-history 实验证明核心数据流可组合。真实跨后端异构硬件验证尚未完成；带宽矩阵只能说明网络性能是关键系统变量，尚无 TCO、能耗或单位吞吐成本实验。3.目标：论文将 HCP 定义为 transport-agnostic 的完整推理生命周期异构 Context Parallelism。算法只要求每个 worker 与 predecessor/successor 进行 P2P 传输；CXL、RDMA/RoCE、InfiniBand、UALink、PCIe peer access 或未来通用高速互联均可作为承载。论文严格区分当前实验性核心证据、开放的跨硬件验证和待验证的成本优势。4.他者：主流 TP、PP、CP 与 collective 通常围绕同构设备、对称分片和一致 kernel 能力优化；异构 serving 常通过请求路由、模型放置或 prefill/decode disaggregation 利用不同资源池。这些方式降低设备内核和负载不对称带来的复杂度，但通常不聚合同一 attention context 的不均等 KV 容量。该研究定位已获用户确认，进入论文时仍需文献与实现审计限定范围。5.本方案：以逐层 position-indexed KV context 为统一对象，prefill 按 capacity-weighted positions 永久分片并环传 KV，decode 保持历史 KV 原地且让 Q、O/LSE 与 activation packet 遍历所有 shard，新 KV 按 capacity-weighted layer×position event 唯一归属；continuation prefill继续追加同一逻辑 context。物理互联只需实现邻居 P2P 合同。6.为什么：HCP 的价值不来自押注某种网卡，而是尝试把难以高效组成同一细粒度并行组的代际混合、品牌混合和容量不对称设备转化为一个 context-capacity pool。互联带宽和时延的持续提升会扩大这一方法的可用区间；长上下文使 KVCache 成为持续增长的主要显存压力，因此是当前最直接的应用目标。边界：模型权重仍由每个 worker 完整持有；真实跨后端硬件协作、成本优势与未来超大模型适应性仍待验证。VERDICT: IMPLEMENT AS REVISED PAPER POSITIONING；DEFER HARDWARE AND LOW-COST CLAIMS UNTIL MATCHING EVIDENCE。

_updated: 2026-08-01 18:40:49_
### 当前全生命周期 HCP 尚缺真实跨后端异构硬件验证

type: `uncertainty` · status: `open` · confidence: 1.0 · importance: 1.0 · source: `user-correction-2026-08-02`

当前修订方案已有数学、Rust tensor correctness、localhost P2P、任意 N/L、capacity-weighted schedule/reservation、prefill-decode-continuation mixed-history 等模块证据，但尚未在真实不同品牌或不同后端加速卡上执行同一版本的完整 prefill + self-driving decode + continuation 流程。在完成对应实验前，论文只能把细粒度异构协作写成方法目标和实验性核心可行性，不能写成当前方案已经跨硬件验证。

_updated: 2026-08-01 18:40:49_
### 用 test-only reserved commit adapter 连接 TCP ring 与原地 KV append

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：24 层实验已证明 exact slab 与 capacity-weighted prefill/decode continuation，但 localhost TCP 两-token ring 的 distributed worker 仍调用 process_layer_packet 中的 Tensor::cat；两条核心证据尚未组合。2.现状：Tensor::cat 位于公开 process_layer_packet 的 tuple history 写入；直接替换需改变公共 cache 合同，过宽。ReservedPositionedKvShard 已在 cfg(test) 中验证。3.目标：TCP worker 使用精确预留 slab，所有 decode growth 原地写且 data_ptr 不变；packet 路由、8 次 send、唯一 assignee、两 token continuation、hidden/logits/token 对齐保持。4.他者：serving engine 用 reserved arena 或 paged cache，再让 attention 读取 committed view；测试通常通过 cache adapter 组合网络与存储路径。5.本方案：提取私有的 post-commit packet continuation 原语供 legacy process_layer_packet 和 test-only reserved adapter 共用；公开函数仍保留 cat。TCP 测试按 layer×domain 冻结 assignee 次数精确预留并记录 cursor/capacity/pointer。6.为什么：它避免复制 online-softmax 与 layer finish 数学，也避免提前设计生产 cache trait；只填补 TCP 数据流与 slab mechanics 的组合缺口。【牺牲四问】legacy cat 为未知长度和简单 tuple ownership 提供动态增长，本节点不删除它；test-only slab 牺牲 horizon 外增长和运行期重分配，这些能力服务开放式生成与生产 allocator；当前固定两-token实验不需要。因此本节点只能证明 reserved 变体可驱动同一 TCP packet 数学，不能声称公共 process_layer_packet 或生产 runtime 已无 cat。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-08-01 18:34:17_
### 旧 1M 工程证据直接退出当前仓库与知识图谱

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-02`

【动机六问】1.问题：旧 1M 工程路线与当前完整生命周期 HCP 的数据流和实现不同，保留结果容易被未来论文或 agent 再次误用。2.现状：虽然旧结果已与当前 decision 脱钩，报告、计划、展示资产、Graph Memory 节点、迁移 seed 和多份文档引用仍在仓库。3.目标：删除旧结果与实验专用材料，同时保留当前算法和独立有效的理论/网络内容；用搜索、图完整性和 Git diff 验证。4.他者：研究项目通常归档被取代的实验以保留 provenance；当旧证据极易污染新方法 claim 时，也会撤回 artifact 并仅保留撤回记录。5.本方案：删除专用 artifact 和结果节点，清理重建脚本与引用；只保留一个不含实验结果细节的 deletion decision，防止自动恢复。6.为什么：当前项目明确选择用完善后的新实现重新产生证据，旧实现数据的误导成本高于其在线可访问价值。【牺牲四问】默认保留历史是为了复现、审计与回归比较；本次牺牲旧实现的就地复现便利和旧结果细节；这些材料在一般研究中用于追踪系统演进；对本项目而言，当前论文必须只消费匹配当前方法的新证据，且 Git 跟踪内容仍可从历史恢复、未跟踪资产先移入系统废纸篓，因此接受该牺牲。VERDICT: IMPLEMENT DELETION。

_updated: 2026-08-01 18:32:53_
### 将 schedule 显存保证限定为完整 horizon reservation

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：旧测试在一个 [1,3,2]、24 units、单 phase 样例上检查前缀比例误差小于等于 1，容易继续暗示该结论对任意 phase 成立；反例已证明这不是普遍定理。2.现状：largest remainder 产生完整 horizon 精确 counts，phase 只循环旋转同一 sequence；exact slab 已证明已知 horizon 可以按 layer×domain 精确预留并原地 append，但 schedule 测试尚未把 counts 明确验证为每域 reservation 上界。3.目标：对多组 tickets/horizon 和所有 phase，任意 prefix 的消费计数都不超过 counts，完整 horizon 后精确等于 counts；现有确定性、容量份额、唯一 assignee和零容量语义不变。4.他者：vLLM 等 serving engine 依赖 admission reservation、block quota 或预分配 arena 保证显存，调度顺序负责平滑吞吐而不是充当显存硬界。5.本方案：不改算法或 API，只用纯单元测试把 counts 解释并验证为完整 horizon reservation，删除旧样例中的 scaled prefix-error 断言。6.为什么：这是把已修订数学结论落实到代码合同的最小方案；无需发明 cyclic discrepancy 算法，也不引入生产 allocator。【牺牲四问】旧前缀检查的目的，是约束单请求短期 event 分布和平滑计算；本节点放弃把小于等于 1 当作普遍保证，但不删除 phase 轮转或 smooth sequence；短期平滑本质上服务并发负载均衡，而不是物理显存安全；本项目现阶段优先保证可证明的 capacity hard bound，多请求效果以后单独实验。VERDICT: IMPLEMENT EXPERIMENT ONLY。

_updated: 2026-08-01 17:31:51_
### HCP 论文核心候选：保持 KV 分片并按阶段切换环上传输对象

type: `hypothesis` · status: `superseded` · confidence: 0.95 · importance: 1.0 · source: `analysis-2026-08-01`

待用户审查的统一方案：HCP 将每层 KV 定义为按 global position 索引的逻辑关系 C_l[p]=(K_l,p,V_l,p)，每个 position×layer 恰好归属一个 worker。Prefill 以 capacity-weighted position/context shards P_i 写入本地 KV；每个 worker 保留本地 Q 与 activation，KV micro-block 沿 predecessor/successor ring 流动并以全局 position 因果掩码和 online softmax 合并，随后各 worker对本地 token chunk 完成 W_o、residual、norm、MLP。Decode 不移动历史 KV：单个 activation packet 携带 residual、normalized hidden、Q、O/LSE 与瞬时角色，访问所有本地 shard；唯一 assignee 按 capacity-weighted token×layer event schedule 投影并原地追加 current K/V，finisher 在 N-1 hop 后完成 W_o、residual、norm、MLP并成为下一层 starter，末层 finisher 完成 final norm、LM head、sampling、embedding。Prefill shard 与 decode layer-striped growth 的物理归属可以不同；每层 position union 完备互斥且保留显式 position 即可，无需 KV reshuffle。Continuation prefill把新 positions 再按 context shard append，同一 positioned online-softmax 可读取 prefill+decode mixed history。Prefill 集群网络量每层约为 (N-1)×S×KVBytesPerToken，decode 每层为 (N-1)×PacketBytes 且与历史长度无关；每 worker 只有两个 peer。模型权重仍复制，capacity guarantee 仅针对 KV。vLLM context-passing 是无法直接暴露 partial-attention 时的后端适配，不作为主算法。真实 runtime 的 prefill tail→decode starter 直接交接、开放式 horizon、多请求与跨机 self-driving decode 仍是边界。

_updated: 2026-08-01 13:31:11_
### HCP 论文主方法：完整推理生命周期的异构 Context Parallelism

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-01`

HCP 将 Context Parallelism 从狭义的 prefill sequence partition 推广到完整推理生命周期：对每层逻辑 context C_l[p]=(K_l,p,V_l,p)，异构 worker 按 capacity-weighted policy 互斥且完备地持有 layer×position shards。Prefill 时，当前 token 序列可切分，各节点计算本地 activation/Q/K/V、永久保存本地 KV，并通过 P2P Ring Attention 聚合其他 context shard；decode 时当前 query 长度为 1，无法再沿 sequence 切 Q，但历史 context/KV 仍保持分片，单个 query、online-softmax accumulator 与 activation packet 访问全部本地 shard，新 KV 按 capacity-weighted token×layer event 分配并原地追加。两阶段都是 HCP：被并行化的本质对象是同一个逻辑 attention context，而不是某一种固定通信张量。Prefill 传 KV block、decode 传 Q/O/LSE/activation 是阶段特定的数据流机制；position-indexed cache 使两种物理布局无需重排即可组合。论文核心算法排除 vLLM、context-passing connector、runtime negotiation、allocator 和其他工程适配；这些只能进入实现或未来工作，不能定义主方法。

_updated: 2026-08-01 13:31:11_
### 论文核心必须覆盖 HCP 端到端推理框架而非仅 decode 实验

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-direction-2026-08-01`

【动机六问】1.问题：若只从最近的自驱动 decode 实验写论文，会遗漏 HCP 原始贡献，即非均等 context-parallel prefill、异构 capacity weighting、P2P ring attention 与 decode 新增长分片如何共同服务同一 KVCache。2.现状：prefill 设计与真实异构证据分散在 DESIGN、RINGATTN_MODEL、INFERENCE_PIPELINE、历史报告和 Graph Memory；decode 数学与 positioned mixed-history 证据集中在近期 self_driving 实验，尚未统一成一套论文级端到端模型。3.目标：给出从输入 token 到 prefill、首 token、逐 token decode、continuation prefill 的完整数学和数据流；明确每阶段传输对象、KV 永久归属、显存/通信复杂度、阶段转换条件、已验证证据和未覆盖边界。4.他者：Ring Attention 用 KV block 环传与 online softmax 完成 context-parallel prefill；常规推理引擎用 paged/block KV 管理 decode 增长，pipeline/collective 体系通常依赖同构拓扑。5.本方案：保留 HCP prefill 的 capacity-weighted token/context shard；decode 将每个 token×layer 新 KV event 按容量权重分配到唯一节点，Q 与 online-softmax accumulator 随 activation packet 遍历所有本地历史；显式 global position 使两类物理布局成为同一逻辑 KV 序列；finisher 原地完成 W_o、residual、norm、MLP 与层间/跨 token continuation。6.为什么：它保持每节点只永久持有自己的 KV 份额、每 worker 仅有 predecessor/successor、单请求通信随 N 线性，同时避免要求 prefill 和 decode 使用相同物理分片坐标。VERDICT: IMPLEMENT ANALYSIS AND PAPER DESIGN FIRST；不把 test-only cache、runtime、多请求或硬件性能写成已完成贡献。

_updated: 2026-08-01 12:40:06_
### 模块化证据已闭合自驱动 decode ring 实验核心

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `code-audit-2026-08-01`

【审查六问】1.问题：判断现有 24 层、TCP、任意 N、reserved slab 与 schedule 证据能否组合，是否必须增加 24 层 TCP。2.现状：各实验分别覆盖规模、网络、存储和生命周期，公共 runtime 仍未接入。3.目标：证明或否定 schedule→reservation→append→packet continuation→逐层与跨 token 递推→positioned continuation prefill 的组合。4.他者：分布式实现通常用局部不变量加少量边界集成测试建立归纳证据，只有规模引入新状态时才追加大集成测试。5.本方案：逐行核对同一 continue_layer_packet、reserved adapter、layer loop、TCP frame 和 24 层四阶段测试，并运行全部 17 项 self-driving 回归。6.为什么：层数只增加相同状态转移次数；layer_idx、shard/model 数组索引和同步 TCP 流没有随 L 改变的额外协议状态。组合结论：FrozenKvAssigneeSchedule 为完整固定 horizon 给出 capacity-weighted counts；每 layer×domain reservation 从实际 prefill split 与 decode assignee 次数导出；reserved append 写前拒绝 overflow 且只暴露 committed prefix；packet 传 residual、normalized、position、Q、O/LSE，不传历史 KV；assignee 唯一投影并留存 current K/V；finisher 唯一执行 W_o、residual、post norm、MLP，末层执行 final norm、head、greedy sampling 和 embedding；两层 TCP 已覆盖 finisher-to-starter 归纳步，任意 L runner 覆盖重复，任意 N TCP 覆盖 successor 与 wrap-around；24 层四阶段覆盖 prefill→decode→continuation prefill→decode 的 positioned mixed history 和最终 [56,168,112]=1:3:2。每请求网络为 L×(N-1) 个固定 context-independent packet，随 N 线性。未覆盖：公共 process_layer_packet 仍有 Tensor::cat、真实 WorkerRuntime、开放式 horizon、byte-level admission、多请求、QUIC、远端异构设备和 GPU 物理显存。VERDICT: MODULAR CORE CLOSED; DEFER 24-LAYER TCP。

_updated: 2026-08-01 11:33:57_
### 以模块不变量组合审查决定是否补 24 层 TCP 实验

type: `decision` · status: `held` · confidence: 1.0 · importance: 1.0 · source: `user-confirmed-2026-08-01`

【动机六问】1.问题：24 层四阶段与两-token TCP 分别成立，但尚需判断它们是否能组合成完整实验性核心，避免遗漏跨模块耦合，也避免为规模而重复测试。2.现状：24 层 in-process 证据覆盖 capacity-weighted positioned KV、prefill-decode-continuation 与 exact slab；TCP 证据覆盖真实 P2P packet、两 token、reserved append、sampling；任意 N 与任意 L 各有独立证据。公共 runtime 仍保留 legacy Tensor::cat。3.目标：逐项审查 schedule、slab、packet continuation、逐层递推、跨 token 与任意 N 的接口不变量；只有存在不能由现有证据推导的新耦合风险才建议最小新实验。4.他者：分布式系统通常以单元/属性测试证明局部不变量，再用少量集成测试证明边界组合；扩大层数的网络集成测试只有在层数改变协议状态或资源布局时才增加信息。5.本方案：只读对照 Graph Memory 证据、实现与测试，构造组合证明和未覆盖边界清单；不改 Rust、不进入 runtime、多请求、QUIC 或远端硬件。6.为什么：它直接回答当前核心是否闭环，并遵守小步与最小证据原则；24 层 TCP 若只是重复相同 continuation 分支，不应作为默认下一步。VERDICT: IMPLEMENT AUDIT ONLY。

_updated: 2026-08-01 11:22:12_
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
### 现有多请求 context 未隔离每层 ring phase state

type: `risk` · status: `open` · confidence: 1.0 · importance: 0.98 · source: `code-audit-2026-08-02`

TchWorkerBackend::RequestContext 只保存 KvCaches/global_seq_len/is_prefill_done；每层 HcpRingAttentionBackend 还持有 request-sensitive prefill_kv_len、is_prefill_done 和 seq_offset。单请求不受影响，但分布式多请求交错可能共享错误 phase。按用户路线先继续单请求完整流程；在多请求节点前必须把这些字段纳入 request-local state，否则属于 correctness blocker。

_updated: 2026-08-01 20:42:55_
### 真实模型 KV dtype 与 reserved slab 固定 Float 不兼容

type: `risk` · status: `resolved` · confidence: 1.0 · importance: 0.98 · source: `hetero-cp-ringattn@b6902ba`

代码审计确认 ReservedPositionedKvShard::new 固定以 Kind::Float 预分配；本地真实 Qwen2-0.5B config 和 LlamaModel dtype 为 BFloat16。append 会严格校验 dtype，因此真实 Qwen K/V 无法进入当前 slab。该问题不否定 reserved/positioned ownership、capacity hard bound 或 ring 数学，只阻断真实模型数据接线。

[2026-08-02 解决] new_with_kind 允许实际模型 dtype 预分配，不改变核心数据流。

_updated: 2026-08-01 20:38:25_
### Reserved positioned KV 由调用者显式指定运行 dtype

type: `decision` · status: `held` · confidence: 1.0 · importance: 0.98 · source: `hetero-cp-ringattn@b6902ba`

【动机六问】1.问题：真实 Qwen2-0.5B 的 current K/V 是 BFloat16，当前 slab 固定 Float，append 必然失败。2.现状：Float 固定值来自 synthetic correctness 测试；shape/device/overflow/position 语义已成立，不应重写。3.目标：调用者可用实际模型 Kind 预分配 slab，BF16 K/V 可原地 append；默认 Float 构造与 dtype mismatch rejection 保持。4.他者：vLLM/paged KV 等缓存按模型或 cache runtime dtype 分配，不根据测试默认值固定。5.本方案：保留 new(config, capacity, device) 作为 Float 兼容入口，新增 new_with_kind(config, capacity, device, kind)，仅替换 Tensor::zeros 的 Kind 来源。6.为什么：显式 runtime Kind 是权重加载后的事实，可兼容未来 dtype override；从 config.torch_dtype 再推导会复制 LlamaModel 映射并可能与实际加载 dtype 偏离。该改动不改变 attention、分片、capacity 或 position 语义。VERDICT: IMPLEMENT。

[2026-08-02 实现] b6902ba 验证显式 runtime Kind；VERDICT: IMPLEMENTED。

_updated: 2026-08-01 20:38:25_
### 现行远程操作只解析 inventory endpoint

type: `decision` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `user-correction+verified-inventory-endpoint-2026-08-02`

【动机六问】
1.问题：项目规则把一个已过期的 GPU LAN 地址声明为当前 host，导致按规则连接时 SSH 超时，并与 inventory 冲突。
2.现状：AGENTS.md、三个 remote smoke 脚本和两份现行操作文档仍含旧默认或示例；inventory 记录 white 的当前 Tailscale SSH endpoint，实测可连接并看到 RTX 4090、仓库和 Qwen2-0.5B 模型。
3.目标：所有新操作先查 inventory；现行规则不固定旧 IP；会发起连接的脚本要求显式 GPU_HOST/CONNECT_ADDR，文档示例指向 inventory 字段。历史实验记录仍保持为带日期证据。
4.他者：基础设施自动化通常把 endpoint 放在 inventory/service discovery/config 中，运行脚本消费显式配置，而不是复制易漂移的地址常量。
5.本方案：AGENTS.md 改为 inventory authority；删除 active script 的旧默认，参数缺失时 fail-fast；更新 PROTOCOL_SMOKE 与 DEPLOYMENT_GUIDE 的现行说明。
6.为什么：这是最小修复，不引入 YAML parser、DNS 或新的配置系统；既利用已有 inventory，也避免脚本再次静默连接错误机器。
【牺牲四问】默认地址原本用于减少手动参数；移除后牺牲无参数启动便利性；该便利性的本质是为单一稳定环境提供快捷入口；本项目节点和网络路径会变化，显式从 inventory 取值比过期默认造成的误诊成本更低。历史证据不删除，避免改写既有实验事实。
VERDICT: IMPLEMENT。

_updated: 2026-08-02 11:12:40_
### Graph Memory 强制使用非 PTY 长文本写入与正文回读校验

type: `decision` · status: `held` · confidence: 1.0 · importance: 0.95 · source: `sqlite-forensics-2026-08-02`

【动机六问】1.问题：skill 只要求事务和节点存在，不能防止正文在到达 SQLite 前已被 PTY 行编辑或 Unicode 传输破坏。2.现状：项目 lesson 已记录 evidence 第 3、4 条被串坏并已修正，但通用 skill 没有禁止 PTY、字面反斜杠换行或盲目重放，也未要求正文级验证。3.目标：skill 明确非 PTY stdin 写入、事务与失败处理、写后原文或 hash/哨兵验证；回归测试锁定合同并执行 Unicode 多行 roundtrip。4.他者：SQLite CLI 常用非交互脚本加 .bail on 和显式事务，应用则用参数绑定；SQLite 只能保证收到的字节，无法识别上游文本语义错误。5.本方案：最小修改现有 graph-memory SKILL.md 和 tests/run.sh，不新增存储层。6.为什么：根因是 transport/process contract 而非 schema；新 writer 框架会扩大范围，仍不能替代正文回读。VERDICT: IMPLEMENT。
[2026-08-02 验证] 规则、回归测试与 skill validator 全部通过；VERDICT: IMPLEMENTED。

_updated: 2026-08-01 21:33:21_
### 历史扩展线：网络、Striped 与 vLLM 探索

type: `task` · status: `superseded` · confidence: 0.8 · importance: 0.95 · source: `user-direction`

当前核心方向：以 Ring Attention 为策略基础，推进与 vLLM 的 Block KV cache 集成。\n\n已完成/持有：\n1. hyp-net-speed：white-pearl 带宽矩阵与稳定性复测证明网络是首要瓶颈。\n2. claim-ring-derivatives：在 HCP 上实现并对比 Vanilla/Striped/ZigZag；Ring Flash 挂起。\n3. decision-ring-attn-chosen：用户确认以 Ring Attention 为模型策略继续推进。\n\n下一步开放工程线：\n- hyp-block-kv-vllm：Block KV cache + vLLM 集成。

_updated: 2026-08-01 18:40:49_
### CXL-class P2P fabric 可使 HCP 的异构 KV 容量聚合获得系统经济性

type: `hypothesis` · status: `superseded` · confidence: 0.75 · importance: 0.95 · source: `user-vision-and-network-evidence-2026-08-01`

待验证假设：当 CXL-class 或同等级 memory-semantic 高带宽低时延 P2P fabric 将邻居通信成本压到足够低时，HCP 能把闲置或代际混合的异构加速卡组织成低连接度的 context-capacity pool，以较低硬件成本服务单设备 KV 显存无法容纳的超长上下文。现有带宽矩阵只证明低带宽传统网络是瓶颈并支持高速互联的必要性；它没有在真实 CXL-class 设备上测量延迟、带宽、memory ordering、peer access、拓扑规模或端到端成本，因此不能证明该硬件条件充分，也不能外推到模型权重本身无法装入单节点的场景。

_updated: 2026-08-01 16:51:10_
### 通用高速 P2P 互联可释放 HCP 的异构资源经济性

type: `hypothesis` · status: `open` · confidence: 0.75 · importance: 0.95 · source: `user-vision-2026-08-02`

待验证假设：只要一种互联能够为相邻 worker 提供足够高带宽、低时延且可用的 P2P tensor 传输，无论其具体是 CXL、RDMA/RoCE、InfiniBand、UALink、PCIe peer access 还是未来通用 fabric，HCP 都可能把闲置、代际混合或品牌混合的加速卡组织成 context-capacity pool。网络演进降低环上传输成本后，异构资源的采购价格、存量复用和容量互补可能转化为更低的长上下文服务成本。现有证据只确认技术可行性和传统网络瓶颈；要证明经济性，还需至少比较单位 token 吞吐、首 token/逐 token 延迟、能耗、互联成本、设备价格、利用率以及同等 KV 容量下的同构基线。

_updated: 2026-08-01 16:51:10_
### 异构 CP 对网络速度敏感，CXL / 类 RDMA 互联可显著突破网线局限

type: `hypothesis` · status: `superseded` · confidence: 0.85 · importance: 0.95 · source: `user-direction`

HCP 跨节点推理性能对网络带宽极度敏感。\n\n证据（正常规模工作负载）：\n1. Qwen2.5-3B/1K 单节点 CUDA 0.14s，分布式 ~12s（~85× 慢）。\n2. Qwen2.5-3B/4K 单节点 CUDA 0.27s，分布式 ~40s（~148× 慢）。\n3. 分布式 3B 甚至慢于单节点 CPU（3B/1K 12s vs 7.8s；3B/4K 40s vs 29s）。\n4. 策略差异仅在 3B/1K 可见（ZigZag ~5%），4K 时被网络完全掩盖。\n5. 7B bf16 无法装入 pearl 16GB HIP，分布式 7B 在当前无量化路径下不可行。\n\n结论：对正常规模的 3B/7B 模型和 1K/4K seq，跨节点网络仍是首要瓶颈；CXL/类 RDMA 高速互联是 HCP 实用的必要前提。

_updated: 2026-08-01 14:42:10_
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
### HCP P2P KV ring 在 ≤1 Gbps 跨节点以太网下会成为端到端瓶颈

type: `belief` · status: `held` · confidence: 0.85 · importance: 0.95 · source: `ev-net-speed-matrix-20260629`

基于 white-pearl 限速矩阵：\n- 2.35 Gbps 基线 20.5 s\n- 1 Gbps 29.5 s（1.44x）\n- 500 Mbps 50 s（2.44x）\n- 100 Mbps 445 s（21.7x）\n\n在 Qwen2-0.5B-1M、seq=4096、max_tokens=5 的异构推理任务中，端到端 latency 随跨节点带宽下降呈非线性增长。低于 1 Gbps 时，P2P KV ring 的通信时间显著超过计算时间；100 Mbps 时通信完全主导总时间。\n\n推论：若要在生产环境中部署异构 CP 推理，需要 CXL / RDMA / 高速 NVLink 等级别的互联带宽，否则网络将把多卡聚合的显存优势抵消为极高的延迟惩罚。

_updated: 2026-06-29 14:32:15_
### Cargo 经 rsproxy 联网运行，不使用 offline

type: `preference` · status: `held` · confidence: 1.0 · importance: 0.9 · source: `user-correction-2026-08-03`

用户确认：Mac、white、pearl 的 Cargo 均使用各机 rsproxy 配置，正常构建和测试不要设置 CARGO_NET_OFFLINE=true，也不要传 --offline。离线模式可能因 replacement registry 的独立索引/cache 缺包而产生假性依赖失败；直接使用 rsproxy。

_updated: 2026-08-02 18:34:07_
### Tch reserved Tensor 分配尚不是可恢复的 admission 操作

type: `risk` · status: `open` · confidence: 0.95 · importance: 0.9 · source: `code-audit-2026-08-02`

ReservedPositionedKvShard::new_with_kind 通过 Tensor::zeros 直接分配 K/V storage，API 不返回 Result；逐层 capacity 合同能在写入前拒绝逻辑 under-capacity，但真实设备 OOM 是否可被 tch 安全转换为 request error 尚未证明。该张力不影响当前 CPU correctness，也不改变 capacity 数学；在稳定服务的物理 admission/release 节点前需要验证或封装 fallible allocation。

_updated: 2026-08-01 21:10:25_
### All-domain positioned helper 不能成为 worker runtime API

type: `risk` · status: `mitigated` · confidence: 1.0 · importance: 0.9 · source: `code-audit-2026-08-02`

24 层 correctness helper 在单进程同时持有所有 domain shard，适合作为数值与位置 union oracle；若直接提升为 runtime API，会违反每 worker 只有 predecessor/successor 且只持本地 KV 的核心边界。处理：保留为 test oracle，真实服务通过 worker-local cache adapter 接线。该风险不阻断框架推进。

_updated: 2026-08-01 20:42:55_
### 主流单请求细粒度分布式推理仍以同构并行组为主要设计点

type: `assumption` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `user-confirmed-research-position-2026-08-02`

用户确认的研究定位：主流 tensor、pipeline、context parallel 和 collective 通常假设设备容量、kernel 能力及通信性能近似对称；异构 serving 更常见的利用方式是将不同请求、模型或 prefill/decode 阶段放到不同资源池，而不是让不同代际或不同品牌加速卡以不均等份额共同处理同一 attention context。论文不得写成异构 GPU 绝对不能合作；正式表述仍需检索并区分框架支持、理论可运行、实际优化目标和公开验证范围。

_updated: 2026-08-01 17:46:39_
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
