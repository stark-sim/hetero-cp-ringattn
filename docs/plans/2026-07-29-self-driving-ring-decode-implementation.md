# Self-Driving Decode Ring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 Rust/tch HCP 路径实现单 packet、每层 `N-1` 跳、KV 显存硬上界内 compute-balanced、支持异步多请求 pipeline，并最终由 worker 自主采样续 token 的 decode ring。

**Architecture:** Coordinator 在 admission 时根据每节点剩余 KV 字节上界和 attention throughput 生成冻结的 request placement；worker 数据面只连接 predecessor/successor。每个 layer packet 携带 `h_residual + Q + O + LSE`，由唯一 starter、KV assignee 和 finisher 逐跳推进；R1 先保留 coordinator 的逐 token 控制，R2 改成有界异步 packet scheduler，R3 再把首 token/后续 token sampling 和 token loop 移到 worker。

**Tech Stack:** Rust 2021、tch-rs/libtorch、QUIC/quinn、serde/bincode、现有 online-softmax ring、Tokio bounded channels、纯 Rust fixed-point placement/counter RNG。

---

## 0. 实施边界与最终合同

计划只修改主仓 Rust HCP 路径，不回到已拆出的 vLLM plugin owner 模型，也不以 plugin 扩展点限制核心设计。旧 `HCP_RING_DECODE_RING` Q-ring 作为显式 fallback 保留到 R4 完成；新路径先由 `HCP_SELF_DRIVING_DECODE=1` 开启，硬件门通过后再改成默认开启、`=0` 回退。

必须始终成立：

```text
每 worker 数据面 peer 数                   = 2
每 request/token/layer 逻辑 packet 数       = 1
每 layer packet hop                         = N - 1
每 token attention hop                      = L * (N - 1)
Q projection / current KV append / MLP       = 各 1 次/layer
LM head / sampling                           = 各 1 次/token
packet tensor bytes                          = O(model_width), 与 context T 无关
durable KV                                   = 互斥、完备、无复制、低于冻结 reservation
实际 KV slab + metadata + decode workspace   = 始终低于节点 memory ledger hard bound
同一 request 在途 layer packet               <= 1
不同 request                                 = 独立异步流，无跨请求 barrier
```

### 动机剖析六问

1. **面对什么问题**：现有 Rust decode Q-ring 虽已把通信降为 `Q+LSE` 并切分增长 KV，但每个 worker 仍同步执行完整 forward、重复 Q/K/V/MLP/logits；coordinator 仍逐 token 采样，且没有真实 ring 内多请求 pipeline。
2. **现状是什么**：`LlamaModel::forward()` 是 embedding 到 LM head 的同步调用栈；`HcpRingAttentionBackend::forward()` 一次性做 Q/K/V/O；`WorkerRuntime::run()` 阻塞调用 `decode_request()`；`RingPacket` 只有 layer/Q/O/LSE/scale；`DecodeDone` 默认所有 worker 都返回 logits。
3. **做完能怎样**：每层仅一个 packet 和一个 logical forward；durable KV 受 admission reservation 硬约束；两个以上请求能不同步地占据不同 ring stage；最终 coordinator 不做 embedding/logits/sample。验证同时覆盖 token、KV、exact-once、hop、topology、lifecycle 和异构 hidden drift。
4. **其他人怎么做**：Ring Attention 用 P2P KV ring 和 online softmax；LoongServe 类 decode 传 Q/accumulator 而不传历史 KV；vLLM 用 paged KV、continuous batching 和 per-request block table；pipeline runtime 用 ingress/ready queue/device executor/egress 解耦阻塞。它们提供数学、缓存和调度机制，但没有可直接复用的“跨 CUDA/ROCm/MPS、两 peer、无 collective、序列维 KV 永久切分”的完整 runtime。
5. **我们怎么做**：复用已验证的 online-softmax merge、QUIC per-layer stream 和 request KV context；新增确定性 placement/reservation、decode packet header、拆分 attention 投影原语、layer continuation、bounded packet scheduler、counter RNG 和自治 token loop。
6. **为什么这样做**：HCP 的产品目标是突破单节点 KV 显存墙，并允许部分可达的异构 P2P 网络。TP/all-reduce、全量 KV owner 或 vLLM P/D 搬移都会破坏这两个约束；单 packet 自驱动 continuation 是同时保持线性拓扑、无冗余 forward 和 KV 硬切分的最小方案。

### 牺牲四问与结论

1. **默认为什么存在**：全节点同步 forward 让每个 worker 都有完整 hidden/logits，控制流简单，并允许每节点独立发 packet。
2. **牺牲什么**：放弃全节点结果复制、单请求内多 packet 并行、故障后继续运行和 v1 retry；一个请求的一层 partial 变成串行关键路径。
3. **被牺牲者的用途**：结果复制简化同步和容错，多 packet 能在同构高速互联上提高单请求并行度，retry 能掩盖短暂链路错误。
4. **对本项目的意义**：HCP 当前优先级是显存语义和拓扑可行性；吞吐由多请求 packet pipeline 恢复，单请求 fail-stop 可接受。结论：**implement**；副本、retry 和 kernel micro-batch 在正确性闭环后另立任务。

## 1. Placement 数学和冻结数据结构

对 worker `i` 和 request `r`，resource profile 必须显式上报或配置：

```text
B_i^K[l]    = layer l 每个本地 position 的 K 字节数
B_i^V[l]    = layer l 每个本地 position 的 V 字节数
G_i         = KV allocation granularity
H_i         = request 在该 worker 分到任意 KV 后的固定 metadata/allocator overhead
W_i(m)      = 扫描最多 m 个本地 KV position 时的保守 decode workspace 上界；
              W_i(0)=0，m>0 时才收 fixed + per-unit
C_i         = 模型加载后，由设备 telemetry 与显式 KV budget 取保守值，
              再扣除 packet queue/static runtime reserve 后的可用字节
```

`C_i/H_i/G_i/W_i` 缺失时 self-driving admission 必须拒绝，不能猜。`C_i` 不能由当前 coarse `capacity_mb()` 直接换算；CUDA/HIP 可取模型加载后的 device-free telemetry 与配置上限的较小值，MPS/无可靠 telemetry 的 backend 必须提供显式 `kv_budget_bytes`。第一版 workspace profile 可用显式保守 affine bound `W_i(0)=0`、`W_i(m>0)=workspace_fixed_i + workspace_per_kv_unit_i*m`；R4 再用 backend high-water telemetry 验证该 bound。

对 request `r`：

```text
T_max(r)      = prompt_len + max_new_tokens
P_i(r)        = T_max(r) * sum_l(B_i^K[l] + B_i^V[l])
R_i(r,z)      = exact rounded persistent bytes when worker i owns fraction z
A_i           = sum of persistent+H reservations for current active requests
W_i_active    = max workspace bound of current active requests, or 0
u_i(r)        = max z in [0,1] where
                A_i + R_i(r,z) + H_i*indicator(z>0)
                    + max(W_i_active, W_i(ceil(z*T_max))) <= C_i
admit iff sum_i(u_i) >= 1

minimize    max_i(x_i / attention_rate_i)
subject to  sum_i(x_i) = 1
            0 <= x_i <= u_i

x_i = min(u_i, lambda * attention_rate_i)
```

`lambda` 用确定性 breakpoint/water-filling 求出；`x_i` 只用于生成整数 tickets，runtime owner 判断不使用浮点。若任一参与节点 throughput 缺失，回退到明确的 capacity-only 目标 `x_i = u_i / sum_j(u_j)` 并记录原因，不混用部分实测、部分猜测的 rate。接近容量墙时 `sum(u_i) -> 1`，唯一可行解自动变成 `x_i=u_i`。

冻结计划：

```rust
pub struct RequestPlacementPlan {
    pub request_id: u64,
    pub ring_epoch: u64,
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub prompt_tokens_per_worker: Vec<usize>,
    pub kv_calendar: Vec<usize>,
    pub kv_phase: usize,
    pub starter_phase: usize,
    pub reserved_bytes: Vec<u64>,
    pub placement_hash: u64,
}

impl RequestPlacementPlan {
    pub fn kv_assignee(&self, token_position: usize, layer_idx: usize) -> usize {
        self.kv_calendar[(self.kv_phase + token_position + layer_idx)
            % self.kv_calendar.len()]
    }
}
```

`u_i` 通过单调整数搜索求出；water-filling 仍只接收最终 hard cap。`H_i` 必须整笔计费，不能放进 `P_i` 后随 `x_i` 缩放。workspace 在 v1 单线程 device executor 下不会并行存在，所以 active request 之间取 `max` 而不是求和；若未来 kernel micro-batch/多 stream 并行，必须先修改 ledger 合同。

`prompt_tokens_per_worker` 保持连续 chunk 的传输/存储形态，但 chunk 长度必须由同一组 bounded target `x_i` 量化，不能继续按纯 capacity 另算；否则长 prompt 会永久决定各节点的 attention scan 负载，使 compute balance 对 decode growth 的优化失效。decode growth 使用二维 calendar。reservation 分别对 prompt chunk 和 `max_new_tokens` 的每个 `(position,layer)` 做精确整数计数；每层 K slab 和 V slab 是两个物理 allocation，必须分别按 `G_i` 向上取整后再求和，不能把 `(K+V)` 合并 round 一次。最后加实际启用节点的一次 `H_i`；ledger 另外维护 active `max(W_i)`。浮点份额只用于求目标，不作为 admission 依据。

## 2. 依赖顺序

```text
Task 1 placement/reservation
   -> Task 2 route/exact-once state machine
   -> Task 3 control-plane admission profile
   -> Task 4 packet wire schema
   -> Task 5 cache read/append split
   -> Task 6 attention primitive split
   -> Task 7 layer continuation
   -> Task 8 R1 worker/coordinator vertical slice
   -> Task 9 bounded async scheduler
   -> Task 10 R2 multi-request pipeline
   -> Task 11 counter RNG
   -> Task 12 R3 autonomous token loop
   -> Task 13 lifecycle/failure/observability
   -> Task 14 R4 hardware ladder and default switch
```

在 Task 8、10、12、14 后停下来审查 checkpoint；不要把四个阶段合成一个不可定位的大提交。

### Task 1: 有显存硬上界的 placement 与 reservation ledger

**Files:**
- Create: `rust/src/distributed/placement.rs`
- Modify: `rust/src/distributed/mod.rs`
- Modify: `rust/src/capacity.rs`

**Step 1: 写 water-filling、固定 overhead 和容量墙失败测试**

在 `placement.rs` 添加：

```rust
#[test]
fn compute_balance_respects_memory_caps() {
    let p = plan(&profiles(
        &[600, 300, 100],      // free KV units
        &[100, 300, 200],      // attention rates
    ), request(600));
    assert!(p.target_units[1] <= 300);
    assert!(p.target_units[2] <= 100);
    assert_eq!(p.target_units.iter().sum::<u64>(), 600);
}

#[test]
fn near_capacity_wall_becomes_capacity_only() {
    let p = plan(&profiles(&[600, 300, 100], &[1, 1000, 1000]), request(1000));
    assert_eq!(p.target_units, vec![600, 300, 100]);
}

#[test]
fn rejects_when_aggregate_free_kv_is_insufficient() {
    assert!(matches!(plan_result(&profiles(&[40, 30, 20], &[1, 1, 1]), request(100)),
        Err(PlacementError::InsufficientKv { .. })));
}

#[test]
fn fixed_overhead_is_zero_for_zero_share_and_charged_once_for_nonzero_share() {
    let profile = test_profile(
        vec![(24, 24), (24, 24)], // (B_i^K[l], B_i^V[l])
        64,           // G_i
        32,           // H_i
        128,          // workspace fixed
        2,            // workspace bytes per local KV position
    );
    let zero = reservation_for_layer_counts(&profile, &[0, 0]).unwrap();
    assert_eq!(zero.additive_bytes, 0);
    assert_eq!(zero.workspace_bytes, 0);

    let nonzero = reservation_for_layer_counts(&profile, &[1, 0]).unwrap();
    assert_eq!(nonzero.persistent_bytes, 128); // round_up(K, G_i) + round_up(V, G_i)
    assert_eq!(nonzero.fixed_overhead_bytes, 32); // exactly once, not per layer
    assert_eq!(nonzero.additive_bytes, 160);
    assert_eq!(nonzero.workspace_bytes, 130);
}

#[test]
fn active_requests_each_pay_metadata_but_share_serial_workspace_by_max() {
    let mut ledger = KvReservationLedger::new(vec![1_000]);
    ledger.reserve(11, vec![breakdown(600, 20, 80)]).unwrap();
    ledger.reserve(12, vec![breakdown(100, 20, 50)]).unwrap();
    assert_eq!(ledger.additive_reserved_bytes(), &[740]);
    assert_eq!(ledger.workspace_reserved_bytes(), &[80]);
    assert_eq!(ledger.total_reserved_bytes(), &[820]);
    ledger.release(11).unwrap();
    assert_eq!(ledger.total_reserved_bytes(), &[170]); // 100 + 20 + max(50)
}
```

**Step 2: 运行并确认失败**

Run: `cargo test --manifest-path rust/Cargo.toml --no-default-features distributed::placement::tests -- --nocapture`

Expected: FAIL，`distributed::placement` 不存在。

**Step 3: 实现 fixed-point water-filling、prompt chunk、smooth calendar 和精确 reservation**

实现 `WorkerKvProfile`、`RequestDemand`、`KvReservationBreakdown`、`RequestPlacementPlan`、`PlacementError`、`reservation_for_layer_counts()` 和 `build_placement_plan()`。profile 明确包含 `request_fixed_kv_overhead_bytes`、`kv_allocation_granularity_bytes`、workspace fixed/per-unit bound；prompt contiguous chunk counts 和 decode tickets 都从同一 `x_i` 受上界量化。tickets 使用固定总 quantum（默认 256），经有 cap 的最大余数法量化后再用 smooth weighted round-robin 铺开。reservation 通过 prompt exact counts 与 calendar 周期前缀和精确计算，而非 `ceil(x*T)`。

量化后必须再次逐节点验证 `active_additive + new_additive + max(active_workspace,new_workspace) <= C_i`。若最大余数舍入或 allocator granularity 使某节点越界，先把剩余 quantum 分配给仍有 byte headroom 的节点；不存在合法整数分配时拒绝 admission，不能用浮点可行性掩盖整数/allocator 不可行。

同时让旧 `allocate_by_capacity()` 继续存在，避免一次改坏 prefill；新 planner 只调用它的最大余数 helper，不复用 CPU/MPS “容量即速度”的旧假设。

**Step 4: 增加 ledger 并验证并发 reservation/release**

```rust
#[test]
fn ledger_counts_all_active_requests_and_releases_exactly_once() {
    let mut ledger = KvReservationLedger::new(vec![1000, 500]);
    ledger.reserve(11, &[600, 200]).unwrap();
    assert!(ledger.reserve(12, &[500, 100]).is_err());
    ledger.release(11).unwrap();
    assert_eq!(ledger.reserved_bytes(), &[0, 0]);
    assert!(matches!(ledger.release(11), Err(PlacementError::UnknownReservation(11))));
}
```

再覆盖：同一 request 的 `H_i` 不得按 layer 重复计费；两个 request 各计一次；释放较大 workspace request 后 ledger 的 `max(W)` 降到剩余 request；`G_i` round-up 后越界必须拒绝。

Run: `cargo test --manifest-path rust/Cargo.toml --no-default-features distributed::placement::tests -- --nocapture`

Expected: PASS；额外断言 calendar 中任意节点的固定层前缀误差不超过一个 ticket quantum，`placement_hash` 对相同输入稳定。

**Step 5: Commit**

```bash
git add rust/src/distributed/placement.rs rust/src/distributed/mod.rs rust/src/capacity.rs
git commit -m "feat: add bounded decode placement planner"
```

### Task 2: 纯 Rust route、角色和 exact-once 状态机

**Files:**
- Create: `rust/src/model/decode.rs`
- Modify: `rust/src/model/mod.rs`

**Step 1: 写 N=1/2/3/4 路由和模数共振失败测试**

```rust
#[test]
fn route_visits_every_worker_once_and_stops_at_predecessor() {
    for n in 1..=4 {
        for starter in 0..n {
            let route = layer_route(starter, n);
            assert_eq!(route.len(), n);
            assert_eq!(route[0], starter);
            assert_eq!(*route.last().unwrap(), (starter + n - 1) % n);
        }
    }
}

#[test]
fn l24_n3_has_fixed_sampler_without_extra_handoff() {
    let request_phase = 0;
    assert_eq!(
        sampler_for_token(request_phase, 0, 24, 3),
        sampler_for_token(request_phase, 1, 24, 3),
    );
    assert_ne!(
        sampler_for_token(0, 0, 24, 3),
        sampler_for_token(1, 0, 24, 3),
    );
    assert_eq!(attention_hops_per_token(24, 3), 48);
}
```

**Step 2: 运行并确认失败**

Run: `cargo test --manifest-path rust/Cargo.toml --no-default-features model::decode::tests -- --nocapture`

Expected: FAIL，模块和函数未定义。

**Step 3: 实现 header、角色纯函数、commit key 和 phase 状态**

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecodeKey {
    pub ring_epoch: u64,
    pub request_id: u64,
    pub token_position: u64,
    pub layer_idx: u32,
}

pub enum DecodePhase { Layer, TokenHandoff, Cancel }
pub enum LocalRole { Starter, Relay, Finisher, StarterFinisher }
```

`validate_header()` 必须重算 finisher/assignee、检查 `placement_hash`、layer 单调性和 `hops_remaining`；不接受 packet 自报角色作为真相。

`model::decode` 在 `model/mod.rs` 中无条件导出；本 Task 只放纯 Rust contract/route/state，不引用 `Tensor`。Task 7 才在同一模块增加 `#[cfg(feature = "tch-backend")]` 的 model driver，因此这里的 no-default test 不加载 libtorch。

**Step 4: 用 mock simulator 覆盖三种 assignee 重合和反例**

测试 `assignee==starter/middle/finisher`，每层断言：partial=N、Q=1、KV project=1、KV append=1、MLP=1、hop=N-1。再注入 duplicate packet、错误 epoch、错误 assignee、finisher 提前 continuation，Expected: 全部被拒绝且 append counter 不增加。

Run: `cargo test --manifest-path rust/Cargo.toml --no-default-features model::decode::tests -- --nocapture`

Expected: PASS。

**Step 5: Commit**

```bash
git add rust/src/model/decode.rs rust/src/model/mod.rs
git commit -m "feat: define self-driving decode state machine"
```

### Task 3: Admission profile、冻结 placement 和控制协议

**Files:**
- Modify: `rust/src/distributed/protocol.rs`
- Modify: `rust/src/worker_sdk/backend.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`
- Modify: `rust/src/worker_sdk/runtime.rs`
- Modify: `rust/src/distributed/coordinator.rs`
- Modify: `rust/src/distributed/scheduler.rs`

**Step 1: 写 versioned hello、mode negotiation 和 admission round-trip 失败测试**

用带固定 envelope 的 bincode `WorkerHelloWire` 替换当前固定 16-byte `WorkerHandshake`。envelope 先读固定 `magic + hello_version + payload_len`，magic 不匹配立即返回 incompatible，不能把旧 `domain_id` 字节误读成长度后等到超时。新增 `WorkerResourceProfileWire`、`RequestPlacementPlanWire`、`SamplingPlanWire` 和：

```rust
WorkerCommand::NegotiateMode { protocol_version, mode, ring_epoch }
WorkerResponse::ModeAccepted { protocol_version, mode, ring_epoch }
WorkerResponse::ModeRejected { supported_versions, capabilities, reason }
WorkerResponse::DataPlaneReady { protocol_version, mode, ring_epoch }
WorkerCommand::AdmitRequest { request_id, placement, sampling }
WorkerResponse::AdmissionAccepted { request_id, placement_hash }
WorkerResponse::AdmissionRejected { request_id, reason }
```

hello 必须携带 control protocol versions、`LegacyQring|SelfDrivingV1` capabilities、model fingerprint、层数，以及 `usable_memory_bytes/B_i^K/B_i^V/G_i/H_i/W_i` profile。测试 bincode round-trip 保留全部 memory bounds、optional attention rate、reservation、两个 phase 和 hash；错误 magic、旧 16-byte hello、oversized payload 或未知 version 必须快速返回明确 incompatibility，不能按长度猜模式或等 QUIC timeout。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend distributed::protocol::tests -- --nocapture`

Expected: FAIL，variants 不存在。

**Step 3: 扩展 handshake 和 backend profile**

把当前 `capacity_mb()` 升级为：

```rust
fn resource_profile(&self) -> Result<WorkerResourceProfile, ResourceProfileError> {
    Ok(WorkerResourceProfile {
        usable_memory_bytes: self.available_kv_budget_bytes()?,
        k_bytes_per_position_per_layer: self.model.k_bytes_per_layer_position(),
        v_bytes_per_position_per_layer: self.model.v_bytes_per_layer_position(),
        kv_allocation_granularity_bytes: configured_allocator_granularity(),
        request_fixed_kv_overhead_bytes: configured_request_overhead(),
        decode_workspace_fixed_bytes: configured_workspace_fixed(),
        decode_workspace_bytes_per_local_kv_unit: configured_workspace_slope(),
        attention_units_per_sec: configured_attention_rate(),
        model_fingerprint: self.model.model_fingerprint(),
        capabilities: self.backend_capabilities(),
    })
}
```

`available_kv_budget_bytes()` 在模型加载后求值，并已扣除 static runtime/packet buffer reserve；无可靠设备 free-memory API 时要求显式配置。`attention_units_per_sec` 第一版只读显式 worker 配置；缺失时 planner 走 capacity-only，不在 admission 临时跑重 benchmark。`C_i/G_i/H_i/W_i` 缺失时 self-driving mode 不可 admission。所有 worker 的 model fingerprint、层数和 KV layout 不一致时拒绝 negotiation/admission。Task 3 只落 wire/negotiator；tch backend 在 Task 4-7 完成前仍不得 advertise `SelfDrivingV1`，避免半成品路径被协商成功。

**Step 4: 由 coordinator 唯一选择 mode，再接入 ledger**

只有 coordinator 读取 `HCP_SELF_DRIVING_DECODE`/CLI 并选择集群 mode；worker 只上报 backend capability，不独立读取该 flag 决定 response 合同。重构当前把 peer/control 混在一起的 `WorkerRuntime::setup_network()`：worker 加载模型后先只建立 coordinator control connection，发送 hello 并完成 negotiation；全员 ack 后才并发建立 predecessor/successor streams，并回报同一 tuple 的 `DataPlaneReady`。coordinator 收齐 ready 后才允许 admission/prefill。请求 self-driving 但任一 worker 不支持、version 不同、ack/ready mode 或 epoch 不同，必须在任何 peer packet 前 fail-stop；不得静默 mixed-mode 或逐 worker fallback。

协商成功后，coordinator 在 prefill 前生成计划、reserve、向所有 worker 发送同一冻结计划；任一 worker 拒绝时向已接受 worker 发送 release 并回滚 ledger。请求正常/失败结束都通过同一 guard 释放。

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend distributed::scheduler::tests -- --nocapture`

Expected: PASS，新增“第二请求超过容量被拒、第一请求释放后可重试”、self-driving coordinator + legacy-only worker、version mismatch、一个 worker 错 epoch、worker 本地 env 不同但 negotiated mode 仍一致等测试；前四个错误场景都必须在 prefill/packet 前拒绝且 ledger 为零。

**Step 5: Commit**

```bash
git add rust/src/distributed/protocol.rs rust/src/worker_sdk/backend.rs rust/src/worker_sdk/tch_backend.rs rust/src/worker_sdk/runtime.rs rust/src/distributed/coordinator.rs rust/src/distributed/scheduler.rs
git commit -m "feat: freeze decode placement at admission"
```

### Task 4: Versioned self-driving packet wire schema

**Files:**
- Modify: `rust/src/model/transport/block.rs`
- Modify: `rust/src/model/transport/trait.rs`
- Modify: `rust/src/model/transport/mock.rs`
- Modify: `rust/src/model/transport/tcp.rs`
- Modify: `rust/src/distributed/transport/quic.rs`
- Modify: `rust/src/model/transport/mod.rs`

**Step 1: 写 TCP/QUIC codec 失败测试**

新 packet：

```rust
pub struct RingPacket {
    pub header: DecodePacketHeader,
    pub h_residual: Tensor,
    pub q: Tensor,
    pub o: Tensor,
    pub lse: Tensor,
    pub scale: f64,
}
```

测试 round-trip 后 header 和四个 tensor 完整；protocol version、dtype、shape、endianness、placement hash 错误均返回具体错误。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend model::transport::tests -- --nocapture`

Expected: FAIL，旧 packet 缺 header/h_residual。

**Step 3: 实现一个共享 codec metadata helper**

TCP 与 QUIC 必须调用同一 `RingPacketMeta::from_packet()/validate()`，避免两套 JSON 字段漂移；只保留流 I/O 差异。旧 Q-ring packet 明确使用 `protocol_version=0` legacy decoder，新 self-driving 使用 v1，禁止按字段缺失猜版本。

**Step 4: 证明 packet size 不随 context 增长**

同一模型 shape，构造 `token_position=16` 和 `1_000_000` 两个 packet，断言 tensor payload 完全相同、总 frame 仅允许十进制/metadata 常数级差异；记录 `tensor_payload_bytes` 为主协议指标。Qwen2-0.5B BF16 预期约 5.3 KiB，不把估算写成硬编码常量。

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_self_driving_packet -- --nocapture`

Expected: PASS。

**Step 5: Commit**

```bash
git add rust/src/model/transport rust/src/distributed/transport/quic.rs
git commit -m "feat: add versioned self-driving ring packet"
```

### Task 5: Reservation-backed KV slab，消除全 shard 临时拼接

**Files:**
- Modify: `rust/src/model/cache.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`

**Step 1: 写预分配、view、append-once 和无增长分配失败测试**

新增 self-driving 专用 `ReservedKvCache`；worker 处理 `AdmitRequest` 时按本 worker 每层 exact planned count 一次性分配 K/V slab，只有全部 allocation 成功且 actual bytes 不超过 plan 才能返回 `AdmissionAccepted`，因此 prefill 前物理内存已经兑现 reservation：

```rust
fn with_capacity(layout: KvLayout, planned_tokens: usize) -> Result<Self, ModelError>;
fn history_view(&self) -> Option<(Tensor, Tensor)>;
fn append_reserved(
    &mut self,
    commit_key: DecodeKey,
    new_k: &Tensor,
    new_v: &Tensor,
) -> Result<(), ModelError>;
```

测试：空 view；prefill bulk copy；decode append 后长度 +1；超 planned count 拒绝；重复 commit key 不写；zero-share 不分配 slab。注入第 `k` 层 slab allocation failure，worker 必须 `AdmissionRejected`，coordinator 回滚其他 worker slab 和 ledger，且不能发送 prefill。instrumented allocation stats 必须证明 slab 只在 init 分配，`history_view`/append 不随 `seq_len` 新建 KV-sized storage。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend model::cache::tests -- --nocapture`

Expected: FAIL，reserved cache 不存在。

**Step 3: 实现固定 slab，禁止 self-driving path 的 `Tensor::cat`**

K/V slab shape 为 `[batch,kv_heads,planned_local_tokens,head_dim]`；append 用 `narrow(...).copy_()` 写入下一 reserved slot，history 用 `narrow` view，不重分配/不拼接。prefill chunk 也写入同一 slab。assignee 先把 current K/V 写 reserved slot，再从包含 current 的 view 算本地 partial；此后任何 compute/send 错误都按 v1 fail-stop 清理，不 retry。

现有 `ContiguousKvCache` 和 `BlockTableKvCache` 的 `Tensor::cat` 仅保留 legacy。`SelfDrivingV1` capability 必须依赖 `ReservedKvCache`；不得把当前 block table 称为 block-aware，因为 `update/get_kv` 仍会 concat。未来真正 paged kernel 或逐 block online merge 另立任务。

**Step 4: 验证 physical high-water 与 legacy 回归**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_reserved_kv_cache -- --nocapture`

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_update_sharded -- --nocapture`

Expected: PASS；reserved cache storage capacity/address 在 append 前后稳定，reported allocation 不超过 rounded slab reservation；legacy contiguous/block-table 行为不变。

**Step 5: Commit**

```bash
git add rust/src/model/cache.rs rust/src/worker_sdk/tch_backend.rs
git commit -m "feat: allocate decode KV from frozen reservations"
```

### Task 6: 拆分 Q、K/V、partial 和 O projection 原语

**Files:**
- Modify: `rust/src/model/attention/backend.rs`
- Modify: `rust/src/model/attention/ring.rs`

**Step 1: 写 primitive composition 等价失败测试**

用小 synthetic layer 比较旧 `forward()` 和组合：

```text
project_q(h_norm) + project_kv(h_norm)
-> local_partial(history + current)
-> output_projection(o_acc)
```

分别覆盖 empty history、GQA heads、assignee starter/middle/finisher 所需调用序列；断言 diff `<1e-5`。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_decode_primitives -- --nocapture`

Expected: FAIL，primitive 方法未暴露。

**Step 3: 提取无 transport 副作用的 `DecodeAttentionOps`**

接口至少包含：

```rust
fn project_q(&self, h_norm: &Tensor, position: &Tensor) -> Tensor;
fn project_kv(&self, h_norm: &Tensor, position: &Tensor) -> (Tensor, Tensor);
fn partial(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> AttentionState;
fn merge(&self, acc: AttentionState, local: AttentionState) -> AttentionState;
fn output_projection(&self, o: &Tensor) -> Tensor;
```

`AttentionState` 统一 `(O,LSE)` identity/empty semantics。旧 `forward()` 改为调用这些原语，避免新旧路径复制数学。

**Step 4: 跑 attention 全模块回归**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend model::attention::ring::tests -- --nocapture`

Expected: PASS，包括已有 Q-ring 三域、threaded forward 和 uneven correctness。

**Step 5: Commit**

```bash
git add rust/src/model/attention/backend.rs rust/src/model/attention/ring.rs
git commit -m "refactor: expose decode attention primitives"
```

### Task 7: Decoder layer continuation 与单请求 model driver

**Files:**
- Modify: `rust/src/model/layers/mod.rs`
- Modify: `rust/src/model/model.rs`
- Modify: `rust/src/model/decode.rs`

**Step 1: 写 finisher continuation 等价失败测试**

新增 `DecoderLayer::finish_decode(h_residual, o_acc)`，与旧 layer forward 的 `O projection + residual + post norm + MLP` 对照。另用 2 层、3 worker synthetic model 断言 hidden 只在每层 finisher 产生一次并转成下一层 starter seed。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_finish_decode -- --nocapture`

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_self_driving_model_driver -- --nocapture`

Expected: FAIL，continuation API 不存在。

**Step 3: 实现 `LlamaModel::decode_ring_request()`**

所有 worker 同时进入该函数；每层每 worker 只处理一个角色动作：starter 用本地 hidden 建 seed，relay 收包后 merge，finisher 完成本层并保留下一层 hidden。assignee 从 packet `h_residual` 重算 input RMSNorm，project current K/V，先用 `append_reserved()` 写入已分配 slot 并登记 commit key，再从包含 current 的 narrow view 计算唯一 local partial；finisher 若也是 assignee，必须先完成 append/partial 再断言 packet commit。append 后任一错误都 fail-stop，不在 v1 重放。

函数返回：

```rust
pub enum DecodeOutcome {
    Finished { request_id: u64, finisher: usize, logits: Tensor },
    Participated { request_id: u64 },
}
```

**Step 4: exact-once counters 与 reference 对比**

对 `N=3,L=3` 强制三种 assignee 重合；对 `N=3,L=24` 断言 48 hops、每节点 starter/finisher 各 8 次、唯一 logits。最终 logits/token 对单节点 synthetic reference。

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_self_driving_model_driver -- --nocapture`

Expected: PASS。

**Step 5: Commit**

```bash
git add rust/src/model/layers/mod.rs rust/src/model/model.rs rust/src/model/decode.rs
git commit -m "feat: continue decode layers at ring finisher"
```

### Task 8: R1 coordinator-triggered 单请求垂直切片

**Files:**
- Modify: `rust/src/worker_sdk/backend.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`
- Modify: `rust/src/worker_sdk/runtime.rs`
- Modify: `rust/src/distributed/protocol.rs`
- Modify: `rust/src/distributed/coordinator.rs`
- Modify: `scripts/run_distributed_2node_smoke.sh`

**Step 1: 写 `Finished/Participated` protocol 和 runtime 失败测试**

把 decode response 改为带 outcome 的 variant；测试三 worker 结果集合必须恰有一个 `Finished`，finisher id 与 route 公式一致，其余为 `Participated`。coordinator 若收到 0 个或多个 finisher 必须报错。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_unique_decode_finisher -- --nocapture`

Expected: FAIL，runtime 仍要求所有 backend 返回 logits。

**Step 3: 接入已协商的 backend 路径**

`WorkerBackend` 新增 capability 和 `decode_ring_request()` 默认 unsupported；tch 实现调用 Task 7 driver。只有本 Task 把 reserved cache、packet codec、layer driver 和 outcome runtime 全部接通后，tch backend 才 advertise `SelfDrivingV1`。`WorkerRuntime` 只根据 Task 3 已确认的 `NegotiatedMode` 选择 outcome response 或 legacy `decode_request()`，不得再读取本地 `HCP_SELF_DRIVING_DECODE` 决定 wire contract。收到与 negotiated mode/version/epoch 不符的命令或 packet 立即返回 protocol error。

Coordinator 仍向所有 worker 单播同一 token command并收齐 response，但只从唯一 finisher 取 logits/sample；这只是 R1 控制面屏障，数据面仍是单 packet。

**Step 4: 跑完整本地单请求回归**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend -- --nocapture`

Expected: 全部 PASS；coordinator negotiated legacy 对照也 PASS；mixed capability/version 在 prefill 前拒绝。然后在非 sandbox host 运行：`HCP_SELF_DRIVING_DECODE=1 HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps bash scripts/run_distributed_2node_smoke.sh`。这里只由 coordinator 进程读取 self-driving flag，worker 以 capability hello + negotiated command 为准。

Expected: token/text 与 legacy A/B 一致；每层 `N-1` hop；唯一 logits producer。

**Step 5: Commit 和 checkpoint 审查**

```bash
git add rust/src/worker_sdk rust/src/distributed scripts/run_distributed_2node_smoke.sh
git commit -m "feat: run single-request self-driving decode ring"
```

Checkpoint R1 不允许用“所有 worker 完成”替代 exact-once/hop/KV 证据。

### Task 9: Bounded packet ingress、ready queue 和 backpressure

**Files:**
- Create: `rust/src/distributed/decode_scheduler.rs`
- Modify: `rust/src/distributed/mod.rs`
- Modify: `rust/src/worker_sdk/backend.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`
- Modify: `rust/src/model/transport/trait.rs`
- Modify: `rust/src/distributed/transport/quic.rs`

**Step 1: 写小队列/慢 relay/公平性失败测试**

用 fake backend 和 depth=2 transport：A 连续注入 10 个 ready packet，B/C 各 1 个；断言 queue 永不超过 2，blocked send 不丢包，B/C 在有限 tick 内执行。加入一个 stalled request，其他 request 仍前进。

**Step 2: 运行并确认失败**

Run: `cargo test --manifest-path rust/Cargo.toml --no-default-features distributed::decode_scheduler::tests -- --nocapture`

Expected: FAIL，scheduler 不存在。

**Step 3: 实现纯控制 scheduler 和单线程 device executor adapter**

```text
control ingress task -> bounded command channel
peer poll            -> per-request ready queues
round-robin arbiter  -> one tensor compute quantum
pending egress       -> try_submit successor packet
completion/error     -> coordinator event channel
```

Tensor 只在 worker 主计算线程访问。网络 task 只处理 bytes/frame。`try_submit_send_packet()` 返回 `Queued|Backpressured`；backpressured packet 留在唯一 pending-egress slot，不复制 tensor。

`distributed::decode_scheduler` 只保存 request key、queue metadata 和 opaque work handle，因此可在 `--no-default-features` 下验证公平性/backpressure；`worker_sdk` 的 tch adapter 才拥有 `Tensor` 并执行 work handle。不要把 tensor 或 libtorch 类型泄漏进纯 scheduler。

**Step 4: 接入 R1 blocking wrapper**

`decode_ring_request()` 改成在单请求场景循环 `scheduler.tick()` 直到得到本 request outcome；这样 R1 行为不变，R2 可直接复用同一 engine，不维护第二套控制流。

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_decode_scheduler -- --nocapture`

Expected: PASS，R1 tests 仍 PASS。

**Step 5: Commit**

```bash
git add rust/src/distributed/decode_scheduler.rs rust/src/distributed/mod.rs rust/src/worker_sdk/backend.rs rust/src/worker_sdk/tch_backend.rs rust/src/model/transport/trait.rs rust/src/distributed/transport/quic.rs
git commit -m "feat: add bounded decode packet scheduler"
```

### Task 10: R2 独立异步多请求 pipeline

**Files:**
- Modify: `rust/src/worker_sdk/runtime.rs`
- Modify: `rust/src/distributed/decode_scheduler.rs`
- Modify: `rust/src/distributed/coordinator.rs`
- Modify: `rust/src/distributed/scheduler.rs`
- Create: `scripts/run_self_driving_ring_concurrent_local.sh`

**Step 1: 写非 lockstep 的 integration 失败测试**

两个不同 prompt、不同 `starter_phase`、不同 max tokens；人为让 reqA 某个 partial 慢，断言 reqB 能在 reqA 下一层前完成至少一个自己的 layer。验证不能只看两个请求最终完成，必须检查 event trace 中存在交错。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_async_request_interleaving -- --nocapture`

Expected: FAIL，当前 `WorkerRuntime` 阻塞于单次 decode。

**Step 3: 把 coordinator receive 与 device tick 解耦**

runtime 启动 control receive task；主线程持续 drain command、poll packet、tick device、flush response。Coordinator 逐 request 发送 decode trigger，不再用 `DecodeBatch` 形成全局 iteration barrier；每个 request 下一 token 何时触发只依赖自己的上一个 outcome。

所有 request state key 至少包含 `(ring_epoch,request_id)`；packet/commit 还包含 token/layer。release 先进入 Draining，只有 in-flight=0 才清 KV 和 reservation。

**Step 4: 跑 concurrency、backpressure 和不同 prompt 正确性**

Run: `HCP_SELF_DRIVING_DECODE=1 bash scripts/run_self_driving_ring_concurrent_local.sh`

Expected: 两请求 token 各自等于独立 reference；trace 证明交错；`max_ready_depth<=configured_depth`；无跨请求 KV/commit/release；不同 request phase 在节点角色计数中可见。

**Step 5: Commit 和 checkpoint 审查**

```bash
git add rust/src/worker_sdk/runtime.rs rust/src/distributed/decode_scheduler.rs rust/src/distributed/coordinator.rs rust/src/distributed/scheduler.rs scripts/run_self_driving_ring_concurrent_local.sh
git commit -m "feat: pipeline independent decode requests"
```

Checkpoint R2 只声明 packet-level pipeline；没有兼容 packet batching 和 kernel 证据时，不称为 kernel-level continuous batching。

### Task 11: 与 worker/调度顺序无关的 counter RNG sampling

**Files:**
- Modify: `rust/src/model/sampling.rs`
- Modify: `rust/src/distributed/protocol.rs`

**Step 1: 写调度无关和节点无关失败测试**

```rust
#[test]
fn counter_rng_is_independent_of_sampler_and_interleaving() {
    let a = sample_with_counter(&logits, cfg(7), 42, 10, 0).unwrap();
    let _other = sample_with_counter(&other_logits, cfg(9), 99, 3, 0).unwrap();
    let b = sample_with_counter(&logits, cfg(7), 42, 10, 0).unwrap();
    assert_eq!(a, b);
}
```

再用相同 seed/position 在两个 sampler id 下比较；top-p cutoff 边界和 greedy 不消耗 draw。

**Step 2: 运行并确认失败**

Run: `cargo test --manifest-path rust/Cargo.toml --no-default-features model::sampling::tests -- --nocapture`

Expected: FAIL，现有实现调用 `rand::random()`。

**Step 3: 实现无全局状态的 SplitMix64 counter mapping**

输入 `(request_seed,token_position,draw_index)` 映射到 `(0,1)`；禁止 thread RNG 和 sampler-local RNG。tch sampling 与 pure slice sampling 共用同一个排序、top-p 和 uniform draw 语义，避免 backend 分叉。

**Step 4: 跑 sampling 回归**

Run: `cargo test --manifest-path rust/Cargo.toml --no-default-features model::sampling::tests -- --nocapture`

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend model::sampling::tests -- --nocapture`

Expected: 两组都 PASS。

**Step 5: Commit**

```bash
git add rust/src/model/sampling.rs rust/src/distributed/protocol.rs
git commit -m "feat: make distributed sampling counter based"
```

### Task 12: R3 worker 自主首 token 和后续 token loop

**Files:**
- Modify: `rust/src/model/model.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`
- Modify: `rust/src/worker_sdk/runtime.rs`
- Modify: `rust/src/distributed/protocol.rs`
- Modify: `rust/src/distributed/coordinator.rs`

**Step 1: 写 coordinator 零模型计算失败测试**

event recorder 断言自治模式下 coordinator 只有 `Admit/PrefillBarrier/Release/Error`，不存在 logits payload、sample、embedding 或逐 token `Decode/DecodeBatch`；worker events 包含 sampled token 和 finish reason。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_autonomous_token_loop -- --nocapture`

Expected: FAIL，coordinator 仍逐 token 收 logits 并采样。

**Step 3: 处理 prefill 到首 token 的边界**

拥有 prompt 最后位置的 worker 暂存 last logits。Coordinator 收齐 `PrefillDone` 后只发送 `BeginGeneration {request_id}` 控制屏障，不接收 logits；prompt-tail worker 用冻结 sampling plan 采样首 token，并按 `starter_phase` 必要时发送一个小型 `TokenHandoff`，随后启动 layer-0 packet。

默认 `k=0`：末层 finisher 本地 final norm、LM head、counter sampling、embedding，直接成为下一 token layer-0 starter。`L%N==0` 的固定 sampler 保留；phase-shift 只有策略接口，不默认发送额外 handoff。

logits 只能活在该 sampler 的一个 compute quantum 内：生成后立即完成过滤/采样并释放，不进入 packet、ready queue 或跨请求缓存。这样固定 sampler 的额外显存是有界 workspace 加至多一个 scheduler quantum 的 logits，而不是并发请求数乘以 vocab 的无界积压。

**Step 4: 验证 greedy、temperature 和 EOS/length**

两个并发 request 使用不同 seed、不同长度；断言 token events 可流式回 coordinator，coordinator 无 logits bytes；同一 seed 在不同 request interleaving 下输出不变；EOS 或 max token 后进入 Draining。

Run: `HCP_SELF_DRIVING_DECODE=1 HCP_AUTONOMOUS_DECODE=1 bash scripts/run_self_driving_ring_concurrent_local.sh`

Expected: PASS。

**Step 5: Commit 和 checkpoint 审查**

```bash
git add rust/src/model/model.rs rust/src/worker_sdk/tch_backend.rs rust/src/worker_sdk/runtime.rs rust/src/distributed/protocol.rs rust/src/distributed/coordinator.rs
git commit -m "feat: drive decode token loop from ring workers"
```

### Task 13: 生命周期、fail-stop 和可审计 observability

**Files:**
- Modify: `rust/src/model/decode.rs`
- Modify: `rust/src/distributed/decode_scheduler.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`
- Modify: `rust/src/worker_sdk/runtime.rs`
- Modify: `rust/src/distributed/coordinator.rs`
- Modify: `rust/src/report.rs`

**Step 1: 写 append 后 send failure 和 release 失败测试**

在 assignee reserved-slab append 成功后注入 successor send error；Expected: request fail-stop，commit key 只出现一次，所有 worker 最终 release KV/commit/queue/stats，coordinator additive/workspace ledger 均归零。重复 release 返回可观测 idempotent completion，但不得重复扣 ledger。

**Step 2: 运行并确认失败**

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend test_append_then_send_failure -- --nocapture`

Expected: FAIL，现有 release 不知道 in-flight/commit 状态。

**Step 3: 实现 request lifecycle 和 fail-stop fanout**

状态严格为 `Admitted -> Prefilled -> DecodeRunning -> Draining -> Released` 或 `Failed -> Draining -> Released`。第一版不 retry；旧 epoch/duplicate packet 安全拒绝。取消时不释放仍被 device executor 或 pending egress 引用的 tensor。

**Step 4: 添加机器可读 counters/report**

每 `(request,token,layer,node)` 记录 hop、role、partial、Q/KV/O/MLP；每 request/node 记录 additive reservation、workspace bound、actual allocator high-water、slab capacity/used bytes、queue depth/wait、attention/dense/LM-head/sample/link 时间、sampler count、hidden NaN/Inf 与抽样 checksum。报告不得依赖日志文本 grep 才能判断核心不变量。

Run: `LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --manifest-path rust/Cargo.toml --features tch-backend -- --nocapture`

Expected: PASS；失败注入 report 自足。

**Step 5: Commit**

```bash
git add rust/src/model/decode.rs rust/src/distributed/decode_scheduler.rs rust/src/worker_sdk/tch_backend.rs rust/src/worker_sdk/runtime.rs rust/src/distributed/coordinator.rs rust/src/report.rs
git commit -m "feat: harden self-driving decode lifecycle"
```

### Task 14: R4 硬件阶梯、异构 drift 和默认开关

**Files:**
- Create: `scripts/run_self_driving_ring_mps_2node.sh`
- Create: `scripts/run_self_driving_ring_cuda_hip_2node.sh`
- Create: `scripts/run_self_driving_ring_3node.sh`
- Create: `scripts/validate_self_driving_ring_report.py`
- Modify: `docs/plans/2026-07-29-self-driving-ring-decode-implementation.md`
- Modify: `graph-memory/graph.db`
- Regenerate: `graph-memory/active.md`
- Regenerate: `graph-memory/progress.md`
- Regenerate: `graph-memory/systemPatterns.md`

**Step 1: 先写 report validator 的失败 fixtures**

fixture 分别缺 hop、重复 MLP、additive/workspace/allocator high-water 任一超限、第三 worker 未计算、hidden 出现 NaN、queue 无界、negotiated mode/version 不一致；validator 必须逐个 FAIL。合法 `N=3,L=24` fixture 必须断言 48 hops/token、每 worker 两 peer、唯一 sampler/logits、三个 worker 都有 partial，且 hello/negotiation/packet version 一致。

**Step 2: 运行并确认 validator 先失败后通过 fixtures**

Run: `python3 scripts/validate_self_driving_ring_report.py --self-test`

Expected: PASS，表示 validator 能拒绝所有反例；不是表示生产实现通过。

**Step 3: 本地 MPS 两 worker**

必须在非 sandbox host 运行：

```bash
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps \
  bash scripts/run_self_driving_ring_mps_2node.sh
```

通过标准：greedy 文本与单节点 reference 一致；packet bytes 与 context 无关；每层 1 hop；slab/additive/workspace/allocator high-water 全部有界；append 不产生 context-sized cat；两个 request trace 交错；release 后归零。CPU-only 不能替代此门。

**Step 4: 跨节点 CUDA+HIP 两 worker**

按项目纪律：本地 commit/push，remote `git pull --ff-only`，不直接改远端源码；非交互 SSH 显式 `PATH=/home/stark/.cargo/bin:$PATH`。运行脚本后验证逐层 hidden checksum/diff、NaN/Inf、最终 argmax/text、每平台至少一个 worker。

通过标准：两种 backend 都执行 partial；无 coordinator compute；hidden drift 不随层数失控；旧 Q-ring A/B 文本一致。

**Step 5: 三 worker CUDA+CUDA+ROCm 真 ring**

使用真实 subnet endpoint，禁止 `127.0.0.1` 伪装跨机。强制 `N=3,L=24`、至少两个不同 phase 的并发 request；验证 48 hops/token、middle relay、固定 sampler 分散到不同 request、每 worker 仅 predecessor/successor、容量非等权、compute balance 在 hard bound 内。

**Step 6: 只有全部通过才切默认并更新记忆**

将 `HCP_SELF_DRIVING_DECODE` 默认改为 enabled，`=0` 保留 legacy；在 graph memory 中把 task 标记完成，写入各硬件 evidence 和 residual risk。若任一硬件门失败，保持 opt-in，不用降低 validator 标准换 PASS。

**Step 7: Commit**

```bash
git add scripts/run_self_driving_ring_mps_2node.sh \
  scripts/run_self_driving_ring_cuda_hip_2node.sh \
  scripts/run_self_driving_ring_3node.sh \
  scripts/validate_self_driving_ring_report.py \
  docs/plans/2026-07-29-self-driving-ring-decode-implementation.md \
  graph-memory/graph.db graph-memory/active.md graph-memory/progress.md \
  graph-memory/systemPatterns.md
git commit -m "test: validate self-driving ring on heterogeneous workers"
```

## 3. 每个 checkpoint 的统一验证

纯状态机改动：

```bash
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo test --manifest-path rust/Cargo.toml --no-default-features
```

tch 改动：

```bash
LIBTORCH=/Users/stark_sim/libtorch \
DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib \
cargo test --manifest-path rust/Cargo.toml --features tch-backend -- --nocapture
```

文档/记忆：

```bash
git diff --check
python3 graph-memory/export.py
sqlite3 graph-memory/graph.db "PRAGMA integrity_check;"
rg -n "memory hard bound|compute-balanced|capacity-only|R0|R1|R2|R3|R4" \
  docs/plans/2026-07-29-self-driving-ring-decode-*.md graph-memory/*.md
```

## 4. 实施期间禁止偷换的完成口径

- `DecodeBatch` 让两个请求都完成，不等于 ring 内 packet pipeline。
- token 一致，不等于 KV ownership/reservation/exact-once 正确。
- coordinator 把 hidden/logits 旁路送给下一节点，不算 P2P-only 自驱动 ring。
- 单请求固定 sampler 不是失败；只有 sampler queue/LM-head 成为实测关键瓶颈才触发 phase-shift 新任务。
- capacity tickets 不能因 request phase 旋转给别的物理节点；phase 只改变 calendar offset。
- active request 的 placement 不因 free-memory/throughput 新观测而变化；没有显式 KV migration 就不能重算。
- packet retry、KV replica、elastic shrink、kernel-level batching 均不在本计划 v1 范围。
- 三节点硬件门未过前，不把新路径称为 HCP 默认完成态。

## 5. 执行入口

计划执行时使用 `executing-plans`，从 Task 1 开始，每个 Task 严格遵循 red -> green -> focused commit。R1、R2、R3、R4 分别提交审查结果后再进入下一阶段；遇到设计合同冲突时先更新本计划和 graph memory，不在实现中静默改变不变量。
