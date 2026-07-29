# HCP 自驱动 Decode Ring 修订方案（审查稿）

日期：2026-07-29

状态：核心策略已由用户确认；本文作为实现计划的设计合同与审查基线

## 0. 审查结论先行

自驱动 ring 适合作为 HCP 的 decode 数据面，但适合的是下面这个精确定义：

- 每层只有一个逻辑 packet，沿单向 ring 访问所有 `N` 个 worker；
- 每个 worker 只连接 predecessor 和 successor 两个数据面 peer；
- 每层用 `N-1` 条物理边得到完整 attention，结果停在最后一个 worker；
- durable KV 按显式 placement policy 互斥、完备地切分，不在任何节点复制完整历史；
- Q、current K/V、`W_o + residual + MLP`、LM head 和 sampling 都有唯一执行者；
- coordinator 只负责 admission、release 和故障控制，最终不参与逐层 decode 数据流。

它解决的是 aggregate KV capacity、P2P-only 拓扑和重复 forward，不是低延迟本身。单请求的一个 attention packet 必须串行经过所有节点；异构节点中最慢的计算或链路仍会进入 token latency。多请求 packet pipeline 才能把各节点重新填满。

经用户澄清，`1/N` 不是 equal placement 的字面产品合同，而是在强调：KV 压力必须可计算、不能塌缩到 owner，也不能让某节点临时物化其他节点随 context 增长的历史 KV shard。最终 placement 使用“显存硬上界 + 上界内 compute-balanced”的约束分配：任何节点都不能越过可用 KV 容量；存在容量余量时按实测 attention throughput 降低慢节点份额；请求逼近聚合容量墙时自然退化为纯 capacity 比例。

因此算法合同是“互斥、完备、无复制、受冻结 quota 约束”；equal-`1/N` 只是容量和吞吐均相同设备上的特例。Decode packet 只携带 `O(model_width)` 状态，不携带远端历史 KV。

## 1. 与 HCP 项目目标的对齐矩阵

| HCP 目标 | 自驱动 ring 如何满足 | 边界或代价 | 验证方式 |
|---|---|---|---|
| 聚合 KV 容量 | 每层历史位置按可用 KV 容量互斥、完备切分 | placement 必须在 request admission 时冻结 | 逐请求、逐层导出 KV position ownership 与 quota |
| 异构 worker | packet 为设备无关协议，各节点用本地域内后端计算 | 语义兼容不等于性能均衡；慢节点不可跳过 | CUDA、ROCm、MPS 分层验证；记录逐节点耗时 |
| 仅 P2P ring | 数据面只走 predecessor/successor | fail-stop；任一唯一 KV shard 丢失即请求失败 | 连接表断言每 worker 恰有两个 peer |
| 无 collective | 不使用 all-reduce/all-gather/all-to-all | R1 有 coordinator 单播控制消息，但不是数据面 collective | 抓取连接和消息类型，禁止新增 collective dependency |
| 网络成本线性 | 物理 ring 为 `N` 条边；每层 packet 走 `N-1` 跳 | 单请求延迟仍随 `N` 线性增长 | hop counter 精确等于 `L*(N-1)` |
| 无 N 倍 full forward | 每层只有一个 distributed logical forward | 单 packet 放弃单请求内 N 包并行 | projection、MLP、LM-head exact-once counters |
| coordinator 不做模型计算 | 最终仅 admission/release/error control | R1-R2 仍暂时触发每个 token | R3 后 coordinator 无 embedding/logits/sample 事件 |
| correctness 优先 | online-softmax 精确归并，不改变 attention 定义 | BF16 跨平台只要求既定 argmax/text 标准 | 单节点 reference、argmax、文本、有限 logits diff |

### 1.1 这里的“线性”具体指什么

对一个 request、一个 token、`L` 层、`N` 节点：

```text
physical data-plane links       = N
peer degree per worker          = 2
attention packet edges/token    = L * (N - 1)
packet payload                  = O(model_width)
total attention bytes/token     = O(L * N * model_width)
```

旧的全节点各发一个等价 packet 是 `N*(N-1)` packet edges/layer；自驱动单 packet 是 `N-1`。因此总网络流量从关于节点数的二次增长降为线性增长。

这不意味着 latency 与 `N` 无关。单请求 packet 在一层内是串行的，近似：

```text
T_layer ~= T_q
         + sum_i(T_local_partial_i)
         + sum_(N-1 edges)(T_link)
         + T_o_proj+residual+mlp
```

## 2. 目标、非目标与牺牲

### 2.1 本计划必须达到

1. Decode 全程不存在完整历史 KV owner。
2. 单个 layer 只有一份 Q、一份 current K/V、一份完整 attention output 和一次 layer continuation。
3. packet 不携带 context-sized KV，大小不随历史长度 `T` 增长。
4. 所有 worker 都参与计算；coordinator 不替代 worker 做 forward。
5. 单请求可运行，多请求状态相互隔离，release 后所有本地 shard 可证明清空。
6. `N=3, L=24` 下默认 48 attention edges/token，不为形式轮转额外增加 token handoff。
7. 多请求是互相独立的异步 packet 流，不要求 token 对齐、等速前进或跨请求同步。

### 2.2 明确不是目标

- 不保证单请求 TBT 比单卡或冗余多 packet 方案更低。
- 不提供节点故障后的请求继续运行。
- 不在本轮做 KV 副本、弹性缩环、投机解码、量化或近似 attention。
- 不承诺 capacity-weighted memory placement 自动等价于 compute-balanced placement。
- R2 先保证多请求正确并发，不等同于已经具备 kernel-level continuous batching。

### 2.3 方案主动牺牲什么

旧的 owner-return 或全节点冗余 forward 提供了更简单的同步调用栈、每个节点都有完整结果、以及多个等价 packet 的并行机会。自驱动 ring 放弃这些能力，换取：

- 不复制完整 attention/MLP/logits 工作；
- 不要求结果返回 starter；
- KV 容量与网络拓扑成为可证明合同。

必须诚实记录的代价是：单 packet 把一个请求的一层 partial 计算串行化。吞吐恢复依赖多个 request 的 packet 同时占据不同 ring stage，而不是靠一个请求复制出 `N` 个 packet。

## 3. 系统角色与所有权

角色都是 request/token/layer 作用域，不是进程的永久身份。

| 角色 | 唯一职责 | 不拥有的东西 |
|---|---|---|
| `starter(r,p,l)` | 持有 `h_l`；做 input RMSNorm 和 Q projection；形成本层 seed | 不一定持有 current K/V；不一定完成本层 |
| normal relay | 用本层本地 durable history 计算一个 partial 并 merge | 不做 Q、K/V projection、MLP 或 sampling |
| `kv_assignee(r,p,l)` | 唯一计算本层 current K/V；让其参与本层 partial；durable append 一次 | 不因该角色自动成为 sampler |
| `finisher(r,p,l)` | 合并最后一个 partial；断言 KV 已提交；做 `W_o + residual + post-norm + MLP`；产生 `h_(l+1)` | 不需要把结果送回 starter |
| token finisher / sampler | 末层后做 final norm、LM head、sampling 一次 | 不额外拥有历史 KV |
| coordinator | admission、初始 phase、release、超时/失败控制 | 不做 embedding、layer forward、logits 或 sampling（目标态） |

在单向 successor ring 中：

```text
finisher(r,p,l)      = starter(r,p,l) - 1 mod N
starter(r,p,l+1)     = finisher(r,p,l)
```

因此 finisher 可以就地继续下一层，省掉返回 starter 的第 `N` 跳。

### 3.1 集群 mode 必须全员协商

Self-driving 和 legacy 的 command/response/packet 合同不同，不能由各 worker 独立读取环境变量后自行分支。启动时 worker 加载模型后先只连接 coordinator，发送带固定 magic/version/length envelope 的 `WorkerHello`，携带 control protocol versions、`LegacyQring|SelfDrivingV1` capability、model fingerprint 和 resource profile；错误 magic/旧 16-byte hello 必须立即 incompatibility，不能误读长度后等超时。只有 coordinator 读取 CLI/`HCP_SELF_DRIVING_DECODE` 并选择一次集群 mode，再向所有 worker 发送同一个 `{protocol_version,mode,ring_epoch}` negotiation。

coordinator 收齐完全一致的 ack 后，worker 才并发建立 predecessor/successor 数据面 streams，并回报同一 tuple 的 `DataPlaneReady`；只有全员 ready 后才允许 admission/prefill。任一 worker 不支持、version/epoch/mode 不一致，必须在 peer stream/第一个数据面 packet 前拒绝；不能静默按 worker fallback，也不能让 mixed-mode 进入 ring。后续 command 和 packet 都校验 negotiated tuple。

## 4. KV placement 与显存合同

### 4.1 不应混在一起的三类显存

1. **模型权重**：当前设计在所有 worker 复制，不属于 `1/N` KV 保证。
2. **durable KVCache**：本计划严格切分的主体，随 context 长度增长。
3. **瞬态状态**：packet、本地 attention workspace、最终 logits 和队列；必须有界，但不要求恰为 `1/N`。其中 attention workspace 可随本地 shard 长度增长，必须按本节点份额建模，不能藏进固定 slack。

所以可审计的表述应是：“任何节点都不持久保存超过 placement quota 的 KV；decode 中任何节点都不瞬态物化其他节点的 context-sized 历史 KV shard。”不能表述成“每个节点的总显存都只有单卡方案的 `1/N`”，因为权重仍然复制。

这里允许有界 packet buffer、当前 token 激活和与本地 shard 成比例的 kernel workspace；后两者都必须进入 resource profile/ledger。禁止的是随远端 context shard 长度增长的临时 KV 副本。Prefill 若仍需传 KV，也必须使用有界 micro-block/window，不能一次 stage 某个 peer 的完整历史 shard。

### 4.2 算法级硬不变量

令 `K_i(r,l)` 是节点 `i` 为请求 `r`、层 `l` 持久保存的位置集合：

```text
K_i(r,l) intersect K_j(r,l) = empty                 (i != j)
union_i K_i(r,l)             = all committed positions of r
owner(r,p,l)                 = exactly one worker
```

每个 worker 的 versioned resource profile 必须给出：每层每 position 的 `B_i^K(l)`、`B_i^V(l)`、allocator granularity `G_i`、request 只要分到任意 KV 就整笔支付的固定 metadata/allocator overhead `H_i`，以及扫描 `m` 个本地位置时的保守 decode workspace 上界 `W_i(m)`，其中 `W_i(0)=0`，只有非空本地 shard 才收 fixed + per-unit workspace。模型加载后，把可靠 device-free telemetry 与显式 KV budget 取较小值，再扣除 packet queue/static runtime reserve，得到 `C_i^KV`；不能把当前 coarse `capacity_mb()` 直接当 hard bytes。MPS 或无可靠 telemetry 的 backend 必须显式配置 budget。`C_i^KV/G_i/H_i/W_i` 缺失时不能启用 self-driving admission。

对新请求 `r`，令 `A_i` 是 active requests 的持久 slab + metadata reservation 之和，`W_i^active` 是 active requests workspace bound 的最大值；单线程 device executor 保证同一时刻只有一个 request 的 tensor compute quantum，因此 workspace 取最大而不是求和：

```text
P_i(r) = T_max(r) * sum_l(B_i^K(l) + B_i^V(l))
R_i(r,z) = worker i 持有份额 z 时，逐 layer 对 K/V 两个 slab
           分别按 G_i 向上取整后的持久 bytes 之和
u_i(r) = max z in [0,1] such that
         A_i + R_i(r,z) + H_i*indicator(z>0)
             + max(W_i^active, W_i(ceil(z*T_max))) <= C_i^KV
admit(r) iff sum_i(u_i(r)) >= 1
```

`u_i` 由单调整数搜索求出。固定 overhead 不能放进 `P_i` 后随份额缩放；若最终整数计划给节点 `i` 的 KV unit 为 0，不收 `H_i`，一旦非零只收一次，不按 layer 重复。若未来允许 kernel micro-batch 或多 stream 同时计算，workspace 不再是 `max`，必须先升级 ledger 合同。

`u_i` 是节点对该请求可承担的最大份额，不是目标份额。目标份额还要结合 decode attention throughput，由 4.4 节的有上界 water-filling 求解。placement quantum 默认是一个 `(position, layer)` KV 单元。容量和吞吐相同时，结果退化为每层 `floor(T/N)` 或 `ceil(T/N)`；异构时不追求节点间 token 数相等，只要求每个节点不超过冻结 quota，并在可行域内降低串行关键路径。

admission 不能只计算比例，还必须做真实 reservation。异构 backend 的 KV dtype/layout 可能不同，所以使用节点实测的每个 layer-token K/V 字节数与请求上界 `T_max = prompt_len + max_new_tokens`：

```text
reserve_i(r) = sum_l(
                 round_up(exact_count_i(r,l) * B_i^K(l), G_i)
               + round_up(exact_count_i(r,l) * B_i^V(l), G_i))
             + H_i * indicator(planned_count_i > 0)
sum_active_requests(reserve_i(r))
  + max_active_requests(W_i(max_local_positions_i(r))) <= C_i^KV
```

`exact_planned_count` 由冻结的 prompt chunk、decode calendar、`kv_phase` 和 `max_new_tokens` 精确计数，不用浮点比例直接估算。prompt 仍用 contiguous chunk，但每个节点的 chunk 长度和 decode calendar tickets 都来自同一组 bounded target `x_i`；若 prompt 继续按纯 capacity 切分，长 prompt 的 attention scan 会抵消 decode growth 的 compute balance。量化后还必须逐节点复核 additive reservation + active max workspace；整数舍入没有可行解就拒绝 admission。请求完成或失败后释放 reservation，并在最大 workspace request 释放后重算 active max。没有真实 ledger 的“capacity-aware”无法保证并发时显存压力可控。

当前 `ContiguousKvCache::update`、`BlockTableKvCache::update/get_kv` 都会 `Tensor::cat` 出本地完整 shard，产生额外 `O(local_T)` storage；现有 block table 不是 block-aware kernel。Self-driving v1 因此必须在 worker 处理 `AdmitRequest` 时按每层 exact count 预分配 K/V slab；只有所有 slab 成功且 actual bytes 不越 reservation 才返回 `AdmissionAccepted`，否则全员回滚且不进入 prefill。prefill/decode append 用 reserved slot `copy_`，history 用 `narrow` view，禁止全 shard concat。旧 cache 只留给 legacy。backend high-water 必须验证实际分配未超过上述 ledger。

### 4.3 已确认的实现选择：二维 weighted assignee 日历

现有理论稿把 assignee 简写为 `kv_assignee(position)`。这能做等权长期均衡，但一个新 token 的全部 `L` 层 K/V 都会由同一节点计算和提交，也不能表达异构 capacity quota。

在 request admission 时，把 4.4 节求得的目标份额量化为整数 tickets `q_i`，构造平滑的 weighted calendar `A_r`。calendar 中节点 `i` 占 `q_i/sum(q)` 的 slot，并用 stable `request_id` hash 选择起始 offset：

```text
weights_snapshot(r) = bounded_water_fill(u_i(r), attention_rate_i)
kv_phase(r)         = stable_hash(request_id, ring_epoch) mod |A_r|
kv_assignee(r,p,l)  = A_r[(kv_phase(r) + p + l) mod |A_r|]
```

calendar 必须用 smooth weighted round-robin/deficit scheduling 一类方法均匀铺开 tickets，不能把同一节点的 slots 全挤在一起。它有五个好处：

1. 对固定 layer，position 长期逼近 bounded target weights，前缀误差受一个 scheduling quantum 约束。
2. 对固定 token，current K/V projection 随 layer 按同一权重平滑分配。
3. 不同 request 的 offset 不同，可平滑短请求的舍入误差和瞬时 commit 峰值。
4. 不增加通信，因为本层 packet 本来就要访问全部节点。
5. 容量相同时，`A=[0,1,...,N-1]`，退化为 `(phase+p+l) mod N`；若 `L%N==0`，每个生成 token 的 layer-KV 恰好均分。

`kv_phase` 与 `starter_phase/sampler_phase` 必须是两个独立字段。前者维护 memory quota，后者用于错开 dense compute 和 sampler；不能为了把 sampler 放到快节点而旋转物理 bounded-target tickets。

`request_id` phase 只改变 weighted calendar 的起点和角色时序，不改变节点在完整 calendar 中的 ticket 数。否则所谓“分散 phase”会意外把大显存节点的 quota 旋转给小显存节点。

weights 和 calendar 对一个已 admission 的 request 必须冻结。设备剩余容量变化只影响新 request；若运行中直接改变 owner 函数，旧 KV 的查找位置会失效，除非另做显式 KV migration，而 migration 不在本轮范围。

### 4.4 显存硬上界内的 compute-balanced 求解

capacity 解决“最多能存多少”，attention throughput 解决“在容量余量内最好存多少”。节点本地 partial 的工作量大致随其 KV quota 增长，因此只按显存比例会让大显存但慢算力的设备成为串行瓶颈。

令 `s_i > 0` 是节点 `i` 经同一模型、dtype、代表性 decode KV 长度测得的 attention KV-unit throughput，`u_i` 是 4.2 节算出的本次请求最大份额。求解：

```text
minimize    max_i(x_i / s_i)
subject to  sum_i(x_i) = 1
            0 <= x_i <= u_i

solution    x_i = min(u_i, lambda * s_i)
            choose lambda so sum_i(x_i) = 1
```

这是带上界的 water-filling，可用排序后的 breakpoint 或单调二分确定 `lambda`。实现使用确定性整数/有理数量化生成 tickets，禁止把浮点误差带进 owner 判断。其行为边界是：

- hard memory upper bound：任何 `x_i` 都不能使 additive reservation + active max workspace 超过 `C_i^KV`；
- performance objective：容量富余时，未饱和节点满足 `x_i/s_i` 近似相等，慢节点份额下降；
- max-context objective：当请求逼近聚合容量墙时，`sum_i(u_i) -> 1`，唯一可行解 `x_i -> u_i`，自然退化为纯 capacity 比例；
- unavailable throughput：缺少或失真的 `s_i` 不可猜测；R0 使用显式配置，生产 admission 只要有参与节点缺值，就回退到 `x_i=u_i/sum_j(u_j)` 的 conservative capacity-only 目标并记录原因，不混用实测 rate 和猜测值。

份额只在 admission 时计算并冻结；新的容量或 throughput 观测只影响后续 request。该求解平衡的是 attention scan，不自动平衡 starter/finisher/LM-head；后者继续由 request initial phase 和多请求调度处理。

## 5. 单层 packet 状态机

### 5.1 推荐 packet 最小字段

```text
protocol_version
phase                         // LayerPacket | TokenHandoff | Cancel 等
request_id
token_position
layer_idx
ring_epoch                    // 拓扑/请求代际，防旧包污染
input_token_id                // layer 0 有效；供审计和轻量 token history 更新
starter_id
hops_remaining                // 初始 N-1，每跳严格减一
kv_assignee
flags.kv_committed

h_residual                    // finisher 做 residual 所需
q
o_acc
lse_acc
```

`kv_assignee` 可由 policy 推导，但仍建议在线上携带，并由接收端重算校验，尽早发现配置或版本不一致。packet 不携带 current K/V，也不携带任何历史 KV。

默认不携带 `h_norm`。若 assignee 不是 starter，它从 `h_residual` 本地重算一次 input RMSNorm，再做 K/V projection。这样每层最多多一次 `O(model_width)` 的逐元素 norm，却让 `N-1` 条边都少传一个 model-width 向量，更符合单请求不浪费带宽的目标。

对 Qwen2-0.5B，现有 Q/O/LSE packet 约 3.6 KiB。加入一个 BF16 `h_residual` 后，tensor payload 估算约 5.3 KiB，且与 context 长度无关。精确字节数应由 R0 序列化测试给出，不把估算写成协议保证。若 profile 证明重算 norm 比多传 `h_norm` 更贵，后者只能作为显式协议策略对照。

### 5.2 starter 分支

starter 先计算：

```text
h_norm = input_rms_norm(h_residual)
q      = rope(q_proj(h_norm), position)
```

然后分两种情况：

**A. `starter != kv_assignee`**

1. 只对 starter 的 durable history 计算 local partial。
2. 以该 partial 作为 `(o_acc,lse_acc)` seed。
3. `kv_committed=false`，发往 successor。

**B. `starter == kv_assignee`**

1. 唯一计算 current K/V，并应用 RoPE。
2. 用 commit key 把 current K/V 写入预分配 slab 的下一个 reserved slot。
3. 从包含 current 的 `narrow` view 对 `durable history + current` 形成一个 local partial，不做 `Tensor::cat`。
4. partial 成功后在首跳前设置 packet `kv_committed=true`；append 后任一失败均 fail-stop，不 retry。
5. 禁止再对 starter history 计算第二个 partial。

### 5.3 relay 分支

普通 relay：只用 durable history 形成一个 partial，与 accumulator 做 online-softmax merge。

assignee relay：

1. 从 packet 的 `h_residual` 重算 input RMSNorm，并唯一计算 current K/V。
2. 用 commit key 写入 reserved slot，再从包含 current 的 view 形成一个 partial。
3. partial 成功后设置 packet `kv_committed=true`；append 后错误直接 fail-stop。
4. merge 后转发；不得再走普通 relay 分支。

### 5.4 finisher 分支

finisher 先完成自己对应的 normal/assignee relay 分支，再检查：

```text
hops_remaining == 0
receiver_id == starter_id - 1 mod N
kv_committed == true
layer_idx matches local continuation
```

然后唯一执行：

```text
attn_out = o_proj(o_acc)
x1       = h_residual + attn_out
h_l+1    = x1 + mlp(post_attention_norm(x1))
```

若不是末层，finisher 就地构造下一层 packet，成为新 starter。若是末层，进入 token 边界状态机。

### 5.5 三种 assignee 重合必须分别测试

以 `N=3` 为例，R0 必须构造：

| Case | assignee 位置 | 容易出现的 bug |
|---|---|---|
| A | starter | history 被 seed 一次、assignee 分支又算一次 |
| B | middle relay | current K/V 未从 `h_residual`/position 正确生成，或 append 次序错误 |
| C | finisher | finisher 先断言 commit，导致合法 packet 被拒；或先做 MLP 再 append |

每个 case 都要断言每个节点 local partial 恰好一次、current K/V project/append 恰好一次。

## 6. 单 token 的全层时序

假设 ring 为 `0 -> 1 -> 2 -> 0`，`N=3`，第一层 starter 为 0。下面只展示 starter/finisher route；KV assignee 另由冻结的 bounded-target calendar 决定：

| Layer | starter | 路径 | finisher / 下一层 starter |
|---|---:|---|---:|
| 0 | 0 | `0 -> 1 -> 2` | 2 |
| 1 | 2 | `2 -> 0 -> 1` | 1 |
| 2 | 1 | `1 -> 2 -> 0` | 0 |
| 3 | 0 | `0 -> 1 -> 2` | 2 |

每 3 层角色完成一个周期。对 `L=24`：

- 每个节点恰好做 8 次 starter 工作；
- 每个节点恰好做 8 次 finisher 的 `W_o + MLP`；
- 每层所有节点各做一次本地 attention partial；
- 在三节点 KV capacity 和 attention throughput 都相同的特例中，每个节点恰好做 8 次 current K/V projection；异构时按 bounded-target tickets 比例分配；
- attention 数据面跳数为 `24 * 2 = 48`。

这说明 `L % N == 0` 时层内重计算分配是均衡的。它不说明 wall time 均衡：同样的工作量在 CUDA、ROCm、MPS 上耗时可能不同。

## 7. Token 边界与固定 sampler

零额外 handoff 时：

```text
starter(t+1,0) = starter(t,0) - L mod N
sampler(t)     = starter(t,0) - L mod N
```

当 `L % N == 0`，单请求 sampler 固定。这是可接受的默认策略，因为：

1. LM-head/embedding 权重已经在所有 worker 复制，不新增持久权重。
2. sampler 与 KV placement 解耦，不会把后续 token 的全部 KV 放到 sampler。
3. 额外持久状态只有小型 request/sampler metadata。
4. 主要瞬态是 `[batch,vocab]` logits。Qwen2-0.5B 的 `151936` 个 fp32 logits 约 `0.58 MiB/request-row`，与 context 长度无关。

实现必须在 sampler 的同一个 compute quantum 内完成 logits 过滤/采样并释放；logits 不得进入 ring packet、ready queue 或 request backlog。否则即使单行显存很小，多请求积压仍会把固定 sampler 变成无界瞬态显存热点。

但“显存不重要”不等于“计算不重要”。Qwen2-0.5B 的单 token LM head 粗略为：

```text
2 * hidden_size * vocab_size
= 2 * 896 * 151936
~= 272 million FLOPs/token
```

它对小 hidden、大 vocab 模型可能是可见比例。固定 sampler 的真实风险是：

- 单请求若固定在慢节点，LM-head latency 每 token 都落在慢节点；
- 多请求若初始 phase 偏斜，会形成 sampler queue 和吞吐热点；
- backend 还可能有 logits kernel workspace，不能只按 `0.58 MiB` 推断峰值。

默认策略：

1. 单请求把初始 sampler phase 放在 LM-head 实测最快的节点。
2. 多请求至少以 stable `request_id` hash 错开 initial starter phase；scheduler 还可结合 sampler queue 和设备能力选择 hash bucket 到物理 starter 的映射。
3. 记录 `lm_head_time`、`sample_time`、`sampler_queue_wait`、峰值显存和每节点 sampler request 数。
4. 只有 profile 证明 sampler 已是关键瓶颈，才启用 token-boundary phase shift。

若 token 边界沿 successor 额外走 `k` 跳：

```text
cross-token phase delta = k - L mod N
full rotation condition = gcd(k - L, N) == 1
```

对 `L=24,N=3`，`k=1` 可全轮转，总成本是 48 个 attention packet edges 加 1 个很小的 token-ID handoff。`+1` 不是通用规则，不能硬编码为所有模型/节点数的默认策略。

### 7.1 Sampling state 不能隐含在节点本地

Greedy sampling 没有 RNG 状态问题。Temperature/top-p 等随机 sampling 若直接使用 sampler 节点的本地 RNG，当 `L%N!=0` 或启用 phase shift 后，输出会依赖 sampler 落在哪个节点和 packet 调度顺序。

推荐把随机性定义为 request 逻辑状态：

```text
random_draw = CounterRng(request_seed, token_position, draw_index)
```

sampling params 和 seed 在 admission 时分发为 request metadata。下一 token 的 layer-0 packet 携带 `input_token_id`；packet 经过全环时，各节点可以顺路更新生成 token 的轻量 history，不增加 hop，也不复制 KV。

需要完整 prompt history 的 repetition/frequency penalty 是单独的 API 合同：可以复制紧凑 token IDs（约 4 bytes/token，远小于 KV），也可以在 v1 明确只支持不依赖完整 prompt history 的 sampling。不能默认依赖某个固定 sampler 的私有 history，否则 `L%N!=0` 时语义会破裂。

## 8. 异构负载的真实边界

自驱动 ring 对异构的支持是“不同设备能共同执行同一语义”，不是“自动达到最优负载均衡”。

### 8.1 constrained capacity/compute KV policy 下

- 每节点 KV token 数逼近 admission 时冻结的 quota，而不是互相相等；
- 每层每节点 attention partial 的 KV 长度按显存上界内的 compute-balanced 份额分布；
- 慢设备会在每层进入串行关键路径；
- `L % N == 0` 时 dense role 次数相同，但耗时未必相同。

### 8.2 可以安全偏置的内容

- request 初始 starter phase，可错开不同请求的 dense role 和固定 sampler；
- 多请求 admission 映射，可让快节点承担更多 sampler queue；
- 本地 kernel 和 micro-batching，可按设备能力分别优化。

### 8.3 不能在不改变合同的情况下偏置

- 跳过慢节点：会丢失其唯一 KV shard；
- 运行中静默改变既有 request 的 KV weights：会使旧 KV owner 映射失效；
- 让某层 finisher 任意选择：在单向 ring、`N-1` 跳下 finisher 由 starter 的 predecessor 唯一确定；
- 复制慢节点 KV 到快节点：会破坏 durable KV 无复制保证。

因此，“memory hard bound + bounded compute balance”是最终默认；纯 capacity placement 是请求接近容量墙或 throughput 数据不可用时的退化路径，且都只对新 request 生效。

### 8.4 跨设备 layer continuation 的数值风险

当前冗余 Q-ring 中，每个 worker 可以在自己的完整 forward 栈里持续计算 hidden state。自驱动 ring 改为本层 finisher 产生 `h_(l+1)`，下一层可能由另一种设备继续；hidden state 会逐层跨 CUDA/ROCm/MPS 边界。

这在数学上成立，但会扩大异构 kernel 舍入次序差异的传播面。协议必须固定 tensor dtype、shape、endianness 和模型/权重 fingerprint；R4 除最终 argmax/文本外，还要逐层采样 hidden-state diff、NaN/Inf 和 drift 趋势。若误差随层数失控，应先定位 kernel/serialization，而不是用 sampler 容忍度掩盖。

## 9. 多请求调度与 backpressure

一个 request 内层依赖严格串行，但不同 request 的 packet 是互相独立的异步流，可以同时处于 ring 的不同节点。下面只是某个瞬间的占用示意，不表示 request 之间存在 barrier、相同 token index 或等速要求：

```text
t0: node0 handles reqA/l7, node1 handles reqB/l2, node2 handles reqC/l19
t1: whichever local work finishes becomes eligible to move; no global tick
```

R2 首先实现 correctness concurrency：

- 所有状态以 `(request_id, token_position, layer_idx)` 隔离；
- 同一 request 最多一个 layer packet 在途；
- 每条 link 使用有界队列，满时 backpressure，不无界缓存 tensor；不同 request 用 round-robin/deficit fairness，避免单个 request 长时间占住链路；
- 每请求保持 token 顺序；取消和 release 不得越过仍在途的 packet；
- stable `request_id` 分别生成 `kv_phase` 与 starter phase；至少两个 request 使用不同 phase 和不同 prompt，避免状态混叠被相同输入掩盖；
- 一个 request stall 时，除共享 node/link 的正常资源竞争外，不得形成跨请求逻辑依赖或全局 barrier。

当前 `WorkerRuntime` 在收到 `Decode` 后同步阻塞于 `backend.decode_request()`，在该调用返回前不能接收或推进另一个 request。R2 因此需要真正的 packet ingress -> ready queue -> device compute -> successor send 事件循环，或语义等价的有界调度器；只在 coordinator 侧一次发送 `DecodeBatch` 不能证明 ring 内 pipeline 已存在。

真正的 kernel-level batching 是后续性能层：只批处理同一节点上 layer、dtype、shape、role 兼容的 packet，并设置最大等待时间，不能为了凑 batch 阻塞本可前进的单请求。不能打破 request 的层依赖，R2 也不能仅凭“两个请求都完成”宣称 continuous batching 已实现。

## 10. 故障、重试和生命周期

### 10.1 本轮故障语义

采用 fail-stop：任一 worker 或 link 失败，请求失败。原因是该节点持有唯一 KV shard；没有副本就无法在保持 exact attention 的情况下继续。

### 10.2 为什么第一版不做 packet retry

assignee 在发送 packet 前已经 durable append。若发送失败后无条件重放，本层 current K/V 可能重复 append。因此第一版网络错误直接 abort request，比伪幂等 retry 更安全。

未来若加入 retry，commit key 至少是：

```text
(ring_epoch, request_id, token_position, layer_idx)
```

重复 packet 必须返回已提交结果或安全拒绝，不能再次 append。

### 10.3 请求生命周期

```text
Admitted -> Prefilled -> DecodeRunning -> Draining -> Released
                                    \-> Failed -> Draining -> Released
```

- `Draining` 阶段阻止新 token，同时等待或作废带相同 epoch 的在途 packet。
- `ReleaseRequest` 必须到达所有 worker，并分别清理 durable KV、commit 表、队列项和统计状态。
- coordinator 可保留 control-plane fanout；这不改变 worker 数据面只有两个 peer 的约束。

## 11. 修订后的实施切片

### R0：协议合同与纯状态机 mock

范围：不切 production path，不依赖真实 tensor kernel。

交付：

- packet version/phase/schema；
- versioned worker hello、全员 mode negotiation 与 mixed-mode 拒绝；
- role/route 纯函数与 capacity-constrained bounded-target calendar；
- online-softmax accumulator mock；
- exact-once KV commit mock；
- N=1/2/3/4、equal/uneven tickets、空 shard、三种角色重合、`L%N==0/!=0` 测试。

退出条件：hop、partial、projection、commit、finisher、sampler counters 全部可精确断言；weighted owner 的前缀误差不超过一个 scheduling quantum；序列化 round-trip 和 packet size 不随 context `T` 增长。

### R1：单请求、单 token 的完整垂直切片

范围：coordinator 在全员成功协商 `SelfDrivingV1` 后，暂时向所有 worker 单播同一 decode command，使其进入一次协调参与的 decode 调用；数据面只有一个 packet。

必须同时修改：

- `RingPacket` 协议；
- attention projection 边界（Q、K/V、O 可由不同节点执行）；
- model layer continuation（finisher 就地做 residual/MLP 并续层）；
- worker response 合同（唯一 `Finished {logits,finisher}`，其他为 `Participated`）。

不能把“只改单 packet attention、其余 worker 继续同步 full `forward()`”作为 checkpoint；只有 finisher 得到 layer output，其余调用栈无法继续。

退出条件：一个 request 的一个 token 在真实 backend 上完成；每层 `N-1` hops；一个 logits producer；无重复 Q/KV/MLP；旧 Q-ring 保留显式 fallback。

### R2：多请求隔离、pipeline 和 backpressure

范围：至少两个不同 prompt、不同 initial phase、不同前进速度的并发请求；先逐 packet 异步调度，再考虑兼容 packet micro-batch。

退出条件：token 各自对 reference；请求状态、KV、commit、release 完全隔离；有界队列在压力下不增长；无 starvation；每节点 packet/role 分布符合预测。

### R3：自主 token loop

范围：末层 finisher 本地 final norm、LM-head、sampling、embedding，并直接启动下一 token；coordinator 退出逐 token 广播。

默认保留固定 sampler；可选 phase shift 只留策略接口，不默认开启。

退出条件：coordinator 日志无逐 token logits/sample/broadcast；worker 只向 coordinator 发送 token event、完成或错误事件；release/cancel 仍由 control plane 正确收口。

### R4：硬件验证阶梯

1. mock：覆盖所有状态机组合和反例。
2. 本地 MPS 双 worker：验证真实 tensor、序列化、进程生命周期。
3. 跨节点 CUDA+HIP 双 worker：验证异构数值与 transport。
4. 三 worker CUDA+CUDA+ROCm 真 ring：强制覆盖 middle relay、`N=3,L=24` sampler resonance、48 hops/token 和每 worker 两 peer。

最终退出条件必须同时满足 correctness、KV、compute、topology、lifecycle 五类证据，不能只凭 token 一致宣称完成。

## 12. 验证矩阵与否决条件

### 12.1 必测矩阵

| 类别 | 必测项 | 通过标准 |
|---|---|---|
| 数值 | greedy token、文本、末步 logits | 遵循现有 BF16 跨平台标准；argmax/文本一致 |
| KV ownership | 每 request/layer/node 的 position 集合 | 互斥、完备、不超过冻结 capacity quota，误差不超过一个 quantum |
| KV reservation | 每节点所有 active request 的最坏情况预留 | `sum(reserve_i) <= C_i^KV`，结束/失败后归零 |
| exact-once | Q、current K/V project/append、O/MLP、LM head | 预期执行者各 1 次，其他为 0 |
| hop | 每 request/token/layer 的 edge count | 精确 `N-1`；`L=24,N=3` 为 48/token |
| packet | 序列化字节数对不同 context 长度 | 保持不变，只随模型宽度/batch 变化 |
| topology | worker peer adjacency | 恰一个 predecessor、一个 successor，无远端直连 |
| protocol mode | hello/negotiation/command/packet | 全 worker version/mode/epoch 一致；mixed capability 在 prefill 前拒绝 |
| sampler | `L%N==0` 和 `!=0` | 单请求分布符合公式；多请求 queue 可观测 |
| sampling state | sampler 轮转、不同 packet interleaving | counter RNG 对同一 seed/position 可复现；history 不绑节点 |
| 并发 | 两个不同 prompt、不同 phase、不同 decode 长度/速度 | token 正确，无 barrier、无 KV/commit/release 混叠 |
| backpressure | 小队列+慢 relay | 有界内存、无死锁、无静默丢包 |
| failure | assignee append 后 send 失败 | 请求 fail-stop，release 完整，无重复 append |
| heterogeneity | 逐节点 attention/dense/LM-head/link 时间 | 找到真实瓶颈，不用总 wall time 掩盖 |
| hidden handoff | CUDA/ROCm/MPS 逐层 hidden diff | 无 NaN/Inf；drift 在既定阈值内；最终 argmax/文本一致 |

### 12.2 任何一项出现就否决当前实现

- packet 携带随 context 增长的 KV 或 hidden history；
- 某节点持久保存未分配给自己的 KV；
- 任一层出现两个 Q、两个 current K/V append、两个 MLP continuation 或两个 logits producer；
- 为得到 `N-1` 跳却把完整结果通过 coordinator/旁路送回；
- worker 数据面连接超过 predecessor/successor；
- R1 的临时 coordinator token fanout 被误当成最终自治状态；
- retry 没有幂等 commit key；
- 只测 token 正确，不测资源与角色不变量；
- capacity-constrained placement 没有记录 weights snapshot/quota，却在运行中依赖可变 free-memory 重新算 owner；
- admission 只算 weights、不按 `prompt+max_new_tokens` 做 per-node KV reservation；
- `request_id` phase 改变物理节点 ticket 数，而不只是 calendar offset/角色时序；
- 把 request pipeline 实现成跨请求 lockstep barrier。
- 继续使用阻塞 `decode_request()`，却仅凭 coordinator 的 `DecodeBatch` 声称 ring 内多请求 pipeline 已完成。
- stochastic sampling 依赖节点本地 RNG 或 sampler 私有 history，导致角色轮转后语义改变。
- worker 依据本地 env 自行选择 legacy/self-driving response，未经过 coordinator 全员协商。

## 13. 已确认策略与实现默认

已由用户确认的项目目标：

- [x] **KV 合同**：最终采用 capacity-constrained weighted placement；`1/N` 只强调无 owner-collapse、无 context-sized 远端临时 KV 峰值和显存压力可计算。
- [x] **多请求语义**：每个 request 独立异步跑自己的 packet 流，不要求同步或等速；用 stable `request_id` 分散初始 phase。
- [x] **sampler 默认**：`L%N==0` 时允许单请求 sampler 固定；只在 profile 证明 queue/compute 瓶颈后启用 phase shift。
- [x] **性能定位**：单请求用一个 packet 避免带宽浪费；多请求通过独立 packet pipeline 并发提速。
- [x] **capacity/compute 关系**：显存容量是 hard bound；上界内按 attention throughput 优化；接近总容量墙时退化为 pure capacity。

实现计划采用以下保守默认；它们不改变上述用户确认的产品合同，可在对应 checkpoint 审查：

- [x] **二维 assignee**：calendar 使用 `(request phase + token position + layer)`，同时平滑每层 KV 份额和 current K/V projection。
- [x] **packet 取舍**：默认只传 `h_residual`，assignee 重算 RMSNorm；`h_norm` 仅保留为 profile 后可评估的协议变体，不在 v1 实现。
- [x] **故障语义**：第一版 fail-stop、无 retry、无 KV replica。
- [x] **R1 临时边界**：coordinator 单播让所有 worker 进入 decode；ring 数据面仍只有一个 packet；R3 移除逐 token coordinator 驱动。
- [x] **最终硬件门**：三节点真异构 ring 是完成条件，双节点只算 smoke。

## 14. 建议的最终裁定模板

审查完成后，把结论压缩为以下合同再更新 graph-memory：

```text
Placement contract:
  memory hard bounds + bounded compute-balanced quota
  frozen at request admission; capacity-only at the aggregate wall

Assignee function:
  smooth weighted calendar by request phase + position + layer

Packet hidden-state policy:
  carry residual and recompute norm at the unique KV assignee

Token-boundary policy:
  default k = 0
  optional k chosen by gcd(k-L,N)=1 after profiling

Failure model:
  fail-stop, no retry in v1

Completion gate:
  R0 mock -> R1 vertical slice -> R2 concurrency -> R3 autonomy
  -> R4 MPS + CUDA/HIP + 3-node heterogeneous ring
```

## 15. 依据与当前代码落点

- 理论依据：`docs/plans/2026-07-29-self-driving-ring-theory.md`
- 计划修订依据：`docs/plans/2026-07-29-self-driving-ring-decode-revision.md`
- 当前 packet：`rust/src/model/transport/block.rs::RingPacket` 只有 layer/Q/O/LSE/scale。
- 当前模型边界：`rust/src/model/model.rs::forward` 仍是 embedding -> 全层同步 forward -> final norm -> LM head。
- 当前 worker 合同：`rust/src/worker_sdk/backend.rs::decode_request` 要求每个 worker 返回 logits。
- 当前 control response：`rust/src/distributed/protocol.rs::WorkerResponse::DecodeDone` 没有 finisher/participant 区分。
- 当前 Q-ring：`rust/src/model/attention/ring.rs::ring_decode_attention` 明确保留全节点冗余 forward；可复用 accumulator 数学，不能直接作为自驱动层间控制流。
