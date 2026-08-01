# HCP：线性 P2P 环上的异构上下文并行

> 方法草稿，2026-08-02。本文件固定当前的问题模型、算法、证明义务、
> 证据边界和评测计划，不是一篇完整论文。本文有意省略摘要、实现结果、
> 性能数字和最终结论。

## 论断分级

本草稿使用四类标签，避免把算法陈述误写成实证结果。

- **方法主张**：HCP 设计必须满足的性质。
- **已证明不变量**：可以从既定模型和假设推导出的性质。
- **原型证据**：当前 Rust 正确性原型已经覆盖的性质，但不代表物理显存或
  生产 runtime 结果。
- **开放实证问题**：必须在当前方案的端到端异构实现上测量的问题。

## 1. 问题定义与研究定位

现代分布式推理系统通常使用显存、算力和集合通信能力对称的设备组成
细粒度并行组。异构资源更常被划分在请求路由、模型放置或 prefill/decode
服务解耦等较粗边界上。这些边界有实际价值，但不能把不同加速器的不等
显存容量聚合起来，共同承载同一个请求的上下文。

异构上下文并行（Heterogeneous Context Parallelism，HCP）研究一个更窄的
问题：

> 显存容量不同、甚至采用不同执行后端的设备，能否仅依靠邻居间通信，
> 共同持有并计算同一个请求的 attention context？

目标压力来自键值缓存（KVCache）。对于自回归 Transformer，即使模型参数
保持不变，持久 KV 状态仍会随上下文长度增长。因此，HCP 把逻辑 attention
context，而不是某一种 tensor collective 或某一个推理阶段，作为被并行化
的对象。

**方法主张。** HCP 将上下文并行扩展到请求的完整生命周期：初始 prefill、
自回归 decode 和 continuation prefill 都在同一个分布式逻辑 KV context 上
运行。

**方法主张。** 每个 worker 只与逻辑环中的 predecessor 和 successor 通信。
HCP 只要求点对点 send/receive 语义，不要求 collective library、全局共享设备
内存或 all-to-all peer graph。

这一逻辑合同与具体传输无关。任何能够实现所需 P2P 顺序和 tensor 表示的
fabric 都可以承载它。更快的通用加速器互联会扩大 HCP 的适用区间，但 HCP
不假设某一种互联天然就能使系统高效。

**开放实证问题。** 异构细粒度协作能否优于请求级或阶段级粗粒度放置，取决于
链路带宽、时延、设备不均衡、kernel 效率和工作负载并发度。本草稿不声称
这一优势已经得到测量。

## 2. 系统模型

### 2.1 Worker 与拓扑

定义 worker 集合

\[
\mathcal{W}=\{0,1,\ldots,N-1\}.
\]

Worker 组成单向逻辑环：

\[
\operatorname{succ}(i)=(i+1)\bmod N,
\qquad
\operatorname{pred}(i)=(i-1)\bmod N.
\]

每个 worker 都持有完整的模型参数副本。Worker 可以具有不同的可用 KV
容量、attention throughput、加速器类型和本地 kernel 实现，但必须执行相同
的 Transformer 语义。当前 HCP 方法聚合的是 context capacity；它不能切分
一个参数量大到无法装入单 worker 的模型。

对 worker \(i\)，令 \(B_i\) 表示扣除模型权重、runtime 状态以及有界通信和
计算 workspace 后，允许用于持久 KV 的字节数。Placement 的硬上界是
\(B_i\)，而不是设备的标称显存。

### 2.2 按位置索引的逻辑 KV context

对 layer \(\ell\) 和全局 token position \(p\)，定义

\[
\mathcal{C}_{\ell}[p]=(K_{\ell,p},V_{\ell,p}).
\]

所有权函数 \(a_{\ell}(p)\in\mathcal{W}\) 产生本地 shard：

\[
S_{i,\ell}=\{p\mid a_{\ell}(p)=i\}.
\]

每个 layer 必须满足

\[
S_{i,\ell}\cap S_{j,\ell}=\varnothing \quad (i\ne j),
\qquad
\bigcup_{i=0}^{N-1}S_{i,\ell}=\mathcal{P},
\]

其中 \(\mathcal{P}\) 是该请求已经提交的全局 position 集合。每个本地条目
都同时保存 global position、K 和 V。物理存储顺序不等于逻辑序列顺序。

**已证明不变量。** 只要各 shard 互斥且完备，并且 causal mask 根据 global
position 计算，attention 正确性就不要求先拼接或排序物理 shard。

### 2.3 Capacity-weighted 所有权

定义归一化容量目标

\[
\alpha_i=\frac{B_i}{\sum_j B_j}.
\]

对于一个有限的、已准入的等尺寸 KV event 集合 \(E\)，HCP 选择整数计数

\[
n_i\in\{\lfloor \alpha_i|E|\rfloor,
          \lceil \alpha_i|E|\rceil\},
\qquad
\sum_i n_i=|E|,
\]

并使用确定性的余数分配。如果不同 layer 的 KV event 字节数不同，则同一
合同按字节而不是原始 event 数量执行：

\[
M_i=\sum_{(\ell,p):a_\ell(p)=i} b_\ell \le B_i,
\]

其中 \(b_\ell\) 是 layer \(\ell\) 上一个 position 的 K+V 存储成本。

分配顺序经过平滑，使每个前缀都尽量贴近其加权目标，而不是先把某个 worker
的全部 event 连续放置。由 request 派生的 phase 会循环旋转这一确定性序列，
从而把不同并发请求的初始角色分散到不同 worker，同时不改变任何单请求的
总 reservation。

**方法主张。** Capacity weighting 是 HCP 的默认显存策略。只有当所有 worker
的可用 KV budget 相等时，均等 \(1/N\) placement 才是它的特例。

**方法主张。** 一个请求在已准入 horizon 内的所有权被冻结。不能仅仅因为
后续推理阶段采用了不同数据流，就迁移已有 KV。扩展 horizon 时，必须先完成
新的 capacity check，才能准入新 position。

**开放实证问题。** 面向开放式生成、request churn、fragmentation 和重新准入
的生产 allocator 不属于当前核心。硬上界目前只在显式预留的有限 horizon 上
得到验证。

## 3. 统一的 HCP 请求生命周期

HCP 根据可用的 query parallelism 改变环上传输的对象，但逻辑 KV 所有权
合同保持不变。

```text
初始 prefill
  本地 Q/activation chunk 留在各自 worker
  有界 KV micro-block 沿环传递
            |
            v
自回归 decode
  历史 KV 留在各自 worker
  单个 activation + Q + softmax accumulator packet 沿环传递
            |
            v
continuation prefill
  新的 Q/activation chunk 再次分布到各 worker
  新旧 positioned KV shard 共同参与同一个 Ring Attention
```

任何阶段转换都不会重建一个稠密、全局有序的 KV tensor。各阶段共同使用的
接口是 \(\mathcal{C}_{\ell}[p]\) 和 global position。

## 4. Capacity-weighted Ring Attention Prefill

### 4.1 本地序列计算

对于 position 集合为 \(P\) 的 prefill block，HCP 将 query position 切分成
互斥的 capacity-weighted 集合 \(P_i\)。Worker \(i\) 持有 \(P_i\) 对应的
activation 和 query row，为这些 position 计算 K/V，并把它们永久提交到本地
shard。这个 prefill block 在各 layer 间保持相同的 position partition，使
layer normalization、residual 和 MLP 都能在对应 token activation 所在的
worker 上本地执行。

因此，prefill 同时分布 attention query 和非 attention token 计算，而不是
让每个 activation 遍历所有 worker。

### 4.2 KV 环传与 online softmax

在 layer \(\ell\)，每个目标 worker 保持自己的本地 Q block，而 KV block
按照环顺序访问全部 worker。大的源 shard 被流式切成有界 micro-block，因此
接收方不需要物化另一个 worker 的完整持久 shard。

对一行 query 和一个 KV 子集 \(A\)，定义 accumulator

\[
\mathcal{A}_A=(m_A,z_A,u_A),
\]

其中

\[
m_A=\max_{p\in A}s_p,
\quad
z_A=\sum_{p\in A}\exp(s_p-m_A),
\quad
u_A=\sum_{p\in A}\exp(s_p-m_A)V_p.
\]

对互斥子集 \(A\) 和 \(B\)，稳定合并为

\[
m=\max(m_A,m_B),
\]

\[
z=\exp(m_A-m)z_A+\exp(m_B-m)z_B,
\]

\[
u=\exp(m_A-m)u_A+\exp(m_B-m)u_B.
\]

精确 attention 输出是 \(u/z\)。等价的 wire 表示使用归一化输出
\(O_A=u_A/z_A\) 和 \(\operatorname{LSE}_A=m_A+\log z_A\)：

\[
\operatorname{LSE}=\operatorname{logaddexp}
  (\operatorname{LSE}_A,\operatorname{LSE}_B),
\]

\[
O=\exp(\operatorname{LSE}_A-\operatorname{LSE})O_A+
  \exp(\operatorname{LSE}_B-\operatorname{LSE})O_B.
\]

Causal mask 由全局 query position 和 key position 决定。这一点是必要的，
因为 capacity-weighted shard，特别是经历 decode growth 后的 shard，不一定
包含连续 position 区间。

**已证明不变量。** 对互斥且完备的 KV 子集重复执行合并，在浮点计算顺序差异
范围内，会得到与对应稠密 causal attention row 相同的结果。

**方法主张。** 只有瞬时、有界的 KV micro-block 可以访问非所有者。持久 KV
cache 在整个 prefill 过程中始终保持 capacity-weighted。

## 5. 自驱动 Decode 环

Decode 对每个请求只有一个 query token，因此不能再沿 query sequence 维度
切分。HCP 改为保持历史 KV 原地不动，并传递完成一个逻辑 Transformer layer
所需的状态。

### 5.1 Layer packet

在 layer \(\ell\)，临时 starter \(s_\ell\) 收到 hidden state \(h_\ell\)，
计算

\[
x_\ell=\operatorname{Norm}_{in}(h_\ell),
\qquad
Q_\ell=W_Qx_\ell,
\]

并创建包含以下字段的 layer packet：

```text
request 与路由状态：
  request_id, global_position, layer_idx, current_worker,
  visited_workers, kv_assignee

tensor 状态：
  residual h_l, normalized x_l, Q_l,
  attention output accumulator O, LSE
```

Packet 携带 normalized hidden state，是因为唯一 KV assignee 不一定是
starter。Packet 大小取决于模型宽度和 query head 数量，而与历史 context
长度无关。

### 5.2 Exact-once layer 状态机

对于 decode position \(p\)，冻结的 capacity-weighted schedule 为每个 layer
选择唯一 assignee \(a_\ell(p)\)。

Packet 到达时，每个 worker 执行以下操作：

1. 如果自己是 assignee，则从 \(x_\ell\) 唯一一次计算
   \((K_{\ell,p},V_{\ell,p})\)，并把它提交到本地 positioned shard。
2. 用 query 计算本 worker 完整本地 shard 的 attention partial；若本 worker
   是 assignee，则该 shard 已包含刚提交的当前 K/V。
3. 把本地 partial 合并进 \((O,\operatorname{LSE})\)。
4. 除非全部 \(N\) 个 shard 都已贡献，否则将 packet 转发给 successor。

最后一个被访问的 worker 是 finisher。它唯一一次计算非 attention layer tail：

\[
a_\ell=W_OO_\ell,
\]

\[
r_\ell=h_\ell+a_\ell,
\]

\[
h_{\ell+1}=r_\ell+
\operatorname{MLP}_\ell(\operatorname{Norm}_{post}(r_\ell)).
\]

Finisher 立即成为下一 layer 的 starter，不把结果返回给上一 starter。

**已证明不变量。** 一个 layer 恰好执行一次 Q projection、一次 current-token
K/V projection 和持久 append、\(N\) 个互斥本地 attention partial、一次
output projection，以及一次 residual/norm/MLP continuation。

**已证明不变量。** Starter 在发送前先完成自己的本地贡献，finisher 直接消费
完整结果，因此一个精确 layer 经过 \(N-1\) 条物理 ring edge。在每个 worker
都持有必须参与计算的 shard 这一假设下，\(N-1\) 也是单向环的下界。

### 5.3 Layer 与 token 角色递推

按照上文定义的 successor 方向，finisher 是 starter 的 predecessor：

\[
s_{\ell+1}=s_\ell-1\pmod N.
\]

如果末层 finisher 不经过额外 handoff，直接执行 final normalization、
language-model head、sampling 和下一 token 的 embedding，则

\[
s_{t+1,0}=s_{t,0}-L\pmod N.
\]

当 \(L\bmod N=0\) 时，同一个 worker 会持续充当该请求的最终 sampler。这
不会形成 KV hotspot，因为 layer-position assignee schedule 与 sampler 相互
独立。模型参数和 language-model-head 参数已经在所有 worker 上复制；集中
在 sampler 的只有与 context 长度无关的 activation/logit workspace 和计算。

因此，HCP 核心规则保留零 handoff 的跨 token continuation，并用 request
派生的初始 phase 把不同请求的固定 sampler 分散到不同 worker。若测量发现
sampler 排队值得付出额外 edge，可以启用可选的 token-boundary handoff 来
轮转 sampler；这只是调度扩展，不是正确性要求。

**原型证据。** Rust tensor model 覆盖任意 \(N\)、任意 \(L\)、非零 starter、
wrap-around edge、assignee/finisher 的全部重叠情况、finisher-to-starter
continuation、final logits 和 localhost TCP packet transport。这些测试建立的
是模块数据流证据，不是当前跨后端硬件性能证据。

## 6. Continuation Prefill 与 Mixed-History KV

一个请求可能在一次或多次 decode 后追加一个多 token continuation。HCP 不会
把旧 cache 规整成新的物理布局。

设 continuation 前已经提交的历史为 \(\mathcal{P}_{old}\)，新 block position
为 \(\mathcal{P}_{new}\)。对每个 layer，

\[
\mathcal{P}_{old}\cap\mathcal{P}_{new}=\varnothing.
\]

新 position 接受 capacity-weighted prefill partition。它们的 activation 和
Q row 在各 layer 间保持本地，其 K/V pair 被追加到对应的 positioned shard。
最终一个 shard 可能同时包含：

- 由 block position partition 分配的初始 prefill position；
- 每个 layer 独立分配的 decode position；
- 由后续 block partition 分配的 continuation-prefill position。

Continuation 的 Ring Attention 使用 \(\mathcal{P}_{new}\) 对应的本地 Q row，
并通过 positioned KV micro-block 访问并集

\[
\mathcal{P}_{old}\cup\mathcal{P}_{new}
\]

Global position 保证位于 \(q\) 的新 query 只能关注满足 \(p\le q\) 的 key。

**已证明不变量。** 如果每个 layer 的 K、V 和 position array 始终对齐，且
global-position union 完备、无重复，那么各阶段历史可以直接组合。Prefill
和 decode 不需要相同的物理 shard shape 或逐 layer ownership function。

**原型证据。** 一个 24-layer Rust 正确性实验执行

```text
prefill -> decode -> continuation prefill -> decode
```

并使用不等 capacity ticket。它在每个 phase 后验证与稠密 reference 对齐的
hidden state 和 logits，仅对新 continuation position 执行 exact-once
projection，验证完整 position coverage、冻结的加权 decode assignment，以及
精确预分配 slab 中不变的 storage address。

**证据边界。** 该 slab 是有限 horizon 的实验性 cache。它证明 mixed-history
语义能够避免反复拼接完整 shard，但不证明生产 allocator 或加速器物理 peak
memory 结果。

## 7. 正确性论证

HCP 的正确性由四个引理推出。

### 引理 1：所有权完备性

对每一个已提交的 \((\ell,p)\)，admission 选择且只选择一个 owner。因此，
分布式 cache 表示的逻辑 K/V pair 集合与稠密 cache 相同，且不存在持久副本。

### 引理 2：分块 attention 等价性

Online-softmax merge 是对互斥 key set 上 softmax 分子和分母的精确重新分组。
Causal mask 根据 global position 计算。因此，在实数运算下，合并后的
attention output 等于稠密 causal attention。

### 引理 3：Decode exact-once 语义

Packet 恰好访问每个 worker 一次。Assignee predicate 仅在一个 worker 上成立，
所以 current K/V 只被 projection 和 commit 一次。由引理 1，本地 partial 恰好
覆盖此前历史和当前 position；由引理 2，它们的合并结果就是稠密 decode
attention output。

### 引理 4：Transformer continuation

Finisher 执行与稠密 Transformer layer 相同的 output projection、residual
addition、normalization 和 MLP。因此，\(h_\ell\) 相等蕴含
\(h_{\ell+1}\) 相等。先对 layer 归纳，再对 prefill/decode/continuation
phase 归纳，可以得到相同的逻辑 hidden state 和 logits；不同 backend 的
浮点计算顺序可能造成数值差异。

**已证明不变量。** 两种物理数据流不是两种 cache format，而是同一个
position-indexed 逻辑 context 上的两种求值策略。

## 8. 显存、通信与计算复杂度

令 \(T\) 为已提交 context length，\(b_\ell\) 为 layer \(\ell\) 上每个
position 的 KV 字节数，\(D\) 为与模型宽度同阶的 packet dimension，\(L\)
为 layer 数量。

### 8.1 持久显存

逻辑 KV 总显存为

\[
M_{KV}(T)=T\sum_{\ell=0}^{L-1}b_\ell.
\]

Worker \(i\) 只保存分配给自己的 pair：

\[
M_i=\sum_{\ell}\sum_{p\in S_{i,\ell}}b_\ell\le B_i.
\]

忽略整数舍入，\(M_i\) 逼近 \(\alpha_i M_{KV}\)。持久 KV 不会集中到
starter、finisher 或 sampler。Prefill 需要一个有界 KV micro-block buffer；
decode 需要 \(O(D)\) packet/activation workspace。模型权重仍在每个 worker
完整复制。

### 8.2 Prefill 通信

如果一个 layer 的全部 KV 都被环传一次并访问每个非 owner target，则该 layer
的总 ring traffic 近似为

\[
(N-1)T b_\ell.
\]

对 continuation block，\(T\) 表示其新 query 可见的历史，其中包含刚提交的
新 position。Context size 固定时，总 traffic 和 hop work 随 \(N\) 线性增长；
每个 worker 仍只有两个逻辑 peer。

### 8.3 Decode 通信

一个 layer 把一个 packet 发送经过 \(N-1\) 条 edge：

\[
C_{decode/token}=L(N-1)\,|P_{layer}|,
\qquad |P_{layer}|=O(D).
\]

Wire payload 与 \(T\) 无关。Attention 计算并不与 \(T\) 无关：每个 layer
上，所有 worker 合计扫描 \(T+1\) 个 KV position，worker \(i\) 的份额由其
shard 决定。

### 8.4 时延结果

单个 decode packet 必须串行依赖全部参与 worker 和 edge。HCP 消除了冗余的
完整模型 forward 和 context-sized decode 传输，但没有从单请求时延中消除
最慢 worker 或最慢 link。未来 scheduler 可以重叠不同请求的独立 packet；
该并发机制不属于当前核心。

## 9. 假设、局限与有效性威胁

1. **参数复制。** 每个 worker 都必须容纳完整模型权重并实现兼容的 layer
   语义。
2. **有限 admission horizon。** 当前 KV 硬上界假设已经声明并预留 horizon。
   开放式增长需要尚未定义的 allocator 和重新准入策略。
3. **Fail-stop 环。** 在没有 KV replication 时，丢失一个 worker 就会丢失
   唯一 shard，并中止受影响请求。弹性移除和容错需要不同的显存权衡。
4. **串行 decode path。** 单请求在每个 layer 都访问全部 worker。即使慢
   worker 提供了有用显存，它或其链路仍可能主导时延。
5. **Prefill traffic。** KV 环传量随可见 context 和 ring size 增长。有界
   streaming 控制瞬时显存，但不会减少总字节数。
6. **Backend 兼容性。** 设备本地 kernel 可能采用不同的浮点计算顺序。
   Wire dtype/layout 转换和数值容差需要端到端验证。
7. **当前不包含生产 runtime 主张。** Admission、request multiplexing、
   backpressure、retry、fragmentation、observability 和多请求 fairness 都不
   属于本方法草稿。
8. **当前不包含经济性主张。** 更低成本、更高能效和更充分利用混合加速器池
   都是假设，必须与适当的同构和粗粒度 baseline 比较后才能成立。

## 10. 评测设计

最终评测应回答研究问题，而不只是证明原型能够运行。

### RQ1：HCP 是否保持模型语义？

- 在每个 phase boundary 与单 worker 的 dense/incremental reference 比较。
- 覆盖初始 prefill、多步 decode、continuation prefill 和第二段 decode。
- 扫描 \(N\)、\(L\)、非零 starter、wrap-around edge、不等 capacity vector，
  以及所有 starter/assignee/finisher 重叠情况。
- 报告 hidden-state/logit error、sampled-token agreement、position-union
  completeness、duplicate position 和 exact-once operation count。

### RQ2：Capacity weighting 是否满足本地显存合同？

- 在权重和固定 workspace 加载后测量真实 peak device memory。
- 逐 worker 比较 admitted bytes、reserved bytes、committed bytes、allocator
  overhead 和瞬时 receive buffer。
- 验证没有 worker 物化远端持久 shard，也没有超过声明的 \(B_i\)。
- 扫描异构 capacity ratio、prompt/continuation 组成、decode horizon 和
  fragmentation pattern。

### RQ3：P2P 数据面是否符合 scaling model？

- 对每条 directed edge 记录 bytes 和 message 数量。
- 验证每 layer \(N-1\) 个 decode hop，以及与 context 无关的 decode packet
  size。
- 在测量总 KV traffic 的同时，验证有界 prefill receive buffer。
- 扫描 ring size、context length、KV micro-block size、bandwidth 和 latency。

### RQ4：异构协作在哪些区域有收益？

至少比较：

- 工作负载可以装入时，速度最快的单 worker；
- worker 数量相同的同构 ring；
- equal-shard P2P ring；
- capacity-weighted HCP；
- 在相同资源池上、存在公平实现时的 request-level 或 prefill/decode-stage
  placement。

只有在测量方法和资源计费固定后，才报告 time to first token、inter-token
latency、throughput、device utilization、link utilization、peak memory、
energy 和 cost。

### RQ5：哪些设计选择控制瓶颈？

消融比较：

- capacity-weighted 与 equal ownership；
- \(N-1\)-hop finisher consumption 与 \(N\)-hop return-to-starter route；
- KV micro-block size；
- 固定的 per-request sampler phase 与可选 token-boundary rotation；
- 单请求执行与独立多请求 packet overlap，后者仅在对应机制存在后评测。

### 必需的证据递进

评测必须依次跨越不同证据层级：

1. 确定性 tensor correctness；
2. 进程内 positioned-cache lifecycle；
3. 真实 P2P packet transport；
4. 每种参与 backend 上的物理显存验证；
5. 完整跨 backend 异构生命周期；
6. 性能、scalability 和 cost 对比。

较低层级通过，不能被报告成更高层级的证据。
