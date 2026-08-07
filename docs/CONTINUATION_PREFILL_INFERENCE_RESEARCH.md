# HCP 推理场景 Continuation Prefill 数据流研究

## 摘要

本研究只讨论推理，不讨论训练反向传播。问题被限定为：同一请求已经有长度为
`T` 的历史 KVCache，本轮追加长度为 `m` 的 prompt segment 时，新增 query
怎样对完整的 `T+m` 个可见 key/value 做精确 causal attention，以及这一过程
在只允许 predecessor/successor 邻接 P2P 的异构 ring 上应移动什么数据。

结论是高置信度的：历史 KV 不需要在数学上沿 ring 移动。只要每层历史与新增 KV
按绝对 position 形成互斥且完备的分片，每个新 query 覆盖所有分片，并用可结合的
online-softmax 状态合并 partial attention，KV 原地与 KV 环传在实数域中等价。
Ring Attention 原论文在 inference 附录中选择 circulating KV cache，是一种有效的
执行路线，不是历史 KV 必须移动的证明。FlashInfer、SGLang、vLLM 的 decode CP
实现以及 Helix Parallelism 从不同侧面给出了 KV-stationary inference 的实现证据。

对于 HCP，continuation prefill 不能脱离整层 activation 数据流单独讨论。新增
K/V 必须由当前层新增 segment 的 normalized activation 生成；attention 完成后，
output projection、residual、Norm 与 MLP 又必须产出下一层 activation。因此原计划
中的 `m` 长度 activation packet 正是 KV-stationary continuation 的一条完整候选
路线，不是与 continuation 无关的 transport 实验。当前证据支持同时保留 KV-ring
基线和 KV-stationary packet 路线，以显式实验模式比较；现阶段不应引入动态 planner。

## 研究范围与证据演进

第一轮研究从 Ring Attention 原论文、作者 JAX inference kernel、PyTorch Context
Parallel 和 Megatron Core Context Parallel 出发。共同模式是 sequence/context
分片后，每个 query 必须获得完整可见 KV 的贡献。Ring Attention 作者实现固定本地
query，并以 `jax.lax.ppermute` 循环 K/V block；PyTorch 和 Megatron 的训练 CP
文档也主要通过 KV all-gather 或 ring P2P 实现的 gather/reduce-scatter 解决这一
依赖。这个证据足以说明 KV-ring 是正统且可复用的基线，但这些资料主要围绕训练或
单 token inference，不能直接回答 request cache 的 continuation 生命周期
（Liu et al., 2023; NVIDIA, n.d.; PyTorch, n.d.）。

第二轮研究转向 inference serving。FlashInfer 的 paged KV append API 只把新增 K/V
写入已分配的 page，并以显式 position 和 page table 访问完整 cache。SGLang 明确
区分 `forward_extend` 与 `forward_decode`；其 extend paged 路径先写新增 K/V，再让
新增 query 访问完整 paged cache。其 ragged 路径则把新增 segment 内部的 causal
partial 与历史 prefix 的 non-causal partial 分开计算，再以 O/LSE 精确合并。
这证明 continuation 的核心合同是“`m` 个新 query 对 `T+m` 个可见 KV”，而不是
“重新拼接或搬运 `T` 个历史 KV”（FlashInfer Project, n.d.; SGLang Project,
n.d.）。

vLLM 的官方 Context Parallel 文档进一步把 prefill CP 与 decode CP 明确分开：
prefill 可以选择 partial-Q/full-KV 或 partial-Q/partial-KV ring，而 decode 将 paged
KVCache 沿时间维分片。其 DCP all-to-all 源码把每个 KV shard 的 partial output 与
LSE 打包交换，再做精确 LSE-weighted combination。Helix Parallelism 采用相同的
KV-stationary 数学方向，并指出 decode 新 KV 可以按 rank 交错追加。这些实现依赖
collective、同构 GPU 和各自的模型并行布局，不能直接移植到 HCP；但它们对“历史
KV 原地是否成立”的证明价值很高（Bhatia et al., 2025; vLLM Project, n.d.）。

两轮资料之间不存在算法矛盾。它们描述的是同一个 attention 等式的两种数据移动
对偶：固定 Q 并移动 KV，或者固定 KV 并移动 Q 与 softmax accumulator。原论文的
inference 建议回答了“Ring Attention 怎样自然延伸到 decode”，serving 系统回答了
“request-owned KVCache 怎样避免重复搬运历史”。HCP 的任务不是在二者中宣告唯一
正确路线，而是在 neighbor-only、异构 capacity-weighted 约束下找出阶段适配点。

## 数学合同

设本轮 continuation 进入某一层前已有历史位置 `0..T-1`，新增位置为
`T..T+m-1`。第 `r` 个新增 query 的绝对位置为 `p_r=T+r`。节点 `i` 在这一层
永久保存的 KV position 集合记为 `S_i`。正确分片必须满足：

```text
S_i intersect S_j = empty,  i != j
union_i S_i = {0, 1, ..., T+m-1}.
```

对 query `r`，因果可见集合为 `V_r={p | p <= T+r}`。节点 `i` 只在
`S_i intersect V_r` 上计算 score。令该局部集合上的最大值、指数和与未归一化
value 和分别为 `M_i(r)`、`L_i(r)` 与 `A_i(r)`：

```text
M_i(r) = max_p s(r,p)
L_i(r) = sum_p exp(s(r,p) - M_i(r))
A_i(r) = sum_p exp(s(r,p) - M_i(r)) V_p.
```

空分片定义为 `M=-inf, L=0, A=0`。两个 partial 状态 `a`、`b` 的合并为：

```text
M = max(M_a, M_b)
L = exp(M_a-M)L_a + exp(M_b-M)L_b
A = exp(M_a-M)A_a + exp(M_b-M)A_b.
```

这个运算在实数域中可结合、可交换。完成所有节点的合并后，`O=A/L`，
`LSE=M+log(L)`，恰好等于在完整 `V_r` 上一次性计算的 softmax attention。
若局部 kernel 输出已经归一化的 `(O_i,LSE_i)`，同一结论可写为：

```text
LSE = logsumexp_i(LSE_i)
O = sum_i exp(LSE_i-LSE) O_i.
```

因此历史 KV 是否移动不会改变结果；唯一数学要求是每个 query 的 accumulator 恰好
消费每个可见 KV 分片一次。绝对 position mask `p_k <= p_q` 同时处理历史 KV、
本轮更早的新增 KV 和非连续 positioned shard。浮点实现会因合并顺序产生舍入差异，
但这不是语义差异，现有 Rust batched positioned oracle 已在 `m=2/3`、非连续
shard、不同 starter 和 wrap-around 上验证到 `1e-5` 以内。

## 新 KV 的就绪条件

历史 KV 原地不意味着当前层的新增 KV 可以凭空就绪。设这一层输入 activation 为
`X`，则 input Norm 后得到 `Xn`，并由 `Xn` 投影 `Q`、`K_new`、`V_new`。
对于 position `p` 被分配到节点 `i` 的情况，节点 `i` 必须在计算自己的 local
attention partial 前拿到 `Xn[p]`，生成并永久 append `K_new[p],V_new[p]`。
它不需要其他节点的新 KV，也不需要全局 barrier；packet 到达一个节点后，该节点
先生成自己负责的位置，再把本地完整 positioned shard 纳入 partial 即可。

segment 内因果性不会引入额外依赖。节点可以一次投影自己负责的所有新增位置，随后
对每个 query 用绝对 position mask 排除未来位置。计算 K/V 不依赖同层 attention
output，所以无需按 token 串行推进 `m` 次。只有在这一层的所有 query attention
完成后，才能执行 output projection、residual 和 MLP，生成下一层的 `X`。

这也是 `m` 长度 activation packet 与 continuation 直接交合的原因。若 packet
携带整段 `X`、`Xn` 和 `Q`，每个节点都能从 `Xn` 中选取自己的 position 子集生成
K/V，并用 packet 中的完整 Q 更新 O/LSE。最后一个节点持有完整 attention output，
执行 `W_o`、residual、post-attention Norm 和 MLP，再以新 hidden 启动下一层。
这里 Norm 和 MLP 没有被拆成 ring partial；它们是逐 token 运算，在 activation
所在节点执行一次即可。随着每层 starter/finisher 轮转，这部分计算跨层和跨请求
分散，但单层内仍集中在 finisher，这是一项必须实测的异构 latency 风险。

当前 Rust `LayerPacket` 正好携带 `residual`、`normalized`、`position_ids`、`Q`、
`attention_output` 和 `LSE`，并在 finisher 执行 `W_o + residual + Norm + MLP`。
但 `validate_route` 仍限制 sequence length 为 1；当前代码只完整证明了 decode，
batched positioned accumulator 只证明了 attention 数学，尚未证明 `m>1` 的整层
K/V 本地生成、wire、MLP 和 24 层递推。研究结论不能把这部分写成已经实现。

## 通信成本模型

统一记号如下。节点数为 `N`，hidden width 为 `H`，query width 为 `D_q`，KV
width 为 `D_kv`，query head 数为 `h_q`，每个元素 `b` bytes。对标准 GQA，通常
`D_q=H`，并令 `g=H/D_kv`。以下成本都是每请求、每层、所有 directed link 上的
byte-hop 总和；它不是单包大小，也不是某条 link 的峰值。所有公式先采用理想的
`N-1` hops，若为了让结果回到原始节点而走完整一圈，则必须显式改为 `N`。

KV-ring 基线保持 query/activation 按 position 分片，让每个 KV shard 访问其他
`N-1` 个节点。其总流量为：

```text
C_KV = (N-1) * 2(T+m)D_kv * b.
```

在单向 ring 中，每条 link 会看到除某一个初始 shard 外的所有 KV；若节点 shard
bytes 为 `B_i`，单层最重 link 为 `sum_i(B_i)-min_i(B_i)`。不均匀 capacity
weight 不改变 byte-hop 总量，但会改变 per-link 峰值。接收端可用固定 chunk buffer
流式处理，使额外显存与 chunk size 而不是 `T` 成正比；不过它仍临时持有一个外来
KV chunk，因此只有 KV-stationary 路线能完全消除历史 KV 的临时传输压力。

当前完整 self-driving packet 在 ring 上携带 `X`、`Xn`、`Q`、`O` 和 `LSE`，忽略
很小的 route metadata 和 position Int64 后，其成本为：

```text
C_packet4 = (N-1) * m(2H + D_q + H + h_q) * b
          = (N-1) * m(4H + h_q) * b,  when D_q=H.
```

第一个 `2H` 分别是 residual 与 normalized activation，后两个 `H` 是 Q 与 O。
如果每个节点从 residual 重算 input Norm，可以不传 `Xn`，得到：

```text
C_packet3 = (N-1) * m(3H + h_q) * b.
```

这会让 Norm 重复 `N` 次，牺牲计算换网络，现阶段只能作为后续优化分支，不能直接
替换正确性基线。若继续去掉 residual，finisher 将无法完成 residual connection；
除非 activation 本来就在 finisher，或额外传输 residual，否则 `Q+O+LSE` 不是
一条完整的层间数据流。

另一条严格成立的路线是 query-shard return。各节点保留自己的 position activation，
本地生成该子集 Q/K/V；每个 Q shard 携带 O/LSE 绕完整一圈，回到原节点后执行
`W_o + residual + Norm + MLP`。所有 query shard 长度之和为 `m`，所以成本为：

```text
C_Q_return = N * m(D_q + H + h_q) * b
           = N * m(2H + h_q) * b.
```

它保留 tokenwise non-attention 计算的并行性，但需要 `N` hops。若坚持 `N-1`
hops，结果停在原节点的 predecessor，必须让 residual 随 Q/O 一起移动并让 activation
归属逐层旋转；这时成本约为：

```text
C_Q_rotate = (N-1) * m(3H + h_q) * b.
```

因此此前保存的 `N-1` 与 `N` 路线选择确实对应两种不同的层间 activation 合同，
而不只是少一次 send 的实现细节。`N` hops 让结果回到 query holder，`N-1` hops
则必须接受 finisher/activation holder 轮转或额外传 residual。两者都只用邻接 P2P，
总网络成本对单请求仍随 `N` 线性增长。

永久 KV 显存对上述路线相同。设节点 `i` 的 capacity weight 为 `w_i`，共有 `L`
层，则目标是：

```text
M_KV_i ~= w_i * 2L(T+m)D_kv * b,
sum_i w_i = 1.
```

整数 position 会带来最多若干 KV slot 的舍入误差。重要的是 assignment 应作用于
`(layer, absolute_position)` 的 append unit，而不是把可变长度的整个 segment
永远当成一个等权事件。把整段 `m` K/V 交给单一节点在数学上可行，也可能在 24 层
总量上接近权重，但当 segment 长度变化时会破坏 byte-level capacity 比例。下一
实验应复用确定性的 capacity-weighted append sequence，为每个新增 position 指定
per-layer assignee；这不是动态 planner，也不要求迁移已有 KV。

## Qwen2-0.5B 数值例子

当前模型参数为 `H=896`、`D_q=896`、`D_kv=2*64=128`、`h_q=14`、`g=7`、
BF16 `b=2`，并取 `N=3`。KV-ring 每个 `T+m` position 每层产生 `1,024`
byte-hops。当前完整 packet 每个新增 token 每层产生 `14,392` byte-hops；重算 Norm
的三张量 packet 为 `10,808` byte-hops；只考虑 Q/O/LSE 的 `N-1` 理想下界为
`7,224` byte-hops，但这个下界本身不含完整 residual/层间合同。

比较当前完整 packet 与 KV-ring，公共的 `(N-1)Hb` 消去后得到：

```text
4m + (h_q/H)m < 2(T+m)/g.
```

因此完整 packet 的纯网络 break-even 为 `T > 13.0546875m`。重算 Norm 的 packet
阈值为 `T > 9.5546875m`。若只比较 Q/O/LSE 的 `N-1` 理想下界，阈值为
`T > 6.0546875m`。query-shard 绕 `N=3` hops 回原节点的实际阈值则是
`T > 9.58203125m`。

当 `T=0,m=128` 时，KV-ring 是 `131,072` byte-hops，完整 packet 是
`1,842,176` byte-hops，前者约低十四倍。这解释了 initial prefill 和无长 prefix
的大段 continuation 为什么应保留 KV-ring。当 `T=4096,m=128` 时，KV-ring 为
`4,325,376` byte-hops，完整 packet 为 `1,842,176` byte-hops，KV-stationary
开始明显占优。当 `T=4096,m=1` 时，两者分别约为 `4.00 MiB` 与 `14.1 KiB`，这正是
单 token decode 必须保留 Q/O/LSE 自驱动路线的原因。

这些阈值只比较 byte-hop，不是性能选择器。KV-ring 可以让多个 query shard 并行
工作并重叠 KV 通信；单个完整 activation packet 在每层按节点顺序执行 attention，
并把当层 MLP 放在一个 finisher。异构节点的带宽、attention 吞吐、MLP 吞吐、
per-link 最慢边和多请求并发都会改变 latency。当前阶段应把公式用作实验分层依据，
不能据此引入运行时动态 planner。

## 对完整请求生命周期的含义

Initial prefill 从 `T=0` 开始，新增 segment 同时也是完整 context。position-sharded
activation 配合 KV-ring 在 GQA 模型上通信紧凑，并允许各节点对自己的 query token
执行 Norm、QKV、attention output 和 MLP。阶段结束后，环传的 foreign KV 临时副本
丢弃，每个 `(request,layer,position)` 只在 capacity-weighted assignee 保留一份。

Decode 是 `m=1` 的极端 continuation。当前 token 的 normalized activation 与 Q
沿 self-driving packet 到达各节点；唯一 assignee 生成并保存当前层 K/V，所有节点
依次用本地 durable history 合并 O/LSE，finisher 完成该层 MLP。历史 KV 不上环，
packet bytes 与 `T` 无关。24 层结束后 sampler 所在节点得到 next token；末层 sampler
即使因 `L mod N = 0` 固定，也只增加很小的 logits/sampling 临时量，不改变 KVCache
capacity pressure，真正需要关注的是重复的末层计算 latency 而非显存集中。

Continuation prefill 的正确语义不是重建 cache，而是只投影新增 `m` 个位置，并把
它们 append 到同一套 positioned shard 中。若走 KV-ring，历史加新增 KV 会作为临时
block 访问 query shard；若走 KV-stationary packet，历史完全原地，packet 让所有新
query 依次消费每个 local shard。无论走哪条路线，阶段结束时的永久 cache 格式都
相同，所以随后的 decode 或下一次 continuation 只依赖 position union 与 capacity
reservation，不依赖上一阶段选择过哪种通信对象。

多请求不改变单请求数学。每个 request_id 有独立 positioned KV ledger 和 packet；
多个请求可以同时在 ring 的不同边和不同层推进，不要求彼此等待同一 phase。按
request_id 分散 starter phase 可以平滑 MLP/finisher 与链路占用，但这是后续服务
调度问题，不属于本研究的下一实现节点。

## 路线裁定与下一实验

当前裁定是保留四种分析路线，但只把两种提升为近期实验模式。KV-ring 是 initial
prefill 与大 `m/T` continuation 的 correctness 和带宽基线。完整 activation packet
是最接近现有 Rust self-driving decode 的 KV-stationary continuation 路线，也是
下一步应验证的对象。重算 Norm 的三张量 packet 是有明确牺牲的优化，应等完整 packet
测出网络瓶颈后再考虑。query-shard return 数学成立且能保留 tokenwise MLP 并行，
但它引入多 packet 与 `N` hops 的层间合同，暂存为后续路线，不与下一节点混做。

下一实验应继续小步推进，且不接真实 runtime、多请求或动态 planner。第一步只把
`LayerPacket` 的 shape 合同从 `m=1` 推广到 `m>1`，以一层 synthetic oracle 验证
residual、normalized、Q、position、O/LSE 与 finisher MLP 的完整 shape 和数值；
这一小步不宣称 capacity 分配完成。第二步加入按 `(absolute_position,layer)` 冻结的
capacity-weighted assignee vector，让每个节点只投影并 append 自己负责的新 K/V，
验证无遗漏、无重复和历史 KV 不上传输。第三步才进入用户要求的 24 层、`N=3`、
`tickets=[1,3,2]` mixed-history oracle，执行已有 prefill/decode 后的 `m=6`
continuation，再 decode 一次，并同时记录 dense-reference 差异、每节点 KV slot、
每层 byte-hop、最重 link 与 finisher 分布。

这个次序不会改变已成立的核心方案。第一步可证伪整层 packet，第二步可证伪本地
new-KV readiness，第三步可证伪跨层递推和 capacity-weighted 永久显存。只有第三步
通过，才能说 KV-stationary continuation 在 HCP Rust 核心中闭环；在此之前，现有
KV-ring continuation baseline 继续作为可运行后备路线。

## 研究限制

本研究证明的是数学可行性、官方实现先例和当前代码映射，不是硬件性能结果。业界
资料大多假设同构 GPU 与 collective；HCP 的 neighbor-only 异构 ring 会呈现不同的
per-link bottleneck。通信公式忽略 frame header、alignment、QUIC/TCP framing 与
kernel launch latency，也没有处理 segment chunking。capacity weight 目前只保证
永久 KV 显存目标；如果高显存节点的计算或入边带宽较弱，stationary attention 会让
它承担更多 local KV dot-product，仍可能成为 latency 瓶颈。这个矛盾应在实验数据
出现后记录并修订，不应提前用生产级调度器掩盖。

## 参考文献

Bhatia, N., More, A., Borkar, R., Mitra, T., Matas, R., Zhao, R., Golub, M.,
Mudigere, D., Pharris, B., & Rouhani, B. (2025). *Helix parallelism: Rethinking
sharding strategies for interactive multi-million-token LLM decoding*.
arXiv. https://arxiv.org/abs/2507.07120

FlashInfer Project. (n.d.). *append_paged_kv_cache*. FlashInfer documentation.
https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html

FlashInfer Project. (n.d.). *Attention APIs*. FlashInfer documentation.
https://docs.flashinfer.ai/api/attention.html

Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with Blockwise
Transformers for near-infinite context. *International Conference on Learning
Representations*. https://arxiv.org/abs/2310.01889

Liu, H. (2023). *ringattention_jax_inference.py*. GitHub.
https://github.com/lhao499/llm_large_context/blob/main/ringattention/ringattention_jax_inference.py

NVIDIA. (n.d.). *Context parallelism*. Megatron Core developer guide.
https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html

PyTorch. (n.d.). *Introduction to Context Parallel*. PyTorch tutorials.
https://docs.pytorch.org/tutorials/unstable/context_parallel.html

SGLang Project. (n.d.). *FlashInfer attention backend*. GitHub.
https://github.com/sgl-project/sglang/blob/master/python/sglang/srt/layers/attention/flashinfer_backend.py

vLLM Project. (n.d.). *Context Parallel deployment*. vLLM documentation.
https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/

vLLM Project. (n.d.). *DCP all-to-all communication backend*. GitHub.
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/dcp_alltoall.py
