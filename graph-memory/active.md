# Active Context

当前活跃的任务、决策、风险和假设。

### 当前焦点：hetero-cp-ringattn 向 vLLM 生态插件收敛 + PageAttn/block KV 整合

type: `task` · status: `ongoing` · confidence: 0.85 · importance: 0.95 · source: `user-direction`

三步顺序完成后，vLLM 线已具备：CUSTOM ring backend(online softmax 显存切分) + HcpRingKvConnector(切分瞬时 peer KV) + 跨节点异构(CUDA↔ROCm)闭环。
下一步（用户给定方向）：
1. 把 hetero-cp-ringattn 分布式调度框架整理成标准 vLLM 生态插件（entry points 注册、配置化、可随 vLLM 官方更新跟进），既有异构长上下文能力又不 fork 内核；
2. 整合 PageAttn 与 hetero-cp-ringattn 的 block KV：现在 ring backend 用 plain-PyTorch fp32 逐请求算 attention，需评估与 vLLM paged attention/flash_attn 内核的融合路径；
3. 解除 PoC 限制：PEER_KV_STAGING 按 layer 键限单并发(max_num_seqs=1)，consumer 必须关 prefix caching；工程化需支持多请求并发 staging（按 request 键）。
[2026-07-21 更新] 执行顺序修正为 3→2→1(原记录 1→2→3)：per-request staging 是地基(数据结构正确性)；paged kernel 化建在其上(按请求取 staging)；插件化最后(对外配置面等二者定型再冻结)。三者动机剖析已落 decision 节点：decision-per-request-staging-20260721 / decision-ring-paged-kernel-20260721 / decision-vllm-plugin-packaging-20260721。

_updated: 2026-07-21 13:28:03_
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
### 步骤2(次做)：ring attention 从 plain-PyTorch 换成原生 paged kernel + cascade 式 LSE 合并

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `user-direction`

【现状】ring_backend._attn_with_lse 为自写 plain PyTorch fp32：每请求每层把 K/V 从 paged cache gather 成连续张量，einsum 物化完整 score 矩阵 [H, Tq, Tk] 再手动 softmax+LSE。2048 token 可跑，但 score 矩阵显存 随长度平方增长，128K/1M(HCP 卖点)直接爆显存；fp32 无 kernel 融合，速度差原生一个量级。
【动机】显存切分省下的显存会被自实现低效吃回去；不长上下文，跨节点能力无实用价值。
【vLLM 怎么做】(a) PagedAttention：KV 按 block 分页，kernel 以 block table 为索引直接读分页内存，不 gather、不物化 score 矩阵，内部本即 online softmax 分块；(b) cascade attention：与 HCP merge 数学同构(见 belief-vllm-cascade-attn-20260721)；(c) FlashAttention kernel 支持输出 LSE。
【目标态】chunk B 走 vLLM 原生 paged kernel(带 LSE)，chunk A 对 staging buffer 跑一次 flash kernel(带 LSE)，再一次 LSE merge。score 矩阵不再物化，长度天花板消失，速度接近原生。

_updated: 2026-07-21 13:28:03_
### 步骤1(最后做)：从研究脚本收敛为标准 vLLM 生态插件

type: `decision` · status: `held` · confidence: 0.85 · importance: 0.9 · source: `user-direction`

【现状】插件能工作但形态是研究脚本：pip install -e 本仓库、手写长 kv_transfer_config dict、环境变量控制行为；验证脚本硬编码模型路径与节点 IP；无版本兼容声明。
【动机】KVConnectorBase_V1 是 experimental API(见 belief-connector-api-experimental-20260721)，不收敛插件边界则 vLLM 升级可能悄悄破坏兼容性；收敛后别人 pip install + 两个参数即可获得异构长上下文能力。
【vLLM 怎么做】官方答案就是插件：两条标准扩展面(KV connector 接口 + attention backend 注册表)，NIXL connector / LMCache / Mooncake 均走同一 KVConnectorBase_V1 接口，无人 fork 内核；我们的 ring backend 注册在 CUSTOM，与官方后端机制平级。此步非发明新东西，是打磨已在正确接口上的代码。
【目标态】entry points 自动注册、配置项收敛为文档化的少数键、声明兼容的 vLLM 版本区间、留最小可跑示例；vLLM 升级时跑兼容性验证脚本即知坏没坏。
【为何最后】插件定义的对外配置面应等 staging(3)与 kernel(2)定型后再冻结，避免刚发布就改配置。

_updated: 2026-07-21 13:28:03_
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
