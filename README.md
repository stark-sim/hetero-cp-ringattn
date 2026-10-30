# hetero-cp-ringattn

**HCP（Heterogeneous Context Parallelism）**：让多个**异构**计算 domain（不同厂商、不同代际、不同显存的 GPU/NPU/Apple Silicon）以**不均分**方式共同完成同一个 attention layer，把超长上下文（200k → 1M → 10M）从"单卡显存墙"变成"可调度问题"。

> *HCP lets heterogeneous accelerators (CUDA / ROCm / Ascend NPU / Apple MPS) cooperate on a single attention layer via uneven sequence-chunk context parallelism over a P2P KV ring with blockwise online softmax — turning the long-context memory wall into a scheduling problem.*

## 为什么存在：产品问题

长上下文需求持续增长（多文档归纳、agent 长链路记忆、大代码仓分析、法务/科研整包推理），而现实算力持续**异构化**。如果不同类型的设备无法合作，系统的长上下文能力就永远受制于"最强单卡"，而不是"系统总算力"——面对更长 context 只剩三条退路：截断输入、有损摘要、拒绝任务。

HCP 的产品假设相反：

> 只要把每个 domain 的 `seq_chunk` 和 `block_size` 压到各自设备吃得下的范围，系统就应该允许多个异构 domain 共同承担一次长上下文 attention 计算；context 变长时，靠增加设备外推可支持长度，而不是等下一代更大的卡。

愿景不是"把所有设备伪装成一台同构超级机"，而是：**承认设备不对称、承认带宽是主要瓶颈，仍然让这些设备在同一个 attention 任务里合作。**

## 核心设计理念

1. **异构是常态** —— 真实部署中 GPU 代际、显存、互联带宽 rarely 一致；调度必须按 capacity 分配，而不是强行对称。
2. **非均等 CP 是可行性前提** —— 均匀分片在异构显存下会让小显存设备先 OOM；分片应匹配设备能力边界（实测：worker 上报空闲显存，coordinator 按预留后容量比例切分）。
3. **数学必须的是 blockwise online softmax；P2P ring 是工程占优调度** —— 跨厂商后端没有共同 collective 栈（CUDA NCCL / ROCm RCCL / NPU HCCL / MPS 无），低带宽 WAN（实测跨网 520 KB/s 级链路）下 all-gather 类通信不成立；ring 用逐跳 P2P 把这些约束变成可调度参数。
4. **Correctness 优先于性能** —— 数值正确性未跨平台稳定通过前，不引入量化、近似 attention、非确定性 kernel。所有结论以 argmax/文本一致性 + logits 噪声地板为准。

架构边界：

```
Coordinator (Rust) ──QUIC──► Worker / Domain ──► 域内后端（tch-rs / transformers / vLLM 插件）
                              ├─ 控制面：WorkerCommand / WorkerResponse (bincode)
                              ├─ 数据面：P2P KV ring + online softmax
                              └─ 模型面：可插拔 WorkerBackend
```

- **跨域协议**：P2P `send_kv_block` / `recv_kv_block`，支持任意 `seq_chunk_len` 与 `block_size`。
- **域内黑盒**：HCP 只定义跨域数据流；域内可以是单卡，也可以是多卡 TP/CP 组成的"抽象 worker"（一个逻辑 domain）。
- **1 GPU = 1 worker**：禁止单卡多 worker 重复加载完整权重。

## 已验证成果（全部有原始数据 + commit 锚点，见 `docs/` 与 `reports/`）

| 里程碑 | 结论 | 证据 |
|---|---|---|
| 跨厂商数值地板 | 同一模型在 Ascend NPU / NVIDIA CUDA / AMD HIP 上单域前向一致到 fp32 噪声地板（max\|Δlogit\| ≈ 5e-5），argmax 全一致 | `docs/HETERO_FEASIBILITY_EVIDENCE.md` |
| 3-domain 精确 CP | 跨三厂商 KV ring 与各单域 golden 差异 ≤ 6.3e-5（同噪声地板），全部 decode 步 argmax 一致 | 同上（`07c6f27` 起） |
| 4-domain 全异构 CP | NPU+CUDA+HIP+MPS 四环 max\|Δ\| ≤ 4.2e-5，**低于** golden 两两互差（≤ 4.9e-5）——ring 引入的误差在跨后端噪声下不可分辨 | R5（`dc8a6b9`/`1b4c328`） |
| Rust↔Python 混合 ring | Python(NPU) + Rust(CUDA) + Rust(HIP) 3-domain ring 与 golden 逐 token 一致 | `af864ed`/`cdcc875` |
| 抽象 worker（域内并行） | 双芯 Ascend 910 经 HCCL 组成单逻辑 domain 接入外层 CP ring（R6）；laptop 4060 + white 4090 经 NCCL 跨机组成单逻辑 domain 4-domain PASS（R7） | `f9f7890`/`6ed8bc7`；`a6b36bd`/`90b9be9` |
| 7B 级模型 CP | Qwen2.5-7B bf16 双逻辑 worker CP PASS（含 TP dtype 一致性检查与 bf16 门加固，Critic 第 5 轮闭环） | `71585a6`/`bcc3601`/`2f80ba4` |
| vLLM 生态插件 | ring-attention CUSTOM backend + `HcpRingKvConnector`（显存切分 CP，KV 瞬时 staging）+ 自研 Triton kernel（LSE 输出，CUDA/ROCm 同一实现），已解耦为独立仓 | [hcp-vllm-plugin](https://github.com/stark-sim/hcp-vllm-plugin) |
| 对抗评审闭环 | Critic 评审 5 轮抓到并修复真实缺陷（RoPE 全局 position bug、周期性 prompt 掩盖、NCCL 版本 wire 不兼容等），形成 falsification 流程资产 | graph-memory lesson 层 |

实验环境的网络是**真实异构 WAN**：容器↔其他节点走自建 Tailscale DERP 中继（RTT ~120ms，实测 ~520 KB/s），white↔pearl 为 2.5GbE LAN——ring 调度在这样的带宽差异下仍然成立。

## 当前主赛道（2026-10）

进入 PD 分离生态的 **P 侧：异构请求级 CP prefill**。定位（经 Critic 对抗分析确认为无先例区间）：

> **跨引擎 PD**——HCP 作为 dedicated prefill tier 完成长上下文 prefill，KV 经 connector 交给 vLLM 等成熟引擎做 decode。卖点：硬件成本低（混合存量卡即可组队）、通信要求低（ring P2P 不需要 RDMA/collective 栈）、不强制同生态。

生态空档依据：PD 分离已主流化（DistServe/SGLang/llm-d/vLLM），prefill CP 近线性扩展且中低带宽可行（MLSys 2025），但业界异构只做到**池级调度**，请求级跨厂商 CP 无人做。

正在推进的 prefill 线 debt：

1. workspace-aware admission（30k context 并发 OOM 闸门，准入按并发计入 attention 瞬态工作区）
2. fused attention kernel（SDPA/flash 桥接，消除 per-block scores ~134MB 中间体）
3. 长上下文异构 prefill 对外证明实验

decode 线已收口：自驱动环已合并为主线默认（`474e9cb`，N=3 WiFi 复测延迟 -6~-11%、带宽 -33.4%），代码与证据保留，不再投入新工作。

## 仓库地图

两个产品级产出已解耦为独立仓库：

| 仓库 | 内容 | 生命周期 |
|------|------|----------|
| [stark-sim/hcp-vllm-plugin](https://github.com/stark-sim/hcp-vllm-plugin) | HCP 的 vLLM 生态插件：`HcpFileConnector`（跨引擎 PD 主路线：载入 HCP 导出的 prefill KV 直接 decode，vLLM 0.31）+ ring-attention CUSTOM backend + HcpRingKvConnector（显存切分 CP，KV 瞬时 staging）+ 自研 Triton kernel（LSE 输出，CUDA/ROCm 同一实现） | 跟随功能路线（PD 闭环、N>2 ring、kernel hardening） |
| [stark-sim/vllm-rocm-gfx1200](https://github.com/stark-sim/vllm-rocm-gfx1200) | RX 9060 XT（gfx1200, RDNA4）适配 vLLM 的补丁与构建/运行脚本（纯补丁仓：vLLM 源码构建时现拉，故仓内只有 patch + shell） | 跟随 ROCm/vLLM 版本 |

本仓保留：Rust/Python 分布式调度核心、transformers 线 worker、跨节点驱动脚本（`scripts/run_cross_node_*.sh`，通过 `*_PLUGIN_REPO` 环境变量指向插件 clone，默认 `/home/stark/hcp-vllm-plugin`）、graph-memory 知识库、docs/reports。

解耦原则：插件功能只在 hcp-vllm-plugin 仓改；gfx1200 兼容性只在 vllm-rocm-gfx1200 仓改；本仓只做调度核心、跨节点驱动与知识库。

## 目录导览

| 路径 | 内容 |
|------|------|
| `rust/` | 调度核心：coordinator、worker SDK、QUIC 控制/数据面、tch 后端、capacity 调度 |
| `python/` | transformers 线 worker（精确 CP：逐层 KV ring + 全局因果注意力）、TP 抽象 worker、PD KV 导出/消费工具 |
| `scripts/` | 跨节点驱动与验证 gate（3/4-domain 异构、混合 ring、TP 抽象 worker） |
| `docs/` | 设计与证据文档（见下） |
| `graph-memory/` | 图数据库知识库（`blueprint.md` 为项目蓝图；任务/决策/证据/教训全部入库） |
| `reports/` | 原始实验数据（日志、logits、SHA-256 清单） |
| `harness/` | 计划/评审/问题的治理记录 |

关键文档：

- [`docs/HETERO_FEASIBILITY_EVIDENCE.md`](docs/HETERO_FEASIBILITY_EVIDENCE.md) —— 跨厂商精确 CP 的可复核证据（论文级）
- [`docs/PRODUCT_THESIS.md`](docs/PRODUCT_THESIS.md) —— 产品问题定义与愿景论证
- [`docs/SCALING_ARGUMENT.md`](docs/SCALING_ARGUMENT.md) —— context 长度与显存/网络/域数的 scaling 分析
- [`docs/HLPP_VS_HCP.md`](docs/HLPP_VS_HCP.md) —— 与 HLPP（层高边界编排）的定位边界
- [`docs/PLUGIN_ARCHITECTURE.md`](docs/PLUGIN_ARCHITECTURE.md) —— 可插拔域内后端架构
- [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md) —— 部署与运行

## 机器上的标准布局

white/pearl（`/home/stark/` 下）：

```
/home/stark/hetero-cp-ringattn   # 本仓（研究/驱动）
/home/stark/hcp-vllm-plugin      # 插件 clone（pip install -e 从这里装）
/home/stark/vllm                 # vLLM 源码树（pearl 上由 vllm-rocm-gfx1200 的补丁构建）
```

本地 Mac（`~/VSCodeProjects/` 下，三仓同级）：

```
~/VSCodeProjects/hetero-cp-ringattn   # 本仓（研究/驱动/知识库）
~/VSCodeProjects/hcp-vllm-plugin      # 插件（功能开发在这里做）
~/VSCodeProjects/vllm-rocm-gfx1200    # gfx1200 补丁/构建脚本
```
