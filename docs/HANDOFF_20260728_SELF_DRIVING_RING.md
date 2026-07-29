# HCP Ring Attention — Session Handoff (2026-07-28)

> **状态（2026-07-30）：历史快照，不能作为当前计划。**
> 其中 plugin Task E、14 Task 生产化路线等后续已被修订；当前以 graph-memory 中的“核心优先、小步验证”决策为准。

## 接手第一步（强制）

按项目 `AGENTS.md` 的 Graph Memory Protocol 恢复上下文，**不要跳过**：

1. 读 `graph-memory/RULES.md`、`blueprint.md`、`active.md`、`progress.md`
2. 重点节点（`sqlite3 graph-memory/graph.db` 查询）：
   - `decision-self-driving-ring-20260728` — **当前冻结的计划**，含完整动机剖析六问 + 牺牲四问 + 排序 + 风险登记。下一个 session 的核心任务来源。
   - `decision-true-memsplit-audit-20260727` — 显存切分审计裁定（A/B/C 修复已完成）
   - `decision-no-star-transport-20260727` — 星形传输禁令（N>3 部分可达网络）
   - `blueprint-two-phase-ring-20260727` — 两阶段统一 ring 架构（prefill 传 KV、decode 传 Q+LSE）

## 当前状态：自驱动环计划已冻结，待开工

**计划（已写入 graph.db，待用户最终确认后按序实施）**：

- **任务E**（`task-plugin-successor-seeded-ring-20260728`，先行）：plugin 线 successor-seeded 优化——ring_decode_step 种子改 q-only，owner 收包后最后归并，每层 N 跳→N-1 跳（72→48/token，N=3）。小改动，验证 owner-最后归并数学，为任务D 的 finisher 语义预演。插件仓：`~/VSCodeProjects/hcp-vllm-plugin`。
- **任务D**（`task-rust-self-driving-ring-20260728`，主体）：Rust 线自驱动环 decode。子步 D1 单包轮转 attention（ring.rs）→ D2 finisher 就地续层（model.rs decode 期事件循环化，**最大工程风险**）→ D3 采样轮转 + coordinator 退位 → D4 验证阶梯（mock→MPS 双节点→跨节点 CUDA+HIP）。
- **关键架构领悟**：decode 期 worker 没有 forward 调用栈，是纯事件循环（收包→算 partial→完整则就地 MLP 续发/否则转发），实现比现状更简单。
- **跳数账**：每 token 下限 24×(N-1)（层依赖决定）；现 plugin N 跳/层 → 自驱动 N-1 跳/层。

## 已完成（本 session，全部有验证 + Reviewer APPROVE）

显存切分审计的 A/B/C 修复，6/8 harness issues resolved（详见 `harness/issues/resolved/` 与 graph.db progress 层）：

- **任务A**：vLLM prefill 逐层流式 staging（峰值 staged 层数 48→4~6），star decode 删除（ring-only），邻接 prefill。插件 `6eacec3`/`fb4a2b0`，三机 `p2p3n-000004` PASS。
- **任务B**：decode ≥2 并发按请求全隔离（metadata 跳池门 + req tag + (req,layer) growth）。插件 `cc733dd`+`453fa1f`，`conc2b` PASS。
- **任务C**：Rust decode Q-ring（Q+LSE 累积器环 + 增长 p%N 零传输分片）。主仓 `c4a3e7f`，cargo 68/68 + MPS A/B + 跨节点 CUDA+HIP PASS。
- 主仓 HEAD `7cf6954`，插件 HEAD `7c059e1` 之后的 `453fa1f`，均已推送。

## 环境速查（别再抓瞎）

- **三节点**（`~/.agents/inventory.yaml` + `harness/infra.yaml`）：white `100.118.253.68`（4090 24G CUDA，`/home/stark/miniconda3/envs/vllm-v1`）、pearl `100.111.242.55`（9060XT 16G ROCm，`vllm-rocm` env + LD_LIBRARY_PATH wrapper，兼 k8s-master 小心）、laptop `100.96.154.1`（4060 8G CUDA，`vllm-v1`）。
- **验证脚本**：plugin 仓 `validate_ring_decode_p2p.py`（`--mode all --concurrent N`，默认已开 decode-ring）；主仓 `scripts/run_3node_decode_p2p.sh`（三机驱动）、`scripts/run_distributed_2node_smoke.sh`（Rust 本地 MPS）、`scripts/run_cross_node_2domain_cuda_hip.sh`（Rust 跨节点）。
- **远程纪律**：代码只走 git 同步（禁止直接改远程）；非交互 SSH 需 `PATH=<conda env>/bin:$PATH`（否则 flashinfer JIT 找不到 ninja）；Rust 远程构建需 `PATH=/home/stark/.cargo/bin:$PATH`。
- **仓库布局**：主仓 `~/VSCodeProjects/hetero-cp-ringattn`（调度核心/驱动/知识库）、`~/VSCodeProjects/hcp-vllm-plugin`（插件）、`~/VSCodeProjects/vllm`（上游参考树，只读）。

## 治理与坑（本 session 教训，graph.db lesson 层有全文）

- **禁止自写自测自报 OK**：新验证必须过 Reviewer subagent 独立复核。
- 探针族四类缺陷：通道不可见（插件 logger 不进捕获日志→用 print/HTTP 请求行）、投影塌缩（slots 集合被复用→直接测 skip 计数）、通道不匹配（探针读单例 vs 直接构造）、归档不自足（verdict 依赖的证据要门控）。
- **N=1→N≥2 竞态相变点**：单实例 PASS 后把代码里所有"隐含的 1"列出来重审（键空间/启发式/文件路径/全局缓冲）。
- 环拓扑顺序启动：init 期任何阻塞等 peer/等运行后数据都会死锁——等待一律放工作线程。

## 建议下个 session 携带的 skill

- `graph-memory`（协议强制，读写记忆）
- `infrastructure-inventory`（涉及远程节点时先读 `~/.agents/inventory.yaml`）
- `harness-governance`（危险操作/issue 管理；active issues 仅剩 ISSUE-006，待任务D 验证后 resolve）
- `verification-before-completion` / Reviewer subagent（任何"完成"声明前）

## 待办一句话

用户确认冻结的计划后：**先任务E（plugin successor-seeded，小步），再任务D（Rust 自驱动环，主体）**；ISSUE-006 随任务D 验证 resolve；遗留评估项：Rust 全节点冗余 logits 计算、coordinator exit code 落盘。
