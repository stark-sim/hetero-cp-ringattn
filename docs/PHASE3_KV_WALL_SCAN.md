# Phase-3 A': KV Capacity-Wall Scan — HCP N=2 Ring vs vLLM PD (Qwen2.5-3B)

**日期**: 2026-08-14
**运行**: `routeb-p3-kvwall-20260814-211513`
**Harness**: `scripts/phase3_10_kv_wall_scan.sh` + `phase3_10_kv_wall_{hcp,pd}_driver.sh`
**定位**: 能力维对照（phase3-9 是开销维）；边界声明见
`docs/PHASE3_VLLM_PD_COMPARISON.md` §1 与 graph `decision-phase3-baseline-overhead-vs-capability-20260814`

---

## 1. 设计

- 模型 Qwen2.5-3B-Instruct（36 层 GQA 16/2，KV/token = 36KB，rope 上限 32768）。
  两侧同权重；HCP 侧 3B 支持先经数值对照验证（max_diff 0.46 vs 同管线 0.5B 对照 0.61，
  管线固有漂移，argmax 一致）+ N=2 ring smoke（36 跳自适应、生成连贯）。
- 负载：30,720-token 随机输入（留 2k 余量给输出，避开 rope 上限的 fail-closed 拒绝）、
  16 输出 tokens、单波 burst、mc ∈ {4, 8, 16, 32}。每请求 KV ≈ 1.09GB。
- HCP coordinator 用新 `--max-batch-size 16`（历史默认 4 会把并发上限伪装成 KV 上限）。
- 墙信号：vLLM 侧抓 `vllm:num_preemptions` 增量 + decode 启动日志的池大小；
  HCP 侧抓 coordinator 的 byte-level admission 行（accepted/rejected）。

## 2. 池与预算（地面真值）

| | KV 容量 | 折算 30k 会话并发上限 |
|---|---|---|
| vLLM PD decode（pearl 16GB） | 池 **7.88 GiB = 229,504 tokens**（vLLM 自报 "Maximum concurrency for 32768 tokens: **7.00x**"） | ~7 |
| vLLM PD prefill（white 24GB） | 池 441,968 tokens | ~14（非瓶颈） |
| HCP ring N=2 聚合 | admission 预算 **17.99GB（white）+ 10.42GB（pearl）≈ 28.4GB** | ~19（KV 字节账） |

理论容量比 ≈ **2.7x**——这正是 HCP 的存在理由（聚合显存 vs 单节点池）。

## 3. 结果

| mc | HCP 完成 | HCP p99 TTFT | vLLM PD 完成 | PD p99 TTFT | PD preemptions |
|---:|---:|---:|---:|---:|---:|
| 4 | 4/4 | 209.3s | 4/4 | 53.0s | 0 |
| 8 | 8/8 | 418.6s | 8/8 | 103.3s | 0 |
| 16 | **14/16** | 740.1s | 16/16 | 205.7s | 0 |
| 32 | **0/32** | — | 32/32 | 408.0s | 0 |

HCP admission：60 accepted / **0 rejected**（预算账面足够，全部放行）。

## 4. 解读（两个意外，都比预期更有信息量）

1. **vLLM 触墙行为是排队而非 preemption**。prefill 主导的巨大 prompt 负载下，v1 调度器
   在准入处排队（KV 块不够就不开始 prefill），preemption 计数全程为 0。p99 TTFT 随队列深度
   线性增长（53→103→206→408s），但 60/60 全部完成——更小的墙，但墙是"软"的，不致命。
2. **HCP 的墙比账面上的低，而且是"硬"的（真 bug）**。admission 只记 KV 字节
   （每域每请求 543MB），没有为 **prefill 激活工作区** 预留 headroom。mc=16 时 pearl 的
   KV 分配把 16GB 卡压到只剩 36MB 空闲，一次 172MB 的激活分配直接 OOM 崩溃
   （worker panic → 连接断开 → coordinator 级联失败，mc=32 全军覆没）。
   理论 2.7x 的容量优势**当前无法兑现**——可执行天花板先于 KV 账面天花板到来。

**结论**：能力维的真实对照结果是——vLLM PD 以 7x 并发 30k 会话的池 + 优雅排队提供了
"低而稳"的天花板；HCP 的聚合容量账（19x）是真实存在的设计优势，但 admission 必须
预留激活余量才能让 fail-closed 拒绝在正确位置发生，否则墙以崩溃形式出现。

## 5. 衍生的工程任务（graph 已建）

`task-phase3-admission-activation-headroom-20260814`：coordinator admission 预算
应减去激活工作区估计（按模型 config + 当前 in-flight prefill 数）或至少减固定安全余量
（~2GB/域），使 HCP 在 mc≈16 处干净地 status=rejected 而非 worker OOM。
验收：mc=32 重跑 = 16 完成 + 16 fail-closed 拒绝，无 worker panic。

## 6. 复现

```bash
bash scripts/phase3_10_kv_wall_scan.sh   # ~50min，顺序跑 HCP 侧 + PD 侧
```


---

## 7. 跟进：admission 修复验证 + 8k 变体（2026-08-14 晚）

**修复落地**：`--activation-reserve-mb`（默认 1536 MiB）在握手后从源头扣减
`worker_capacities`（commit `ec59df8`；首次尝试只扣了 HTTP ledger 而 per-request
准入路径从原始容量建账，重跑证实无效后改为源头扣减）。

**30k 重跑**（`routeb-p3-kvwall-20260814-232932`）：预算正确缩到 [15586, 8406] MiB，
但 mc=16 仍 OOM（15/16）、mc=32 全灭——**1.5GB 余量盖不住 30k prefill 的激活峰值**。
根因深一层：HCP ring 的 attention 每跳物化全量 [heads, shard, kv_len] 分数矩阵
（无 online/chunked softmax），30k 时 shard=15k 的分数矩阵 bf16 约 7.5GB/跳——
**激活工作区（非 KV 字节）才是 30k 下的真实约束**。这是 A 类工程债：
online/chunked attention 正是 ring-attention 论文的标准轮子，HCP 尚未实现。

**8k 变体**（`routeb-p3-kvwall-20260815-000103`，mc 8/16/32/64，max-batch-size 64）：

| mc | HCP 完成 | HCP p99 TTFT | HCP TPOT | vLLM PD 完成 | PD p99 TTFT | PD TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8/8 | 113.1s | — | 8/8 | 12.4s | — |
| 16 | 16/16 | 229.9s | — | 16/16 | 24.2s | — |
| 32 | 32/32 | 465.7s | — | 32/32 | 47.9s | — |
| 64 | **64/64** | 946.3s | 14.5s | 64/64 | 95.3s | 0.36s |

- **双方 120/120 全完成**。8k 下 HCP 的 KV 账面墙 ≈ 55 并发会话/域预算（pearl 8.36GB ÷
  150MB），但准入节奏（prefill 串行、短输出快速释放）使在册并发从未触顶——账本健康、零 OOM。
- vLLM 侧 64×8k=524k tokens 远超其 229.5k-token 池 → 排队加深（p99 48s→95s），依然零 preemption。
- HCP mc=64 的 TPOT 14.5s/token（vLLM 0.36s）——这是 A 类引擎差距在高并发下的真实体感。

**今日可辩护的能力结论**：
1. HCP N=2 聚合 KV 账面 28.4GB vs vLLM PD decode 池 7.88GiB（2.7x），机制真实且入账正确；
2. 8k 下 HCP 全量通过 mc≤64（聚合在册 KV 峰值超 vLLM 硬池上限）；
3. 30k 下 HCP 的兑现被激活工作区卡住——修复路径明确（online attention），属于生态轮子线。
