# Phase-3 A: HCP N=2 vs vLLM PD-Disaggregation — Controlled Interleaved Baseline

**日期**: 2026-08-14
**运行**: `routeb-p3-pd-baseline-20260814-194248`（10+10 交错，20/20 PASS）
**Harness**: `scripts/phase3_9_pd_interleaved_baseline.sh`（Mac 控制器）+ `scripts/phase3_9_pd_driver.sh`（white 上的 PD 驱动）
**原始产物**: `reports/routeb-p3-pd-baseline-20260814-194248/`（gitignored；comparison.json 为聚合真源）

---

## 1. 这份对照回答什么、不回答什么（边界声明，先于一切数字）

本实验分两个正交维度，**本报告只覆盖 (a)**：

- **(a) 引擎开销对照**：同负载（32-token 随机输入，KV/请求 ≈ 0.4MB，远离 KV 墙）下，
  HCP P2P ring 与 vLLM PD 分离的延迟/吞吐差异。量化的是 **HCP 执行器相对主流引擎的成熟度差距**。
- **(b) KV 容量能力对照**：长 context × 并发递增把两栈逼到各自的 KV 天花板。
  vLLM PD 把单请求全量 KV 压在 decode 单节点（天花板 = 单节点 VRAM 池，触墙 = preemption 重算）；
  HCP ring 把每请求 KV 分片到全环（天花板 = 聚合 VRAM，触墙 = byte-level admission fail-closed 拒绝）。
  **ring attention 的核心价值在 (b)**，由后续的 KV 墙扫描实验量化
  （graph: `task-phase3-kv-wall-capacity-20260814`，模型换 Qwen2.5-3B 让墙更低更可见）。

> **引用纪律**：(a) 的任何数字脱离本边界引用即为误读。
> "vLLM PD 快 N 倍"的完整说法是"在 32-token 小负载、双方均远离 KV 墙时，
> vLLM 成熟引擎比 HCP 研究级 eager 执行器快 N 倍"。
> 对应 owner 裁决：`decision-phase3-baseline-overhead-vs-capability-20260814`。

## 2. 对照形态（无 Ray，无额外调度面）

| | HCP N=2 | vLLM PD |
|---|---|---|
| white (RTX 4090, CUDA) | coordinator + worker0 + bench 客户端 | prefill 实例（默认编译）+ proxy + bench 客户端 |
| pearl (RX 9060 XT, ROCm gfx1200) | worker1 | decode 实例（`--enforce-eager`，见 §5） |
| 跨机 KV/数据面 | QUIC P2P ring，enp10s0↔enp8s0 直连 | NIXL/UCX（NixlConnector pull），同一条直连线 |
| 调度面 | coordinator（KVring 合作所必需） | 无（disagg_proxy_demo.py 纯转发，无调度） |
| 版本 | commit `bcf999d`（tch-rs eager，永远 eager） | 两端同 commit `3f99883d9` 源码构建；pearl 侧 UCX 1.19.1 `--with-rocm` + nixl 1.4.0 `wheel_variant=rocm` 源码链 |
| bench 客户端 | `~/venv-bench/bin/vllm` 0.27.1 | 同一个，同一参数 |

**Workload 阶梯**（两侧完全相同，client 同参）：
L1 = 8 prompts rate=1；L2 = 8 prompts rate=inf mc=2；L3 = 16 prompts rate=inf mc=4；
random input 32 / output 16 / range-ratio 0.5 / seed 42；`/v1/completions`。

**交错设计**：rep 内交替先手（奇数 rep HCP 先，偶数 rep PD 先）抵消时间窗漂移；
每 rep 全新栈（HCP：coordinator+2 workers 重启；PD：prefill+decode+proxy 重启）；
每侧 rep 前互杀对方栈并等待双 GPU VRAM 释放（防止 0.92 显存占用串扰）。

**网络门**（campaign 开头采集）：RTT avg 0.193ms；iperf3 goodput 2350 Mbps；0 重传。
与既有 HCP 20-rep 有线基线（0.17ms / 2350 / 0）等价。

**正确性门**：HCP rep 走 7a 断言（trace 24 跳、reserved==released、metrics failed=0）；
PD rep 走 bench 完整性（32/32 completed、指标正常）。每侧允许一次环境性重试（本次未触发）。

## 3. 结果（median；括号内 min-max 离散度；10 reps/侧）

| Level | 指标 | HCP N=2 | vLLM PD | 比值 (HCP/PD) |
|---|---|---:|---:|---:|
| L1 | TTFT ms | 333.58 (5.4%) | 49.26 (3.1%) | 6.77x |
| L1 | TPOT ms | 72.55 (4.5%) | 8.32 (5.2%) | 8.72x |
| L1 | 输出吞吐 tok/s | 15.05 (1.1%) | 17.20 (0.2%) | 0.88x |
| L2 | TTFT ms | 150.82 (10.0%) | 30.45 (9.5%) | 4.95x |
| L2 | TPOT ms | 57.42 (3.2%) | 8.59 (7.0%) | 6.69x |
| L2 | 输出吞吐 tok/s | 31.02 (3.0%) | 191.39 (6.6%) | 0.16x |
| L3 | TTFT ms | 282.59 (6.1%) | 33.55 (5.7%) | 8.42x |
| L3 | TPOT ms | 114.91 (3.3%) | 8.72 (5.3%) | 13.18x |
| L3 | 输出吞吐 tok/s | 30.30 (2.5%) | 355.51 (5.2%) | 0.09x |

## 4. 解读

1. **方向完全符合预期**：vLLM 引擎（连续批处理、融合 kernel、prefill 侧 cudagraph、
   成熟的调度循环）对 HCP 的研究级 eager tch-rs 执行器。这不是"ring 错了"，
   是"执行器还没工程化"。
2. **并发吞吐的差距（L2 0.16x / L3 0.09x）远大于单流差距**：vLLM 的连续批处理在
   mc≥2 时线性放大吞吐，HCP 当前每请求串行占用执行管线，基本没有批间复用。
   这是引擎差距的最大单项，也是未来优化清单的第一优先级候选。
3. **L1 输出吞吐基本持平（0.88x）**：rate=1 单流下 vLLM 的批处理优势无的放矢，
   差距只剩逐 token 的 kernel/调度开销。这给了我们一个干净的"每 token 固定开销"读数。
4. **HCP 侧离散度（1-10%）与 PD 侧（0.2-9.5%）同量级**：两套栈在交错时间窗内都是稳定的，
   比值不是噪声。（HCP 相对自身 20-rep 基线的中位数 334.78/72.23/114.96 完全落在既往 min-max 带内。）

## 5. 已知配置不对称（如实记录，不隐藏）

- pearl decode 强制 `--enforce-eager`（hipblaslt 在 gfx1200 上 FULL cudagraph 捕获报
  `operation not permitted when stream is capturing`）；white prefill 用默认编译（PIECEWISE graph）。
  这**压低**了 PD 侧数字——若 gfx1200 cudagraph 可用，PD 还会更快一点。HCP/tch-rs 永远 eager。
- proxy 的 TTFT 包含 prefill 调用 + KV 传输 + decode 首 token 的完整 honest 成本。
- PD 侧无 HCP 的 trace 断言门（vLLM 无此概念）；等价门是 bench 32/32 完整性。

## 6. 复现

```bash
# Mac 控制器（会驱动 white/pearl）：
REPS=10 bash scripts/phase3_9_pd_interleaved_baseline.sh
# 单 rep 试点：
REPS=1 bash scripts/phase3_9_pd_interleaved_baseline.sh
```

前置条件：pearl 的 ROCm NIXL 链（UCX 1.19.1 `--with-rocm` + nixl 1.4.0 `wheel_variant=rocm`
源码构建 + site-packages `rixl.py` 别名 shim）已在位；两端 `~/vllm` 同 commit。

## 7. 后续

- **KV 容量墙对照**（本报告的 (b) 维）：Qwen2.5-3B-Instruct（128k context，KV/token=36KB，
  墙更低更可见）× 并发递增扫描；抓 vLLM `vllm:num_preemptions` 与 HCP admission 拒绝。
  graph 任务：`task-phase3-kv-wall-capacity-20260814`。
- Dynamo 生态级对比：已裁决延后（三期再看生态层差异；Dynamo 是 vLLM 之上的编排层，
  底层同为 NIXL，且引入额外调度面，与"无必要时勿增实体"原则冲突）。
