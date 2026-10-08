# HCP 异构可行性实验证据（NPU + CUDA + HIP + MPS）

> 目的：为学术论文提供可复核的异构 context parallelism（CP）可行性证据。
> 所有原始数据（日志、logits 二进制、prompt 文本）保存在 `reports/` 对应目录，
> 关键工件附 SHA-256。记录日期：2026-10-05/06。

## 1. 结论摘要

在 Qwen2-0.5B（fp32）上，**同一模型在 Ascend NPU / NVIDIA CUDA / AMD HIP 三家
硬件栈上的单域前向数值一致到 float32 噪声地板（max|Δlogit| ≈ 5e-5）**；
在此之上，**跨三厂商的 3-domain 精确 CP ring 的 decode logits 与各单域 golden
的差异同样处于该噪声地板（≤ 6.3e-5），全部 decode 步 argmax 一致**。
即：HCP 协议层（QUIC 控制面 + KV 环传输）可以承载跨厂商精确 CP 推理。

2026-10-06 起扩展为**四后端全覆盖**：加入 Apple Silicon MPS 后的 4-domain
ring（NPU+CUDA+HIP+MPS）对全部四个单域 golden 的 max|Δ| ≤ 4.2e-5，**低于
golden 两两互差（≤ 4.9e-5）**，ring 引入的额外误差在跨后端噪声下不可分辨
（§4.6）。Rust↔Python 混合 ring 亦已打通（§6）。

## 2. 实验环境

### 2.1 硬件与软件栈

| 角色 | 节点 | 硬件 | 计算栈 | transformers | 上报 capacity |
|---|---|---|---|---|---|
| domain 0 | AtomGit CANN 容器 | Ascend910_9362（64GB HBM），driver 25.5.5，CANN 9.0.0 | torch 2.7.1+cpu + torch_npu 2.7.1.post4，Python 3.11.4 | 4.45.2 | 60354 MB |
| domain 1 | white | RTX 4090 24GB | torch 2.13.0+cu130，Python 3.12 (venv-bench) | 5.15.0 | 21134 MB |
| domain 2 | pearl | RX 9060 XT 16GB（RDNA4, gfx1200 target） | torch 2.13.0a0+rocm7.13.0a20260416，Python 3.11 (conda vllm-rocm) | 5.12.1 | 14060 MB |
| domain 3 | Mac（兼 coordinator） | Apple Silicon MPS（16GB 统一内存） | torch 2.11.0，Python 3.12 (miniconda) | 4.57.6 | 12124 MB（`torch.mps.recommended_max_memory`） |
| coordinator | Mac | Apple Silicon（仅 tokenizer/调度/采样，无模型计算） | Rust 二进制（tch/libtorch 仅用于 tokenizer 路径） | — | — |

coordinator 按 capacity-aware 切分：各 worker 上报空闲显存（握手 16 字节包），
扣除 1536 MiB 激活预留后按比例分配 prompt chunk。实测预留后容量：
[58818, 19598, 12524] MiB（NPU/CUDA/HIP）。

### 2.2 网络拓扑（实测）

| 链路 | 路径 | RTT | 带宽（100MB HTTP 实测） |
|---|---|---|---|
| container ↔ Mac / white / pearl | Tailscale，自建 DERP 中继（derp.starksim.com, szx），NAT 打洞失败 | ~120-135 ms | 容器→Mac 581 KB/s，Mac→容器 520 KB/s |
| white ↔ pearl | 2.5GbE 有线 LAN（192.168.100.0/24） | <1 ms | ~2.5 Gbps 链路 |
| pearl ↔ Mac | Tailscale（不同 LAN 子网：Mac 192.168.8.0/24） | min 5.8 / avg 21.7 ms（ping ×5 实测） | 未测吞吐 |
| Mac ↔ white | Tailscale | min 41.2 / avg 97.9 ms（ping ×5 实测） | 未测吞吐 |
| 容器管理面 | AtomGit VS Code 插件 WebSocket 隧道 | — | scp ~1.2 MB/s |

3-domain ring 边：container→white（DERP）、white→pearl（LAN）、pearl→container
（DERP）。4-domain ring 边：container→white（DERP）、white→pearl（LAN）、
pearl→mac（tailnet）、mac→container（DERP）。
容器无 LAN 可达性，任意排序下恰好两条 ring 边跨 DERP。

### 2.3 代码版本

| 组件 | commit |
|---|---|
| NPU 支持 + capacity 上报 | `ee7bd96` |
| DynamicCache 版本兼容（4.44-4.52 `key_cache`） | `2bf893f` |
| N-domain ring 泛化（dial+accept 双工） | `ce48033` |
| RoPE 全局 position_ids 修复（bincode Option<u8>） | `a03b93a` |
| 非周期 prompt 生成器 + logits 比对工具 | `6120747` |
| **精确 CP prefill（逐层 KV ring）** | `07c6f27` |

## 3. 实验设计

### 3.1 采样与对照

- 采样：greedy argmax（Rust 分布式路径确定性采样，`tch_backend.rs:1293`）。
- prompt：`scripts/gen_natural_prompt.py` 生成的**非周期**词序列（seed=20261005，
  自检无 < N/4 的周期）。非周期是关键：周期 prompt 会因 argmax margin 巨大而
  掩盖位置编码类 bug（见 §4.2 的 falsification 记录）。
- 对照：每个 backend 各跑一次 1-domain golden（同 prompt、同 tokenizer、
  同 coordinator），ring 输出与各 golden 做文本级 + logits 级双重比对。
- logits 采集：coordinator `--export-logits-dir`，逐步 f32 LE 全词表向量
  （vocab=151936）。首 token 的 logits 来自 ring 中最后一个 domain（持有 prompt
  末尾 token，HIP/pearl），后续 decode 步来自 domain 0（NPU）——采样管线的
  每步来源均在 §4 的比对中按此对应。

### 3.2 精确 CP 算法（Python worker，`07c6f27`）

与 Rust Q-ring 同算法：每层 (a) 本地 chunk 计算 Q/K/V（RoPE 用全局位置）；
(b) 该层 KV 绕环 N-1 轮（发送本地块、接收并转发前序块）；(c) 按全局位置排序
拼接全量 KV，以"远端全可见 + 本地因果"掩码做精确注意力。各 worker 缓存全量
精确 KV，decode 在完整 cache 上进行。Python 侧以"全量 KV + 精确掩码"替代字面
online softmax（worker 本就持有全量 KV，二者数学等价）。

KV 帧格式（Rust/Python 互通）：`[4B BE meta_len][JSON meta][f32 LE K][f32 LE V]`。

## 4. 实验结果

### 4.1 R1：跨厂商单域数值地板

同一 prompt、同一权重 fp32，三个 backend 各自 1-domain golden 逐步比对
（16 个 decode 步）：

| prompt 长度 | golden-NPU vs golden-CUDA max|Δ| | golden-NPU vs golden-HIP max|Δ| | argmax 一致 |
|---|---|---|---|---|
| 64 | 3.43e-05 | 4.86e-05 | 16/16 |
| 512 | — | 5.36e-05 | 16/16 |
| 2050 | — | 5.05e-05 | 16/16 |

### 4.2 R2：RoPE 位置 bug 的 falsification 闭环（修复前证据）

Critic 评审发现 Python worker prefill 忽略全局 position_ids（非 0 domain 的
RoPE 相位错误）。周期 prompt 下该 bug 完全不可见（此前 gate 全 PASS）。

| 实验 | prompt | ring 状态 | ring vs golden max|Δ| | argmax |
|---|---|---|---|---|---|
| E0 `231505` | 自然 64 | bug 版 | **13.61** | **step 1 起发散** |
| E1 `232431` | 自然 64（同 E0，逐字节相同 prompt：SHA 6e2344ce…） | RoPE 修复，仍为 KV-swap 近似 | 4.06 | 16/16（min margin 0.0295，侥幸） |
| E1c `000805` | 同上 | **精确 CP** | **4.05e-05** | 16/16（margin 与 golden 逐位一致） |

### 4.3 R3：KV-swap 近似的机制确认

"chunk 本地 prefill + 事后 KV 拼接"的系统性误差（~4.0）来自 chunk i>0 第 2 层
起的 K/V 缺少跨 chunk 注意力上下文。该机制在 Mac CPU 单进程模拟中独立复现：
max|Δ| = 3.822，与真实 3-domain ring 的 3.819 四位有效数字一致
（`scripts/test_ring_exact_prefill_local.py` 的前身实验）。精确 CP 版在同模拟
下所有 domain 最后位置 max|Δ| ≤ 2.1e-5。

### 4.4 R4：精确 CP 3-domain 异构规模矩阵（最终 gate）

| Run | prompt tokens | decode 步 | ring vs golden-NPU max|Δ| | ring vs golden-HIP max|Δ| | argmax | 文本 |
|---|---|---|---|---|---|---|---|
| E1c `000805` | 64 | 16 | 4.05e-05 | 5.63e-05 | 16/16 | 一致 |
| E2-512 `001100` | 512 | 16 | 3.91e-05 | 6.29e-05 | 16/16 | 一致 |
| E2-2048 `001343` | 2050 | 16 | 3.72e-05 | 4.88e-05 | 16/16 | 16/16 |

ring 与 golden 的差异与"同一 backend 单域重复计算 + 跨后端"噪声地板同量级，
即 **ring 引入的额外误差在 float32 噪声下不可分辨**。

### 4.5 R0：NPU 双芯 2-domain（路线 A，前置验证）

`reports/npu-2domain-atomgit-20261005-220300`：容器内 npu:0 + npu:1，
coordinator 在 Mac（tailscale DERP 控制面），ring 输出与单芯 golden 逐 token
一致。首次打通 2-domain Rust↔Python KV 环。

### 4.6 R5：4-domain 四后端全覆盖（NPU + CUDA + HIP + MPS）

`reports/pyring4-npu-cuda-hip-mps-20261006-022629`（64 tokens，decode 16 步，
prompt 与 §4 各 run 逐字节相同：SHA `6e2344ce…`）。拓扑：container(NPU, d0) →
white(CUDA, d1) → pearl(HIP, d2) → mac(MPS, d3) → container；capacity 上报
[60353, 21134, 14060, 12124] MB，不均分切分。

logits 级比对（ring 对全部四个单域 golden，以及 golden 两两互差作为噪声地板）：

| 比对 | max|Δ| | mean|Δ| |
|---|---|---|
| ring vs golden-MPS（主 gate，末域后端） | 3.20e-05 | 4.24e-06 |
| ring vs golden-NPU | 3.34e-05 | 4.31e-06 |
| ring vs golden-CUDA | 3.39e-05 | 4.08e-06 |
| ring vs golden-HIP | 4.20e-05 | 5.72e-06 |
| golden 两两互差（6 对，噪声地板） | 2.55e-05 ~ 4.86e-05 | — |

**ring 对任一 golden 的偏差 ≤ golden 两两互差**：4-domain 精确 CP 引入的额外
误差在跨后端 float32 噪声下不可分辨。文本输出与 MPS golden 逐 token 一致，
四个后端 golden 亦全部一致（argmax 跨后端确定性对该 prompt 成立）。

附带改动：worker `capacity_mb` 增加 MPS 分支
（`torch.mps.recommended_max_memory()`，此前落默认 4096 MB 占位）。

### 4.7 R6：抽象 worker——双芯 NPU 经 HCCL TP 聚合为单逻辑 domain

`python/hcp_tp_engine.py`（N1，`f9f7890`）+ `python/hcp_tp_worker.py`（N2，
`6ed8bc7`）：Megatron 式 head 并行 TP over torch.distributed（HCCL；gloo/NCCL
仅换 backend 字符串）。Qwen2-0.5B TP=2：每芯 7 Q heads + 1 KV head；q/k/v 按
head 切行、o_proj/down_proj 切列 + all_reduce(SUM)（每层 2 次 collective）；
embed/norm 复制、lm_head 仅 rank0。rank 0 兼跑 HCP 协议面（QUIC 控制面 + ring
数据面，wire 格式零变化）；每 rank 只缓存自己 head 切片的全序列 KV
（**每芯 KV 显存减半**，抽象 worker 的核心收益）。capacity 按两芯合计上报
（121576 MB）。

| Gate | 拓扑 | 结果 |
|---|---|---|
| N1 单元 | Mac gloo/CPU TP=2 vs HF 参考 | argmax 128/128，max\|Δ\|=5.26e-4（fp32 求和重结合地板；world_size=1 与 HF bit 级一致） |
| N1 单元 | 容器 HCCL npu:0+npu:1 vs 单芯 | argmax 128/128，max\|Δ\|=1.46e-4 |
| N2 本地 | TP 抽象 domain + 普通 CPU domain 2-domain loopback | TEXT MATCH，worst max\|Δ\|=6.48e-05 |
| N2 远程 | `pyring3-abstract-tp-20261006-031155`：抽象(NPU×2, d0) + white(CUDA, d1) + pearl(HIP, d2) | 文本逐 token 一致；ring vs 三 golden max\|Δ\| = 6.1e-5 ~ 7.9e-5 |

注：含 TP 域的 ring 的 logits 偏差（≤ 7.9e-5）略高于纯单芯 3/4-domain ring
（≤ 6.3e-5），增量来自 TP all_reduce 的 fp32 求和重结合（N1 独立测得
~1.5e-4），与跨后端噪声同量级；argmax 全部一致。decode 的 Q-ring packet
模式在 tp_size>1 下硬报错（护栏；decode 生态位留给 PD 分离线）。

工件 SHA-256（pyring3-abstract-tp-20261006-031155）：prompt.txt
`d9adca8a9ae5f67c`、logits_ring/logits_1.bin `89f7e991919fe8e7`、
coordinator_ring.log `a5241a1b47c2c837`。复现：
`bash scripts/run_pyring_3domain_abstract_tp.sh`（本地单测
`scripts/test_tp_ring_local.sh`、TP 单元 `scripts/test_tp_dual_chip.py`）。

### 4.8 R7：抽象 worker 泛化到 NCCL 跨机异构卡（laptop 4060 + white 4090）

`reports/pyring4-nccl-abstract-20261006-074637`：d0 = 抽象 worker
（laptop RTX 4060 8GB rank0 + white RTX 4090 rank1，**NCCL over tailscale**），
d1 = 容器 NPU、d2 = pearl HIP、d3 = Mac MPS 的 4-domain ring（64 tokens）。

- **主 gate PASS**：ring 与 MPS golden 逐 token 一致；四后端 golden 亦全部一致。
- logits：ring vs 四 golden max|Δ| = 3.53e-5 ~ 7.41e-5，argmax 全步一致
  （min_margin 0.0598）。
- **异构 TP capacity 实证**：all_gather 求和上报 [6414, 21722] = 28136 MB
  （4060 实测 6.4GB + 4090 实测 21.7GB；旧的 rank0×2 外推会错报 12828MB）。
- **NCCL 版本教训**：2.28.9 ↔ 2.29.7 bootstrap wire 结构体差 4 字节，首个
  all_reduce 静默挂死（须 `NCCL_DEBUG_FILE` 取证）；对齐到双端 2.29.7 后
  smoke 6 秒通过。跨机 NCCL 组队前必须对齐小版本。
- **collective 开销**（`HCP_TP_TIMING=1`，rank0 视角）：prefill 每层
  all_gather 0.32ms + remote bcast 1.31ms + 2×all_reduce 1.40ms/call，
  24 层合计 106.5ms；decode 每 token collective 5-7ms。**测量口径注记**：
  该数字由 issue #23 修正前的打点测得（计时区间含部分层内计算且未做
  stream 同步），是**上界**；修正后本地 gloo 对照 all_reduce 从
  6.25ms/call 降到 0.66ms/call，远程干净数值待重测。低延迟因本次
  laptop↔white tailscale 走了 direct LAN（~5ms）；跨地域 DERP/WiFi
  （25-70ms RTT）场景未实测，按 RTT 预估 decode 每 token +100-300ms——
  跨机 TP 是延迟敏感路径，WAN 下不划算（符合"TP 属同机房"的业界共识）。
- 坑：d3→d0 边 Mac→laptop 的 LAN 直连 UDP "No route to host"（WiFi 客户端
  隔离），脚本改用 tailnet IP 后通过。

工件 SHA-256：prompt.txt `d9adca8a9ae5f67c`（与 R6 相同）、
logits_ring/logits_1.bin `41f1e00b4d92e412`、coordinator_ring.log
`4471fb4228339cdb`。复现：`bash scripts/run_pyring_4domain_nccl_abstract.sh`
（NCCL 对齐 smoke：`scripts/nccl_smoke_2node.py`）。

### 4.9 R8：双逻辑 worker 7B CP（Qwen2.5-7B bf16，`71585a6`）

`reports/pyring2-abstract7b-20261008-134844`：d0 = 容器双芯 910（HCCL TP=2
抽象 worker），d1 = white 4090 单卡，2-domain ring，512 tokens / 16 decode 步。
模型体量从 0.5B 升到 7B（bf16，~15GB——white 24GB 是卡点位，fp32 28GB 超界）。

- **主 gate PASS**：ring 与末域（white CUDA）golden 逐 token 一致；
  golden-npu / golden-cuda / ring 三者文本全同。
- logits（bf16 容差，ulp≈0.0625@17）：ring vs golden-npu max|Δ|=0.3125，
  ring vs golden-cuda max|Δ|=0.34375，argmax 全步一致（min_margin 0.125）。
  隔离证据：NPU 单进程 bf16 引擎 vs HF 参考 bit 级一致（max|Δ|=0.0），
  TP=2 偏差纯来自 bf16 求和重结合。
- **抽象 worker capacity 聚合实证**：d0 上报 per-rank [53854, 55145] =
  108999 MB（all_gather 求和），d1 = 7956 MB。
- **诚实注记**：本轮 chunk 分配是均分（vanilla ring 路径不做 capacity
  加权分片；capacity 加权在 continuation/admission 路径）。「聚合显存 →
  不均分 chunk」的完整展示需走 continuation 路径或补分片策略。
- 协议演进：wire 帧新增 k_dtype/v_dtype 标签（payload 仍 f32 LE，接收端
  cast）；无标签帧按 f32 处理，fp32 路径字节级零变化（回归逐位一致）。
- 模型分发：modelscope（容器 45MB/s，~6min）；hf-mirror 不代理 Xet CAS，
  该路径对 huggingface_hub≥1.24 已废。

工件 SHA-256：prompt.txt `610fa3dd86634892`、coordinator_ring.log
`5fc99c49fd733800`。复现：`bash scripts/run_pyring_2domain_abstract7b.sh`。

## 5. 工件完整性与复现

### 5.1 关键工件 SHA-256

| run | 文件 | SHA-256（前 16 位） |
|---|---|---|
| E0 | prompt.txt | 6e2344ce408538d8 |
| E0 | logits_ring/logits_1.bin | 9b32ad619ed2e39d |
| E1 | prompt.txt | 6e2344ce408538d8（与 E0 相同） |
| E1 | logits_ring/logits_1.bin | 70eb7f3268b20c9f |
| E1c | prompt.txt | 6e2344ce408538d8 |
| E1c | logits_ring/logits_1.bin | e1e302117d862459 |
| E2-512 | prompt.txt | a0a537eaa640c69b |
| E2-512 | logits_ring/logits_1.bin | c60adce84cf07565 |
| E2-2048 | prompt.txt | d53d069df9a8c10d |
| E2-2048 | logits_ring/logits_1.bin | 062904182abd20a6 |
| R5-4domain `022629` | prompt.txt | 6e2344ce408538d8（与 E0/E1 相同） |
| R5-4domain | logits_ring/logits_1.bin | 07bd07c2ae93de2e |
| R5-4domain | logits_golden-mps/logits_1.bin | 2f800e9b3acf940d |
| R5-4domain | coordinator_ring.log | e372cca4c145bff6 |

### 5.2 复现命令

```bash
# 主 gate（3 goldens + 3-domain ring，自然 prompt，logits 导出）
SEQ_LEN=2048 MAX_NEW_TOKENS=16 bash scripts/run_pyring_3domain_npu_cuda_hip.sh

# 4-domain 四后端 gate（4 goldens + 4-domain ring，含 Mac MPS）
bash scripts/run_pyring_4domain_npu_cuda_hip_mps.sh

# logits 级比对（ring vs 各 golden）
python3 scripts/compare_logits_dir.py reports/<run>/logits_ring reports/<run>/logits_golden-npu

# 本地精确性单元验证（无远程依赖，Mac CPU）
python3.12 scripts/test_ring_exact_prefill_local.py

# NPU 双芯 2-domain 回归
bash scripts/run_npu_2domain_atomgit.sh
```

## 6. 混合后端 ring（Rust↔Python worker 混编，`af864ed`）

Python worker 以 `--ring-mode rust` 加入 Rust per-layer-stream 环（每条流 1 dummy
字节、帧按 layer_idx 路由、micro-block 重组、GQA 线上扩展、wire dtype 标签
bfloat16），decode 阶段完整参与 Q-ring RingPacket 在线 softmax 合并（owned shard
= 自己的 prefill chunk + growth 位置 p % N == domain_id + 当前 token）。

| Run | 拓扑 | 结果 |
|---|---|---|
| `mixed-local-20261006-011932` | Mac loopback：Rust(CPU, d0) + Python(CPU, d1) | 与 golden 逐 token 一致，argmax 16/16 |
| `mixed3-npuPy-cudaRs-hipRs-20261006-013939` | Python(NPU, d0) + Rust(CUDA 4090, d1) + Rust(HIP 9060XT, d2) | 与 NPU golden 逐 token 一致，argmax 16/16，max\|Δlogit\|=0.68 |

数值域说明：Rust worker 按 config `torch_dtype=bfloat16` 运行，Python worker 跑
fp32；ring decode 的 packet 合并发生在 Rust 侧 bf16 域，因此混合 ring vs fp32
golden 的 logits 差异为 bf16 量级（0.2~0.7），argmax 全部一致。混合 ring 的
正确性判据是 argmax/文本级一致性 + ring 不挂起，logits 地板级等价主张仅适用于
同精度的全 Python ring（§4）。

附带修复（`38f5cf8`）：Python bincode 命令/响应枚举标签与当前 Rust 源码对齐
（SyncGlobalSeqLen=5 / ReleaseRequest=6 / Shutdown=9 / Error=4）。旧标签只在
旧编译二进制下碰巧正确——这是一次"重建即坏"的隐性协议漂移，已回归验证
（route A + 全 Python 3-domain 在新二进制下双双 PASS）。

工件 SHA-256（mixed3-20261006-013939）：prompt.txt `6e2344ce408538d8`、
logits_ring/logits_1.bin `78f3b663dd91c2ed`、coordinator_ring.log `eef5095d062769`。

## 7. 已知限制

1. Python worker 仅支持 vanilla 连续分片；striped/zigzag 会硬报错（护栏，
   见 `a03b93a`）。
2. Python 侧精确 CP 用"全量 KV + 精确掩码"，显存模型是全量 KV 每节点一份
   （与 KV-swap 相同）；显存分片主张仍由 Rust Q-ring 线承载。
3. 跨 DERP 链路的 ring 边带宽 ~4.5 Mbps，2048-token 规模可用；更大规模的
   性能画像（非正确性）未在本证据范围内。
4. torch_npu 2.7.1 的 sdpa 不支持 `enable_gqa`，GQA 以 repeat_interleave
   展开（语义相同，内存略增）。
5. AtomGit 容器代码同步走 git commit + rsync（容器无 git remote 凭据），
   报告中所有 commit 均以 GitHub main 为准。
