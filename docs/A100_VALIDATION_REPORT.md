# A100 4× HCP Ring Attention 验证报告

> **日期**: 2026-06-13  
> **主机**: `ssh root@223.109.239.30 -p 25412`  
> **硬件**: 4× NVIDIA A100-SXM4-40GB, driver 610.43.02, compute capability 8.0, NVLink/SXM 互联  
> **软件**: Rust 1.96.0, libtorch 2.12.0+cu126, CUDA runtime/cuDNN/NCCL/cuSPARSELt/NVSHMEM 提取到 `third_party_libs/`  
> **模型**: Qwen/Qwen2.5-7B-Instruct BF16 (~15 GB, 28 layers, max_position_embeddings=32768)

---

## 1. 验证目标

在 A100 4 卡高端同构 GPU 平台上验证 HCP Ring Attention 的：
1. 单节点 7B 推理正确性
2. 4-domain 同节点分布式 7B 推理正确性
3. 长上下文（~4k / ~8k / ~16k / ~32k tokens）分布式推理稳定性
4. Serial vs Pipeline overlap 在 NVLink 高速互联下的实际收益

---

## 2. 环境准备

- libtorch 2.12.0+cu126 部署在 `/root/libtorch`
- 从 PyPI wheel 提取 NVIDIA 运行时库到项目 `third_party_libs/`，避免污染系统环境
- 环境脚本: `run_a100_env.sh`
- 运行脚本: `run_a100_single_node.sh`, `run_a100_4domain.sh`, `run_a100_long_context.sh`, `run_a100_tests.sh`, `run_a100_serial_vs_overlap.sh`

关键环境变量：

```bash
export LIBTORCH=/root/libtorch
export LD_LIBRARY_PATH="/root/libtorch/lib:/usr/local/cuda/lib64:<third_party_libs>${LD_LIBRARY_PATH}"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cuda:0}"
```

---

## 3. 验证结果

### 3.1 单节点 7B (cuda:0)

| 项目 | 值 |
|------|-----|
| Prompt | `"The quick brown fox jumps over the lazy dog."` |
| max_tokens | 10 |
| temperature | 0.0 |
| 输出 | ` The quick brown fox jumps over the lazy dog.` |
| 耗时 | ~57 s |
| 状态 | ✅ PASS, exit=0 |

### 3.2 4-domain 同节点分布式 7B

| Worker | GPU | Capacity |
|--------|-----|----------|
| domain 0 | cuda:0 | 23419 MB |
| domain 1 | cuda:1 | 24979 MB |
| domain 2 | cuda:2 | 23569 MB |
| domain 3 | cuda:3 | 24089 MB |

- Prompt / max_tokens / temperature 与单节点一致
- 输出：` The quick brown fox jumps over the lazy dog.`（与单节点一致）
- 耗时：~56 s
- 状态：✅ PASS, exit=0
- 28 layers ring attention 全通，3 rounds KV ring 正常转发

### 3.3 长上下文 4-domain 7B（Pipeline overlap 模式）

实际 prompt tokens 因 tokenizer round-trip 略低于目标值：

| 目标长度 | 实际 tokens | 输出 | 状态 |
|----------|-------------|------|------|
| ~4k | 3724 | `jumps over the lazy dog` | ✅ |
| ~8k | 7448 | `dog. The quick brown` | ✅ |
| ~16k | 14895 | `over the lazy dog.` | ✅ |
| ~32k | 29790 | `The quick brown fox jumps` | ✅ |

- 4/4 全部通过，exit=0
- ring attention 3 rounds 全通，workers 优雅退出

### 3.4 Serial vs Pipeline Overlap A/B（29.8k tokens, max_tokens=5）

| 模式 | 环境变量 | 耗时 | 输出 | 状态 |
|------|----------|------|------|------|
| Serial | `HCP_DISABLE_OVERLAP=1` | 325 s | `The quick brown fox jumps` | ✅ |
| Pipeline | `HCP_DISABLE_OVERLAP=0` | 317 s | `The quick brown fox jumps` | ✅ |

- Pipeline 仅快 ~2.5%（325s → 317s）
- 两者输出完全相同

**根因分析**：A100 SXM4 NVLink 互联带宽极高，单机四卡传输太快，network time 占比小；recv/compute 在 pipeline 下仍高达 ~80-200x，compute 远未成为瓶颈。

**结论**：对 A100 单机四卡这种高速互联环境，当前 7B/32k 规模下 pipeline overlap 收益有限；overlap 的真正价值预计在跨节点/慢网络或更大规模（>32k / 更多 domain）场景。

---

## 4. 工程纪律复验

- **1 GPU = 1 worker**：每个 worker 显式绑定 `HCP_TORCH_DEVICE=cuda:$domain`，避免多 worker 挤在同一 GPU 上 OOM。
- **显式 `--num-domains 4`**：coordinator CLI 默认 `num_domains=2`，多 domain 启动必须显式传递。
- **长上下文边界意识**：prompt + decode tokens 总数不能超过 `max_position_embeddings=32768`。

---

## 5. 与 A800 的对比

| 维度 | A800 (4× A800-SXM4-40GB) | A100 (4× A100-SXM4-40GB) |
|------|--------------------------|--------------------------|
| 单节点 7B | ✅ ~36 s | ✅ ~57 s |
| 4-domain 7B | ✅ | ✅ ~56 s |
| 长上下文 4-domain | ✅ 4k/8k/16k/32k | ✅ 3.7k/7.4k/14.9k/29.8k |
| Serial vs Pipeline @ 32k | 未做 | Pipeline 快 ~2.5% |
| 关键洞察 | 32k 时 compute 与 network 可比 | NVLink 太快，overlap 收益有限 |

> 注：A800 的 ~36s 单节点速度明显快于 A100 的 ~57s，可能与时钟/内存频率、驱动/CUDA 版本、或测量时的系统负载有关，非本文重点。

---

## 6. 未完成的探索

- **Qwen2.5-7B-Instruct-1M（1M context）**：下载因 xet-hub 不稳定暂停于 ~2.2 GB，已清理 partial cache。该模型含 `dual_chunk_attention_config`（chunk_size=262144），标准 attention 在 256k 内应可工作，262k+ 需额外支持。
- **跨节点 A100 验证**：当前仅在单机 4 卡完成。多机 A100 场景下的 overlap 收益有待验证。
- **更大 scale**：64k/128k 序列、8+ domain 规模未测试。

---

## 7. 相关文件

- `run_a100_env.sh`
- `run_a100_single_node.sh`
- `run_a100_4domain.sh`
- `run_a100_long_context.sh`
- `run_a100_tests.sh`
- `run_a100_serial_vs_overlap.sh`
- `monitor_a100_long_context.sh`
- `monitor_a100_ab.sh`
- `harness/infra.yaml`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`
