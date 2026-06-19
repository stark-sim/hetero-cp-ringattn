# 1M Context 本地异构分布式推理报告

**white RTX 4090 CUDA + pearl RX 9060 XT HIP**

- **日期**：2026-06-19
- **运行编号**：v9
- **网络**：2.5G 有线直连（white `192.168.100.1`，pearl `192.168.100.2`）
- **报告路径**：`reports/1m-white-pearl-20260619/README.md`

---

## 1. 摘要

本次运行首次在本地消费级异构设备上通过 HCP Ring Attention 完成 **1,000,000 token context** 的端到端推理：

- **模型**：`Qwen2-0.5B-1M`（0.5B，24 layers，BF16，权重 ~1GB）
- **Prompt**：精确 1,000,000 tokens
- **分片**：white 750,000 tokens / pearl 250,000 tokens（3:1 不均等分片）
- **输出**：`generated:  the.`（5 decode tokens，greedy）
- **结果**：prefill 24/24 层全通，decode 5/5 tokens 全通，exit=0，workers 优雅退出

这是 HCP「单节点显存墙 + 高速 P2P 异构扩展」路径的重要可行性证明。

---

## 2. 硬件与环境

| 节点 | 平台 | GPU | 显存 | OS / 关键软件 |
|------|------|-----|------|---------------|
| white | CUDA | NVIDIA RTX 4090 | 24 GB | Linux, libtorch 2.11.0+cu130, ROCm/CUDA 环境 |
| pearl | HIP | AMD Radeon RX 9060 XT | 16 GB | Linux, libtorch 2.11.0+rocm7.2 |

网络基线（2026-06-16）：

| Path | RTT | 带宽 | 备注 |
|------|-----|------|------|
| white ↔ pearl（2.5G 有线直连） | ~0.1-0.2 ms | ~2.35 Gbps 双向对称 | 0 retransmits |
| Tailscale | 0.77-39 ms | 300 Mbps-1.8 Gbps 严重不对称 | 不使用 |

---

## 3. 关键配置

### 3.1 模型配置 patch

原 `Qwen2-0.5B-1M` config 的 `max_position_embeddings=1_000_000` 会在 decode 第一个新 token（position = 1,000,000）时触发 RoPE `index_select` 越界。运行前已 patch 为：

```json
"max_position_embeddings": 1048576
```

### 3.2 环境变量

**white（coordinator + worker 0）**

```bash
export HCP_KV_CHANNEL_BUFFER_SIZE=512
export HCP_QUIC_TIMEOUT_SECS=14400
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**pearl（worker 1）**

```bash
export HCP_KV_CHANNEL_BUFFER_SIZE=512
export HCP_QUIC_TIMEOUT_SECS=14400
export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so
```

### 3.3 命令示例

**white — coordinator**

```bash
./hcp-ringattn-rust \
  --distributed-role coordinator \
  --listen-addr 0.0.0.0:9000 \
  --worker-addrs 192.168.100.1:9100,192.168.100.2:9100 \
  --num-domains 2 \
  --capacity-aware \
  --chunk-sizes 750000,250000 \
  --model-path models/Qwen2-0.5B-1M \
  --prompt-file prompt_1m.txt \
  --max-tokens 5 --temperature 0.0
```

**white — worker 0**

```bash
HCP_TORCH_DEVICE=cuda:0 ./hcp-ringattn-rust \
  --distributed-role worker \
  --domain-id 0 \
  --listen-addr 0.0.0.0:9100 \
  --next-peer-addr 192.168.100.2:9100 \
  --coordinator-addr 192.168.100.1:9000 \
  --model-path models/Qwen2-0.5B-1M
```

**pearl — worker 1**

```bash
HCP_TORCH_DEVICE=cuda:0 LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so \
  ./hcp-ringattn-rust \
  --distributed-role worker \
  --domain-id 1 \
  --listen-addr 0.0.0.0:9100 \
  --next-peer-addr 192.168.100.1:9100 \
  --coordinator-addr 192.168.100.1:9000 \
  --model-path models/Qwen2-0.5B-1M
```

---

## 4. 结果

### 4.1 输出

```
generated:  the.
```

5 个 decode tokens 全部生成，无重复 / 乱码。

### 4.2 退出状态

- coordinator：exit=0
- worker 0 / worker 1：优雅退出，无 panic

### 4.3 进度确认

- prefill：24/24 layers 全通
- decode：5/5 tokens 全通

---

## 5. 性能数据

| 指标 | 数值 |
|------|-----|
| 总耗时 | ~2 小时 8 分钟（04:08 UTC → 06:16 UTC） |
| prefill | ~1 小时 52 分钟 |
| decode | ~16 分钟（~3 分钟 / token） |
| white 显存峰值 | **23,999 MB**（RTX 4090 24GB 几乎占满） |
| pearl 显存 | 未 OOM，16GB 内完成 |
| decode 阶段 GPU 利用率 | ~5%（memory-bound 特征） |

### 5.1 为什么 0.5B 会接近 24GB？

虽然 0.5B 权重仅 ~1GB，1M token KV cache 总量仅 ~12GB，但 white 峰值仍达到 23,999 MB，主要原因：

1. **KV cache 工作集**：prefill 过程中需要同时保留本地 KV 与接收到的 peer KV 中间状态。
2. **Activation 与中间 tensor**：24 层 attention / FFN 的中间结果。
3. **CUDA allocator fragmentation**：大块连续分配加剧显存碎片。
4. **expandable_segments:True**：帮助 allocator 扩展 segment，最终刚好 fit。

---

## 6. 攻关的关键问题

### 6.1 1M prefill 死锁

**现象**：prefill 到某个 layer 后 coordinator / worker 全部 hang 住。

**根因**：QUIC KV transport 的 `mpsc` channel buffer 默认为 64；1M context 下单层产生超过 64 个 micro block，send 端阻塞等待 recv 端消费，recv 端又依赖 send 端继续发送，形成死锁。

**解决**：引入环境变量 `HCP_KV_CHANNEL_BUFFER_SIZE=512`（默认值也调整为 512）。

### 6.2 QUIC 600s 超时

**现象**：大 context 传输中单步耗时超过 600s，触发 `recv_kv_block` timeout panic。

**解决**：引入环境变量 `HCP_QUIC_TIMEOUT_SECS=14400`（4 小时）。

### 6.3 Decode 位置越界

**现象**：prefill 24/24 成功后，decode 第一步即触发 `index_select` CUDA assert：

```
index out of bounds at position 1000000
```

**根因**：原 config `max_position_embeddings=1_000,000`，而 decode 第一个新 token 的 position 恰好为 1,000,000，超出 RoPE 缓存上标。

**解决**：patch config 为 `max_position_embeddings=1048576`。

### 6.4 pearl 16GB 碎片化 OOM

**现象**：2:1 split（white 666K / pearl 333K）在 layer 23/24 时 pearl 显存分配失败；3:2 split 同样失败。

**根因**：pearl 16GB 在容纳 250K+ tokens 的 KV 工作集 + 权重 + activation 后剩余空间不足，allocator 无法找到足够大的连续块。

**解决**：改为 **3:1 split**（white 750K / pearl 250K），显著降低 pearl 压力；white 24GB 刚好吸收更多 KV。

---

## 7. 观察与洞见

### 7.1 计算与网络瓶颈

- **prefill 阶段**：white 与 pearl 交替处于 compute / wait 状态。由于 pearl 分到的 chunk 更小但相对其算力仍重，且 ring attention Phase 2 中较小 domain 需要接收更多 remote micro block，pearl 常成为跨层瓶颈；white 在某些时段 GPU 利用率下降。
- **decode 阶段**：每 token ~3 分钟，GPU 利用率仅 ~5%，系统明显 **memory-bound**（1M KV cache 的读取和写入主导）。
- **2.5G 网络不是瓶颈**：~2h 总耗时主要由显存容量、allocator 行为和 memory-bound compute 决定。

### 7.2 不均等分片的价值

- 若严格按 1:1 分片，pearl 16GB 在 1M context 下无法承受。
- 3:1 capacity-aware 分片让两台设备的显存都得到充分利用，是成功关键。
- 这也说明：异构集群中 **按设备显存容量比例分片** 比均匀分片更接近可行性边界。

### 7.3 超长 context 的工程现实

- 即使 0.5B 模型，1M context 的 KV cache 也已占满 24GB+16GB 显存总和。
- 每 decode token ~3 分钟，对实时交互不可接受，但对批处理、长文档摘要等离线场景已有工程意义。
- 要进一步扩展到 7B 1M，必须引入更多 device / 权重分片 / KV 量化。

---

## 8. 复现建议

1. 两台机器配置 2.5G 有线直连，静态 IP 互通。
2. 两边都下载 `Qwen2-0.5B-1M` 并 patch `max_position_embeddings=1048576`。
3. 使用 `scripts/generate_long_prompt.py` 或 `gen_prompt` 生成精确 1M token prompt，并做 decode→encode round-trip 校验。
4. 设置环境变量（见 §3.2）。
5. 先启动 coordinator，再启动两个 worker。
6. 监控日志中的 `layer X/Y completed` 和显存峰值。
7. 如 pearl OOM，可继续增大 white 比例（如 4:1），但需 white 显存 headroom。

---

## 9. 结论

本次运行证明：在本地消费级异构硬件（RTX 4090 CUDA + RX 9060 XT HIP）和 2.5G 有线直连环境下，HCP Ring Attention 可以通过 capacity-aware 不均等分片完成 **1,000,000 token context** 的端到端推理。

这是 HCP 产品叙事「单节点显存墙 + 高速 P2P 异构扩展」的有力证据。下一步可评估：

- 引入第三台设备做 3-domain 1M，进一步降低单设备压力。
- 更大模型（7B）1M context 的可行性，需至少 4-8 台设备或权重分片。
- KV cache 量化 / 压缩，以缩短 decode 时间和降低显存占用。
