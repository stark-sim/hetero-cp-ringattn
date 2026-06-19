# 1M Context + 2.5G Wired 本地异构验证计划

> **目标**：在 white（RTX 4090 CUDA）和 pearl（RX 9060 XT HIP）两台本地机器上，通过 **2.5G 有线直连** 验证 HCP Ring Attention 在 1M context 级别的可行性。
> **战略意义**：证明当单节点显存墙不可避免时，异构设备 + 高速 P2P 互联是通向百万 token 的差异化路径。
> **优先级**：这是当前最高优先级；vLLM CUDA E2E、量化、多请求并发等工程完善项后置。
> **更新**：[2026-06-16] 原计划使用 Thunderbolt 5/4，但两台主板无雷雳口；改为 2.5G 网口直连。网络基线已完成，结果见 §2.4。

---

## 1. 研究问题

我们要回答三个问题：

1. **可行性**：在本地两台异构机器（CUDA + HIP）上，能否用 HCP Ring Attention 协同完成 1M context 推理？
2. **高速互联价值**：2.5G 有线直连提供的低延迟、高带宽 P2P 链路，能否显著改善当前 WiFi / Tailscale 下的网络瓶颈？
3. **未来外推性**：该验证能否支撑 HCP「通过增加 domain / 高速互联来无限外推 context」的产品叙事？

---

## 2. 关键可行性约束

### 2.1 显存墙：7B 模型 1M context 无法在两台机器上完成

以 **Qwen2.5-7B-Instruct** 为例（BF16 权重 ~14GB）：

| 参数 | 值 |
|------|-----|
| num_layers | 28 |
| num_kv_heads | 4 |
| head_dim | 128 |
| KV bytes/token (BF16) | 2 × 28 × 4 × 128 × 2 = **57,344 bytes ≈ 56 KB/token** |

1M tokens 总 KV cache：

```
1,000,000 × 56 KB = 56 GB
```

分 domain 后的显存需求（每 domain 权重 + 本地 KV）：

| Domain 数 | 每 domain KV | 每 domain 权重 | 每 domain 总显存 | 是否 fit white (24GB) | 是否 fit pearl (16GB) |
|-----------|--------------|----------------|------------------|------------------------|------------------------|
| 2 | 28 GB | 14 GB | **42 GB** | ❌ | ❌ |
| 4 | 14 GB | 14 GB | **28 GB** | ❌ | ❌ |
| 8 | 7 GB | 14 GB | **21 GB** | ✅ | ❌ |
| 16 | 3.5 GB | 14 GB | **17.5 GB** | ✅ | ❌ |

**结论**：7B 模型 1M context 在当前 BF16 / 每 worker 全量权重加载的前提下，**无法让 pearl（16GB）参与**。即使 white 也无法在 2/4 domain 下完成。

> 注：如果未来支持「每层权重只加载到负责该层计算的 worker」或模型并行，此约束可放宽，但属于远期工程。

### 2.2 可行路径：用 0.5B 模型在 white + pearl 上验证 1M context

以 **Qwen2-0.5B / Qwen2.5-0.5B** 为参考：

| 参数 | 值 |
|------|-----|
| num_layers | 24 |
| num_kv_heads | 2 |
| head_dim | 64 |
| KV bytes/token (BF16) | 2 × 24 × 2 × 64 × 2 = **12,288 bytes ≈ 12 KB/token** |

1M tokens 总 KV cache：

```
1,000,000 × 12 KB = 12 GB
```

| Domain 数 | 每 domain KV | 每 domain 权重 (~1GB) | 每 domain 总显存 | 是否 fit white (24GB) | 是否 fit pearl (16GB) |
|-----------|--------------|------------------------|------------------|------------------------|------------------------|
| 2 | 6 GB | 1 GB | **7 GB** | ✅ | ✅ |
| 4 | 3 GB | 1 GB | **4 GB** | ✅ | ✅ |

**结论**：0.5B 模型 1M context 在 white + pearl 两台机器上做 2-domain 分布式**完全可行**，且仍有充足 headroom。

### 2.3 模型选择建议

| 模型 | 1M context 可行性 on white+pearl | 建议 |
|------|-----------------------------------|------|
| Qwen2.5-7B-Instruct-1M | ❌ 不可行（显存不足） | 延后，待 4+ 设备或权重分片支持 |
| Qwen2.5-0.5B-Instruct + RoPE / ALiBI 扩展 | ✅ 可行 | **首选验证目标** |
| Qwen2.5-3B-Instruct | ⚠️ 边缘（3B 权重 ~6GB，1M KV ~24GB，2-domain 需 18GB/设备） | pearl 可能 OOM，不建议作为首目标 |

> 用户提到的 **Qwen2.5-7B-Instruct-1M** 仍可作为 A100 平台后续目标（4× A100 40GB 或 8× 设备），但不在 white+pearl 本地验证范围内。

### 2.4 2.5G 有线直连网络现实与基线结果

由于两台机器主板均无雷雳口，改用 **2.5G 以太网口直连**。实测基线（2026-06-16）：

| Path | Direction | Avg RTT (ms) | Send (Mbps) | Receive (Mbps) | Retransmits |
|------|-----------|--------------|-------------|----------------|-------------|
| **wired (2.5G)** | white→pearl | **0.206** | **2354.5** | **2353.6** | 0 |
| **wired (2.5G)** | pearl→white | **0.131** | **2353.9** | **2353.1** | 0 |
| **wired (2.5G)** | bidirectional | N/A | **2350.4** | **2349.7** | 0 |
| wifi | white→pearl | 36.694 | 254.0 | 252.7 | 6 |
| wifi | pearl→white | 89.159 | 256.7 | 256.1 | 0 |
| tailscale | white→pearl | 0.774 | 1779.7 | 1779.3 | 0 |
| tailscale | pearl→white | 39.563 | 303.4 | 302.9 | 1815 |

**关键发现**：
- **2.5G 有线直连**：接近满速（~2.35 Gbps），双向对称，RTT ~0.1-0.2 ms，0 retransmits。
- **WiFi**：~250 Mbps，RTT 36-89 ms，明显受限。
- **Tailscale 严重不对称**：white→pearl 方向接近 1.8 Gbps（可能走有线直连），但 pearl→white 方向仅 ~300 Mbps / 39 ms，且有 1815 次 retransmits，疑似走了 DERP relay 或 WiFi。

**决策**：HCP 分布式验证应直接使用 **2.5G 有线直连 IP**（white: `192.168.100.1`，pearl: `192.168.100.2`），而非 Tailscale 地址。

**期望收益**：
- 网络 latency 从 Tailscale/WiFi 的数十毫秒降到 **亚毫秒级**
- 有效带宽从 ~250 Mbps / ~300 Mbps 提升到 **~2.35 Gbps**
- 当前被 network-bound 的场景（大 KV block、长序列）将显著改善

---

## 3. 分阶段实验计划

### Phase 0：环境准备（✅ 网络已就绪，模型待确认）

1. **网络连接**（✅ 已完成）
   - 2.5G 有线直连：white `192.168.100.1` ↔ pearl `192.168.100.2`
   - iperf3 基线完成，结果见 §2.4
   - HCP 验证将使用有线直连 IP

2. **模型准备**
   - 确认 white 和 pearl 上都有 Qwen2.5-0.5B-Instruct（或 Qwen2-0.5B）
   - 确认 tokenizer 和 config 一致
   - 生成 1M token 级别的长 prompt（复用 `scripts/generate_long_prompt.py`）

### Phase 1：单节点与分布式 0.5B 长 context 验证（✅ 已完成）

1. **单节点 0.5B 1M context**
   - white 单节点尝试 1M tokens（24GB 应可 fit，但可能较慢）
   - pearl 单节点尝试 1M tokens（16GB 应可 fit）
   - 记录峰值显存、总耗时、输出文本

2. **2-domain 异构分布式 0.5B 1M context**
   - white (CUDA, domain 0) + pearl (HIP, domain 1)
   - 使用 **2.5G 有线直连 IP**
   - 目标：prefill 完成 + 至少 1-5 decode tokens
   - 记录：总时间、每步时间、recv/compute ratio、显存峰值

3. **里程碑推进**
   - 先跑 256K / 512K context 确认链路和代码稳定
   - 再推进到 1M context

### Phase 2：极限与正确性（2-3 天）

1. **更大 domain 数**
   - 如果 white / pearl 支持 `--local-domain-ids`，可尝试单机多 domain（但受显存限制）
   - 或引入第三台设备（如 Mac MPS）做 3-domain 1M context

2. **Correctness 验证**
   - 与单节点参考输出对比 token 序列
   - 使用 `compare_logits.py`（BF16 异构下放宽阈值）
   - 文本级一致性是核心指标

3. **A/B：Serial vs Pipeline overlap**
   - 在 2.5G 有线直连高速链路上，overlap 收益可能不同于 WiFi/Tailscale
   - 预期：高速网络下 network time 占比降低，overlap 收益可能减小（与 A100 NVLink 类似）

### Phase 3：报告与叙事（✅ 已完成）

1. 生成结构化报告：
   - `reports/1m-context-thunderbolt-<timestamp>/`
   - config、log、correctness summary、transport metrics
2. 更新文档：
   - `docs/A100_VALIDATION_REPORT.md` 或新报告
   - `memory-bank/progress.md`
   - `docs/SCALING_ARGUMENT.md` 补充 2.5G 有线直连数据点
3. 明确下一步：
   - 是否继续推 7B 1M（需要更多设备）
   - 是否引入真实 RDMA 硬件（InfiniBand / RoCE）

---

## 4. 技术风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| 2.5G 网卡驱动/直连配置不顺利 | 低 | 延误 | 网络已配置完成，静态 IP 已分配；如后续重连需检查网口和线缆 |
| 0.5B 1M context 单节点 prefll 极慢 | 高 | 体验差 | 允许只测分布式；或缩短到 512K/256K 作为里程碑 |
| pearl 16GB 在 0.5B 1M 下仍 OOM | 低 | 无法完成 | 先用 512K 验证；或进一步减少 batch / 使用更激进 chunking |
| 2.5G 实际带宽远低于理论值 | 低 | 收益不明显 | iperf3 实测已达 ~2.35 Gbps，风险排除 |
| HCP QUIC transport 无法直接利用 2.5G 低延迟 | 中 | 需要调参 | 调整 QUIC 窗口、idle timeout、micro block size |
| dual_chunk_attention / 长模型特殊结构 | 中 | 需要代码改动 | 先避开 1M-specific 模型，用 0.5B 标准模型 + 长 prompt |

---

## 5. 需要新增/修改的代码与脚本

1. **启动脚本**
   - `scripts/run_white_pearl_1m_0.5b.sh`：自动化 white+pearl 1M context 启动（使用 2.5G 有线直连 IP）
   - `scripts/benchmark_network_white_pearl.sh`：`iperf3` 基线测试（已完成）

2. **Prompt 生成**
   - `scripts/generate_1m_prompt.py`：生成 1M token 长 prompt
   - 或复用现有 `generate_long_prompts.py`

3. **监控脚本**
   - `monitor_white_pearl_1m.sh`：长时间运行监控

4. **可能的代码调整**
   - 调整默认 `HCP_MICRO_KV_BLOCK_SIZE` 适配高速网络
   - 调整 QUIC timeout 参数适配低延迟链路
   - 如需要，优化长序列 prefill 的 chunking 策略

---

## 6. 成功标准

### 最小成功

- white + pearl 通过 **2.5G 有线直连**完成 **0.5B 模型 1M context 2-domain 分布式推理**
- prefill 完成，decode 至少生成 1 token
- 无 OOM、无 crash
- 输出与单节点参考在文本级一致

### 完整成功

- 完成 Serial vs Pipeline A/B
- 量化 2.5G 有线直连相比 WiFi/Tailscale 的延迟/带宽收益
- 生成正式报告并更新 scaling argument
- 明确 7B 1M context 需要多少设备 / 什么网络

---

## 7. 执行结果（2026-06-19）

### 7.1 最终验证

**1M context 本地异构分布式推理成功**（white RTX 4090 CUDA + pearl RX 9060 XT HIP，2.5G 有线直连）：

| 配置 | 值 |
|------|-----|
| 模型 | `Qwen2-0.5B-1M`（0.5B，24 layers，BF16，权重 ~1GB） |
| Prompt | 精确 1,000,000 tokens（`gen_prompt` 生成，decode→encode round-trip 校验） |
| 分片 | capacity-aware **3:1**（white 750,000 / pearl 250,000） |
| KV buffer | `HCP_KV_CHANNEL_BUFFER_SIZE=512` |
| QUIC timeout | `HCP_QUIC_TIMEOUT_SECS=14400` |
| 显存策略 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（white） |
| Config patch | `max_position_embeddings=1048576`（原 1000000 会在 decode 时越界） |
| 输出 | `generated:  the.`（5 decode tokens，temperature=0.0） |
| 结果 | prefill 24/24 全通，decode 5/5 全通，exit=0，workers 优雅退出 ✅ |

### 7.2 关键性能数据

| 指标 | 数值 |
|------|-----|
| 总耗时 | ~2 小时 8 分钟（04:08 UTC → 06:16 UTC） |
| prefill | ~1 小时 52 分钟 |
| decode | ~16 分钟（~3 分钟 / token） |
| white 显存峰值 | **23,999 MB**（RTX 4090 24GB 几乎占满） |
| pearl 显存 | 未 OOM，16GB 内完成 |
| decode GPU 利用率 | ~5%（memory-bound 特征） |

### 7.3 攻克的关键问题

1. **1M prefill 死锁**：KV transport `mpsc` buffer 默认 64，对 >64 micro blocks 不足，导致 prefill 死锁。通过环境变量 `HCP_KV_CHANNEL_BUFFER_SIZE=512` 解决。
2. **QUIC 600s 超时**：大 context 下单层 KV 传输 + compute 可能超过 600s。通过 `HCP_QUIC_TIMEOUT_SECS=14400` 解决。
3. **Decode 位置越界**：原 config `max_position_embeddings=1_000_000`，decode 第 1 个新 token 的位置为 1,000,000，触发 RoPE `index_select` 越界。patch 为 `1048576` 后解决。
4. **pearl 16GB 碎片化 OOM**：2:1 split 在 layer 23/24 因 pearl 显存分配失败；改用 3:1 split 降低 pearl 压力，white 24GB 刚好 fit。

### 7.4 与原始预期的差异

- **原计划 1:1 分片可行**：实际因 activation / 工作集 / 显存碎片，pearl 16GB 无法承载 500K chunk；最终使用 **3:1 不均等分片**。
- **white 显存接近 24GB 上限**：远超 §2.2 表格估算的 7GB，主要因为 CUDA allocator fragmentation、activation、中间 tensor 以及 KV cache 连续分配。`expandable_segments:True` 帮助 white 刚好 fit。
- **2.5G 网络不是瓶颈**：prefill 耗时 ~1h52m，说明在 0.5B/1M 规模下 system 主要受显存容量和 memory-bound compute 限制，而非网络带宽。

### 7.5 报告

- 完整报告：`reports/1m-white-pearl-20260619/README.md`
- 状态：**最小成功 ✅ 达成；完整成功中 Serial vs Pipeline A/B 与更大模型验证暂未执行。**

---

## 8. 与 A100 1M 模型的关系

- **white+pearl 本地验证**：聚焦「异构 + 高速 P2P」这个差异化命题，模型用 0.5B。
- **A100 1M 模型（Qwen2.5-7B-Instruct-1M）**：作为后续规模化验证目标，需要 4-8× A100 或支持权重分片。
- 两者不冲突：white+pearl 证明「高速互联 + 异构」的可行性；A100 证明「大模型 + 长 context」的工程上限。
