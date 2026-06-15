# 1M Context + Thunderbolt RDMA 本地异构验证计划

> **目标**：在 white（RTX 4090 CUDA）和 pearl（RX 9060 XT HIP）两台本地机器上，通过 Thunderbolt 5/4 高速互联，验证 HCP Ring Attention 在 1M context 级别的可行性。
> **战略意义**：证明当单节点显存墙不可避免时，异构设备 + 高速 P2P 互联是通向百万 token 的差异化路径。
> **优先级**：这是当前最高优先级；vLLM CUDA E2E、量化、多请求并发等工程完善项后置。

---

## 1. 研究问题

我们要回答三个问题：

1. **可行性**：在本地两台异构机器（CUDA + HIP）上，能否用 HCP Ring Attention 协同完成 1M context 推理？
2. **高速互联价值**：Thunderbolt 5/4 提供的低延迟、高带宽 P2P 链路，能否显著改善当前 Tailscale VPN 下的网络瓶颈？
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

### 2.4 Thunderbolt 5/4 网络现实

**关键认知：Thunderbolt 本身不直接提供 RDMA  verbs。**

- Thunderbolt 4：理论带宽 40 Gbps（双向）
- Thunderbolt 5：理论带宽 80 Gbps（双向），可 boost 到 120 Gbps
- 操作系统通常把 Thunderbolt 视为一个高速以太网接口（IP over Thunderbolt）
- 真正的 RDMA（InfiniBand / RoCE）需要 RDMA-capable NIC 和相应驱动，Thunderbolt 控制器不暴露 RDMA verbs

**实际可做的差异化尝试：**

1. **IP over Thunderbolt**：在两台机器间建立 10GbE/25GbE/40GbE 甚至更高速度的 IP 链路，HCP 的 QUIC/TCP transport 直接跑在上面。
2. **Thunderbolt 直连桥接**：如果两台机器都支持 Thunderbolt Networking，可用雷雳线直接连接，形成点对点高速链路，无需交换机。
3. **对比基线**：
   - Tailscale VPN（当前基线，RTT ~1-380ms，有效带宽 ~7-9 MB/s）
   - 千兆以太网（~100 MB/s）
   - Thunderbolt 4/5（理论上 GB/s 级）

**期望收益**：
- 网络 latency 从 1-380ms 降到亚毫秒级
- 有效带宽从 7-9 MB/s 提升到数百 MB/s 甚至 GB/s
- 当前被 network-bound 的场景（大 KV block、长序列）将显著改善

---

## 3. 分阶段实验计划

### Phase 0：环境准备（1-2 天）

1. **硬件连接**
   - 确认 white 和 pearl 的 Thunderbolt 接口版本
   - 准备 Thunderbolt 4/5 线缆（注意：不是所有 USB-C 线都支持 Thunderbolt）
   - 直连两台机器，配置 IP over Thunderbolt
   - 验证连通性：`ping`、`iperf3` 测带宽和延迟

2. **网络配置**
   - 为 Thunderbolt 接口分配静态 IP（如 white: `192.168.100.1`，pearl: `192.168.100.2`）
   - 关闭/绕过 Tailscale，确保 HCP 走 Thunderbolt 链路
   - 防火墙放行 HCP 端口（9000, 9100, peer 端口等）

3. **模型准备**
   - 下载 Qwen2.5-0.5B-Instruct 到 white 和 pearl
   - 确认 tokenizer 和 config 一致
   - 生成 1M token 级别的长 prompt（可用重复文本或 LongBench 超长样本）

### Phase 1：基线验证（2-3 天）

1. **单节点 0.5B 1M context**
   - white 单节点尝试 1M tokens（24GB 应可 fit，但可能较慢）
   - pearl 单节点尝试 1M tokens（16GB 应可 fit）
   - 记录峰值显存、耗时、输出

2. **2-domain 异构分布式 0.5B 1M context**
   - white (CUDA) domain 0 + pearl (HIP) domain 1
   - 使用 Thunderbolt 链路
   - 目标：prefill 完成 + 至少 1-5 decode tokens
   - 记录：总时间、每步时间、recv/compute ratio、显存峰值

3. **对比基线**
   - 同一配置走 Tailscale VPN，量化 Thunderbolt 带来的收益
   - 同一配置走千兆以太网（如果方便）

### Phase 2：极限与正确性（2-3 天）

1. **更大 domain 数**
   - 如果 white / pearl 支持 `--local-domain-ids`，可尝试单机多 domain（但受显存限制）
   - 或引入第三台设备（如 Mac MPS）做 3-domain 1M context

2. **Correctness 验证**
   - 与单节点参考输出对比 token 序列
   - 使用 `compare_logits.py`（BF16 异构下放宽阈值）
   - 文本级一致性是核心指标

3. **A/B：Serial vs Pipeline overlap**
   - 在 Thunderbolt 高速链路上，overlap 收益可能不同于 Tailscale
   - 预期：高速网络下 network time 占比降低，overlap 收益可能减小（与 A100 NVLink 类似）

### Phase 3：报告与叙事（1-2 天）

1. 生成结构化报告：
   - `reports/1m-context-thunderbolt-<timestamp>/`
   - config、log、correctness summary、transport metrics
2. 更新文档：
   - `docs/A100_VALIDATION_REPORT.md` 或新报告
   - `memory-bank/progress.md`
   - `docs/SCALING_ARGUMENT.md` 补充 Thunderbolt 数据点
3. 明确下一步：
   - 是否继续推 7B 1M（需要更多设备）
   - 是否引入真实 RDMA 硬件（InfiniBand / RoCE）

---

## 4. 技术风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| Thunderbolt 驱动/网络配置不顺利 | 中 | 延误 | 先确认两台机器 Thunderbolt 版本和 OS 支持；准备 USB4/Thunderbolt 认证线缆 |
| 0.5B 1M context 单节点 prefll 极慢 | 高 | 体验差 | 允许只测分布式；或缩短到 512K/256K 作为里程碑 |
| pearl 16GB 在 0.5B 1M 下仍 OOM | 低 | 无法完成 | 先用 512K 验证；或进一步减少 batch / 使用更激进 chunking |
| Thunderbolt 实际带宽远低于理论值 | 中 | 收益不明显 | 用 `iperf3` 先测；如果只有几百 MB/s，仍然比 VPN 好一个数量级 |
| HCP QUIC transport 无法直接利用 Thunderbolt 低延迟 | 中 | 需要调参 | 调整 QUIC 窗口、idle timeout、micro block size |
| dual_chunk_attention / 长模型特殊结构 | 中 | 需要代码改动 | 先避开 1M-specific 模型，用 0.5B 标准模型 + 长 prompt |

---

## 5. 需要新增/修改的代码与脚本

1. **启动脚本**
   - `scripts/run_white_pearl_1m_thunderbolt.sh`：自动化 white+pearl 1M context 启动
   - `scripts/setup_thunderbolt_network.sh`：IP over Thunderbolt 配置
   - `scripts/benchmark_network.sh`：`iperf3` 基线测试

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

- white + pearl 通过 Thunderbolt 完成 **0.5B 模型 1M context 2-domain 分布式推理**
- prefill 完成，decode 至少生成 1 token
- 无 OOM、无 crash
- 输出与单节点参考在文本级一致

### 完整成功

- 完成 Serial vs Pipeline A/B
- 量化 Thunderbolt 相比 Tailscale 的延迟/带宽收益
- 生成正式报告并更新 scaling argument
- 明确 7B 1M context 需要多少设备 / 什么网络

---

## 7. 与 A100 1M 模型的关系

- **white+pearl 本地验证**：聚焦「异构 + 高速 P2P」这个差异化命题，模型用 0.5B。
- **A100 1M 模型（Qwen2.5-7B-Instruct-1M）**：作为后续规模化验证目标，需要 4-8× A100 或支持权重分片。
- 两者不冲突：white+pearl 证明「高速互联 + 异构」的可行性；A100 证明「大模型 + 长 context」的工程上限。
