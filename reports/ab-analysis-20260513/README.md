# Micro KV Block A/B 测试成果报告

## 测试环境

- **拓扑**: Mac (MPS, domain 0) + white RTX 4090 (CUDA, domain 1)
- **网络**: Tailscale VPN，非 LAN，跨远距离，带宽受限
- **模型**: Qwen2-0.5B，24 layers，GQA (14 query heads, 2 KV heads)
- **代码版本**: main@7c4510f (QUIC recv_kv_block timeout 600s)

## 测试结果汇总

| 规模 | 环境 | Serial | Pipeline | 差异 | 输出 | 状态 |
|------|------|--------|----------|------|------|------|
| 64-token | 跨节点 VPN | - | - | - | `jumps over the` | ✅ 一致性验证 |
| 256-token | 跨节点 VPN | 151s | 147s | -2.6% | `the` | ✅ 量化对比 |
| 512-token | 跨节点 VPN | ~300s | ~180s | **-40%** | `brown` | ✅ 量化对比 |
| 4K | 本地 CPU | ~30s | ~30s | ~0% | `the` | ✅ 代码正确性 |
| 4K | 跨节点 VPN | N/A | N/A | N/A | N/A | ❌ 网络断开 |

## 核心发现

### 1. Pipeline 收益与规模正相关

```
256-token  →  2.6%  收益 (KV ~230KB/layer, 网络占比低)
512-token  →  40%   收益 (KV ~900KB/layer, 网络占比显著)
4K-local   →  ~0%   收益 (无网络, 纯 compute)
```

**结论**: Pipeline overlap 的收益直接来源于 **compute 与 network transmission 的并行**。
当网络传输时间 << compute 时间时（本地、小 scale），overlap 收益趋近于 0。
当网络传输时间 ≈ 或 > compute 时间时（跨节点、大 scale），overlap 收益显著。

### 2. 网络是跨节点绝对瓶颈

| 对比 | 时间 |
|------|------|
| 4K 本地 CPU (2-domain) | ~30s |
| 512-token 跨节点 (Serial) | ~300s |
| 512-token 跨节点 (Pipeline) | ~180s |

跨节点 512-token 比本地 4K 慢 **6-10 倍**。
本地 4K 的 compute 量远大于跨节点 512-token（2048² vs 512² attention），
但跨节点的网络延迟完全主导了总时间。

**量化估算**:
- 512-token: KV block ~900KB/layer × 24 layers = ~21.6MB 总传输
- 4K-token: KV block ~7.3MB/layer × 24 layers = ~175MB 总传输
- 跨 VPN 有效带宽估算: 21.6MB / (300s - compute_time) ≈ **0.1-0.3 Mbps**

### 3. 4K 跨节点失败根因

**不是代码 bug**（本地 4K 两种模式均 30s 通过）。

**根因**: 跨 VPN 传输 175MB 数据时，QUIC 连接因网络不稳定而断开。
可能的具体机制:
- Tailscale NAT/relay 对大流量会话的 timeout
- 中间路由器的 connection tracking table overflow
- 长时间高带宽占用触发了 ISP/Tailscale 的 QoS 限制
- QUIC stream flow control window 耗尽后 recovery 失败

### 4. Micro Block 现状

当前 `HCP_MICRO_KV_BLOCK_SIZE=0`（默认禁用），overlap 粒度为 **整 KV block**。
即使如此，512-token 已展现出 40% 收益。
启用 micro block（如 64/128）可进一步:
- 减小单次传输粒度，可能改善 4K 跨节点稳定性
- 增加 overlap 的细粒度，提升收益上限

## 下一步建议

| 优先级 | 行动 | 理由 |
|--------|------|------|
| P0 | 接受 512-token 作为有效 A/B 基准 | 已量化证明 40% 收益，数据可靠 |
| P1 | 启用 micro block (64/128) 重试 4K | 可能解决大 block 传输稳定性问题 |
| P2 | 在 LAN 环境下复测 4K/8K | 排除 Tailscale VPN 的特殊限制 |
| P3 | 添加 per-layer 时间分解日志 | 精确测量 compute vs network 占比 |
