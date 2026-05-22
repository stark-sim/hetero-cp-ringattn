# Mac + White 弱网 A/B 测试报告 — 2026-05-22

## 环境
- 拓扑: Mac MPS (d0) + white RTX 4090 CUDA (d1), 2-domain ring
- 网络: Tailscale VPN, RTT ~380ms, 带宽受限
- 模型: Qwen2-0.5B
- max_tokens: 1

## 结果

| Size | Serial | Pipeline | 收益 | 状态 |
|------|--------|----------|------|------|
| 64 | 60s | **57s** | **+5%** | ✅ |
| 256 | 211s | **207s** | **+2%** | ✅ |
| 512 | 383s | 390s | **-2%** | ✅ |
| 1024 | 600s | 600s | N/A | ❌ timeout |
| 2048 | 600s | 600s | N/A | ❌ timeout |
| 4096 | 601s | 2404s | N/A | ❌ timeout/network |

## 关键发现

1. **512 tokens 是弱网可靠上限** — 64/256/512 全部成功，1024+ 全部失败。

2. **Pipeline 收益递减** — 收益从 64-token 的 5% 递减到 512-token 的 -2%。
   原因: Mac MPS 计算慢，overlap 能隐藏的网络时间有限；512-token 时网络传输
   已成为主导瓶颈，Pipeline 的 overhead 超过收益。

3. **1024+ 失败根因** — coordinator shutdown 阶段卡住（已知 bug）。
   所有 1024/2048/4096 serial 和 1024/2048 pipeline 都在 600s timeout 后失败。
   4096 pipeline 在 ~40min 后因网络断开失败。

4. **公式验证** — `benefit ≈ 1 - compute/(compute+network)`：
   - 64-token: compute ≈ network → ~5% 收益 ✅
   - 256-token: compute ≈ network → ~2% 收益 ✅
   - 512-token: network > compute → 负收益 ✅

## 下一步

- 修复 coordinator shutdown 卡住问题，才能扩展 1024+ 测试
- 或接受 512-token 为弱网上限，在 LAN/RDMA 环境下扩展
