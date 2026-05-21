# 4-Domain 4K Serial Cross-Node Test — 2026-05-22

## Summary

**First successful 4-domain 4K-token cross-node Serial mode run.**

- **Total time**: 4988s (1h 23m 8s)
- **Mode**: Serial (`HCP_DISABLE_OVERLAP=1`, no micro block)
- **Model**: Qwen2-0.5B
- **Prompt**: 4096 tokens (4 chunks × 1024)
- **Generated**: 1 token (`over`)
- **Status**: ✅ Success, exit code 0

## Topology

```
Mac MPS (d0) → sd-1 CUDA 4080S (d1) → sd-2 CUDA 4080S (d2) → white CUDA 4090 (d3) → Mac MPS (d0)
```

## Key Observations

1. **Serial mode works for N-domain ring** — The channel buffer fix (2 → 64) successfully resolved the distributed deadlock that previously hung Serial mode in multi-domain configurations.

2. **Network is the dominant bottleneck** — With no overlap, ~168MB per worker (24 layers × ~7MB/layer) must be exchanged sequentially across Tailscale VPN. At typical VPN bandwidth (1–5 Mbps), this alone accounts for 10–30 minutes.

3. **Mac MPS is the straggler** — In Serial mode, all nodes wait for the slowest compute node between rounds. The Mac MPS device's slow compute amplifies total time beyond pure network transfer.

4. **Correctness validated** — All 4 workers completed prefill, global_seq_len=4096, decode generated 1 token successfully.

## Comparison Context

| Test | Time | Status | Notes |
|------|------|--------|-------|
| 512-token 2-domain (Mac+white) Serial | ~300s | ✅ | Mac+white Tailscale |
| 512-token 2-domain (Mac+white) Pipeline | ~180s | ✅ | 40% faster than Serial |
| 512-token 2-domain (sd-1+white) Serial micro=64 | 330s | ✅ | Dual CUDA |
| 512-token 2-domain (sd-1+white) Pipeline micro=64 | 319s | ✅ | Only 3.3% faster |
| **4K 4-domain Serial** | **4988s** | **✅** | **This run** |
| 4K 4-domain Pipeline | 2166s (then ❌) | ❌ Connection lost | VPN instability under ~528MB/worker |

## Formula Validation

Pipeline benefit ≈ 1 − compute/(compute+network)

- **512-token Mac+white**: compute ≈ network → ~40% benefit ✅
- **512-token sd-1+white**: compute >> network → ~3% benefit ✅
- **4K 4-domain**: network >> compute (especially with Mac straggler) → Pipeline should show **massive** benefit, but VPN stability prevents measurement

## Next Steps

1. Retry Pipeline 4K 4-domain with improved network stability (e.g., LAN instead of VPN)
2. Add progress logging to Serial mode for better observability
3. Consider LAN-only 4K+ tests — Tailscale VPN is a hard ceiling for large-scale distributed validation
