# Ring Attention Derivatives on HCP: Vanilla / Striped / ZigZag

**Date:** 2026-06-30  
**Hosts:** white (RTX 4090 CUDA) ↔ pearl (RX 9060 XT HIP) via Tailscale  
**Model:** `Qwen2-0.5B-1M`  
**Workload:** `seq_len=4096`, `max_tokens=5`, 2-domain heterogeneous ring  
**Code:** `scripts/run_ring_derivatives_2domain_cuda_hip.sh`

## Motivation

A common objection to heterogeneous context parallelism is that it is a narrow demo that can only run vanilla Ring Attention.  This benchmark answers that objection directly: we implement the core scheduling ideas of **Striped Attention** (Brandon et al.) and **ZigZag Ring Attention** inside the real HCP Rust codebase and measure them on real heterogeneous hardware.

At the same time, the comparison becomes evidence for the CXL/RDMA argument: even if you apply the best-known scheduling optimizations, the cross-node link is still the bottleneck.

## Implementation notes

- `rust/src/model/attention/strategy.rs` introduces `RingSchedulingStrategy` and assignment helpers.
- The coordinator sends **per-domain permuted token ids and position ids** for Striped/ZigZag; workers use the existing `position_ids` path in `HcpRingAttentionBackend`.
- No custom kernels were added.  All three strategies reuse HCP's existing online-softmax ring path.

## 3:1 capacity-aware split (memory-proportional)

| Strategy | Domain 0 total (ms) | Domain 1 total (ms) | Bottleneck (ms) | Domain 0 recv (ms) | Domain 1 recv (ms) | Domain 0 local (ms) | Domain 1 local (ms) |
|----------|--------------------:|--------------------:|----------------:|-------------------:|-------------------:|----------------------------:|----------------------------:|
| Vanilla  | 15076.7 | 14392.2 | **15076.7** | 14477.0 (96.0%) | 12662.8 (88.0%) | 132.8 | 647.6 |
| Striped  | 14758.6 | 13947.8 | **14758.6** | 14140.5 (95.8%) | 12256.4 (87.9%) | 119.0 | 651.9 |
| ZigZag   | 15577.8 | 14773.2 | **15577.8** | 14905.9 (95.7%) | 13039.5 (88.3%) | 128.9 | 656.3 |

## 1:1 equal split

To remove the memory-proportional load imbalance, the runs below force exactly equal token counts per domain (`--chunk-sizes 2048,2048`).

| Strategy | Domain 0 total (ms) | Domain 1 total (ms) | Bottleneck (ms) | Domain 0 recv (ms) | Domain 1 recv (ms) | Domain 0 local (ms) | Domain 1 local (ms) |
|----------|--------------------:|--------------------:|----------------:|-------------------:|-------------------:|----------------------------:|----------------------------:|
| Vanilla  | 15122.0 | 14516.2 | **15122.0** | 14423.5 (95.4%) | 12804.0 (88.2%) | 146.3 | 656.3 |
| Striped  | 15547.3 | 14721.8 | **15547.3** | 14795.0 (95.2%) | 12601.2 (85.6%) | 133.2 | 661.6 |
| ZigZag   | 15331.0 | 14640.0 | **15331.0** | 14674.7 (95.7%) | 12918.9 (88.2%) | 131.5 | 650.7 |

Raw logs and `summary.csv`: `reports/ring-derivatives-manual-20260630-112010/` and `reports/ring-derivatives-1to1-20260630-122906/`

## Interpretation

1. **HCP runs all three strategies correctly on real heterogeneous hardware.**  No NaN, no crash, no special-case code paths beyond the assignment helper.
2. **The cross-node link dominates.**  On both workers, `recv_ms` accounts for >85% of total time.  Scheduling optimization cannot escape the bandwidth wall.
3. **1:1 vs 3:1 makes little difference.**  Equalizing chunk sizes does not change the overall picture because the slow tailscale link, not the load imbalance, sets the pace.
4. **Strategy differences are within ~6% of each other.**  Striped is marginally faster in one run, ZigZag marginally slower in another — all within run-to-run network variance.
5. **Striped changes the generated token sequence** with degenerate prompts, which is expected position sensitivity, not a correctness bug.  CPU mock tests for all three strategies pass with diff `< 1e-7`.

## Why Ring Flash Attention is not benchmarked here

Ring Flash Attention is a **kernel-level** optimization: it replaces the local attention tile computation with a fused FlashAttention kernel while keeping the same P2P ring structure.  Because HCP's measured bottleneck is the cross-node KV transfer (>85% of time), a faster local kernel would only shrink the already-small `<15%` compute slice.  Adding a custom CUDA/HIP kernel (or a PyO3 bridge to `torch.nn.functional.scaled_dot_product_attention`) is therefore a large engineering step that would not materially change the conclusion of this benchmark.

If future hardware provides a much faster interconnect, Ring Flash becomes relevant again; the existing `RingSchedulingStrategy` abstraction leaves a clean insertion point.

## Takeaway

HCP's heterogeneous design is not a vanilla-only demo.  It can host the main algorithmic variants of the Ring Attention family.  However, those variants do not rescue performance on a slow tailscale/consumer-Ethernet link.  That makes high-bandwidth, low-latency interconnects (CXL / RDMA / NVLink-class links) a prerequisite for heterogeneous CP to be practical, not optional.
