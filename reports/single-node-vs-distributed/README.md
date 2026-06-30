# Single-node vs distributed: 4096-token Qwen2-0.5B

All numbers are wall-clock time to process a 4095-token prompt and generate 5 tokens.

| Configuration | Hardware | Time |
|---------------|----------|-----:|
| Single-node CUDA | RTX 4090 (white) | **0.12 s** |
| Single-node CPU | MacBook Pro M-series | 4.5 s |
| Single-node MPS | MacBook Pro M-series | 5.2 s |
| Distributed 2-domain (vanilla, 1:1) | RTX 4090 CUDA + RX 9060 XT HIP | ~15.1 s |
| Distributed 2-domain at 100 Mbps | RTX 4090 CUDA + RX 9060 XT HIP | ~206 s |

## Interpretation

- The GPU itself is extremely fast: a single RTX 4090 finishes the whole 4096-token forward in 0.12 s.
- HCP's distributed ring on the same GPU pair takes ~15 s — **~125× slower** — because the cross-node KV transfers dominate.
- At 100 Mbps, it is **~1700× slower** than single-node CUDA.
- This is not a CPU-vs-GPU issue.  It is a "single-node local memory" vs "multi-node network" issue.  HCP's value is to break the memory wall at very long contexts; at 4K tokens the network overhead is not yet amortized.
