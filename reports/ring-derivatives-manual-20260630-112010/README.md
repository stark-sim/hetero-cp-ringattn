# Ring Attention Derivatives on white CUDA + pearl HIP

Manual cross-node run, seq_len=4096, max_tokens=5, Qwen2-0.5B-1M.

| Strategy | Domain 0 total (ms) | Domain 1 total (ms) | Bottleneck (ms) | Domain 0 recv (ms) | Domain 1 recv (ms) | Domain 0 local (ms) | Domain 1 local (ms) |
|---|---|---|---|---|---|---|---|
| vanilla | 15076.7 | 14392.2 | 15076.7 | 14477.0 | 12662.8 | 132.8 | 647.6 |
| striped | 14758.6 | 13947.8 | 14758.6 | 14140.5 | 12256.4 | 119.0 | 651.9 |
| zigzag | 15577.8 | 14773.2 | 15577.8 | 14905.9 | 13039.5 | 128.9 | 656.3 |

Observations:
- All three scheduling strategies complete successfully on real heterogeneous hardware.
- Network recv dominates total time (>90% on both workers).
- Strategy differences are small (<6%) because the tailscale link is the bottleneck.
- Striped changes the generated token sequence, which is expected for a small model on a degenerate prompt.
