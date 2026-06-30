# Ring Attention Derivatives: 3:1 vs 1:1 chunk split on white+pearl

 white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP), Qwen2-0.5B-1M, seq_len=4096, max_tokens=5.

| Strategy | Domain 0 total (ms) | Domain 1 total (ms) | Bottleneck (ms) | Domain 0 recv (ms) | Domain 1 recv (ms) | Domain 0 local (ms) | Domain 1 local (ms) |
|---|---|---|---|---|---|---|---|
| vanilla (3:1) | 15076.7 | 14392.2 | 15076.7 | 14477.0 | 12662.8 | 132.8 | 647.6 |
| striped (3:1) | 14758.6 | 13947.8 | 14758.6 | 14140.5 | 12256.4 | 119.0 | 651.9 |
| zigzag (3:1) | 15577.8 | 14773.2 | 15577.8 | 14905.9 | 13039.5 | 128.9 | 656.3 |
| vanilla (1:1) | 15122.0 | 14516.2 | 15122.0 | 14423.5 | 12804.0 | 146.3 | 656.3 |
| striped (1:1) | 15547.3 | 14721.8 | 15547.3 | 14795.0 | 12601.2 | 133.2 | 661.6 |
| zigzag (1:1) | 15331.0 | 14640.0 | 15331.0 | 14674.7 | 12918.9 | 131.5 | 650.7 |

Observations:
- 1:1 capacity-aware split makes vanilla/striped/zigzag operate on equal token counts, removing the 3:1 load imbalance.
- Strategy differences remain small (<4% between fastest and slowest in 1:1), confirming that the cross-node link is the bottleneck.
- ZigZag does not provide a measurable win on this network; its value would appear only when compute and network are better balanced.
