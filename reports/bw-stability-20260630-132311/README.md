# Cross-node bandwidth stability: baseline vs 100 Mbps

 white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP), Qwen2-0.5B-1M, seq_len=4096, max_tokens=5.
 Traffic shaped with `tc tbf` on `enp10s0` / `enp8s0`.

## Baseline (no tc)

| rep | elapsed (s) |
|---|---|
| 1 | 17 |
| 2 | 18 |
| 3 | 17 |
| mean | 17.3 |

## 100 Mbps

| rep | elapsed (s) |
|---|---|
| 1 | 204 |
| 2 | 205 |
| 3 | 217 |
| 4 | 203 |
| 5 | 203 |
| mean | 206.4 |

**Slowdown at 100 Mbps:** 11.9× (baseline mean 17.3s).

## Takeaway

After 5 repetitions at 100 Mbps, the wall-clock time is stable around 206s (stddev ~5s).
The earlier observed 38s and 604s points were transient outliers, not the true distribution.
