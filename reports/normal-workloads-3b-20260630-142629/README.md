# Normal workload comparison: Qwen2.5-3B / 1K / 4K

 white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP), 1:1 chunk split, max_tokens=5.

## Single-node baselines (white)

| Model | Seq | CUDA | CPU |
|---|---|---:|---:|
| 3B | 1024 | 0.14s | 7.78s |
| 3B | 4096 | 0.27s | 29.26s |
| 7B | 1024 | 0.22s | 17.58s |
| 7B | 4096 | 0.52s | 64.09s |

## Distributed 3B strategy comparison

### seq=1024 (3 reps vanilla, 2 reps striped/zigzag)

| Strategy | Bottleneck total (ms) | mean (ms) | min (ms) | max (ms) | vs vanilla |
|---|---:|---:|---:|---:|---:|
| vanilla | [11436.800000000003, 13329.296999999997, 13382.123999999998] | 12716 | 11437 | 13382 | +0.0% |
| striped | [12710.57, 12074.557999999994] | 12393 | 12075 | 12711 | -2.5% |
| zigzag | [11495.118000000002, 12569.778000000008] | 12032 | 11495 | 12570 | -5.4% |

### seq=4096 (single run each)

| Strategy | Bottleneck total (ms) | vs vanilla |
|---|---:|---:|
| vanilla | 39818 | +0.0% |
| striped | 39818 | +0.0% |
| zigzag | 39553 | -0.7% |

## Observations

- At 3B/1K, ZigZag is ~5-6% faster than Vanilla on average, but the variance between runs is comparable to the gain.
- At 3B/4K, strategy differences collapse to <1% because the cross-node transfer volume dominates.
- Distributed 3B GPU is still slower than single-node CPU at these normal seq lengths: 3B/1K ~12s vs CPU 7.8s; 3B/4K ~40s vs CPU 29s.
- Qwen2.5-7B does not fit on the 16GB HIP card on pearl in bf16, so distributed 7B cannot run without quantization support.
