# Phase-3 N=2 Performance Baseline (white + pearl, 2.5GbE)

Date: 2026-08-14. Two independent 10-repetition runs (raw reports are not
committed), both produced by `scripts/phase3_8_perf_baseline_n2.sh` with
`REPS=10`:

- run 1: `reports/routeb-p3-baseline-20260814-134121/` (05:41Z, commit `e07be07`)
- run 2: `reports/routeb-p3-baseline-20260814-155401/` (07:54Z, commit `79cb7a7`,
  after an unattended white reboot that verified network boot persistence)

Together they form the current N=2 comparison baseline (20 repetitions per
load level). They supersede the 2026-08-13 WiFi-bound five-repetition table
for performance comparisons. The old run is retained below only as
network-environment evidence.

## Result validity

All ten repetitions passed on the first attempt. Every repetition launched a
fresh coordinator and worker stack and passed the phase-3 7a correctness gates:

- all three `vllm bench serve` levels completed every request (32 total);
- trace request ids were complete and unique;
- `reserved_bytes == released_bytes` for every request;
- `prefill_hops = 24` and `decode_hops = steps * 24`;
- service metrics reported `failed = 0`.

The report stores `controller_script.sha256` and `controller_script.diff`
because the data-plane parameterization used by this run was not committed with
the model binary commit.

## Configuration

| Item | Value |
|---|---|
| Git commit on white and pearl | `e07be073345566a49fa54bccd92de62a4ff8d5ed` |
| Topology | N=2: coordinator, worker 0, and bench client on white; worker 1 on pearl; Mac control-only |
| Data path | white `enp10s0` `192.168.100.1/24` to pearl `enp8s0` `192.168.100.2/24` |
| white | RTX 4090, CUDA, domain 0 |
| pearl | RX 9060 XT, HIP (`LD_PRELOAD=libtorch_hip.so`), domain 1 |
| Model | Qwen2-0.5B, bfloat16, 24 layers, hidden 896, GQA 14/2 |
| Binary | `rust/target/release/hcp-ringattn-rust --features tch-backend` |
| Client | vLLM 0.27.1 `bench serve`, OpenAI backend, `/v1/completions` |
| Dataset | random, input 32, output 16, range ratio 0.5, seed 42 |
| L1 | 8 prompts, request rate 1, no concurrency cap |
| L2 | 8 prompts, request rate inf, max concurrency 2 |
| L3 | 16 prompts, request rate inf, max concurrency 4 |
| Repetitions | 10, all PASS without retry |
| Wall time | 2026-08-14 05:41:21Z to 05:56:31Z (15m10s) |

## Network gate

The ring data path was confirmed to use the direct 2.5GbE cable rather than
WiFi or Tailscale.

| Fact | Value |
|---|---|
| white link | `enp10s0`, 2500Mb/s, full duplex, link detected |
| pearl link | `enp8s0`, 2500Mb/s, full duplex, link detected |
| RTT white to pearl | min 0.108 / avg 0.172 / max 0.241 ms, 0% loss |
| iperf3 sender | 2.36 Gbit/s |
| iperf3 receiver | 2.35 Gbit/s |
| retransmits | 0 |
| separate stability gate | five 10-second single-stream runs all received 2.35 Gbit/s with zero retransmits; four streams also received 2.35 Gbit/s |

A future candidate is comparable only when its `network.json` confirms the same
media, routed interfaces, and a similar RTT/goodput range.

## Aggregated results (10 reps)

### L1 - 8 prompts, request rate 1

| Metric | Median | Min | Max | Mean | Spread |
|---|---:|---:|---:|---:|---:|
| mean TTFT (ms) | 334.44 | 324.03 | 342.06 | 333.12 | 5.4% |
| mean TPOT (ms) | 69.94 | 68.98 | 72.48 | 70.10 | 5.0% |
| mean ITL (ms) | 73.88 | 72.84 | 76.62 | 74.06 | 5.1% |
| output throughput (tok/s) | 15.18 | 15.10 | 15.22 | 15.17 | 0.8% |

### L2 - 8 prompts, max concurrency 2

| Metric | Median | Min | Max | Mean | Spread |
|---|---:|---:|---:|---:|---:|
| mean TTFT (ms) | 149.92 | 145.86 | 158.77 | 151.01 | 8.6% |
| mean TPOT (ms) | 55.79 | 54.93 | 56.20 | 55.71 | 2.3% |
| mean ITL (ms) | 51.65 | 50.67 | 51.82 | 51.45 | 2.2% |
| output throughput (tok/s) | 31.68 | 31.31 | 32.31 | 31.73 | 3.2% |

### L3 - 16 prompts, max concurrency 4

| Metric | Median | Min | Max | Mean | Spread |
|---|---:|---:|---:|---:|---:|
| mean TTFT (ms) | 279.81 | 275.51 | 297.45 | 281.60 | 7.8% |
| mean TPOT (ms) | 112.18 | 111.03 | 117.25 | 112.82 | 5.5% |
| mean ITL (ms) | 106.49 | 104.84 | 110.68 | 107.16 | 5.5% |
| output throughput (tok/s) | 31.02 | 29.94 | 31.21 | 30.83 | 4.1% |

The result is materially more stable than the old WiFi run: the widest metric
spread is now 8.6%, versus 21-47% for latency metrics on WiFi.

## Reproducibility across independent runs (20 reps pooled)

Run 2 repeated the full 10-rep ladder on a fresh time window (and after an
unattended white reboot in between). All 10 run-2 repetitions passed on the
first attempt under the same correctness gates, with an equivalent network
gate (RTT avg 0.185 ms, iperf3 receiver 2.35 Gbit/s, 0 retransmits).

Cross-run median deltas (run 2 vs run 1) stay inside +/-1.4% for every metric
at every level:

| Level / metric | Run 1 median | Run 2 median | Delta | Pooled 20-rep median | Pooled min-max spread |
|---|---:|---:|---:|---:|---:|
| L1 TTFT (ms) | 334.44 | 333.75 | -0.21% | 334.22 | 5.4% |
| L1 TPOT (ms) | 69.94 | 70.66 | +1.02% | 70.37 | 5.0% |
| L1 ITL (ms) | 73.88 | 74.58 | +0.94% | 74.38 | 5.2% |
| L1 output tok/s | 15.18 | 15.12 | -0.34% | 15.16 | 0.8% |
| L2 TTFT (ms) | 149.92 | 151.98 | +1.37% | 151.26 | 10.0% |
| L2 TPOT (ms) | 55.79 | 55.82 | +0.07% | 55.82 | 2.3% |
| L2 ITL (ms) | 51.65 | 51.56 | -0.16% | 51.59 | 2.4% |
| L2 output tok/s | 31.68 | 31.52 | -0.49% | 31.62 | 3.2% |
| L3 TTFT (ms) | 279.81 | 277.75 | -0.73% | 279.09 | 8.0% |
| L3 TPOT (ms) | 112.18 | 113.35 | +1.04% | 112.43 | 5.6% |
| L3 ITL (ms) | 106.49 | 107.90 | +1.32% | 106.94 | 5.5% |
| L3 output tok/s | 31.02 | 30.64 | -1.21% | 30.84 | 4.4% |

The baseline is therefore reproducible across time windows and node reboots:
per-run medians move by at most ~1.4%, far inside the per-run min-max band.

## Comparison rule

1. Use the same N=2 topology, model, workload ladder, and correctness gates.
2. Compare `network.json` before comparing model metrics. Re-baseline if media,
   routed interface, RTT, or goodput differs materially.
3. Use at least 10 repetitions for a claimed optimization. Interleave baseline
   and candidate runs when they are collected in different time windows.
4. Treat a movement inside the current min-max band as noise. A claimed change
   must move the median outside the band in the claimed direction and retain
   correctness in every repetition.
5. Do not compare a single-node vLLM run against HCP. The controlled reference
   must include distributed communication and scheduling overhead.

## Superseded WiFi baseline

The 2026-08-13 run `routeb-p3-baseline-20260813-175408` used the
`192.168.8.x` WiFi path. Its single-stream TCP ceiling was about 44 Mbit/s with
heavy retransmission and RTT around 4.3 ms. Representative medians were:

| Level | TTFT (ms) | TPOT (ms) | Throughput (tok/s) |
|---|---:|---:|---:|
| L1 | 22006.26 | 2158.76 | 1.55 |
| L2 | 3932.06 | 1135.85 | 1.53 |
| L3 | 8038.79 | 2498.73 | 1.33 |

The wired run is 26-66x lower in TTFT, 20-31x lower in TPOT, and 10-23x higher
in output throughput. These ratios demonstrate network sensitivity, not an HCP
algorithmic speedup: the code/workload boundary is comparable, but the network
environment is intentionally different.
