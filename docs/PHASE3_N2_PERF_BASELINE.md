# Phase-3 N=2 Performance Baseline (white + pearl, LAN)

Date: 2026-08-13. Run: `reports/routeb-p3-baseline-20260813-175408/`
(not committed — raw reports stay out of git). Produced by
`scripts/phase3_8_perf_baseline_n2.sh` with `REPS=5`.

**Status 2026-08-14: this 5-rep table is provisional.** The link between the
nodes turned out to be WiFi with a ~44 Mbit/s TCP goodput ceiling (see
[Network environment](#network-environment-measured-2026-08-14)); the
baseline script now captures network metadata per run and the baseline is
being re-recorded with more reps. Do not compare against this table without
checking the run's `network.json` first.

## Why a repeated baseline

Single `vllm bench serve` runs on this LAN vary by ~35%, so one run cannot
distinguish a real optimization from noise. This baseline runs the phase-3 7a
N=2 ladder 5 times back-to-back — a **fresh coordinator + workers stack per
rep** via `scripts/phase3_7a_n2_driver.sh` — and aggregates per load level.
Every rep had to pass the 7a correctness assertions (bench
completed==num_prompts, trace reserved==released, prefill_hops=24,
decode_hops=steps*24, metrics failed=0) with zero retries; a rep that fails
twice aborts the whole baseline.

## Configuration

| Item | Value |
|---|---|
| Git commit (all 3 nodes verified) | `9b48d8f861fca8ab960c76c821b216e5e0bec0d1` |
| Topology | N=2: coordinator + worker 0 + bench client on **white**; worker 1 on **pearl**; Mac control-only (launch/poll/fetch) |
| white | `stark@100.118.253.68` (LAN 192.168.8.172), RTX 4090, CUDA, domain 0 |
| pearl | `stark@100.111.242.55` (LAN 192.168.8.176), RX 9060 XT, HIP (`LD_PRELOAD=libtorch_hip.so`), domain 1 |
| Model | Qwen2-0.5B (`Qwen2ForCausalLM`), `torch_dtype=bfloat16`, 24 layers, hidden 896, GQA 14/2 |
| Binary | `rust/target/release/hcp-ringattn-rust --features tch-backend` |
| Bench client | `~/venv-bench/bin/vllm bench serve` (vllm 0.27.1) on white, `--backend openai --endpoint /v1/completions` |
| Dataset | `--dataset-name random --random-input-len 32 --random-output-len 16 --random-range-ratio 0.5 --seed 42` |
| L1 | num_prompts=8, request_rate=1, no concurrency cap |
| L2 | num_prompts=8, request_rate=inf, max_concurrency=2 |
| L3 | num_prompts=16, request_rate=inf, max_concurrency=4 |
| Reps | 5, all PASS on first attempt; total wall 09:54:08Z → 10:29:40Z (~35.5 min, ~7 min/rep) |

## Network environment (measured 2026-08-14)

The 192.168.8.x "LAN" path between white and pearl is **WiFi on both ends**,
not wired ethernet. This dominates the baseline numbers and is the main
source of the observed variance.

| Fact | Value |
|---|---|
| white egress | `wlp11s0` (WiFi 6, 5 GHz ch48, 80 MHz), PHY rx 1080.6 / tx 960.7 Mbit/s, signal −32 dBm |
| pearl egress | `wlo1`, link quality 70/70, signal −38 dBm (driver exposes no PHY bitrate via `iw`) |
| RTT white→pearl | min 2.8 / avg 4.3 / max 6.7 ms, 0% loss (20 pkts) |
| **TCP goodput, 1 stream (iperf3, 5×10 s runs)** | **43.3–45.7 Mbit/s (~5.5 MB/s)**, ~800–1050 retransmits per 10 s |
| TCP goodput, 4 parallel streams | 75.4–76.3 Mbit/s (~9.5 MB/s) |
| Wired alternative | Not currently usable: both nodes have 2.5GbE NICs up (`enp10s0` / `enp8s0`), but white's has no IPv4 and pearl's is on a different subnet (192.168.100.2/24, direct-link) |

Key implications:

- Single-TCP-stream ceiling is **~44 Mbit/s** — about 4% of the WiFi PHY
  rate, with heavy retransmission (interference / half-duplex contention).
  All ring KV traffic in the baseline crossed this link, so the numbers above
  are WiFi-bound, not compute-bound.
- From 2026-08-14 the baseline script captures `network.json` (link state,
  RTT, iperf3 ceiling) before the rep loop and folds it into `baseline.json`.
  **A baseline comparison is only valid if the candidate run's `network.json`
  shows an equivalent link** (same media, comparable goodput/RTT). If the
  link changes (e.g. nodes move to wired), discard old bands and re-baseline.

## Aggregated results (5 reps)

### L1 — num_prompts=8, rate=1 (sequential)

| Metric | Median | Min | Max | Mean | Spread (min–max as % of median) |
|---|---|---|---|---|---|
| mean_ttft_ms | 22006.26 | 19459.02 | 29820.05 | 24138.91 | 47.1% |
| mean_tpot_ms | 2158.76 | 1928.03 | 2788.44 | 2336.31 | 39.9% |
| mean_itl_ms | 1928.59 | 1759.84 | 2461.11 | 2095.54 | 36.4% |
| output_throughput (tok/s) | 1.55 | 1.25 | 1.71 | 1.48 | 29.6% |

### L2 — num_prompts=8, rate=inf, max_concurrency=2

| Metric | Median | Min | Max | Mean | Spread (min–max as % of median) |
|---|---|---|---|---|---|
| mean_ttft_ms | 3932.06 | 3559.09 | 4680.71 | 4013.96 | 28.5% |
| mean_tpot_ms | 1135.85 | 1041.80 | 1289.37 | 1146.97 | 21.8% |
| mean_itl_ms | 1025.88 | 965.98 | 1191.18 | 1055.79 | 22.0% |
| output_throughput (tok/s) | 1.53 | 1.31 | 1.63 | 1.50 | 21.2% |

### L3 — num_prompts=16, rate=inf, max_concurrency=4

| Metric | Median | Min | Max | Mean | Spread (min–max as % of median) |
|---|---|---|---|---|---|
| mean_ttft_ms | 8038.79 | 6548.49 | 8470.79 | 7702.95 | 23.9% |
| mean_tpot_ms | 2498.73 | 2190.82 | 2720.86 | 2431.71 | 21.2% |
| mean_itl_ms | 2375.12 | 2085.46 | 2609.76 | 2315.71 | 22.1% |
| output_throughput (tok/s) | 1.33 | 1.22 | 1.53 | 1.38 | 23.7% |

## Variance observations

- The ~35% single-run variance is confirmed and is worst at L1 (TTFT spread
  47%); concurrent levels (L2/L3) are more stable at ~21–29% spread.
- Reps were **not** identically distributed: L1 TTFT rose monotonically-ish
  across reps (rep3 19.5s fastest → rep5 29.8s slowest), i.e. there is slow
  drift on top of run-to-run noise (LAN contention / thermal / background
  load on the shared machines — not investigated here). Conclusion: never
  compare "before" reps run in one time window against "after" reps run in a
  different window without interleaving or re-baselining.
- Correctness was flat across all reps: 0 failed requests, trace
  reserved==released, hop formulas exact — the variance is purely in
  latency/throughput.

## Comparison rule for future optimization work

1. Future perf-affecting changes rerun **this same script**
   (`REPS=5 bash scripts/phase3_8_perf_baseline_n2.sh`) on the **same
   topology** (white + pearl, same models, same load ladder), with remotes
   pinned to the candidate commit.
2. **Check `network.json` first**: if the candidate run's link differs
   materially from the baseline run's (different media, or iperf3 goodput /
   RTT outside the baseline's range), the bands are not comparable —
   re-baseline instead of comparing.
3. A perf change is considered **real** only if the candidate **median** for
   a metric moves outside the baseline **min–max band** recorded above, in
   the claimed direction (lower for ttft/tpot/itl, higher for throughput).
   Movement inside the band is noise.
4. If a change plausibly shifts the band itself (e.g. a large win), record a
   **new baseline** with this script and replace the table above; do not mix
   bands from different days.
5. Any rep that fails correctness gates invalidates the run — performance
   numbers from a correctness-failing stack are meaningless. The script
   enforces this (one environmental retry per rep, then abort).
