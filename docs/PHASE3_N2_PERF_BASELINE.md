# Phase-3 N=2 Performance Baseline (white + pearl, LAN)

Date: 2026-08-13. Run: `reports/routeb-p3-baseline-20260813-175408/`
(not committed — raw reports stay out of git). Produced by
`scripts/phase3_8_perf_baseline_n2.sh` with `REPS=5`.

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
2. A perf change is considered **real** only if the candidate **median** for
   a metric moves outside the baseline **min–max band** recorded above, in
   the claimed direction (lower for ttft/tpot/itl, higher for throughput).
   Movement inside the band is noise.
3. If a change plausibly shifts the band itself (e.g. a large win), record a
   **new baseline** with this script and replace the table above; do not mix
   bands from different days.
4. Any rep that fails correctness gates invalidates the run — performance
   numbers from a correctness-failing stack are meaningless. The script
   enforces this (one environmental retry per rep, then abort).
