#!/bin/bash
# Phase-3 8: stable, recordable N=2 performance baseline (white + pearl).
#
# Problem: single vllm bench serve runs on this LAN vary ~35%, so a single
# 7a-style run cannot serve as a baseline for future optimization work.
# This harness runs the phase-3 7a N=2 bench ladder REPS times back-to-back
# (fresh stack per rep, driven by scripts/phase3_7a_n2_driver.sh on white),
# requires every rep to pass the 7a correctness assertions (0 failures,
# trace reserved==released, hop formulas), and aggregates per load level:
# median/min/max/mean of mean_ttft_ms, mean_tpot_ms, mean_itl_ms,
# output_throughput. A failed rep gets ONE environmental retry (fresh
# relaunch); if the retry also fails the whole baseline aborts — a
# correctness-failing stack must not enter the baseline record.
#
# Usage:
#   REPS=5 bash scripts/phase3_8_perf_baseline_n2.sh
#
# Env:
#   REPS            number of repetitions (default 5)
#   EXPECTED_COMMIT commit the remote repos MUST be on (default: local HEAD)
#
# Output: reports/routeb-p3-baseline-<ts>/
#   rep<i>/bench/n2-l{1,2,3}.json, rep<i>/trace-n2.jsonl, rep<i>/metrics-n2.json
#   rep<i>/*.log (coordinator/workers/driver)
#   network.json + network/ — link state, RTT, iperf3 goodput ceiling captured
#     before the rep loop (comparability metadata; see docs/PHASE3_N2_PERF_BASELINE.md)
#   baseline.json — config + network + per-rep raw values + per-level aggregates
#
# Raw reports are NOT committed (reports/**/*.json is gitignored).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
WHITE_REPO_DIR="${WHITE_REPO_DIR:-hetero-cp-ringattn}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

REPS="${REPS:-5}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(cd "${REPO_ROOT}" && git rev-parse HEAD)}"

RUN_ID="routeb-p3-baseline-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

# Per-level load params (identical to the 7a N=2 ladder).
L1_PARAMS="num_prompts=8 request_rate=1 max_concurrency=none"
L2_PARAMS="num_prompts=8 request_rate=inf max_concurrency=2"
L3_PARAMS="num_prompts=16 request_rate=inf max_concurrency=4"
COMMON_PARAMS="dataset=random random_input_len=32 random_output_len=16 random_range_ratio=0.5 seed=42 endpoint=/v1/completions"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote_white() {
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${WHITE_SSH}" "bash -lc $(shell_quote "$1")"
}

cleanup() {
    echo "=== Cleanup ==="
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Phase-3 8: N=2 perf baseline (white + pearl), REPS=${REPS} ==="
echo "RUN_ID=${RUN_ID}"
echo "EXPECTED_COMMIT=${EXPECTED_COMMIT}"
echo "Reports: ${REPORT_DIR}"
echo "START=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${REPORT_DIR}/START"

# === Preflight: remotes on the expected commit + binary present ===
echo "=== Preflight: sync + verify remote commits ==="
for node in white pearl; do
    if [ "${node}" = white ]; then ssh_target="${WHITE_SSH}"; repo="${WHITE_REPO_DIR}"; else ssh_target="${PEARL_SSH}"; repo="${PEARL_REPO_DIR}"; fi
    remote_head=$(ssh -o ConnectTimeout=30 "${ssh_target}" "cd ~/${repo} && git checkout main >/dev/null 2>&1 && git pull --ff-only origin main >/dev/null 2>&1; git rev-parse HEAD")
    echo "  ${node} HEAD=${remote_head}"
    if [ "${remote_head}" != "${EXPECTED_COMMIT}" ]; then
        echo "ERROR: ${node} on ${remote_head}, expected ${EXPECTED_COMMIT}" >&2
        exit 1
    fi
done
echo "${EXPECTED_COMMIT}" > "${REPORT_DIR}/git_commit.txt"

echo "=== Preflight: remote release binaries (incremental build if stale) ==="
white_build="cd $(shell_quote "${WHITE_REPO_DIR}") && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
run_remote_white "${white_build}" 2>&1 | tail -2
pearl_build="cd $(shell_quote "${PEARL_REPO_DIR}") && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
ssh -o ConnectTimeout=30 "${PEARL_SSH}" "bash -lc $(shell_quote "${pearl_build}")" 2>&1 | tail -2

# === Network metadata ===
# This baseline is only comparable against runs on an equivalent link. The
# 192.168.8.x path between white and pearl is WiFi (wlp11s0 / wlo1), so
# record interface/PHY state, RTT, and the measured TCP goodput ceiling
# (iperf3, one-shot) before the rep loop; the aggregate step folds
# network.json into baseline.json.
echo "=== Network metadata (link info + RTT + iperf3 ceiling) ==="
NET_DIR="${REPORT_DIR}/network"
mkdir -p "${NET_DIR}"

ssh -o ConnectTimeout=20 "${WHITE_SSH}" '
    dev=$(ip -4 route get 192.168.8.176 | sed -n "s/.*dev \([^ ]*\).*/\1/p" | head -1)
    echo "egress_dev=${dev}"
    ip -4 addr show "${dev}" | grep inet || true
    iw dev "${dev}" link 2>/dev/null || { echo "(no iw link info)"; cat /proc/net/wireless; }
' > "${NET_DIR}/white-link.txt" 2>&1 || true

ssh -o ConnectTimeout=20 "${PEARL_SSH}" '
    dev=$(ip -4 route get 192.168.8.172 | sed -n "s/.*dev \([^ ]*\).*/\1/p" | head -1)
    echo "egress_dev=${dev}"
    ip -4 addr show "${dev}" | grep inet || true
    iw dev "${dev}" link 2>/dev/null || { echo "(no iw link info)"; cat /proc/net/wireless; }
' > "${NET_DIR}/pearl-link.txt" 2>&1 || true

ssh -o ConnectTimeout=20 "${WHITE_SSH}" 'ping -c 20 -i 0.2 192.168.8.176' \
    > "${NET_DIR}/ping-white-to-pearl.txt" 2>&1 || true

# iperf3 one-shot server on pearl (-1: exit after first test), 5s client on white.
ssh -o ConnectTimeout=20 "${PEARL_SSH}" 'pkill iperf3 2>/dev/null; sleep 1; nohup iperf3 -s -B 192.168.8.176 -p 5201 -1 >/tmp/hcp-iperf3-baseline.log 2>&1 & sleep 2; ss -tln | grep -q 5201' || true
ssh -o ConnectTimeout=30 "${WHITE_SSH}" 'iperf3 -c 192.168.8.176 -p 5201 -t 5' \
    > "${NET_DIR}/iperf3-white-to-pearl.txt" 2>&1 || true
ssh -o ConnectTimeout=20 "${PEARL_SSH}" 'pkill iperf3 2>/dev/null || true' || true

python3 - "${NET_DIR}" <<'NETPY'
import json, re, sys, os

net_dir = sys.argv[1]
def read(name):
    try:
        return open(os.path.join(net_dir, name)).read()
    except OSError:
        return ""

ping = read("ping-white-to-pearl.txt")
iperf = read("iperf3-white-to-pearl.txt")
out = {
    "purpose": "baseline comparability metadata: ring traffic crosses this link; only compare against baselines with an equivalent network block",
    "captured_at": "baseline start (before rep loop)",
    "white_link": read("white-link.txt"),
    "pearl_link": read("pearl-link.txt"),
    "ping_white_to_pearl_raw": ping,
    "iperf3_white_to_pearl_raw": iperf,
}
m = re.search(r"rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", ping)
if m:
    out["rtt_ms"] = {"min": float(m.group(1)), "avg": float(m.group(2)),
                     "max": float(m.group(3)), "mdev": float(m.group(4))}
m = re.search(r"([\d.]+) Mbits/sec\s+receiver", iperf)
if m:
    out["iperf3_tcp_goodput_mbps_receiver"] = float(m.group(1))
m = re.search(r"([\d.]+) Mbits/sec\s+(\d+)\s+sender", iperf)
if m:
    out["iperf3_tcp_goodput_mbps_sender"] = float(m.group(1))
    out["iperf3_retransmits"] = int(m.group(2))
json.dump(out, open(os.path.join(net_dir, "..", "network.json"), "w"), indent=2)
print("network metadata: rtt_ms=%s goodput_mbps=%s retr=%s" %
      (out.get("rtt_ms"), out.get("iperf3_tcp_goodput_mbps_receiver"),
       out.get("iperf3_retransmits")))
NETPY

# === Per-rep validation: mirror the 7a N=2 assertions ===
validate_rep() { # rep_dir
    python3 - "$1" <<'PY'
import json, sys, glob

rep_dir = sys.argv[1]
TOTAL, HOPS = 32, 24

bench_files = sorted(glob.glob(f"{rep_dir}/bench/n2-l*.json"))
assert len(bench_files) == 3, f"expected 3 bench result files, got {bench_files}"
prompts_seen = 0
for f in bench_files:
    b = json.load(open(f))
    label = f.split("/")[-1]
    completed = b.get("completed", 0)
    prompts_seen += completed
    assert completed == b["num_prompts"], f"{label}: completed {completed} != num_prompts {b['num_prompts']}"
    for key in ("mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
                "request_throughput", "output_throughput"):
        v = b.get(key)
        assert v is not None and v == v and v > 0, f"{label}: bad metric {key}={v}"
assert prompts_seen == TOTAL, f"bench completed total {prompts_seen} != {TOTAL}"

records = [json.loads(l) for l in open(f"{rep_dir}/trace-n2.jsonl") if l.strip()]
assert len(records) == TOTAL, f"trace records {len(records)} != {TOTAL}"
ids = sorted(r["request_id"] for r in records)
assert ids == list(range(1, TOTAL + 1)), f"request_id set broken: {ids[:5]}...{ids[-3:]}"
for r in records:
    assert r["error"] is None, f"req {r['request_id']} error {r['error']}"
    assert r["reserved_bytes"] == r["released_bytes"], f"req {r['request_id']} release mismatch"
    assert r["prefill_hops"] == HOPS, f"req {r['request_id']} prefill_hops {r['prefill_hops']} != {HOPS}"
    assert r["decode_hops"] == r["decode_steps"] * HOPS, f"req {r['request_id']} decode_hops"

m = json.load(open(f"{rep_dir}/metrics-n2.json"))
assert m["total_requests"] == TOTAL, f"metrics total {m['total_requests']} != {TOTAL}"
assert m["completed_requests"] == TOTAL, f"metrics completed {m['completed_requests']}"
assert m["failed_requests"] == 0, f"metrics failed {m['failed_requests']}"
assert m["active_requests"] == 0, f"metrics active {m['active_requests']}"
print("  rep validation OK: bench completed=32, trace ids/hops/reserved==released, metrics failed=0")
PY
}

# === Run one rep on white via the 7a N=2 driver; returns 0 on PASS ===
run_rep_once() { # rep_label run_suffix
    local rep_label=$1 suffix=$2
    local rep_run_id="${RUN_ID}-${suffix}"
    local state_dir="/tmp/hcp-n2-${rep_run_id}"
    local rep_dir="${REPORT_DIR}/${rep_label}"

    echo "=== ${rep_label} (${suffix}) start=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    date -u +%Y-%m-%dT%H:%M:%SZ > "${REPORT_DIR}/${rep_label}.start"
    # Pre-clean in case a previous attempt left daemons behind.
    ssh -o ConnectTimeout=15 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=15 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    sleep 3

    # ssh -n -f returns right after launching the driver; the driver owns the
    # full N=2 lifecycle on white even if the Mac goes away.
    ssh -n -f -o ConnectTimeout=20 "${WHITE_SSH}" "mkdir -p ${state_dir} && setsid bash ~/${WHITE_REPO_DIR}/scripts/phase3_7a_n2_driver.sh ${rep_run_id} > ${state_dir}/driver.log 2>&1 </dev/null"
    echo "  driver launched on white (state: ${state_dir}); polling STATUS..."

    local status=""
    for _ in $(seq 1 30); do
        sleep 60
        status=$(ssh -o ConnectTimeout=20 "${WHITE_SSH}" "cat ${state_dir}/STATUS 2>/dev/null" 2>/dev/null || true)
        [ -n "${status}" ] && break
        echo "  ... still running"
    done
    date -u +%Y-%m-%dT%H:%M:%SZ > "${REPORT_DIR}/${rep_label}.end"

    mkdir -p "${rep_dir}/bench"
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/bench/"*.json "${rep_dir}/bench/" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/trace-n2.jsonl" "${rep_dir}/trace-n2.jsonl" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/metrics-n2.json" "${rep_dir}/metrics-n2.json" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/coordinator-n2.log" "${rep_dir}/coordinator-n2.log" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/driver.log" "${rep_dir}/driver.log" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/worker0-white-n2.log" "${rep_dir}/worker0-white-n2.log" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/worker1-pearl-n2.log" "${rep_dir}/worker1-pearl-n2.log" 2>/dev/null || true

    if [ "${status}" != "DONE" ]; then
        echo "  ${rep_label}: driver STATUS='${status}'"
        return 1
    fi
    if ! validate_rep "${rep_dir}"; then
        echo "  ${rep_label}: validation FAILED"
        return 1
    fi
    echo "  ${rep_label}: PASS"
    return 0
}

# === Rep loop with one environmental retry ===
declare -a REP_RESULTS
overall_fail=0
for i in $(seq 1 "${REPS}"); do
    if run_rep_once "rep${i}" "rep${i}"; then
        REP_RESULTS+=("rep${i}:PASS")
        continue
    fi
    echo "rep${i} failed; allowing ONE environmental retry (fresh relaunch)..."
    if run_rep_once "rep${i}" "rep${i}-retry"; then
        REP_RESULTS+=("rep${i}:PASS(after retry)")
        continue
    fi
    echo "ERROR: rep${i} failed twice — aborting baseline (a failing stack must not enter the record)." >&2
    REP_RESULTS+=("rep${i}:FAIL")
    overall_fail=1
    break
done

printf '%s\n' "${REP_RESULTS[@]}" > "${REPORT_DIR}/rep_results.txt"
echo "END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${REPORT_DIR}/END"

if [ "${overall_fail}" -ne 0 ]; then
    echo "=== BASELINE ABORTED (see ${REPORT_DIR}) ===" >&2
    exit 1
fi

# === Aggregate ===
python3 - "${REPORT_DIR}" "${REPS}" "${EXPECTED_COMMIT}" <<PY
import json, statistics, sys, glob, os

report_dir, reps, commit = sys.argv[1], int(sys.argv[2]), sys.argv[3]
levels = ["n2-l1", "n2-l2", "n2-l3"]
metrics = ["mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "output_throughput"]

per_rep = {}
for i in range(1, reps + 1):
    rep = f"rep{i}"
    per_rep[rep] = {}
    for lv in levels:
        files = glob.glob(f"{report_dir}/{rep}/bench/{lv}.json")
        assert len(files) == 1, f"{rep}/{lv}: {files}"
        b = json.load(open(files[0]))
        per_rep[rep][lv] = {k: b[k] for k in metrics}
        per_rep[rep][lv]["request_throughput"] = b["request_throughput"]
        per_rep[rep][lv]["duration_s"] = b.get("duration")

def agg(vals):
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.mean(vals),
        "spread_pct_of_median": round(100.0 * (max(vals) - min(vals)) / statistics.median(vals), 1),
        "n": len(vals),
    }

levels_out = {}
for lv in levels:
    levels_out[lv] = {k: agg([per_rep[r][lv][k] for r in per_rep]) for k in metrics}

read_ts = lambda name: open(os.path.join(report_dir, name)).read().strip()
try:
    network = json.load(open(os.path.join(report_dir, "network.json")))
except OSError:
    network = {"warning": "network.json missing — this run predates network metadata capture; compare with caution"}
baseline = {
    "run_id": os.path.basename(report_dir),
    "kind": "phase3 N=2 performance baseline (white + pearl LAN)",
    "config": {
        "git_commit": commit,
        "topology": {
            "coordinator": "white (also worker 0 + bench client)",
            "worker0": "white stark@100.118.253.68 (LAN 192.168.8.172), RTX 4090, CUDA, domain 0",
            "worker1": "pearl stark@100.111.242.55 (LAN 192.168.8.176), RX 9060 XT, HIP (libtorch_hip LD_PRELOAD), domain 1",
            "mac": "control-only (launch/poll/fetch); not in the service path",
        },
        "network": network,
        "model": "Qwen2-0.5B (Qwen2ForCausalLM), torch_dtype=bfloat16, 24 layers, hidden 896, GQA 14/2",
        "binary": "rust/target/release/hcp-ringattn-rust --features tch-backend",
        "bench_client": "~/venv-bench/bin/vllm bench serve on white, backend=openai",
        "common_params": "${COMMON_PARAMS}",
        "levels": {
            "n2-l1": "${L1_PARAMS}",
            "n2-l2": "${L2_PARAMS}",
            "n2-l3": "${L3_PARAMS}",
        },
        "reps": reps,
        "rep_independence": "fresh coordinator+workers stack per rep via phase3_7a_n2_driver.sh",
        "rep_gate": "each rep must pass 7a assertions (bench completed==num_prompts, trace reserved==released, prefill_hops=24, decode_hops=steps*24, metrics failed=0); one environmental retry allowed",
    },
    "timestamps": {
        "start": read_ts("START").replace("START=", ""),
        "end": read_ts("END").replace("END=", ""),
        "per_rep": {f"rep{i}": {"start": read_ts(f"rep{i}.start"), "end": read_ts(f"rep{i}.end")} for i in range(1, reps + 1)},
    },
    "rep_results": open(os.path.join(report_dir, "rep_results.txt")).read().split(),
    "per_rep_metrics": per_rep,
    "aggregate": levels_out,
}
out = os.path.join(report_dir, "baseline.json")
json.dump(baseline, open(out, "w"), indent=2)
print(f"wrote {out}")

print("")
print("=== AGGREGATE (median / min / max / mean) ===")
for lv in levels:
    print(f"  {lv}:")
    for k in metrics:
        a = levels_out[lv][k]
        print(f"    {k:18s} median={a['median']:.2f} min={a['min']:.2f} max={a['max']:.2f} mean={a['mean']:.2f} spread={a['spread_pct_of_median']}%")
PY

echo ""
echo "=== PHASE-3 8 N=2 PERF BASELINE PASSED (${REPS} reps) ==="
echo "Reports: ${REPORT_DIR}"
