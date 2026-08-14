#!/bin/bash
# Phase-3 9: interleaved controlled baseline — HCP N=2 vs vLLM PD-disaggregation,
# white + pearl over the direct 2.5GbE link.
#
# Problem: the 20-rep HCP wired baseline (routeb-p3-baseline-20260814-134121 /
# -155401) established HCP absolute numbers, but without a mainstream-engine
# reference under the same link/workload they cannot answer "how much overhead
# does HCP's P2P ring add". This harness interleaves HCP reps (fresh
# coordinator+workers via phase3_7a_n2_driver.sh) with vLLM PD reps (fresh
# prefill+decode+proxy via phase3_9_pd_driver.sh) in one time window,
# alternating which side goes first each rep to cancel drift.
#
# Fairness contract:
#   same link (192.168.100.1<->192.168.100.2, network.json gate),
#   same model (Qwen2-0.5B bf16), same client (vllm bench serve on white),
#   same workload ladder (L1: 8 prompts rate=1; L2: 8 prompts mc=2;
#   L3: 16 prompts mc=4; random input 32 output 16 seed 42),
#   same metrics (TTFT/TPOT/ITL/throughput), REPS pairs each side.
# Known config asymmetry (recorded, not hidden): vLLM decode runs
# --enforce-eager because hipblaslt FULL cudagraph capture fails on gfx1200;
# HCP's tch-rs backend is always eager. vLLM prefill uses default compilation.
#
# Usage:
#   REPS=10 bash scripts/phase3_9_pd_interleaved_baseline.sh
#
# Env:
#   REPS            number of interleaved pairs (default 10)
#   EXPECTED_COMMIT commit the remote HCP repos MUST be on (default: local HEAD)
#
# Output: reports/routeb-p3-pd-baseline-<ts>/
#   rep<i>-hcp/{bench/n2-l*.json, trace-n2.jsonl, metrics-n2.json, *.log}
#   rep<i>-pd/{bench/pd-l*.json, *.log}
#   network.json + network/, comparison.json (side-by-side aggregates)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
WHITE_DATA_IP="${WHITE_DATA_IP:-192.168.100.1}"
PEARL_DATA_IP="${PEARL_DATA_IP:-192.168.100.2}"
WHITE_REPO_DIR="${WHITE_REPO_DIR:-hetero-cp-ringattn}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

REPS="${REPS:-10}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(cd "${REPO_ROOT}" && git rev-parse HEAD)}"

RUN_ID="routeb-p3-pd-baseline-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
shasum -a 256 "$0" > "${REPORT_DIR}/controller_script.sha256"
git -C "${REPO_ROOT}" diff -- "$0" > "${REPORT_DIR}/controller_script.diff"

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
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true; pkill -f 'vllm serve' || true; pkill -f disagg_proxy || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true; pkill -f 'vllm serve' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Phase-3 9: interleaved HCP vs vLLM-PD baseline, REPS=${REPS} pairs ==="
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

# Record the vLLM side's commits too (it lives in ~/vllm, outside the repo).
{
    echo -n "white ~/vllm: "; ssh -o ConnectTimeout=20 "${WHITE_SSH}" "git -C ~/vllm rev-parse HEAD"
    echo -n "pearl ~/vllm: "; ssh -o ConnectTimeout=20 "${PEARL_SSH}" "git -C ~/vllm rev-parse HEAD"
} > "${REPORT_DIR}/vllm_commits.txt" 2>&1
cat "${REPORT_DIR}/vllm_commits.txt"

echo "=== Preflight: remote HCP release binaries (incremental build if stale) ==="
white_build="cd $(shell_quote "${WHITE_REPO_DIR}") && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
run_remote_white "${white_build}" 2>&1 | tail -2
pearl_build="cd $(shell_quote "${PEARL_REPO_DIR}") && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
ssh -o ConnectTimeout=30 "${PEARL_SSH}" "bash -lc $(shell_quote "${pearl_build}")" 2>&1 | tail -2

# === Network metadata (same gate as phase3_8) ===
echo "=== Network metadata (link info + RTT + iperf3 ceiling) ==="
NET_DIR="${REPORT_DIR}/network"
mkdir -p "${NET_DIR}"

ssh -o ConnectTimeout=20 "${WHITE_SSH}" "
    dev=\$(ip -4 route get ${PEARL_DATA_IP} | sed -n 's/.*dev \([^ ]*\).*/\1/p' | head -1)
    echo \"egress_dev=\"\$dev
    ip -4 addr show \"\$dev\" | grep inet || true
    ethtool \"\$dev\" 2>/dev/null | grep -E 'Speed:|Duplex:|Link detected:' || true
" > "${NET_DIR}/white-link.txt" 2>&1 || true

ssh -o ConnectTimeout=20 "${PEARL_SSH}" "
    dev=\$(ip -4 route get ${WHITE_DATA_IP} | sed -n 's/.*dev \([^ ]*\).*/\1/p' | head -1)
    echo \"egress_dev=\"\$dev
    ip -4 addr show \"\$dev\" | grep inet || true
    ethtool \"\$dev\" 2>/dev/null | grep -E 'Speed:|Duplex:|Link detected:' || true
" > "${NET_DIR}/pearl-link.txt" 2>&1 || true

ssh -o ConnectTimeout=20 "${WHITE_SSH}" "ping -c 20 -i 0.2 ${PEARL_DATA_IP}"     > "${NET_DIR}/ping-white-to-pearl.txt" 2>&1 || true

ssh -o ConnectTimeout=20 "${PEARL_SSH}" "pkill iperf3 2>/dev/null; sleep 1; nohup iperf3 -s -B ${PEARL_DATA_IP} -p 5201 -1 >/tmp/hcp-iperf3-baseline.log 2>&1 & sleep 2; ss -tln | grep -q 5201" || true
ssh -o ConnectTimeout=30 "${WHITE_SSH}" "iperf3 -c ${PEARL_DATA_IP} -p 5201 -t 5"     > "${NET_DIR}/iperf3-white-to-pearl.txt" 2>&1 || true
ssh -o ConnectTimeout=20 "${PEARL_SSH}" 'pkill iperf3 2>/dev/null || true' || true

python3 - "${NET_DIR}" "${WHITE_DATA_IP}" "${PEARL_DATA_IP}" <<'NETPY'
import json, re, sys, os

net_dir, white_data_ip, pearl_data_ip = sys.argv[1:]
def read(name):
    try:
        return open(os.path.join(net_dir, name)).read()
    except OSError:
        return ""

ping = read("ping-white-to-pearl.txt")
iperf = read("iperf3-white-to-pearl.txt")
out = {
    "purpose": "baseline comparability metadata: ring traffic crosses this link; only compare against baselines with an equivalent network block",
    "captured_at": "campaign start (before rep loop)",
    "white_data_ip": white_data_ip,
    "pearl_data_ip": pearl_data_ip,
    "white_link": read("white-link.txt"),
    "pearl_link": read("pearl-link.txt"),
    "ping_white_to_pearl_raw": ping,
    "iperf3_white_to_pearl_raw": iperf,
}
m = re.search(r"rtt min/avg/max/mdev = ([d.]+)/([d.]+)/([d.]+)/([d.]+) ms", ping)
if m:
    out["rtt_ms"] = {"min": float(m.group(1)), "avg": float(m.group(2)),
                     "max": float(m.group(3)), "mdev": float(m.group(4))}
def to_mbps(value, unit):
    return float(value) * {"K": 0.001, "M": 1.0, "G": 1000.0}[unit]

m = re.search(r"([d.]+) ([KMG])bits/secs+receiver", iperf)
if m:
    out["iperf3_tcp_goodput_mbps_receiver"] = to_mbps(m.group(1), m.group(2))
m = re.search(r"([d.]+) ([KMG])bits/secs+(d+)s+sender", iperf)
if m:
    out["iperf3_tcp_goodput_mbps_sender"] = to_mbps(m.group(1), m.group(2))
    out["iperf3_retransmits"] = int(m.group(3))
json.dump(out, open(os.path.join(net_dir, "..", "network.json"), "w"), indent=2)
print("network metadata: rtt_ms=%s goodput_mbps=%s retr=%s" %
      (out.get("rtt_ms"), out.get("iperf3_tcp_goodput_mbps_receiver"),
       out.get("iperf3_retransmits")))
NETPY

# === Validators ===
validate_hcp_rep() { # rep_dir  — mirror the 7a N=2 assertions
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
print("  hcp rep validation OK: bench completed=32, trace ids/hops/reserved==released, metrics failed=0")
PY
}

validate_pd_rep() { # rep_dir — vLLM side has no HCP trace; bench completeness + sane metrics
    python3 - "$1" <<'PY'
import json, sys, glob

rep_dir = sys.argv[1]
TOTAL = 32

bench_files = sorted(glob.glob(f"{rep_dir}/bench/pd-l*.json"))
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
print("  pd rep validation OK: bench completed=32, metrics sane")
PY
}

# === Run one HCP rep on white via the 7a N=2 driver; returns 0 on PASS ===
run_hcp_rep_once() { # rep_label run_suffix
    local rep_label=$1 suffix=$2
    local rep_run_id="${RUN_ID}-${suffix}"
    local state_dir="/tmp/hcp-n2-${rep_run_id}"
    local rep_dir="${REPORT_DIR}/${rep_label}"

    echo "=== ${rep_label} (${suffix}) start=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    date -u +%Y-%m-%dT%H:%M:%SZ > "${REPORT_DIR}/${rep_label}.start"
    ssh -o ConnectTimeout=15 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=15 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    sleep 3

    ssh -n -f -o ConnectTimeout=20 "${WHITE_SSH}" "mkdir -p ${state_dir} && setsid env WHITE_LAN=${WHITE_DATA_IP} PEARL_LAN=${PEARL_DATA_IP} bash ~/${WHITE_REPO_DIR}/scripts/phase3_7a_n2_driver.sh ${rep_run_id} > ${state_dir}/driver.log 2>&1 </dev/null"
    echo "  hcp driver launched on white (state: ${state_dir}); polling STATUS..."

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

    if [ "${status}" != "DONE" ]; then
        echo "  ${rep_label}: driver STATUS='${status}'"
        return 1
    fi
    if ! validate_hcp_rep "${rep_dir}"; then
        echo "  ${rep_label}: validation FAILED"
        return 1
    fi
    echo "  ${rep_label}: PASS"
    return 0
}

# === Run one PD rep on white via the PD driver; returns 0 on PASS ===
run_pd_rep_once() { # rep_label run_suffix
    local rep_label=$1 suffix=$2
    local rep_run_id="${RUN_ID}-${suffix}"
    local state_dir="/tmp/vllm-pd-${rep_run_id}"
    local rep_dir="${REPORT_DIR}/${rep_label}"

    echo "=== ${rep_label} (${suffix}) start=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    date -u +%Y-%m-%dT%H:%M:%SZ > "${REPORT_DIR}/${rep_label}.start"

    ssh -n -f -o ConnectTimeout=20 "${WHITE_SSH}" "mkdir -p ${state_dir} && setsid env WHITE_LAN=${WHITE_DATA_IP} PEARL_LAN=${PEARL_DATA_IP} bash ~/${WHITE_REPO_DIR}/scripts/phase3_9_pd_driver.sh ${rep_run_id} > ${state_dir}/driver.log 2>&1 </dev/null"
    echo "  pd driver launched on white (state: ${state_dir}); polling STATUS..."

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
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/prefill.log" "${rep_dir}/prefill.log" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/decode.log" "${rep_dir}/decode.log" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/proxy.log" "${rep_dir}/proxy.log" 2>/dev/null || true
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/driver.log" "${rep_dir}/driver.log" 2>/dev/null || true

    if [ "${status}" != "DONE" ]; then
        echo "  ${rep_label}: driver STATUS='${status}'"
        return 1
    fi
    if ! validate_pd_rep "${rep_dir}"; then
        echo "  ${rep_label}: validation FAILED"
        return 1
    fi
    echo "  ${rep_label}: PASS"
    return 0
}

# === Interleaved pair loop; one environmental retry per side ===
declare -a REP_RESULTS
overall_fail=0
run_side_with_retry() { # side(hcp|pd) rep_idx
    local side=$1 i=$2
    if [ "${side}" = hcp ]; then runner=run_hcp_rep_once; else runner=run_pd_rep_once; fi
    if ${runner} "rep${i}-${side}" "rep${i}-${side}"; then
        REP_RESULTS+=("rep${i}-${side}:PASS")
        return 0
    fi
    echo "rep${i}-${side} failed; allowing ONE environmental retry..."
    if ${runner} "rep${i}-${side}" "rep${i}-${side}-retry"; then
        REP_RESULTS+=("rep${i}-${side}:PASS(after retry)")
        return 0
    fi
    echo "ERROR: rep${i}-${side} failed twice — aborting campaign." >&2
    REP_RESULTS+=("rep${i}-${side}:FAIL")
    return 1
}

for i in $(seq 1 "${REPS}"); do
    if [ $((i % 2)) -eq 1 ]; then order="hcp pd"; else order="pd hcp"; fi
    for side in ${order}; do
        if ! run_side_with_retry "${side}" "${i}"; then
            overall_fail=1
            break 2
        fi
    done
done

printf '%s\n' "${REP_RESULTS[@]}" > "${REPORT_DIR}/rep_results.txt"
echo "END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${REPORT_DIR}/END"

if [ "${overall_fail}" -ne 0 ]; then
    echo "=== CAMPAIGN ABORTED (see ${REPORT_DIR}) ===" >&2
    exit 1
fi

# === Aggregate: side-by-side ===
python3 - "${REPORT_DIR}" "${REPS}" "${EXPECTED_COMMIT}" <<PY
import json, statistics, sys, glob, os

report_dir, reps, commit = sys.argv[1], int(sys.argv[2]), sys.argv[3]
metrics = ["mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "output_throughput"]

def collect(side_tag, file_glob):
    per_rep = {}
    for i in range(1, reps + 1):
        rep = f"rep{i}-{side_tag}"
        per_rep[rep] = {}
        for lv in ("l1", "l2", "l3"):
            files = glob.glob(f"{report_dir}/{rep}/bench/{file_glob}-{lv}.json")
            assert len(files) == 1, f"{rep}/{lv}: {files}"
            b = json.load(open(files[0]))
            per_rep[rep][lv] = {k: b[k] for k in metrics}
            per_rep[rep][lv]["request_throughput"] = b["request_throughput"]
    return per_rep

def agg(vals):
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.mean(vals),
        "spread_pct_of_median": round(100.0 * (max(vals) - min(vals)) / statistics.median(vals), 1),
        "n": len(vals),
    }

hcp = collect("hcp", "n2")
pd = collect("pd", "pd")

def agg_side(per_rep):
    return {lv: {k: agg([per_rep[r][lv][k] for r in per_rep]) for k in metrics}
            for lv in ("l1", "l2", "l3")}

agg_hcp, agg_pd = agg_side(hcp), agg_side(pd)

try:
    network = json.load(open(os.path.join(report_dir, "network.json")))
except OSError:
    network = {"warning": "network.json missing"}

comparison = {
    "run_id": os.path.basename(report_dir),
    "kind": "phase3-9 interleaved controlled baseline: HCP N=2 P2P ring vs vLLM PD-disaggregation (white + pearl, 2.5GbE)",
    "config": {
        "git_commit_hcp": commit,
        "vllm_commits": open(os.path.join(report_dir, "vllm_commits.txt")).read().strip().splitlines(),
        "network": network,
        "model": "Qwen2-0.5B, bfloat16, 24 layers, hidden 896, GQA 14/2",
        "hcp_side": "coordinator+worker0+bench on white (RTX 4090 CUDA), worker1 on pearl (RX 9060 XT HIP); QUIC P2P ring, 24 hops/prefill; tch-rs eager",
        "pd_side": "vllm serve prefill on white (CUDA, default compilation) + vllm serve decode on pearl (ROCm, --enforce-eager: hipblaslt FULL cudagraph capture fails on gfx1200) + disagg proxy on white; KV transfer NixlConnector pull mode over NIXL/UCX on the wired link",
        "bench_client": "~/venv-bench/bin/vllm bench serve on white (0.27.1 client)",
        "ladder": "l1: 8 prompts rate=1; l2: 8 prompts mc=2; l3: 16 prompts mc=4; random input 32 output 16 ratio 0.5 seed 42",
        "reps_per_side": reps,
        "interleave": "odd reps run HCP first, even reps run PD first",
        "gates": "HCP rep: 7a assertions (trace hops, reserved==released, metrics failed=0); PD rep: bench 32/32 + sane metrics; one environmental retry each",
    },
    "rep_results": open(os.path.join(report_dir, "rep_results.txt")).read().split(),
    "aggregate_hcp": agg_hcp,
    "aggregate_pd": agg_pd,
    "median_ratio_hcp_over_pd": {
        lv: {k: round(agg_hcp[lv][k]["median"] / agg_pd[lv][k]["median"], 2) for k in metrics}
        for lv in ("l1", "l2", "l3")
    },
}
out = os.path.join(report_dir, "comparison.json")
json.dump(comparison, open(out, "w"), indent=2)
print(f"wrote {out}")

print("")
print("=== SIDE-BY-SIDE (median; min-max spread) ===")
for lv in ("l1", "l2", "l3"):
    print(f"  {lv}:")
    for k in metrics:
        h, p = agg_hcp[lv][k], agg_pd[lv][k]
        print(f"    {k:18s} HCP={h['median']:9.2f} ({h['spread_pct_of_median']:5.1f}%)  PD={p['median']:9.2f} ({p['spread_pct_of_median']:5.1f}%)  ratio={h['median']/p['median']:6.2f}x")
PY

echo ""
echo "=== PHASE-3 9 INTERLEAVED BASELINE PASSED (${REPS} pairs) ==="
echo "Reports: ${REPORT_DIR}"
