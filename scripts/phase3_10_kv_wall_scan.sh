#!/bin/bash
# Phase-3 10: KV capacity-wall scan — HCP N=2 ring vs vLLM PD, Qwen2.5-3B.
#
# Complements phase3-9 (engine-overhead axis): this scan measures the
# capability axis — where each stack hits its KV capacity wall with long
# contexts. 30k-token prompts (rope cap 32768), single-wave bursts at
# mc in {4, 8, 16, 32}:
#   - vLLM PD: all KV lives on the decode node (pearl ~7GB pool) ->
#     preemption onset observable via vllm:num_preemptions deltas.
#   - HCP: KV sharded across the ring (aggregate ~28GB budget) with
#     byte-level fail-closed admission -> rejections visible in the
#     coordinator admission log; --max-batch-size raised to 16 so the
#     batch cap does not mask the KV ceiling.
#
# Usage: bash scripts/phase3_10_kv_wall_scan.sh
# Output: reports/routeb-p3-kvwall-<ts>/ with wall_table.json + raw artifacts.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(cd "${REPO_ROOT}" && git rev-parse HEAD)}"
LEVELS="${LEVELS:-4 8 16 32}"
INPUT_LEN="${INPUT_LEN:-30720}"
MAX_BATCH="${MAX_BATCH:-16}"

RUN_ID="routeb-p3-kvwall-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
shasum -a 256 "$0" > "${REPORT_DIR}/controller_script.sha256"

cleanup() {
    echo "=== Cleanup ==="
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'distributed-rol[e]' || true; pkill -f 'vllm ser[v]e' || true; pkill -f 'disagg_pro[x]y' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 stark@192.168.100.2 "true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 stark@100.111.242.55 "pkill -f 'distributed-rol[e]' || true; pkill -f 'vllm ser[v]e' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Phase-3 10: KV-wall scan, RUN_ID=${RUN_ID} ==="
echo "EXPECTED_COMMIT=${EXPECTED_COMMIT}"
echo "START=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${REPORT_DIR}/START"

echo "=== Preflight: sync + rebuild remotes ==="
for node in "white ${WHITE_SSH}" "pearl stark@100.111.242.55"; do
    set -- ${node}
    remote_head=$(ssh -o ConnectTimeout=30 "$2" "cd ~/hetero-cp-ringattn && git checkout main >/dev/null 2>&1 && git pull --ff-only origin main >/dev/null 2>&1; git rev-parse HEAD")
    echo "  $1 HEAD=${remote_head}"
    [ "${remote_head}" = "${EXPECTED_COMMIT}" ] || { echo "ERROR: $1 commit mismatch" >&2; exit 1; }
    ssh -o ConnectTimeout=120 "$2" "cd ~/hetero-cp-ringattn/rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -1"
done
echo "${EXPECTED_COMMIT}" > "${REPORT_DIR}/git_commit.txt"

echo "=== Network gate ==="
ssh -o ConnectTimeout=20 "${WHITE_SSH}" "ping -c 20 -i 0.2 192.168.100.2" > "${REPORT_DIR}/ping.txt" 2>&1 || true
tail -2 "${REPORT_DIR}/ping.txt"

run_driver() { # side label
    local side=$1
    local state_dir="/tmp/kvwall-${side}-${RUN_ID}"
    local out_dir="${REPORT_DIR}/${side}"
    echo "=== ${side} side: launching driver on white ==="
    ssh -n -f -o ConnectTimeout=20 "${WHITE_SSH}" "mkdir -p ${state_dir} && setsid env LEVELS='${LEVELS}' INPUT_LEN='${INPUT_LEN}' MAX_BATCH='${MAX_BATCH}' bash ~/hetero-cp-ringattn/scripts/phase3_10_kv_wall_${side}_driver.sh ${RUN_ID} > ${state_dir}/driver.log 2>&1 </dev/null"
    local status=""
    for _ in $(seq 1 90); do
        sleep 60
        status=$(ssh -o ConnectTimeout=20 "${WHITE_SSH}" "cat ${state_dir}/STATUS 2>/dev/null" 2>/dev/null || true)
        [ -n "${status}" ] && break
        ssh -o ConnectTimeout=20 "${WHITE_SSH}" "tail -1 ${state_dir}/driver.log 2>/dev/null" 2>/dev/null || true
    done
    mkdir -p "${out_dir}/bench"
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/bench/"*.json "${out_dir}/bench/" 2>/dev/null || true
    for f in driver.log coordinator.log prefill.log decode.log proxy.log worker0.log worker1.log admissions.log preemptions.jsonl kv-pool.txt; do
        scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/${f}" "${out_dir}/${f}" 2>/dev/null || true
    done
    scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${state_dir}/metrics-mc4.json" "${out_dir}/" 2>/dev/null || true
    echo "  ${side} STATUS='${status}'"
    [ "${status}" = "DONE" ]
}

overall=0
run_driver hcp || overall=1
run_driver pd || overall=1

echo "END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${REPORT_DIR}/END"
[ "${overall}" = 0 ] || { echo "=== SCAN INCOMPLETE (see ${REPORT_DIR}) ===" >&2; exit 1; }

python3 - "${REPORT_DIR}" <<PY
import json, glob, os, re, sys

report_dir = sys.argv[1]

def bench_summary(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(report_dir, pattern))):
        mc = int(re.search(r"mc(\d+)", f).group(1))
        b = json.load(open(f))
        out[mc] = {
            "completed": b.get("completed"),
            "num_prompts": b.get("num_prompts"),
            "failed": b.get("num_prompts", 0) - b.get("completed", 0),
            "duration_s": round(b.get("duration", 0), 1),
            "mean_ttft_ms": round(b.get("mean_ttft_ms") or 0, 1),
            "p99_ttft_ms": round(b.get("p99_ttft_ms") or 0, 1),
            "mean_tpot_ms": round(b.get("mean_tpot_ms") or 0, 1),
            "output_throughput": round(b.get("output_throughput") or 0, 2),
        }
    return out

hcp = bench_summary("hcp/bench/hcp-mc*.json")
pd = bench_summary("pd/bench/pd-mc*.json")

preemptions = {}
try:
    for line in open(os.path.join(report_dir, "pd", "preemptions.jsonl")):
        d = json.loads(line)
        preemptions[d["mc"]] = d["preemptions_after"] - d["preemptions_before"]
except OSError:
    pass

admissions = {"accepted": 0, "rejected": 0}
try:
    for line in open(os.path.join(report_dir, "hcp", "admissions.log")):
        if "status=accepted" in line: admissions["accepted"] += 1
        elif "status=rejected" in line: admissions["rejected"] += 1
except OSError:
    pass

kv_pool = ""
try:
    kv_pool = open(os.path.join(report_dir, "pd", "kv-pool.txt")).read().strip()
except OSError:
    pass

table = {
    "run_id": os.path.basename(report_dir),
    "kind": "phase3-10 KV capacity-wall scan: HCP N=2 ring vs vLLM PD, Qwen2.5-3B, 30k-token prompts, single-wave mc sweep",
    "workload": {"input_len": 30720, "output_len": 16, "range_ratio": 0.05, "seed": 42,
                 "levels_mc": [4, 8, 16, 32], "kv_per_req_bytes": "~1.2GB (36 layers x 2 x 2 heads x 128 dim x bf16 x ~30.7k tokens)"},
    "vllm_pd_kv_pool_log": kv_pool,
    "hcp_side": {"levels": hcp, "admissions": admissions,
                  "note": "fail-closed byte-level admission; in-flight KV sharded across ring (budgets ~17.9GB white + ~10.4GB pearl)"},
    "pd_side": {"levels": pd, "preemptions_delta_per_level": preemptions,
                 "note": "all KV on decode node (pearl); preemption = KV pool exhausted, requests recomputed"},
}
out = os.path.join(report_dir, "wall_table.json")
json.dump(table, open(out, "w"), indent=2)
print(f"wrote {out}")

print("")
print("=== WALL TABLE ===")
print(f"{'mc':>4} | {'HCP done/total':>14} {'HCP TTFT p99 s':>14} | {'PD done/total':>13} {'PD TTFT p99 s':>13} {'PD preempts':>11}")
for mc in sorted(set(hcp) | set(pd)):
    h, p = hcp.get(mc, {}), pd.get(mc, {})
    print(f"{mc:>4} | {str(h.get('completed'))+'/'+str(h.get('num_prompts')):>14} {h.get('p99_ttft_ms',0)/1000:>13.1f}s | {str(p.get('completed'))+'/'+str(p.get('num_prompts')):>13} {p.get('p99_ttft_ms',0)/1000:>12.1f}s {preemptions.get(mc, 0):>11}")
print(f"HCP admissions: {admissions}")
PY

echo ""
echo "=== PHASE-3 10 KV-WALL SCAN DONE ==="
echo "Reports: ${REPORT_DIR}"
