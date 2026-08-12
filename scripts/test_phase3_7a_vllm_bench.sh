#!/bin/bash
# Phase-3 7a: drive the HCP Rust service with the real `vllm bench serve`
# black-box client (white venv-bench), N=2 then N=3.
#
# Topology N=2: white (worker 0, CUDA) + pearl (worker 1, HIP),
#   coordinator on Mac (control plane only, no model compute).
# Topology N=3: Mac MPS worker 0 + white worker 1 + pearl worker 2
#   (same as phase-2 6d).
#
# Ladder per topology (client on white -> Mac Tailscale HTTP :8082):
#   L1: request-rate=1,   num-prompts=8
#   L2: request-rate=inf, max-concurrency=2, num-prompts=8
#   L3: request-rate=inf, max-concurrency=4, num-prompts=16
#
# Validates per topology: bench JSON failed==0 with TTFT/TPOT/ITL/E2EL
# present, HCP trace record counts, reserved==released, hop formulas
# (prefill = L*(N-1), decode = steps*L*(N-1)), and /metrics totals.
# This is a service-metric baseline, NOT a performance claim; no vLLM
# baseline comparison is performed in this node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

# === Configuration ===
WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_USER="${WHITE_USER:-stark}"
WHITE_SSH="${WHITE_USER}@${WHITE_HOST}"
WHITE_REPO_DIR="${WHITE_REPO_DIR:-hetero-cp-ringattn}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_USER="${PEARL_USER:-stark}"
PEARL_SSH="${PEARL_USER}@${PEARL_HOST}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

MAC_ADDR="${MAC_ADDR:-$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')}"
if [ -z "${MAC_ADDR}" ]; then
    echo "ERROR: Could not find local 100.x Tailscale address. Set MAC_ADDR explicitly." >&2
    exit 1
fi
WHITE_ADDR="${WHITE_ADDR:-${WHITE_HOST}}"
PEARL_ADDR="${PEARL_ADDR:-${PEARL_HOST}}"
# white/pearl sit on the same LAN; using LAN addresses keeps the N=2 service
# path (coordinator, ring, bench) entirely off the flaky cross-internet
# Tailscale segment from the Mac.
WHITE_LAN="${WHITE_LAN:-192.168.8.172}"
PEARL_LAN="${PEARL_LAN:-192.168.8.176}"

MAC_MODEL_DIR="${MAC_MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
WHITE_MODEL_DIR="${WHITE_MODEL_DIR:-~/models/Qwen2-0.5B}"
PEARL_MODEL_DIR="${PEARL_MODEL_DIR:-~/hetero-cp-ringattn/models/Qwen2-0.5B}"

BENCH_VLLM="${BENCH_VLLM:-~/venv-bench/bin/vllm}"
BENCH_MODEL_NAME="${BENCH_MODEL_NAME:-hcp-qwen2-0.5b}"
INPUT_LEN="${INPUT_LEN:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"

COORD_PORT=29800
W0_PORT=29801
W1_PORT=29802
W2_PORT=29803
HTTP_PORT=8082

RUN_ID="routeb-p3-bench-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
REMOTE_RESULT_DIR="/tmp/hcp-bench-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote_white() {
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${WHITE_SSH}" "bash -lc $(shell_quote "$1")"
}

run_remote_pearl() {
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${PEARL_SSH}" "bash -lc $(shell_quote "$1")"
}

echo "=== Phase-3 7a: vllm bench serve black-box vs HCP ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC=${MAC_ADDR} (coord) | WHITE=${WHITE_ADDR} (CUDA+bench) | PEARL=${PEARL_ADDR} (HIP)"
echo "Reports: ${REPORT_DIR}"

# === Preflight builds ===
echo "=== Preflight: local build ==="
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -2

echo "=== Preflight: remote builds (white + pearl) ==="
# Remote repos must track main explicitly: they were found checked out on the
# stale codex branch, silently missing main-only commits ("Already up to date").
white_build="cd $(shell_quote "${WHITE_REPO_DIR}") && git checkout main 2>&1 | tail -1 && git pull --ff-only origin main 2>&1 | tail -1 && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
run_remote_white "${white_build}" 2>&1 | tail -3
pearl_build="cd $(shell_quote "${PEARL_REPO_DIR}") && git checkout main 2>&1 | tail -1 && git pull --ff-only origin main 2>&1 | tail -1 && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
run_remote_pearl "${pearl_build}" 2>&1 | tail -3

echo "=== Preflight: bench client on white ==="
run_remote_white "${BENCH_VLLM} --version && mkdir -p ${REMOTE_RESULT_DIR}"

# === Cleanup ===
# N=2 runs coordinator + worker on white as detached daemons (setsid), so the
# pkill patterns must cover both roles; N=3 coordinator/Mac worker are local jobs.
cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
export HCP_TCH_DEVICE=mps

# === Stack launchers ===
launch_coordinator() { # num_domains trace_file log_file
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MAC_MODEL_DIR}" \
        --num-domains "$1" \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        --http-addr "0.0.0.0:${HTTP_PORT}" \
        --trace-jsonl "$2" \
        >"$3" 2>&1 &
    sleep 2
}

launch_white_worker_full() { # domain_id listen_port next_peer_addr next_peer_port num_domains log_file
    local cmd="cd $(shell_quote "${WHITE_REPO_DIR}") && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
      ./rust/target/release/hcp-ringattn-rust \
        --distributed-role worker \
        --domain-id $1 \
        --model-dir ${WHITE_MODEL_DIR} \
        --listen-addr 0.0.0.0:$2 \
        --next-peer-addr $3:$4 \
        --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
        --num-domains $5"
    run_remote_white "${cmd}" >"$6" 2>&1 &
    sleep 5
}

launch_pearl_worker_full() { # domain_id listen_port next_peer_addr next_peer_port num_domains log_file
    local cmd="cd $(shell_quote "${PEARL_REPO_DIR}") && export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
      ./rust/target/release/hcp-ringattn-rust \
        --distributed-role worker \
        --domain-id $1 \
        --model-dir ${PEARL_MODEL_DIR} \
        --listen-addr 0.0.0.0:$2 \
        --next-peer-addr $3:$4 \
        --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
        --num-domains $5"
    run_remote_pearl "${cmd}" >"$6" 2>&1 &
    sleep 5
}

wait_health() { # expected_workers
    echo "Waiting 30s for workers to connect and load model..."
    sleep 30
    local health
    health=$(curl -s --max-time 10 "http://localhost:${HTTP_PORT}/health" || echo '{}')
    echo "$health"
    local connected
    connected=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('workers_connected',0))" 2>/dev/null || echo 0)
    if [ "$connected" -ne "$1" ]; then
        echo "ERROR: expected $1 workers connected, got $connected" >&2
        exit 1
    fi
}

stop_stack() {
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    sleep 3
}

# === N=2 note ===
# The N=2 phase is driven entirely on white by scripts/phase3_7a_n2_driver.sh
# (setsid daemons on the white/pearl LAN). The Mac only polls STATUS and
# fetches artifacts, so Mac-side network drops cannot abort the bench.

# === Bench ladder ===
run_bench() { # label num_prompts rate max_concurrency base_url
    local label=$1 np=$2 rate=$3 mc=$4 base_url=$5
    local mc_arg=""
    if [ "$mc" -gt 0 ]; then
        mc_arg="--max-concurrency $mc"
    fi
    echo "=== bench ${label}: num_prompts=${np} rate=${rate} max_concurrency=${mc} url=${base_url} ==="
    run_remote_white "${BENCH_VLLM} bench serve \
        --backend openai \
        --base-url ${base_url} \
        --endpoint /v1/completions \
        --model ${BENCH_MODEL_NAME} \
        --tokenizer ${WHITE_MODEL_DIR} \
        --dataset-name random \
        --random-input-len ${INPUT_LEN} \
        --random-output-len ${OUTPUT_LEN} \
        --random-range-ratio 0.5 \
        --num-prompts ${np} \
        --request-rate ${rate} \
        ${mc_arg} \
        --seed 42 \
        --save-result \
        --result-dir ${REMOTE_RESULT_DIR} \
        --result-filename ${label}.json" 2>&1 | tail -40
}

run_ladder() { # phase_tag base_url
    run_bench "$1-l1" 8 1 0 "$2"
    run_bench "$1-l2" 8 inf 2 "$2"
    run_bench "$1-l3" 16 inf 4 "$2"
}

fetch_results() { # phase_tag
    mkdir -p "${REPORT_DIR}/bench-$1"
    scp -q -o ConnectTimeout=15 "${WHITE_SSH}:${REMOTE_RESULT_DIR}/$1-*.json" "${REPORT_DIR}/bench-$1/" || true
    run_remote_white "rm -f ${REMOTE_RESULT_DIR}/$1-*.json"
}

# === Validation ===
validate_phase() { # phase_tag expected_total_requests hops_per_iter trace_file metrics_source(local|white)
    local tag=$1 total=$2 hops=$3 trace=$4 msrc=$5
    echo ""
    echo "=== Validation ${tag} ==="
    if [ "$msrc" = "white" ]; then
        run_remote_white "curl -s --max-time 10 http://127.0.0.1:${HTTP_PORT}/metrics" > "${REPORT_DIR}/metrics-${tag}.json"
    elif [ "$msrc" = "file" ]; then
        : # metrics file already fetched into REPORT_DIR by the caller
    else
        curl -s --max-time 10 "http://localhost:${HTTP_PORT}/metrics" > "${REPORT_DIR}/metrics-${tag}.json"
    fi
    python3 - "${REPORT_DIR}" "${tag}" "${total}" "${hops}" "${trace}" <<'PY'
import json, sys, glob

report_dir, tag, total, hops, trace_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]

# --- bench JSON plane ---
bench_files = sorted(glob.glob(f"{report_dir}/bench-{tag}/{tag}-*.json"))
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
    print(f"  {label}: completed={completed} ttft={b['mean_ttft_ms']:.1f}ms "
          f"tpot={b['mean_tpot_ms']:.2f}ms itl={b['mean_itl_ms']:.2f}ms "
          f"req_thru={b['request_throughput']:.3f}")
assert prompts_seen == total, f"bench completed total {prompts_seen} != {total}"

# --- HCP trace plane ---
records = [json.loads(l) for l in open(trace_path) if l.strip()]
assert len(records) == total, f"trace records {len(records)} != {total}"
# Concurrent requests complete out of order; trace rows are written at
# completion time, so only the id SET must be exact.
ids = sorted(r["request_id"] for r in records)
assert ids == list(range(1, total + 1)), f"request_id set broken: {ids[:5]}...{ids[-3:]}"
for r in records:
    assert r["error"] is None, f"req {r['request_id']} error {r['error']}"
    assert r["reserved_bytes"] == r["released_bytes"], f"req {r['request_id']} release mismatch"
    assert r["prefill_hops"] == hops, f"req {r['request_id']} prefill_hops {r['prefill_hops']} != {hops}"
    assert r["decode_hops"] == r["decode_steps"] * hops, f"req {r['request_id']} decode_hops"
print(f"  trace: {len(records)} records, ids 1..{total}, reserved==released, hops ok ({hops}/iter)")

# --- metrics plane ---
m = json.load(open(f"{report_dir}/metrics-{tag}.json"))
assert m["total_requests"] == total, f"metrics total {m['total_requests']} != {total}"
assert m["completed_requests"] == total, f"metrics completed {m['completed_requests']}"
assert m["failed_requests"] == 0, f"metrics failed {m['failed_requests']}"
assert m["active_requests"] == 0, f"metrics active {m['active_requests']}"
print(f"  metrics: total=completed={total} failed=0 active=0")
PY
}

# =====================================================================
# Phase N=2: white (worker 0, CUDA) + pearl (worker 1, HIP),
# coordinator + bench client also on white => entire service path on the
# white/pearl LAN. The white-side driver (phase3_7a_n2_driver.sh) runs
# the stack and ladder; the Mac polls STATUS and fetches artifacts.
# =====================================================================
PHASES="${PHASES:-n2 n3}"
if [[ " ${PHASES} " == *" n2 "* ]]; then
echo ""
echo "########## Phase N=2: white + pearl (LAN-resident, white-driven) ##########"
N2_STATE_DIR="/tmp/hcp-n2-${RUN_ID}"
# ssh -n -f returns right after launching the driver; the driver then owns
# the full N=2 lifecycle on white even if the Mac goes away.
ssh -n -f -o ConnectTimeout=20 "${WHITE_SSH}" "mkdir -p ${N2_STATE_DIR} && setsid bash $(shell_quote "${WHITE_REPO_DIR}")/scripts/phase3_7a_n2_driver.sh ${RUN_ID} > ${N2_STATE_DIR}/driver.log 2>&1 </dev/null"
echo "N=2 driver launched on white; polling STATUS (Mac network drops tolerated)..."

n2_status=""
for i in $(seq 1 90); do
    sleep 60
    n2_status=$(ssh -o ConnectTimeout=20 "${WHITE_SSH}" "cat ${N2_STATE_DIR}/STATUS 2>/dev/null" 2>/dev/null || true)
    if [ -n "${n2_status}" ]; then
        break
    fi
    echo "  ... still running (poll $i)"
done
if [ "${n2_status}" != "DONE" ]; then
    echo "ERROR: N=2 driver finished with STATUS='${n2_status}'" >&2
    ssh -o ConnectTimeout=20 "${WHITE_SSH}" "tail -40 ${N2_STATE_DIR}/driver.log" 2>/dev/null || true
    exit 1
fi
echo "N=2 driver DONE; fetching artifacts..."

mkdir -p "${REPORT_DIR}/bench-n2"
scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${N2_STATE_DIR}/bench/"*.json "${REPORT_DIR}/bench-n2/"
scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${N2_STATE_DIR}/trace-n2.jsonl" "${REPORT_DIR}/trace-n2.jsonl"
scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${N2_STATE_DIR}/metrics-n2.json" "${REPORT_DIR}/metrics-n2.json"
scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${N2_STATE_DIR}/coordinator-n2.log" "${REPORT_DIR}/coordinator-n2.log"
scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${N2_STATE_DIR}/driver.log" "${REPORT_DIR}/driver-n2.log" || true
scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${N2_STATE_DIR}/worker0-white-n2.log" "${REPORT_DIR}/worker0-white-n2.log" || true
scp -q -o ConnectTimeout=20 "${WHITE_SSH}:${N2_STATE_DIR}/worker1-pearl-n2.log" "${REPORT_DIR}/worker1-pearl-n2.log" || true
validate_phase n2 32 24 "${REPORT_DIR}/trace-n2.jsonl" file
stop_stack
echo "=== N=2 PHASE PASSED ==="
fi

# =====================================================================
# Phase N=3: Mac MPS worker 0 + white worker 1 + pearl worker 2
# =====================================================================
if [[ " ${PHASES} " == *" n3 "* ]]; then
echo ""
echo "########## Phase N=3: Mac + white + pearl ##########"
TRACE_N3="${REPORT_DIR}/trace-n3.jsonl"
launch_coordinator 3 "${TRACE_N3}" "${REPORT_DIR}/coordinator-n3.log"
launch_white_worker_full 1 "${W1_PORT}" "${PEARL_ADDR}" "${W2_PORT}" 3 "${REPORT_DIR}/worker1-white-n3.log"
launch_pearl_worker_full 2 "${W2_PORT}" "${MAC_ADDR}" "${W0_PORT}" 3 "${REPORT_DIR}/worker2-pearl-n3.log"
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MAC_MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${WHITE_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 3 \
    >"${REPORT_DIR}/worker0-mac-n3.log" 2>&1 &
wait_health 3

run_ladder n3 "http://${MAC_ADDR}:${HTTP_PORT}"
fetch_results n3
validate_phase n3 32 48 "${TRACE_N3}" local
stop_stack
echo "=== N=3 PHASE PASSED ==="
fi

echo ""
echo "=== PHASE-3 7a VLLM BENCH PASSED ==="
echo "Phases [${PHASES}]: vllm bench serve ladder 0 failures, HCP trace/metrics consistent."
echo "Reports: ${REPORT_DIR}"
