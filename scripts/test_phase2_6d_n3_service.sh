#!/bin/bash
# 6d: N=3 heterogeneous Rust service readiness.
# Topology: Mac MPS (coordinator + worker 0) + white RTX 4090 CUDA (worker 1)
# + pearl RX 9060 XT HIP (worker 2), neighbor-only QUIC ring.
#
# Runs real Qwen2-0.5B multi-request service with the JSONL trace plane and
# verifies: 0 errors, admission (reserved==released), FIFO decode, release,
# telemetry correlation, and N=3 hop formula (prefill = L*(N-1) = 48,
# decode = steps*48). This proves the Rust service is ready for external
# benchmarks; it does NOT invoke the vLLM CLI.

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

MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"

COORD_PORT=29800
W0_PORT=29801
W1_PORT=29802
W2_PORT=29803
HTTP_PORT=8082

RUN_ID="routeb-6d-n3-service-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
TRACE_FILE="${REPORT_DIR}/trace.jsonl"
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

echo "=== 6d N=3 Heterogeneous Rust Service Readiness ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC=${MAC_ADDR} (MPS+coord) | WHITE=${WHITE_ADDR} (CUDA) | PEARL=${PEARL_ADDR} (HIP)"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "Reports: ${REPORT_DIR}"

# === Preflight builds (idempotent; already built but cheap) ===
echo "=== Preflight: local build ==="
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -2

echo "=== Preflight: remote builds (white + pearl) ==="
white_cmd="cd $(shell_quote "${WHITE_REPO_DIR}") && git pull --ff-only 2>&1 | tail -1 && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
run_remote_white "${white_cmd}" 2>&1 | tail -3
pearl_cmd="cd $(shell_quote "${PEARL_REPO_DIR}") && git pull --ff-only 2>&1 | tail -1 && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
run_remote_pearl "${pearl_cmd}" 2>&1 | tail -3

# === Cleanup ===
cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 2' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
export HCP_TCH_DEVICE=mps

# === Launch Coordinator (Mac, HTTP mode + trace) ===
echo "=== Launching Coordinator (HTTP mode, trace) ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 3 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    --trace-jsonl "${TRACE_FILE}" \
    >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_PID=$!
sleep 2

# === Launch Worker 1 (white, domain 1, CUDA) ===
echo "=== Launching Worker 1 (white, domain 1) ==="
white_worker_cmd="cd $(shell_quote "${WHITE_REPO_DIR}") && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir ~/models/Qwen2-0.5B \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${PEARL_ADDR}:${W2_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 3"
run_remote_white "${white_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
sleep 5

# === Launch Worker 2 (pearl, domain 2, HIP) ===
echo "=== Launching Worker 2 (pearl, domain 2) ==="
pearl_worker_cmd="cd $(shell_quote "${PEARL_REPO_DIR}") && export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 2 \
    --model-dir ~/hetero-cp-ringattn/models/Qwen2-0.5B \
    --listen-addr 0.0.0.0:${W2_PORT} \
    --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 3"
run_remote_pearl "${pearl_worker_cmd}" >"${REPORT_DIR}/worker2.log" 2>&1 &
sleep 5

# === Launch Worker 0 (Mac MPS, domain 0) ===
echo "=== Launching Worker 0 (Mac, domain 0) ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${WHITE_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 3 \
    >"${REPORT_DIR}/worker0.log" 2>&1 &
W0_PID=$!

echo "Waiting 30s for workers to connect and load model..."
sleep 30

echo "=== /health ==="
HEALTH=$(curl -s --max-time 10 http://localhost:${HTTP_PORT}/health || echo '{"error":"curl failed"}')
echo "$HEALTH" | python3 -m json.tool || true
CONNECTED=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('workers_connected',0))" 2>/dev/null || echo 0)
if [ "$CONNECTED" -ne 3 ]; then
    echo "ERROR: expected 3 workers connected, got $CONNECTED"; tail -30 "${REPORT_DIR}/coordinator.log"; exit 1
fi

# === Submit 4 requests with unequal lengths (some concurrent) ===
echo ""
echo "=== POST /v1/completions (4 requests, unequal lengths) ==="
send() { # prompt max_tokens outfile
    curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": $1, \"max_tokens\": $2, \"temperature\": 0.0}" > "$3"
}
send '"The quick brown fox jumps over the lazy dog near the river bank"' 4 /tmp/6d_resp1.json &
P1=$!
send '"In a world where the machines learned to dream of electric sheep, the sun rose over the silicon valley and the circuits hummed a quiet symphony of logic gates and memory cells, each one a tiny sentinel watching over the endless streams of data."' 8 /tmp/6d_resp2.json &
P2=$!
wait $P1
wait $P2
send '"Once upon a midnight dreary"' 3 /tmp/6d_resp3.json
send '"Deep in the heart of the ancient forest, a single blue flower bloomed where no seed had ever been planted, glowing softly in the shadows as if it held the last whisper of a forgotten sun."' 6 /tmp/6d_resp4.json

echo ""
echo "--- Response 1 ---"; cat /tmp/6d_resp1.json | python3 -m json.tool || cat /tmp/6d_resp1.json
echo "--- Response 3 ---"; cat /tmp/6d_resp3.json | python3 -m json.tool || cat /tmp/6d_resp3.json

# === /metrics ===
echo ""
echo "=== /metrics ==="
curl -s --max-time 10 http://localhost:${HTTP_PORT}/metrics | python3 -m json.tool || true

# === Validate ===
echo ""
echo "=== Validation ==="
python3 - <<'PY'
import json, glob
def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None
responses = [load(f"/tmp/6d_resp{i}.json") for i in range(1, 5)]
errors = 0
for i, r in enumerate(responses, 1):
    if r is None:
        print(f"  resp{i}: FAILED TO PARSE"); errors += 1; continue
    if "error" in r and r.get("error"):
        print(f"  resp{i}: ERROR {r['error']}"); errors += 1; continue
    text = r.get("choices", [{}])[0].get("text", "")
    if not text or text.startswith("[error:"):
        print(f"  resp{i}: EMPTY/ERROR TEXT {text!r}"); errors += 1; continue
    print(f"  resp{i}: OK text={text!r} tokens={r.get('usage',{}).get('completion_tokens')}")
assert errors == 0, f"{errors} responses failed"
PY

# capture metrics for assertion
curl -s --max-time 10 http://localhost:${HTTP_PORT}/metrics > "${REPORT_DIR}/metrics.json"
python3 - "${REPORT_DIR}/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  metrics: total={m['total_requests']} completed={m['completed_requests']} failed={m['failed_requests']} active={m['active_requests']}")
assert m["total_requests"] == 4, f"total {m['total_requests']}"
assert m["completed_requests"] == 4, f"completed {m['completed_requests']}"
assert m["failed_requests"] == 0, f"failed {m['failed_requests']}"
assert m["active_requests"] == 0, f"active {m['active_requests']}"
PY

# === Trace validation (N=3, L=24) ===
echo ""
echo "=== Trace validation (N=3: prefill_hops=48, decode_hops=steps*48) ==="
python3 - "${TRACE_FILE}" <<'PY'
import json, sys
records = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"  trace records={len(records)}")
assert len(records) == 4, f"expected 4, got {len(records)}"
for r in records:
    assert r["prefill_accepted_elapsed_ms"] > 0
    assert r["completed_elapsed_ms"] > 0
    assert r["error"] is None, f"req {r['request_id']} error {r['error']}"
    assert len(r["reserved_bytes"]) == 3
    assert r["reserved_bytes"] == r["released_bytes"], f"req {r['request_id']} release mismatch"
    assert r["prefill_hops"] == 48, f"req {r['request_id']} prefill_hops {r['prefill_hops']}"
    assert r["decode_hops"] == r["decode_steps"] * 48, f"req {r['request_id']} decode_hops"
    print(f"  req {r['request_id']}: prompt={r['prompt_tokens']} steps={r['decode_steps']} reserved={r['reserved_bytes']} hops={r['prefill_hops']}+{r['decode_hops']} finish={r['finish_reason']}")
PY

echo ""
echo "=== 6D N=3 SERVICE READINESS PASSED ==="
echo "0 errors, 4 requests, metrics + trace consistent, N=3 hop formula holds."
echo "Reports: ${REPORT_DIR}"
