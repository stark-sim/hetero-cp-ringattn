#!/bin/bash
# Cross-node 2-domain HTTP API concurrent request end-to-end test.
# Topology: Mac MPS (coordinator + worker 0) + pearl RX 9060 XT HIP (worker 1)
# Submits 2 requests concurrently and verifies both complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

# === Configuration ===
PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_USER="${PEARL_USER:-stark}"
PEARL_SSH="${PEARL_USER}@${PEARL_HOST}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

MAC_ADDR="${MAC_ADDR:-$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')}"
if [ -z "${MAC_ADDR}" ]; then
    echo "ERROR: Could not find local 100.x Tailscale address. Set MAC_ADDR explicitly." >&2
    exit 1
fi
PEARL_ADDR="${PEARL_ADDR:-${PEARL_HOST}}"

MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"

COORD_PORT=29500
W0_PORT=29501
W1_PORT=29502
HTTP_PORT=8080

RUN_ID="cross-node-http-concurrent-2domain-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote_pearl() {
    local command="$1"
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${PEARL_SSH}" "bash -lc $(shell_quote "${command}")"
}

echo "=== Cross-Node 2-Domain HTTP API Concurrent E2E Test ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC=${MAC_ADDR} (MPS + coordinator) | PEARL=${PEARL_ADDR} (HIP)"
echo "SEQ_LEN=${SEQ_LEN}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "Reports: ${REPORT_DIR}"

# === Preflight: build ===
echo "=== Preflight: local build ==="
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -5

echo "=== Preflight: remote build (pearl) ==="
pearl_cmd="cd $(shell_quote "${PEARL_REPO_DIR}") && git pull --ff-only && cd rust && PATH=/home/stark/.cargo/bin:\$PATH cargo build --features tch-backend --release"
run_remote_pearl "${pearl_cmd}" 2>&1 | tail -5

# === Generate prompt ===
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
cargo run --bin gen_prompt -- "${MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tail -3
scp "${PROMPT_FILE}" "${PEARL_SSH}:~/hcp_prompt_${RUN_ID}.txt" >/dev/null 2>&1 || true

# === Cleanup ===
cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
export HCP_TCH_DEVICE=mps

# === Launch Coordinator (Mac, HTTP mode) ===
echo "=== Launching Coordinator (HTTP mode) ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 2 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_PID=$!
echo "Coordinator PID: ${COORD_PID}"
sleep 2

# === Launch Worker 1 (pearl RX 9060 XT, domain 1) ===
echo "=== Launching Worker 1 (pearl, domain 1) ==="
pearl_worker_cmd="cd $(shell_quote "${PEARL_REPO_DIR}") && export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir ~/hetero-cp-ringattn/models/Qwen2-0.5B \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 2"

run_remote_pearl "${pearl_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
sleep 5

# === Launch Worker 0 (Mac MPS, domain 0) ===
echo "=== Launching Worker 0 (Mac, domain 0) ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${PEARL_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    >"${REPORT_DIR}/worker0.log" 2>&1 &
W0_PID=$!
echo "Worker 0 PID: ${W0_PID}"

# === Wait for model load + connection ===
echo "Waiting 25s for workers to connect and load model..."
sleep 25

# === Test 1: GET /health ===
echo ""
echo "=== Test 1: GET /health ==="
HEALTH=$(curl -s --max-time 10 http://localhost:${HTTP_PORT}/health || echo '{"error":"curl failed"}')
echo "$HEALTH" | python3 -m json.tool || true
WORKERS_CONNECTED=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workers_connected',0))" 2>/dev/null || echo "0")

# === Test 2: Concurrent /v1/completions ===
echo ""
echo "=== Test 2: Concurrent POST /v1/completions ==="
PROMPT_TEXT=$(head -c 200 "${PROMPT_FILE}" | tr '\n' ' ')

# Submit 2 requests in background
REQ1_OUTPUT="/tmp/req1_${RUN_ID}.json"
REQ2_OUTPUT="/tmp/req2_${RUN_ID}.json"

curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${PROMPT_TEXT}\", \"max_tokens\": ${MAX_NEW_TOKENS}, \"temperature\": 0.0}" > "${REQ1_OUTPUT}" &
CURL1_PID=$!

curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"The quick brown fox jumps over the lazy dog.\", \"max_tokens\": ${MAX_NEW_TOKENS}, \"temperature\": 0.0}" > "${REQ2_OUTPUT}" &
CURL2_PID=$!

echo "Waiting for both requests to complete..."
wait ${CURL1_PID}
wait ${CURL2_PID}

# Validate req1
GEN1=$(python3 -c "import json; d=json.load(open('${REQ1_OUTPUT}')); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
ERR1=$(python3 -c "import json; d=json.load(open('${REQ1_OUTPUT}')); print(d.get('error',''))" 2>/dev/null || echo "")
echo "Request 1 output:"
cat "${REQ1_OUTPUT}" | python3 -m json.tool || true

# Validate req2
GEN2=$(python3 -c "import json; d=json.load(open('${REQ2_OUTPUT}')); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
ERR2=$(python3 -c "import json; d=json.load(open('${REQ2_OUTPUT}')); print(d.get('error',''))" 2>/dev/null || echo "")
echo "Request 2 output:"
cat "${REQ2_OUTPUT}" | python3 -m json.tool || true

# === Validate ===
PASS=false
if [ -n "$GEN1" ] && [ -n "$GEN2" ] && [ -z "$ERR1" ] && [ -z "$ERR2" ] && [ "$WORKERS_CONNECTED" = "2" ]; then
    PASS=true
fi

if [ "$PASS" = true ]; then
    echo ""
    echo "=== CROSS-NODE 2-DOMAIN CONCURRENT HTTP API TEST PASSED ==="
    echo "Reports: ${REPORT_DIR}"
    exit 0
else
    echo ""
    echo "=== CROSS-NODE 2-DOMAIN CONCURRENT HTTP API TEST FAILED ==="
    if [ -z "$GEN1" ]; then echo "Request 1: no generated text"; fi
    if [ -n "$ERR1" ]; then echo "Request 1 error: ${ERR1}"; fi
    if [ -z "$GEN2" ]; then echo "Request 2: no generated text"; fi
    if [ -n "$ERR2" ]; then echo "Request 2 error: ${ERR2}"; fi
    if [ "$WORKERS_CONNECTED" != "2" ]; then echo "workers_connected=${WORKERS_CONNECTED} (expected 2)"; fi
    echo "--- Coordinator log ---"
    tail -30 "${REPORT_DIR}/coordinator.log" || true
    echo "--- Worker 0 log ---"
    tail -20 "${REPORT_DIR}/worker0.log" || true
    echo "--- Worker 1 log ---"
    tail -20 "${REPORT_DIR}/worker1.log" || true
    exit 1
fi
