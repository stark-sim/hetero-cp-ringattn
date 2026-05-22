#!/bin/bash
# Cross-node HTTP API concurrent request end-to-end test.
# Topology: Mac MPS (coordinator + worker 0) + white RTX 4090 CUDA (worker 1)
# Submits 2 requests concurrently and verifies both complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

# === Configuration ===
GPU_HOST="${GPU_HOST:-100.64.0.2}"
GPU_USER="${GPU_USER:-stark}"
GPU_SSH="${GPU_USER}@${GPU_HOST}"
GPU_REPO_DIR="${GPU_REPO_DIR:-hetero-cp-ringattn}"

MAC_ADDR="${MAC_ADDR:-$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')}"
if [ -z "${MAC_ADDR}" ]; then
    echo "ERROR: Could not find local 100.x Tailscale address. Set MAC_ADDR explicitly." >&2
    exit 1
fi
GPU_ADDR="${GPU_ADDR:-${GPU_HOST}}"

MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"

COORD_PORT=29500
W0_PORT=29501
W1_PORT=29502
HTTP_PORT=8080

RUN_ID="cross-node-http-concurrent-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote() {
    local command="$1"
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${GPU_SSH}" "bash -lc $(shell_quote "${command}")"
}

echo "=== Cross-Node HTTP API Concurrent E2E Test ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC=${MAC_ADDR} (MPS + coordinator) | GPU=${GPU_ADDR} (CUDA)"
echo "SEQ_LEN=${SEQ_LEN}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "Reports: ${REPORT_DIR}"

# === Preflight: build ===
echo "=== Preflight: local build ==="
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -5

echo "=== Preflight: remote build ==="
remote_cmd="cd $(shell_quote "${GPU_REPO_DIR}") && git pull --ff-only && cd rust && PATH=/home/stark/.cargo/bin:\$PATH cargo build --features tch-backend --release"
run_remote "${remote_cmd}" 2>&1 | tail -5

# === Generate prompts ===
echo "=== Generating prompts (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
cargo run --bin gen_prompt -- "${MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tail -3
scp "${PROMPT_FILE}" "${GPU_SSH}:~/hcp_prompt_${RUN_ID}.txt" >/dev/null 2>&1

# === Cleanup ===
cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${GPU_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
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

# === Launch Worker 1 (white RTX 4090, domain 1) ===
echo "=== Launching Worker 1 (GPU, domain 1) ==="
remote_worker_cmd="cd $(shell_quote "${GPU_REPO_DIR}") && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir ~/models/Qwen2-0.5B \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 2"

run_remote "${remote_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
sleep 5

# === Launch Worker 0 (Mac MPS, domain 0) ===
echo "=== Launching Worker 0 (Mac, domain 0) ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${GPU_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    >"${REPORT_DIR}/worker0.log" 2>&1 &
W0_PID=$!
echo "Worker 0 PID: ${W0_PID}"

# === Wait for model load + connection ===
echo "Waiting 20s for workers to connect and load model..."
sleep 20

# === Test /health ===
echo ""
echo "=== Test 1: GET /health ==="
HEALTH=$(curl -s --max-time 10 http://localhost:${HTTP_PORT}/health || echo '{"error":"curl failed"}')
echo "$HEALTH" | python3 -m json.tool || true

# === Test /metrics (initial) ===
echo ""
echo "=== Test 2: GET /metrics (before requests) ==="
curl -s --max-time 10 http://localhost:${HTTP_PORT}/metrics | python3 -m json.tool || true

# === Submit 2 concurrent /v1/completions requests ===
echo ""
echo "=== Test 3: POST /v1/completions (2 concurrent requests) ==="
PROMPT_TEXT=$(head -c 200 "${PROMPT_FILE}" | tr '\n' ' ')

curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${PROMPT_TEXT}\", \"max_tokens\": ${MAX_NEW_TOKENS}, \"temperature\": 0.0}" \
    > /tmp/resp1_${RUN_ID}.json &
PID1=$!

curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${PROMPT_TEXT}\", \"max_tokens\": ${MAX_NEW_TOKENS}, \"temperature\": 0.0}" \
    > /tmp/resp2_${RUN_ID}.json &
PID2=$!

echo "Submitted 2 requests concurrently, waiting for completion..."
wait $PID1
wait $PID2

echo ""
echo "--- Response 1 ---"
cat /tmp/resp1_${RUN_ID}.json | python3 -m json.tool || cat /tmp/resp1_${RUN_ID}.json

echo ""
echo "--- Response 2 ---"
cat /tmp/resp2_${RUN_ID}.json | python3 -m json.tool || cat /tmp/resp2_${RUN_ID}.json

GENERATED1=$(python3 -c "import json; d=json.load(open('/tmp/resp1_${RUN_ID}.json')); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
GENERATED2=$(python3 -c "import json; d=json.load(open('/tmp/resp2_${RUN_ID}.json')); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")

echo ""
echo "Generated text 1: '${GENERATED1}'"
echo "Generated text 2: '${GENERATED2}'"

# === Test /metrics (after) ===
echo ""
echo "=== Test 4: GET /metrics (after requests) ==="
curl -s --max-time 10 http://localhost:${HTTP_PORT}/metrics | python3 -m json.tool || true

# Validate
PASS1=false
PASS2=false

ERROR1=$(python3 -c "import json; d=json.load(open('/tmp/resp1_${RUN_ID}.json')); print(d.get('error',''))" 2>/dev/null || echo "parse_error")
ERROR2=$(python3 -c "import json; d=json.load(open('/tmp/resp2_${RUN_ID}.json')); print(d.get('error',''))" 2>/dev/null || echo "parse_error")

if [ -n "$GENERATED1" ] && [ -z "$ERROR1" ]; then
    PASS1=true
fi
if [ -n "$GENERATED2" ] && [ -z "$ERROR2" ]; then
    PASS2=true
fi

if [ "$PASS1" = true ] && [ "$PASS2" = true ]; then
    echo ""
    echo "=== CROSS-NODE CONCURRENT HTTP API E2E TEST PASSED ==="
    echo "Both requests completed successfully."
    echo "Reports: ${REPORT_DIR}"
    exit 0
else
    echo ""
    echo "=== CROSS-NODE CONCURRENT HTTP API E2E TEST FAILED ==="
    if [ "$PASS1" != true ]; then echo "Request 1 failed"; fi
    if [ "$PASS2" != true ]; then echo "Request 2 failed"; fi
    echo "--- Coordinator log ---"
    tail -30 "${REPORT_DIR}/coordinator.log" || true
    echo "--- Worker 0 log ---"
    tail -20 "${REPORT_DIR}/worker0.log" || true
    echo "--- Worker 1 log ---"
    tail -20 "${REPORT_DIR}/worker1.log" || true
    exit 1
fi
