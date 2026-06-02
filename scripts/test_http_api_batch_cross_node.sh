#!/bin/bash
# Cross-node HTTP API batch E2E test for Continuous Batching (M13).
# Topology: Mac MPS (coordinator + worker 0) + white RTX 4090 CUDA (worker 1)
# Test: Two sequential requests with overlapping decode iterations.

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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3}"

COORD_PORT=29520
W0_PORT=29521
W1_PORT=29522
HTTP_PORT=8082

RUN_ID="cross-node-batch-$(date +%Y%m%d-%H%M%S)"
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

echo "=== Cross-Node HTTP API Batch E2E Test (M13) ==="
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

# === Generate prompt ===
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
cargo run --bin gen_prompt -- "${MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tail -3
scp "${PROMPT_FILE}" "${GPU_SSH}:~/hcp_prompt_${RUN_ID}.txt" >/dev/null 2>&1 || true

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

# === Test /metrics (baseline) ===
echo ""
echo "=== Test 2: GET /metrics (baseline) ==="
METRICS_BASE=$(curl -s --max-time 10 http://localhost:${HTTP_PORT}/metrics || echo '{"error":"curl failed"}')
echo "$METRICS_BASE" | python3 -m json.tool || true

# === Test 3: Single request (regression) ===
echo ""
echo "=== Test 3: Single Request (regression) ==="
PROMPT_TEXT=$(head -c 200 "${PROMPT_FILE}" | tr '\n' ' ')
RESULT1=$(curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${PROMPT_TEXT}\", \"max_tokens\": ${MAX_NEW_TOKENS}, \"temperature\": 0.0}" || echo '{"error":"curl failed"}')
echo "$RESULT1" | python3 -m json.tool || echo "$RESULT1"
TEXT1=$(echo "$RESULT1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
echo "Generated: '${TEXT1}'"

# === Test 4: Two sequential requests (batch overlap) ===
echo ""
echo "=== Test 4: Two Requests with Batch Overlap ==="
# Send request A
RESULT2A=$(curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"The quick brown fox\", \"max_tokens\": ${MAX_NEW_TOKENS}, \"temperature\": 0.0}" || echo '{"error":"curl failed"}')

# Wait 1s so request A is in active pool, then send request B
sleep 1

RESULT2B=$(curl -s --max-time 300 -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Once upon a time\", \"max_tokens\": ${MAX_NEW_TOKENS}, \"temperature\": 0.0}" || echo '{"error":"curl failed"}')

TEXT2A=$(echo "$RESULT2A" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
TEXT2B=$(echo "$RESULT2B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")

echo "Request A generated: '${TEXT2A}'"
echo "Request B generated: '${TEXT2B}'"

# === Metrics after all requests ===
echo ""
echo "=== Metrics after all requests ==="
sleep 2
METRICS_FINAL=$(curl -s --max-time 10 http://localhost:${HTTP_PORT}/metrics || echo '{"error":"curl failed"}')
echo "$METRICS_FINAL" | python3 -m json.tool || true

TOTAL=$(echo "$METRICS_FINAL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_requests',-1))" 2>/dev/null || echo "-1")
COMPLETED=$(echo "$METRICS_FINAL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('completed_requests',-1))" 2>/dev/null || echo "-1")
FAILED=$(echo "$METRICS_FINAL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('failed_requests',-1))" 2>/dev/null || echo "-1")

echo ""
echo "=== Summary ==="
echo "Total requests: ${TOTAL}"
echo "Completed: ${COMPLETED}"
echo "Failed: ${FAILED}"

# === Validation ===
ERROR1=$(echo "$RESULT1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null || echo "parse-error")
ERROR2A=$(echo "$RESULT2A" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null || echo "parse-error")
ERROR2B=$(echo "$RESULT2B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null || echo "parse-error")

PASS=1
if [ -z "$TEXT1" ] || [ -n "$ERROR1" ]; then
    echo "FAIL: Single request failed or empty output"
    PASS=0
fi
if [ -z "$TEXT2A" ] || [ -n "$ERROR2A" ]; then
    echo "FAIL: Request A failed or empty output"
    PASS=0
fi
if [ -z "$TEXT2B" ] || [ -n "$ERROR2B" ]; then
    echo "FAIL: Request B failed or empty output"
    PASS=0
fi
if [ "${TOTAL}" != "3" ] || [ "${COMPLETED}" != "3" ]; then
    echo "FAIL: Metrics mismatch (expected total=3, completed=3, got total=${TOTAL}, completed=${COMPLETED})"
    PASS=0
fi

if [ "$PASS" -eq 1 ]; then
    echo "=== CROSS-NODE BATCH E2E TEST PASSED ==="
    echo "Reports: ${REPORT_DIR}"
    exit 0
else
    echo "=== CROSS-NODE BATCH E2E TEST FAILED ==="
    echo "--- Coordinator log ---"
    tail -40 "${REPORT_DIR}/coordinator.log" || true
    echo "--- Worker 0 log ---"
    tail -20 "${REPORT_DIR}/worker0.log" || true
    echo "--- Worker 1 log ---"
    tail -20 "${REPORT_DIR}/worker1.log" || true
    exit 1
fi
