#!/bin/bash
# Local HTTP API batch E2E test for Continuous Batching (M13).
# Launches coordinator (HTTP mode) + 2-domain worker, then sends multiple requests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"

COORD_PORT=29510
W0_PORT=29511
W1_PORT=29512
HTTP_PORT=8081

echo "=== HTTP API Batch E2E Test (M13 Continuous Batching) ==="

cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
export HCP_TCH_DEVICE=mps

# Launch coordinator (HTTP mode — no prompt source given)
echo "=== Launching Coordinator (HTTP mode) ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 2 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    > /tmp/hcp_http_batch_coord.log 2>&1 &
COORD_PID=$!
echo "Coordinator PID: ${COORD_PID}"
sleep 2

# Launch Worker 0 (domain 0)
echo "=== Launching Worker 0 ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "127.0.0.1:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    > /tmp/hcp_http_batch_w0.log 2>&1 &
W0_PID=$!
echo "Worker 0 PID: ${W0_PID}"
sleep 2

# Launch Worker 1 (domain 1)
echo "=== Launching Worker 1 ==="
"${BINARY}" --distributed-role worker \
    --domain-id 1 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W1_PORT}" \
    --next-peer-addr "127.0.0.1:${W0_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    > /tmp/hcp_http_batch_w1.log 2>&1 &
W1_PID=$!
echo "Worker 1 PID: ${W1_PID}"

# Wait for workers to connect and model to load
echo "Waiting 15s for workers to connect and load model..."
sleep 15

# Test 1: Single request (regression)
echo ""
echo "=== Test 1: Single Request (regression) ==="
RESULT1=$(curl -s -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The quick brown fox", "max_tokens": 3, "temperature": 0.0}')
echo "$RESULT1" | python3 -m json.tool || echo "$RESULT1"
TEXT1=$(echo "$RESULT1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
echo "Generated: '${TEXT1}'"

# Test 2: Two sequential requests (both should complete)
echo ""
echo "=== Test 2: Two Requests ==="
RESULT2A=$(curl -s -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The quick brown fox", "max_tokens": 3, "temperature": 0.0}')
RESULT2B=$(curl -s -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Once upon a time", "max_tokens": 3, "temperature": 0.0}')

TEXT2A=$(echo "$RESULT2A" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
TEXT2B=$(echo "$RESULT2B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")

echo "Request A generated: '${TEXT2A}'"
echo "Request B generated: '${TEXT2B}'"

# Check metrics
echo ""
echo "=== Metrics after 2 requests ==="
curl -s http://localhost:${HTTP_PORT}/metrics | python3 -m json.tool || true

# Verify results
PASS=true
if [ -z "$TEXT1" ] || [ "$TEXT1" = "" ]; then
    echo "FAIL: Single request returned empty"
    PASS=false
fi

if [ -z "$TEXT2A" ] || [ "$TEXT2A" = "" ]; then
    echo "FAIL: Request A returned empty"
    PASS=false
fi

if [ -z "$TEXT2B" ] || [ "$TEXT2B" = "" ]; then
    echo "FAIL: Request B returned empty"
    PASS=false
fi

if [ "$PASS" = true ]; then
    echo "=== HTTP API BATCH E2E TEST PASSED ==="
else
    echo "=== HTTP API BATCH E2E TEST FAILED ==="
    echo "--- Coordinator log ---"
    tail -30 /tmp/hcp_http_batch_coord.log || true
    echo "--- Worker 0 log ---"
    tail -20 /tmp/hcp_http_batch_w0.log || true
    echo "--- Worker 1 log ---"
    tail -20 /tmp/hcp_http_batch_w1.log || true
    exit 1
fi
