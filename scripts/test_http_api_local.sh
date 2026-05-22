#!/bin/bash
# Local HTTP API end-to-end test.
# Launches coordinator (HTTP mode) + 2-domain worker, then curls the API.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"

COORD_PORT=29500
W0_PORT=29501
W1_PORT=29502
HTTP_PORT=8080

echo "=== HTTP API Local E2E Test ==="
echo "Building release binary..."
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -3

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
    > /tmp/hcp_http_coord.log 2>&1 &
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
    > /tmp/hcp_http_w0.log 2>&1 &
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
    > /tmp/hcp_http_w1.log 2>&1 &
W1_PID=$!
echo "Worker 1 PID: ${W1_PID}"

# Wait for workers to connect and model to load
echo "Waiting 15s for workers to connect and load model..."
sleep 15

# Test /health
echo ""
echo "=== Test 1: GET /health ==="
curl -s http://localhost:${HTTP_PORT}/health | python3 -m json.tool || true

# Test /metrics
echo ""
echo "=== Test 2: GET /metrics ==="
curl -s http://localhost:${HTTP_PORT}/metrics | python3 -m json.tool || true

# Test /v1/completions
echo ""
echo "=== Test 3: POST /v1/completions ==="
RESULT=$(curl -s -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The quick brown fox", "max_tokens": 5, "temperature": 0.0}')
echo "$RESULT" | python3 -m json.tool || echo "$RESULT"

# Extract generated text
GENERATED=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
echo ""
echo "Generated text: '${GENERATED}'"

if [ -n "$GENERATED" ] && [ "$GENERATED" != "[error:"* ]; then
    echo "=== HTTP API E2E TEST PASSED ==="
else
    echo "=== HTTP API E2E TEST FAILED ==="
    echo "--- Coordinator log ---"
    tail -20 /tmp/hcp_http_coord.log || true
    echo "--- Worker 0 log ---"
    tail -20 /tmp/hcp_http_w0.log || true
    echo "--- Worker 1 log ---"
    tail -20 /tmp/hcp_http_w1.log || true
    exit 1
fi
