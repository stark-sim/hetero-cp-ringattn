#!/bin/bash
# Local VllmWorkerBackend transformers E2E test.
# Launches coordinator (HTTP mode) + 1-domain transformers worker, then curls the API.
# Uses HuggingFace transformers on Mac CPU — no vLLM needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"

COORD_PORT=29610
W0_PORT=29611
HTTP_PORT=8091

echo "=== VllmWorkerBackend Transformers Local E2E Test ==="
echo "Building release binary..."
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -3

cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
# Use transformers backend on Mac CPU
export HCP_PYTHON_BACKEND_TYPE=transformers
export HCP_PYTHON_PATH="/Users/stark_sim/.venv-vllm-metal/bin/python3"
export HCP_VLLM_WORKER_PATH="${REPO_ROOT}/python/hcp_worker_process.py"

# Launch coordinator (HTTP mode — no prompt source given)
echo "=== Launching Coordinator (HTTP mode, 1 domain) ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 1 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    > /tmp/hcp_vllm_tf_coord.log 2>&1 &
COORD_PID=$!
echo "Coordinator PID: ${COORD_PID}"
sleep 2

# Launch Worker 0 (domain 0, vllm backend with transformers)
echo "=== Launching Worker 0 (transformers backend, CPU) ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --backend-type vllm \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "127.0.0.1:${W0_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 1 \
    > /tmp/hcp_vllm_tf_w0.log 2>&1 &
W0_PID=$!
echo "Worker 0 PID: ${W0_PID}"

# Wait for worker to connect and model to load
echo "Waiting 30s for worker to connect and load model..."
sleep 30

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
    echo "=== TRANSFORMERS E2E TEST PASSED ==="
else
    echo "=== TRANSFORMERS E2E TEST FAILED ==="
    echo "--- Coordinator log ---"
    tail -30 /tmp/hcp_vllm_tf_coord.log || true
    echo "--- Worker 0 log ---"
    tail -30 /tmp/hcp_vllm_tf_w0.log || true
    exit 1
fi
