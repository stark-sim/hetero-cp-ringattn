#!/bin/bash
# Local VllmWorkerBackend vllm-metal E2E test (Mac MPS).
# Uses vllm-metal plugin for GPU acceleration on Apple Silicon.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"

COORD_PORT=29620
W0_PORT=29621
HTTP_PORT=8092

echo "=== VllmWorkerBackend vllm-metal (MPS) Local E2E Test ==="
echo "Building release binary..."
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -3

cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    # vllm-metal EngineCore cleanup
    pkill -9 -f "EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
# Use vllm-metal backend on Mac MPS
export HCP_PYTHON_BACKEND_TYPE=vllm
export HCP_PYTHON_PATH="/Users/stark_sim/.venv-vllm-metal/bin/python3"
export HCP_VLLM_WORKER_PATH="${REPO_ROOT}/python/hcp_worker_process.py"
# MPS capacity estimate (conservative)
export HCP_MPS_CAPACITY_MB=4096

# Launch coordinator (HTTP mode — no prompt source given)
echo "=== Launching Coordinator (HTTP mode, 1 domain) ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 1 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    > /tmp/hcp_vllm_metal_coord.log 2>&1 &
COORD_PID=$!
echo "Coordinator PID: ${COORD_PID}"
sleep 2

# Launch Worker 0 (domain 0, vllm backend with vllm-metal/MPS)
echo "=== Launching Worker 0 (vllm-metal backend, MPS) ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --backend-type vllm \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "127.0.0.1:${W0_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 1 \
    > /tmp/hcp_vllm_metal_w0.log 2>&1 &
W0_PID=$!
echo "Worker 0 PID: ${W0_PID}"

# Wait for worker to connect and vllm-metal to initialize
# First init can take 60-90s (gloo init + kernel warmup)
echo "Waiting 90s for worker to connect and vllm-metal to initialize..."
sleep 90

# Test /health
echo ""
echo "=== Test 1: GET /health ==="
curl -s http://localhost:${HTTP_PORT}/health | python3 -m json.tool || true

# Test /metrics
echo ""
echo "=== Test 2: GET /metrics ==="
curl -s http://localhost:${HTTP_PORT}/metrics | python3 -m json.tool || true

# Test 3: Single request
echo ""
echo "=== Test 3: POST /v1/completions (single request) ==="
RESULT=$(curl -s -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The quick brown fox", "max_tokens": 3, "temperature": 0.0}')
echo "$RESULT" | python3 -m json.tool || echo "$RESULT"

GENERATED=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
echo ""
echo "Generated text: '${GENERATED}'"

# Test 4: Two concurrent requests (batch decode verification)
echo ""
echo "=== Test 4: POST /v1/completions (2 concurrent requests) ==="
curl -s -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "The quick brown fox", "max_tokens": 3, "temperature": 0.0}' \
    > /tmp/vllm_metal_resp1.json &
PID1=$!

curl -s -X POST http://localhost:${HTTP_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Once upon a time", "max_tokens": 3, "temperature": 0.0}' \
    > /tmp/vllm_metal_resp2.json &
PID2=$!

echo "Submitted 2 requests concurrently, waiting..."
wait $PID1
wait $PID2

echo "--- Response 1 ---"
cat /tmp/vllm_metal_resp1.json | python3 -m json.tool || true

echo ""
echo "--- Response 2 ---"
cat /tmp/vllm_metal_resp2.json | python3 -m json.tool || true

GEN1=$(python3 -c "import json; d=json.load(open('/tmp/vllm_metal_resp1.json')); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")
GEN2=$(python3 -c "import json; d=json.load(open('/tmp/vllm_metal_resp2.json')); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "")

echo ""
echo "Generated text 1: '${GEN1}'"
echo "Generated text 2: '${GEN2}'"

# Validate
PASS1=false
PASS2=false
PASS3=false

if [ -n "$GENERATED" ] && [ "$GENERATED" != "[error:"* ]; then
    PASS1=true
fi
if [ -n "$GEN1" ] && [ "$GEN1" != "[error:"* ]; then
    PASS2=true
fi
if [ -n "$GEN2" ] && [ "$GEN2" != "[error:"* ]; then
    PASS3=true
fi

if [ "$PASS1" = true ] && [ "$PASS2" = true ] && [ "$PASS3" = true ]; then
    echo ""
    echo "=== VLLM-METAL E2E TEST PASSED ==="
    echo "Single request + 2 concurrent requests all completed successfully."
else
    echo ""
    echo "=== VLLM-METAL E2E TEST FAILED ==="
    [ "$PASS1" != true ] && echo "Single request failed"
    [ "$PASS2" != true ] && echo "Concurrent request 1 failed"
    [ "$PASS3" != true ] && echo "Concurrent request 2 failed"
    echo "--- Coordinator log ---"
    tail -30 /tmp/hcp_vllm_metal_coord.log || true
    echo "--- Worker 0 log ---"
    tail -30 /tmp/hcp_vllm_metal_w0.log || true
    exit 1
fi
