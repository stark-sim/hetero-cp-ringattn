#!/bin/bash
# Cross-backend correctness comparison: tch vs vLLM (transformers or vllm-metal).
# Runs the same prompt through both backends and compares generated tokens.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"

PROMPT="The quick brown fox"
MAX_TOKENS=5
TEMPERATURE=0.0

echo "=== Cross-Backend Correctness Comparison ==="
echo "Prompt: '${PROMPT}'"
echo "Max tokens: ${MAX_TOKENS}"
echo ""

# Build release binary
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -3

run_backend() {
    local backend_type="$1"
    local backend_name="$2"
    local http_port="$3"
    local coord_port="$4"
    local w0_port="$5"
    local log_suffix="$6"

    export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
    export HCP_TCH_DEVICE=mps

    # Set Python backend for vLLM types
    if [ "${backend_type}" = "vllm" ]; then
        # Default to transformers on Mac (faster init than vllm-metal)
        export HCP_PYTHON_BACKEND_TYPE="${HCP_PYTHON_BACKEND_TYPE:-transformers}"
        export HCP_PYTHON_PATH="${HCP_PYTHON_PATH:-/Users/stark_sim/.venv-vllm-metal/bin/python3}"
        export HCP_VLLM_WORKER_PATH="${REPO_ROOT}/python/hcp_worker_process.py"
    fi

    # Launch coordinator
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MODEL_DIR}" \
        --num-domains 1 \
        --listen-addr "0.0.0.0:${coord_port}" \
        --http-addr "0.0.0.0:${http_port}" \
        > "/tmp/hcp_cross_coord_${log_suffix}.log" 2>&1 &
    local coord_pid=$!
    sleep 2

    # Launch worker
    "${BINARY}" --distributed-role worker \
        --domain-id 0 \
        --model-dir "${MODEL_DIR}" \
        --backend-type "${backend_type}" \
        --listen-addr "0.0.0.0:${w0_port}" \
        --next-peer-addr "127.0.0.1:${w0_port}" \
        --coordinator-addr "127.0.0.1:${coord_port}" \
        --num-domains 1 \
        > "/tmp/hcp_cross_w0_${log_suffix}.log" 2>&1 &
    local w0_pid=$!

    # Wait for initialization
    if [ "${backend_type}" = "vllm" ]; then
        echo "Waiting 30s for ${backend_name} to initialize..." >&2
        sleep 30
    else
        echo "Waiting 10s for ${backend_name} to initialize..." >&2
        sleep 10
    fi

    # Query API
    local result
    result=$(curl -s -X POST "http://localhost:${http_port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"${PROMPT}\", \"max_tokens\": ${MAX_TOKENS}, \"temperature\": ${TEMPERATURE}}")

    local generated
    generated=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "[error]")

    # Cleanup
    kill $w0_pid 2>/dev/null || true
    kill $coord_pid 2>/dev/null || true
    wait 2>/dev/null || true
    # EngineCore cleanup for vllm-metal
    pkill -9 -f "EngineCore" 2>/dev/null || true

    echo "$generated"
}

# Run tch backend
echo "=== Backend 1: tch (Mac MPS) ==="
TCH_OUTPUT=$(run_backend "tch" "tch" 8093 29630 29631 "tch")
echo "Generated: '${TCH_OUTPUT}'"
echo ""

# Run vLLM backend
echo "=== Backend 2: vLLM (${HCP_PYTHON_BACKEND_TYPE:-transformers}) ==="
VLLM_OUTPUT=$(run_backend "vllm" "vllm" 8094 29632 29633 "vllm")
echo "Generated: '${VLLM_OUTPUT}'"
echo ""

# Compare
echo "=== Comparison ==="
if [ "$TCH_OUTPUT" = "$VLLM_OUTPUT" ]; then
    echo "✅ CORRECTNESS MATCH: both backends generated identical text"
    echo "Text: '${TCH_OUTPUT}'"
    exit 0
else
    echo "❌ CORRECTNESS MISMATCH"
    echo "tch output:    '${TCH_OUTPUT}'"
    echo "vLLM output:   '${VLLM_OUTPUT}'"
    exit 1
fi
