#!/bin/bash
# BlockTableKvCache end-to-end validation.
# Runs the standard HTTP API local E2E twice:
#   1. Baseline: ContiguousKvCache (default)
#   2. BlockTable: HCP_KV_CACHE_BLOCK_TABLE=1
# Compares generated text for correctness.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"

PROMPT="The quick brown fox"
MAX_TOKENS=5
TEMPERATURE=0.0

echo "=== BlockTableKvCache E2E Validation ==="
echo "Prompt: '${PROMPT}'"
echo "Max tokens: ${MAX_TOKENS}"
echo ""

# Build release binary
echo "Building release binary..."
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -3

run_e2e() {
    local cache_type="$1"
    local http_port="$2"
    local coord_port="$3"
    local w0_port="$4"
    local w1_port="$5"
    local log_suffix="$6"

    export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
    export HCP_TCH_DEVICE=mps

    if [ "${cache_type}" = "block_table" ]; then
        export HCP_KV_CACHE_BLOCK_TABLE=1
        export HCP_KV_CACHE_BLOCK_SIZE=4
    else
        unset HCP_KV_CACHE_BLOCK_TABLE || true
        unset HCP_KV_CACHE_BLOCK_SIZE || true
    fi

    # Launch coordinator
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MODEL_DIR}" \
        --num-domains 2 \
        --listen-addr "0.0.0.0:${coord_port}" \
        --http-addr "0.0.0.0:${http_port}" \
        > "/tmp/hcp_bt_coord_${log_suffix}.log" 2>&1 &
    local coord_pid=$!
    sleep 2

    # Launch Worker 0
    "${BINARY}" --distributed-role worker \
        --domain-id 0 \
        --model-dir "${MODEL_DIR}" \
        --listen-addr "0.0.0.0:${w0_port}" \
        --next-peer-addr "127.0.0.1:${w1_port}" \
        --coordinator-addr "127.0.0.1:${coord_port}" \
        --num-domains 2 \
        > "/tmp/hcp_bt_w0_${log_suffix}.log" 2>&1 &
    local w0_pid=$!
    sleep 2

    # Launch Worker 1
    "${BINARY}" --distributed-role worker \
        --domain-id 1 \
        --model-dir "${MODEL_DIR}" \
        --listen-addr "0.0.0.0:${w1_port}" \
        --next-peer-addr "127.0.0.1:${w0_port}" \
        --coordinator-addr "127.0.0.1:${coord_port}" \
        --num-domains 2 \
        > "/tmp/hcp_bt_w1_${log_suffix}.log" 2>&1 &
    local w1_pid=$!
    sleep 2

    # Wait for model load
    echo "Waiting 15s for ${cache_type} workers to load..." >&2
    sleep 15

    # Query API
    local result
    result=$(curl -s -X POST "http://localhost:${http_port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"${PROMPT}\", \"max_tokens\": ${MAX_TOKENS}, \"temperature\": ${TEMPERATURE}}")

    local generated
    generated=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" 2>/dev/null || echo "[error]")

    # Cleanup
    kill $w1_pid 2>/dev/null || true
    kill $w0_pid 2>/dev/null || true
    kill $coord_pid 2>/dev/null || true
    wait 2>/dev/null || true

    echo "$generated"
}

# Run baseline
echo "=== Run 1: ContiguousKvCache (baseline) ==="
BASELINE_OUTPUT=$(run_e2e "contiguous" 8100 29700 29701 29702 "baseline")
echo "Generated: '${BASELINE_OUTPUT}'"
echo ""

# Run BlockTable
echo "=== Run 2: BlockTableKvCache (block_size=4) ==="
BLOCK_OUTPUT=$(run_e2e "block_table" 8101 29710 29711 29712 "block")
echo "Generated: '${BLOCK_OUTPUT}'"
echo ""

# Compare
echo "=== Comparison ==="
if [ "$BASELINE_OUTPUT" = "$BLOCK_OUTPUT" ]; then
    echo "✅ BlockTable E2E PASSED: identical output to baseline"
    echo "Text: '${BASELINE_OUTPUT}'"
    exit 0
else
    echo "❌ BlockTable E2E FAILED: output mismatch"
    echo "Baseline: '${BASELINE_OUTPUT}'"
    echo "BlockTable: '${BLOCK_OUTPUT}'"
    exit 1
fi
