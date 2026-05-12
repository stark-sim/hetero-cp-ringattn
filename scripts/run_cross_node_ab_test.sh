#!/bin/bash
# Cross-node A/B comparison test for Micro KV Block overlap optimization.
# Tests multiple configurations and compares performance & correctness.
#
# Usage:
#   GPU_HOST=100.118.253.68 GPU_USER=stark \
#     SEQ_LEN=4096 MAX_NEW_TOKENS=5 \
#     bash scripts/run_cross_node_ab_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-cross-node-ab-$(date +%Y%m%d-%H%M%S)}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

# === Configuration ===
GPU_HOST="${GPU_HOST:-100.118.253.68}"
GPU_USER="${GPU_USER:-stark}"
GPU_SSH="${GPU_USER}@${GPU_HOST}"
GPU_REPO_DIR="${GPU_REPO_DIR:-hetero-cp-ringattn}"

MAC_ADDR="${MAC_ADDR:-$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')}"
if [ -z "${MAC_ADDR}" ]; then
    echo "ERROR: Could not find local 100.x Tailscale address. Set MAC_ADDR explicitly." >&2
    exit 1
fi
GPU_ADDR="${GPU_ADDR:-${GPU_HOST}}"

# Test parameters
SEQ_LEN="${SEQ_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"

# A/B configurations: "name|env_vars"
# Baseline: serial (no overlap)
# Optimized: pipeline overlap with various micro block sizes
CONFIGS=(
    "baseline_serial|HCP_DISABLE_OVERLAP=1 HCP_MICRO_KV_BLOCK_SIZE=0"
    "overlap_default|HCP_MICRO_KV_BLOCK_SIZE=0"
    "overlap_micro_64|HCP_MICRO_KV_BLOCK_SIZE=64"
    "overlap_micro_256|HCP_MICRO_KV_BLOCK_SIZE=256"
    "overlap_micro_512|HCP_MICRO_KV_BLOCK_SIZE=512"
)

PORTS=(29500 29501 29502)
COORD_PORT=${PORTS[0]}
W0_PORT=${PORTS[1]}
W1_PORT=${PORTS[2]}

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote() {
    local command="$1"
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${GPU_SSH}" "bash -lc $(shell_quote "${command}")"
}

# === Preflight: build ===
echo "=== A/B Test: ${RUN_ID} ==="
echo "MAC=${MAC_ADDR} (MPS) | GPU=${GPU_ADDR} (CUDA)"
echo "SEQ_LEN=${SEQ_LEN}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "Reports: ${REPORT_DIR}"
echo ""

echo "=== Preflight: local build ==="
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tail -5

echo "=== Preflight: remote build ==="
remote_cmd="cd $(shell_quote "${GPU_REPO_DIR}") && git pull --ff-only && cd rust && cargo build --features tch-backend --release"
run_remote "${remote_cmd}" 2>&1 | tail -5

# === Generate prompt once ===
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
cd "${REPO_ROOT}/rust"
cargo run --bin gen_prompt -- "${MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tail -3
scp "${PROMPT_FILE}" "${GPU_SSH}:~/hcp_prompt_${RUN_ID}.txt" >/dev/null 2>&1

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

# === Run each configuration ===
RESULTS_FILE="${REPORT_DIR}/ab_results.tsv"
echo -e "config\tmode\tmicro_block_size\tstatus\telapsed_sec\tgenerated_text" > "${RESULTS_FILE}"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r cfg_name cfg_env <<< "$cfg"
    cfg_report_dir="${REPORT_DIR}/${cfg_name}"
    mkdir -p "${cfg_report_dir}"

    echo ""
    echo "========================================"
    echo "Config: ${cfg_name}"
    echo "Env: ${cfg_env}"
    echo "========================================"

    # Determine micro block size from env for reporting
    micro_size="$(echo "${cfg_env}" | grep -o 'HCP_MICRO_KV_BLOCK_SIZE=[0-9]*' | cut -d= -f2 || echo "default")"
    mode="$(echo "${cfg_env}" | grep -q 'HCP_DISABLE_OVERLAP=1' && echo "serial" || echo "pipeline")"

    # Cleanup previous processes
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${GPU_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
    sleep 1

    # Launch Coordinator
    export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
    export HCP_TCH_DEVICE=mps
    # Apply config env to coordinator too (for consistency logging)
    eval "export ${cfg_env}"

    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MODEL_DIR}" \
        --prompt-file "${PROMPT_FILE}" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --num-domains 2 \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        >"${cfg_report_dir}/coordinator.log" 2>&1 &
    COORD_PID=$!
    sleep 2

    # Launch Worker 1 (GPU)
    remote_worker_cmd="cd $(shell_quote "${GPU_REPO_DIR}") && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && export ${cfg_env} && \
      ./rust/target/release/hcp-ringattn-rust \
        --distributed-role worker \
        --domain-id 1 \
        --model-dir ~/models/qwen2-0.5b \
        --listen-addr 0.0.0.0:${W1_PORT} \
        --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
        --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
        --num-domains 2"

    run_remote "${remote_worker_cmd}" >"${cfg_report_dir}/worker1.log" 2>&1 &
    sleep 5

    # Launch Worker 0 (Mac)
    "${BINARY}" --distributed-role worker \
        --domain-id 0 \
        --model-dir "${MODEL_DIR}" \
        --listen-addr "0.0.0.0:${W0_PORT}" \
        --next-peer-addr "${GPU_ADDR}:${W1_PORT}" \
        --coordinator-addr "127.0.0.1:${COORD_PORT}" \
        --num-domains 2 \
        >"${cfg_report_dir}/worker0.log" 2>&1 &
    WORKER0_PID=$!

    # Time the run
    start_time=$(date +%s)
    coord_status=0
    wait "${COORD_PID}" || coord_status=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    # Extract generated text
    generated=""
    if [ -f "${cfg_report_dir}/coordinator.log" ]; then
        generated=$(grep -o 'generated:.*' "${cfg_report_dir}/coordinator.log" | sed 's/generated: //' | tr '\n' ' ' | head -c 200)
    fi

    status_str="PASS"
    [ "${coord_status}" -ne 0 ] && status_str="FAIL(${coord_status})"

    echo -e "${cfg_name}\t${mode}\t${micro_size}\t${status_str}\t${elapsed}\t${generated}" >> "${RESULTS_FILE}"

    echo "Result: ${status_str}, elapsed=${elapsed}s, generated='${generated}'"

    # Cleanup
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${GPU_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
    sleep 2

done

# === Summary ===
echo ""
echo "========================================"
echo "A/B Test Summary"
echo "========================================"
cat "${RESULTS_FILE}"
echo ""
echo "Full reports: ${REPORT_DIR}"
