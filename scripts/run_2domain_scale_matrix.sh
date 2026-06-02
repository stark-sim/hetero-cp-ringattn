#!/bin/bash
# 2-domain Mac MPS + pearl HIP scale matrix test.
# Runs multiple sequence lengths and collects results.
# Usage: bash scripts/run_2domain_scale_matrix.sh [seq_lengths...]
# Example: bash scripts/run_2domain_scale_matrix.sh 64 256 512 1024 2048

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_USER="${PEARL_USER:-stark}"
PEARL_SSH="${PEARL_USER}@${PEARL_HOST}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

MAC_ADDR="${MAC_ADDR:-$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')}"
if [ -z "${MAC_ADDR}" ]; then
    echo "ERROR: Could not find local 100.x Tailscale address." >&2
    exit 1
fi
PEARL_ADDR="${PEARL_ADDR:-${PEARL_HOST}}"

MAC_MODEL_DIR="${MAC_MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
PEARL_MODEL_DIR="${PEARL_MODEL_DIR:-models/Qwen2-0.5B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"

# Sequence lengths to test
SEQ_LENGTHS=("${@:-64 256 512}")

# Base ports (incremented per test)
BASE_COORD_PORT=29500
BASE_W0_PORT=29501
BASE_W1_PORT=29502

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote_pearl() {
    local command="$1"
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${PEARL_SSH}" "bash -lc $(shell_quote "${command}")"
}

RESULTS_FILE="${REPO_ROOT}/reports/2domain-scale-matrix-$(date +%Y%m%d-%H%M%S).tsv"
mkdir -p "${REPO_ROOT}/reports"
echo -e "seq_len\texit_code\tgenerated\tmac_cap\tpearl_cap\tduration_sec\treport_dir" > "${RESULTS_FILE}"

echo "=== 2-Domain Scale Matrix Test ==="
echo "Platforms: Mac MPS + pearl HIP"
echo "Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "Results: ${RESULTS_FILE}"
echo ""

for i in "${!SEQ_LENGTHS[@]}"; do
    SEQ_LEN="${SEQ_LENGTHS[$i]}"
    OFFSET=$((i * 10))
    COORD_PORT=$((BASE_COORD_PORT + OFFSET))
    W0_PORT=$((BASE_W0_PORT + OFFSET))
    W1_PORT=$((BASE_W1_PORT + OFFSET))

    RUN_ID="2domain-mps-hip-${SEQ_LEN}-$(date +%Y%m%d-%H%M%S)"
    REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
    mkdir -p "${REPORT_DIR}"

    echo ""
    echo "========== SEQ_LEN=${SEQ_LEN} (ports: ${COORD_PORT}/${W0_PORT}/${W1_PORT}) =========="

    # Generate prompt
    PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
    cd "${REPO_ROOT}/rust"
    cargo run --bin gen_prompt -- "${MAC_MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tail -2
    cd "${REPO_ROOT}"

    # Launch coordinator
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MAC_MODEL_DIR}" \
        --prompt-file "${PROMPT_FILE}" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --num-domains 2 \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        >"${REPORT_DIR}/coordinator.log" 2>&1 &
    COORD_PID=$!
    sleep 2

    # Launch pearl worker
    pearl_worker_cmd="cd ${PEARL_REPO_DIR} && export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
      ./rust/target/release/hcp-ringattn-rust \
        --distributed-role worker \
        --domain-id 1 \
        --model-dir ${PEARL_MODEL_DIR} \
        --listen-addr 0.0.0.0:${W1_PORT} \
        --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
        --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
        --num-domains 2"
    run_remote_pearl "${pearl_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
    W1_SSH_PID=$!
    sleep 5

    # Launch Mac worker
    export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
    "${BINARY}" --distributed-role worker \
        --domain-id 0 \
        --model-dir "${MAC_MODEL_DIR}" \
        --listen-addr "0.0.0.0:${W0_PORT}" \
        --next-peer-addr "${PEARL_ADDR}:${W1_PORT}" \
        --coordinator-addr "127.0.0.1:${COORD_PORT}" \
        --num-domains 2 \
        >"${REPORT_DIR}/worker0.log" 2>&1 &
    W0_PID=$!

    START_TIME=$(date +%s)
    echo "Waiting for inference (coordinator PID: ${COORD_PID})..."
    wait ${COORD_PID}
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Collect results
    GENERATED=$(grep "generated:" "${REPORT_DIR}/coordinator.log" 2>/dev/null | sed 's/.*generated: //' || echo "N/A")
    MAC_CAP=$(grep "worker 0.*capacity=" "${REPORT_DIR}/coordinator.log" 2>/dev/null | sed 's/.*capacity=//' || echo "N/A")
    PEARL_CAP=$(grep "worker 1.*capacity=" "${REPORT_DIR}/coordinator.log" 2>/dev/null | sed 's/.*capacity=//' || echo "N/A")

    echo "Result: exit=${EXIT_CODE}, duration=${DURATION}s, generated='${GENERATED}'"
    echo -e "${SEQ_LEN}\t${EXIT_CODE}\t${GENERATED}\t${MAC_CAP}\t${PEARL_CAP}\t${DURATION}\t${REPORT_DIR}" >> "${RESULTS_FILE}"

    # Cleanup between runs
    kill ${W0_PID} 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
    sleep 3

done

echo ""
echo "=== Scale Matrix Complete ==="
cat "${RESULTS_FILE}"
