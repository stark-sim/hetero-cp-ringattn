#!/bin/bash
# Cross-node 2-domain distributed inference smoke test.
# Topology: Mac (MPS, domain 0) + Remote RTX 4090 (CUDA, domain 1)
# Coordinator runs on Mac.
#
# KV ring: domain 0 -> domain 1 -> domain 0 (ring)
# QUIC peer connections over Tailscale VPN.
#
# Usage:
#   GPU_HOST=100.118.253.68 GPU_USER=stark \
#     bash scripts/run_cross_node_2domain_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-cross-node-2domain-$(date +%Y%m%d-%H%M%S)}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

# === Configuration ===
GPU_HOST="${GPU_HOST:-100.118.253.68}"
GPU_USER="${GPU_USER:-stark}"
GPU_SSH="${GPU_USER}@${GPU_HOST}"
GPU_REPO_DIR="${GPU_REPO_DIR:-hetero-cp-ringattn}"

# Tailscale addresses
MAC_ADDR="${MAC_ADDR:-$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')}"
if [ -z "${MAC_ADDR}" ]; then
    echo "ERROR: Could not find local 100.x Tailscale address. Set MAC_ADDR explicitly." >&2
    exit 1
fi
GPU_ADDR="${GPU_ADDR:-${GPU_HOST}}"

echo "=== HCP Cross-Node 2-Domain Smoke ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC=${MAC_ADDR} (MPS)"
echo "GPU=${GPU_ADDR} (CUDA)"
echo "Reports: ${REPORT_DIR}"

# Ports
COORD_PORT=29500
W0_PORT=29501   # Mac worker 0 (domain 0)
W1_PORT=29502   # GPU worker 1 (domain 1)

# Model & prompt
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
SEQ_LEN="${SEQ_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote() {
    local command="$1"
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${GPU_SSH}" "bash -lc $(shell_quote "${command}")"
}

# === Preflight: build locally ===
echo "=== Preflight: local build ==="
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tee "${REPORT_DIR}/local_build.log"

# === Preflight: remote build ===
echo "=== Preflight: remote build ==="
remote_cmd="cd $(shell_quote "${GPU_REPO_DIR}") && git pull --ff-only && cd rust && cargo build --features tch-backend --release"
run_remote "${remote_cmd}" 2>&1 | tee "${REPORT_DIR}/remote_build.log"

# === Generate prompt ===
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
cd "${REPO_ROOT}/rust"
cargo run --bin gen_prompt -- "${MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tee "${REPORT_DIR}/gen_prompt.log"

# Copy prompt to remote
scp "${PROMPT_FILE}" "${GPU_SSH}:~/hcp_prompt_${RUN_ID}.txt"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

# === Cleanup function ===
cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${GPU_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# === Launch Coordinator (Mac) — start first so workers can connect ===
echo "=== Launching Coordinator ==="
DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}" "${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --prompt-file "${PROMPT_FILE}" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --num-domains 2 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_PID=$!
echo "Coordinator PID: ${COORD_PID}"

# Give coordinator time to bind
echo "Waiting 2s for coordinator to bind..."
sleep 2

# === Launch Worker 1 (GPU, domain 1) — MUST start before domain 0 ===
echo "=== Launching Worker 1 (GPU, domain 1) ==="
remote_worker_cmd="cd $(shell_quote "${GPU_REPO_DIR}") && HCP_TCH_DEVICE=cuda:0 \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir ~/models/Qwen2-0.5B \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 2"

run_remote "${remote_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
WORKER1_SSH_PID=$!
echo "Worker 1 (GPU) SSH PID: ${WORKER1_SSH_PID}"

# Give GPU worker time to bind and start accepting
echo "Waiting 5s for GPU worker to bind..."
sleep 5

# === Launch Worker 0 (Mac, domain 0) — dials to domain 1 ===
echo "=== Launching Worker 0 (Mac, domain 0) ==="
DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}" HCP_TCH_DEVICE=mps "${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${GPU_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    >"${REPORT_DIR}/worker0.log" 2>&1 &
WORKER0_PID=$!
echo "Worker 0 (Mac) PID: ${WORKER0_PID}"

# === Wait for coordinator to finish ===
echo "=== Waiting for coordinator ==="
coord_status=0
wait "${COORD_PID}" || coord_status=$?

echo "=== Coordinator exited: ${coord_status} ==="

trap - EXIT INT TERM
cleanup

# === Summary ===
echo ""
echo "=== HCP Cross-Node 2-Domain Summary ==="
echo "Reports: ${REPORT_DIR}"

if [ -f "${REPORT_DIR}/coordinator.log" ]; then
    echo "--- Coordinator output ---"
    cat "${REPORT_DIR}/coordinator.log"
fi

if [ -f "${REPORT_DIR}/worker0.log" ]; then
    echo "--- Worker 0 (Mac) output ---"
    tail -30 "${REPORT_DIR}/worker0.log"
fi

if [ -f "${REPORT_DIR}/worker1.log" ]; then
    echo "--- Worker 1 (GPU) output ---"
    tail -30 "${REPORT_DIR}/worker1.log"
fi

echo "=== Done ==="
exit "${coord_status}"
