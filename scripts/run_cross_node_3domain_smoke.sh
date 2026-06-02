#!/bin/bash
# Cross-node 3-domain heterogeneous distributed inference smoke test.
# Topology: Mac (MPS, domain 0) + white RTX 4090 (CUDA, domain 1) + pearl RX 9060 XT (HIP, domain 2)
# Coordinator runs on Mac.
#
# KV ring: domain 0 -> domain 1 -> domain 2 -> domain 0
# QUIC peer connections over Tailscale VPN.
#
# Usage:
#   bash scripts/run_cross_node_3domain_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-cross-node-3domain-$(date +%Y%m%d-%H%M%S)}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

# === Configuration ===
WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_USER="${WHITE_USER:-stark}"
WHITE_SSH="${WHITE_USER}@${WHITE_HOST}"
WHITE_REPO_DIR="${WHITE_REPO_DIR:-hetero-cp-ringattn}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_USER="${PEARL_USER:-stark}"
PEARL_SSH="${PEARL_USER}@${PEARL_HOST}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

# Tailscale addresses
MAC_ADDR="${MAC_ADDR:-$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')}"
if [ -z "${MAC_ADDR}" ]; then
    echo "ERROR: Could not find local 100.x Tailscale address. Set MAC_ADDR explicitly." >&2
    exit 1
fi
WHITE_ADDR="${WHITE_ADDR:-${WHITE_HOST}}"
PEARL_ADDR="${PEARL_ADDR:-${PEARL_HOST}}"

echo "=== HCP Cross-Node 3-Domain Heterogeneous Smoke ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC=${MAC_ADDR} (MPS, domain 0)"
echo "WHITE=${WHITE_ADDR} (CUDA, domain 1)"
echo "PEARL=${PEARL_ADDR} (HIP, domain 2)"
echo "Reports: ${REPORT_DIR}"

# Ports
COORD_PORT=29500
W0_PORT=29501   # Mac worker 0 (domain 0)
W1_PORT=29502   # white worker 1 (domain 1)
W2_PORT=29503   # pearl worker 2 (domain 2)

# Model & prompt
MAC_MODEL_DIR="${MAC_MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
WHITE_MODEL_DIR="${WHITE_MODEL_DIR:-~/models/Qwen2-0.5B}"
PEARL_MODEL_DIR="${PEARL_MODEL_DIR:-~/hetero-cp-ringattn/models/Qwen2-0.5B}"
SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"

# === Dry-run mode ===
if [ "${DRY_RUN:-}" = "1" ]; then
    echo "=== DRY RUN CONFIG ==="
    echo "Ring topology:"
    echo "  Mac    (d0): ${MAC_ADDR}:${W0_PORT}  -> next=${WHITE_ADDR}:${W1_PORT}"
    echo "  White  (d1): ${WHITE_ADDR}:${W1_PORT}  -> next=${PEARL_ADDR}:${W2_PORT}"
    echo "  Pearl  (d2): ${PEARL_ADDR}:${W2_PORT}  -> next=${MAC_ADDR}:${W0_PORT}"
    echo "  Coordinator: 0.0.0.0:${COORD_PORT}"
    echo "  Model dirs:"
    echo "    Mac:   ${MAC_MODEL_DIR}"
    echo "    White: ${WHITE_MODEL_DIR}"
    echo "    Pearl: ${PEARL_MODEL_DIR}"
    echo "  Prompt: ${SEQ_LEN} tokens, max_new_tokens=${MAX_NEW_TOKENS}"
    exit 0
fi

# === SSH host key preflight ===
echo "=== SSH host key preflight ==="
ssh-keyscan -H "${WHITE_HOST}" >> ~/.ssh/known_hosts 2>/dev/null || true
ssh-keyscan -H "${PEARL_HOST}" >> ~/.ssh/known_hosts 2>/dev/null || true

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

run_remote_white() {
    local command="$1"
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${WHITE_SSH}" "bash -lc $(shell_quote "${command}")"
}

run_remote_pearl() {
    local command="$1"
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${PEARL_SSH}" "bash -lc $(shell_quote "${command}")"
}

# === Verify pearl model path ===
if ! ssh -o ConnectTimeout=10 "${PEARL_SSH}" "test -f ${PEARL_MODEL_DIR}/model.safetensors" 2>/dev/null; then
    echo "WARNING: Model not found on pearl at ${PEARL_MODEL_DIR}, trying fallback..."
    PEARL_MODEL_DIR="~/models/Qwen2-0.5B"
    if ! ssh -o ConnectTimeout=10 "${PEARL_SSH}" "test -f ${PEARL_MODEL_DIR}/model.safetensors" 2>/dev/null; then
        echo "ERROR: Model not found on pearl at fallback path either. Set PEARL_MODEL_DIR." >&2
        exit 1
    fi
fi

# === Preflight: build locally ===
echo "=== Preflight: local build ==="
cd "${REPO_ROOT}/rust"
cargo build --features tch-backend --release 2>&1 | tee "${REPORT_DIR}/local_build.log"

# === Preflight: remote build (white) ===
echo "=== Preflight: remote build (white) ==="
white_cmd="cd $(shell_quote "${WHITE_REPO_DIR}") && git pull --ff-only && cd rust && cargo build --features tch-backend --release"
run_remote_white "${white_cmd}" 2>&1 | tee "${REPORT_DIR}/white_build.log"

# === Preflight: remote build (pearl) ===
echo "=== Preflight: remote build (pearl) ==="
pearl_cmd="cd $(shell_quote "${PEARL_REPO_DIR}") && git pull --ff-only && cd rust && cargo build --features tch-backend --release"
run_remote_pearl "${pearl_cmd}" 2>&1 | tee "${REPORT_DIR}/pearl_build.log"

# === Generate prompt ===
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
cd "${REPO_ROOT}/rust"
cargo run --bin gen_prompt -- "${MAC_MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tee "${REPORT_DIR}/gen_prompt.log"

# Copy prompt to remotes
scp "${PROMPT_FILE}" "${WHITE_SSH}:~/hcp_prompt_${RUN_ID}.txt"
scp "${PROMPT_FILE}" "${PEARL_SSH}:~/hcp_prompt_${RUN_ID}.txt"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

# === Cleanup function ===
cleanup() {
    echo "=== Cleaning up ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 1' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*domain-id 2' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# === Environment (must be exported before any libtorch-linked binary) ===
export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
export HCP_TCH_DEVICE=mps

# === Coordinator flags ===
COORD_FLAGS=""
if [ "${CAPACITY_AWARE:-}" = "1" ]; then
    COORD_FLAGS="${COORD_FLAGS} --capacity-aware"
fi

# === Launch Coordinator (Mac) — start first so workers can connect ===
echo "=== Launching Coordinator ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MAC_MODEL_DIR}" \
    --prompt-file "${PROMPT_FILE}" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --num-domains 3 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    ${COORD_FLAGS} \
    >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_PID=$!
echo "Coordinator PID: ${COORD_PID}"

# Give coordinator time to bind
echo "Waiting 2s for coordinator to bind..."
sleep 2

# === Launch Worker 1 (white, domain 1) ===
echo "=== Launching Worker 1 (white, domain 1) ==="
white_worker_cmd="cd $(shell_quote "${WHITE_REPO_DIR}") && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir ${WHITE_MODEL_DIR} \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${PEARL_ADDR}:${W2_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 3"

run_remote_white "${white_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
WORKER1_SSH_PID=$!
echo "Worker 1 (white) SSH PID: ${WORKER1_SSH_PID}"

# === Launch Worker 2 (pearl, domain 2) ===
echo "=== Launching Worker 2 (pearl, domain 2) ==="
pearl_worker_cmd="cd $(shell_quote "${PEARL_REPO_DIR}") && export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 2 \
    --model-dir ${PEARL_MODEL_DIR} \
    --listen-addr 0.0.0.0:${W2_PORT} \
    --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 3"

run_remote_pearl "${pearl_worker_cmd}" >"${REPORT_DIR}/worker2.log" 2>&1 &
WORKER2_SSH_PID=$!
echo "Worker 2 (pearl) SSH PID: ${WORKER2_SSH_PID}"

# Give remote workers time to bind and start accepting
echo "Waiting 10s for remote workers to bind..."
sleep 10

# === Launch Worker 0 (Mac, domain 0) ===
echo "=== Launching Worker 0 (Mac, domain 0) ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MAC_MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${WHITE_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 3 \
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
echo "=== HCP Cross-Node 3-Domain Heterogeneous Summary ==="
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
    echo "--- Worker 1 (white) output ---"
    tail -30 "${REPORT_DIR}/worker1.log"
fi

if [ -f "${REPORT_DIR}/worker2.log" ]; then
    echo "--- Worker 2 (pearl) output ---"
    tail -30 "${REPORT_DIR}/worker2.log"
fi

echo "=== Done ==="
exit "${coord_status}"
