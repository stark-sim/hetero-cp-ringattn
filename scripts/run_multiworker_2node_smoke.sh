#!/bin/bash
# Launcher for single-GPU multi-domain worker smoke test.
# Topology: GPU node runs 2 workers (2 domains) on one 4090.
# Mac node runs coordinator + optionally a 3rd domain.
#
# Environment:
#   GPU: 100.118.253.68, RTX 4090 24GB, CUDA
#   Mac: 100.121.35.138, M1 Pro 16GB, MPS
#
# Usage:
#   GPU_HOST=100.118.253.68 GPU_USER=user MAC_NODE_ADDR=100.121.35.138 \
#     PORT_BASE=29450 SEQ_LEN=8192 NUM_DOMAINS=2 \
#     bash scripts/run_multiworker_2node_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-rust-multiworker-smoke-local}"
GPU_HOST="${GPU_HOST:-100.118.253.68}"
GPU_USER="${GPU_USER:-stark}"
GPU_SSH="${GPU_SSH:-${GPU_USER}@${GPU_HOST}}"
GPU_REPO_DIR="${GPU_REPO_DIR:-hetero-cp-ringattn}"
PORT_BASE="${PORT_BASE:-29450}"

COORD_PORT="${COORD_PORT:-${PORT_BASE}}"
GPU_WORKER_PORT_0="${GPU_WORKER_PORT_0:-$((PORT_BASE + 1))}"
GPU_WORKER_PORT_1="${GPU_WORKER_PORT_1:-$((PORT_BASE + 2))}"

SEQ_LEN="${SEQ_LEN:-8192}"
NUM_DOMAINS="${NUM_DOMAINS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"

LOCAL_CARGO_OFFLINE="${LOCAL_CARGO_OFFLINE:-0}"
REMOTE_CARGO_OFFLINE="${REMOTE_CARGO_OFFLINE:-0}"

if [ -z "${MAC_NODE_ADDR:-}" ]; then
    MAC_NODE_ADDR="$(ifconfig | awk '/inet / && ($2 ~ /^100\./) { print $2; exit }')"
fi
if [ -z "${MAC_NODE_ADDR}" ]; then
    echo "Could not find a local 100.x address. Set MAC_NODE_ADDR explicitly." >&2
    exit 1
fi

REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\\\''/g"
    printf "'"
}

run_remote() {
    local command="$1"
    ssh -o ConnectTimeout=30 "${GPU_SSH}" "bash -lc $(shell_quote "${command}")"
}

echo "=== HCP Multi-Domain Worker 2-Node Smoke ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC_NODE_ADDR=${MAC_NODE_ADDR}"
echo "GPU_HOST=${GPU_HOST}"
echo "SEQ_LEN=${SEQ_LEN} NUM_DOMAINS=${NUM_DOMAINS}"
echo "coordinator: bind=0.0.0.0:${COORD_PORT}"
echo "gpu-worker:  domain_ids=0,1 ports=${GPU_WORKER_PORT_0},${GPU_WORKER_PORT_1}"
echo "Reports: ${REPORT_DIR}"

# Preflight: local cargo build
echo "Preflight local cargo build"
(
    cd "${REPO_ROOT}/rust"
    cargo build --features tch-backend
)

# Preflight: remote git pull and cargo build
echo "Preflight remote git pull and cargo build"
remote_repo_q="$(shell_quote "${GPU_REPO_DIR}")"
remote_command="cd ${remote_repo_q} && git pull --ff-only && cd rust && cargo build --features tch-backend"
run_remote "${remote_command}" 2>&1 | tee "${REPORT_DIR}/remote_preflight.log"

# Prompt generation (local)
PROMPT_LEN=$((SEQ_LEN - MAX_NEW_TOKENS))
echo "Generating prompt of ${PROMPT_LEN} tokens..."
cd "${REPO_ROOT}/rust"
cargo run --bin gen_prompt -- --model-dir models/Qwen2-0.5B --output /tmp/hcp_prompt_${RUN_ID}.txt --num-tokens "${PROMPT_LEN}" 2>&1 | tee "${REPORT_DIR}/gen_prompt.log"

# Copy prompt to remote
scp "/tmp/hcp_prompt_${RUN_ID}.txt" "${GPU_SSH}:~/hcp_prompt_${RUN_ID}.txt"

# Start coordinator (local Mac)
coordinator_log="${REPORT_DIR}/coordinator.log"
(
    cd "${REPO_ROOT}/rust"
    cargo run --bin hcp-ringattn-rust -- \
        --distributed-role coordinator \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        --num-domains "${NUM_DOMAINS}" \
        --model-dir models/Qwen2-0.5B \
        --prompt-file "/tmp/hcp_prompt_${RUN_ID}.txt" \
        --max-tokens "${MAX_NEW_TOKENS}"
) >"${coordinator_log}" 2>&1 &
coordinator_pid="$!"
echo "Started coordinator pid=${coordinator_pid} log=${coordinator_log}"

# Start GPU multi-domain worker (remote)
gpu_worker_log="${REPORT_DIR}/gpu_worker.log"
remote_repo_q="$(shell_quote "${GPU_REPO_DIR}")"
remote_run_id_q="$(shell_quote "${RUN_ID}")"
remote_coord_q="$(shell_quote "${MAC_NODE_ADDR}:${COORD_PORT}")"
remote_prompt_q="$(shell_quote "~/hcp_prompt_${RUN_ID}.txt")"
remote_command="cd ${remote_repo_q} && cd rust && HCP_TCH_DEVICE=cuda:0 cargo run --bin hcp-ringattn-rust --release --features tch-backend -- \
    --distributed-role worker \
    --local-domain-ids 0,1 \
    --listen-addrs 0.0.0.0:${GPU_WORKER_PORT_0},0.0.0.0:${GPU_WORKER_PORT_1} \
    --next-peer-addrs 127.0.0.1:${GPU_WORKER_PORT_1},127.0.0.1:${GPU_WORKER_PORT_0} \
    --coordinator-addr ${remote_coord_q} \
    --num-domains ${NUM_DOMAINS} \
    --model-dir models/Qwen2-0.5B"
run_remote "${remote_command}" >"${gpu_worker_log}" 2>&1 &
gpu_worker_pid="$!"
echo "Started GPU worker pid=${gpu_worker_pid} log=${gpu_worker_log}"

# Cleanup on exit
cleanup() {
    local status="$1"
    echo "Cleaning up..."
    kill "${coordinator_pid}" >/dev/null 2>&1 || true
    ssh "${GPU_SSH}" "pkill -f 'hcp-ringattn-rust.*worker.*${RUN_ID}' || true" >/dev/null 2>&1 || true
}
trap 'status=$?; cleanup "${status}"; exit "${status}"' INT TERM EXIT

# Wait for coordinator
if wait "${coordinator_pid}"; then
    echo "Coordinator exited successfully"
else
    coord_status="$?"
    echo "Coordinator failed with status ${coord_status}" >&2
fi

trap - INT TERM EXIT

echo "=== HCP Multi-Domain Worker 2-Node Summary ==="
echo "Coordinator log: ${coordinator_log}"
echo "GPU worker log: ${gpu_worker_log}"

if [ -f "${coordinator_log}" ]; then
    grep -E "(generated|elapsed|output)" "${coordinator_log}" || true
fi

echo "=== HCP Multi-Domain Worker 2-Node Done ==="
