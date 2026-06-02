#!/bin/bash
# Cross-node 2-domain smoke: white CUDA (domain 0) + pearl HIP (domain 1)
# Uses Qwen2.5-3B-Instruct model.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# === Configuration ===
WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_USER="${WHITE_USER:-stark}"
WHITE_SSH="${WHITE_USER}@${WHITE_HOST}"
WHITE_REPO_DIR="${WHITE_REPO_DIR:-hetero-cp-ringattn}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_USER="${PEARL_USER:-stark}"
PEARL_SSH="${PEARL_USER}@${PEARL_HOST}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

WHITE_ADDR="${WHITE_ADDR:-${WHITE_HOST}}"
PEARL_ADDR="${PEARL_ADDR:-${PEARL_HOST}}"

WHITE_MODEL_DIR="${WHITE_MODEL_DIR:-~/models/Qwen2.5-3B-Instruct}"
PEARL_MODEL_DIR="${PEARL_MODEL_DIR:-~/models/Qwen2.5-3B-Instruct}"

SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"

COORD_PORT=29500
W0_PORT=29501
W1_PORT=29502

RUN_ID="2domain-cuda-hip-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

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

echo "=== 2-Domain CUDA+HIP Smoke ==="
echo "RUN_ID=${RUN_ID}"
echo "WHITE=${WHITE_ADDR} (CUDA) | PEARL=${PEARL_ADDR} (HIP)"
echo "SEQ_LEN=${SEQ_LEN}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "Reports: ${REPORT_DIR}"

# === Cleanup trap ===
cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_SSH_PID:-}" ] && kill "${COORD_SSH_PID}" 2>/dev/null || true
    [ -n "${W0_SSH_PID:-}" ] && kill "${W0_SSH_PID}" 2>/dev/null || true
    [ -n "${W1_SSH_PID:-}" ] && kill "${W1_SSH_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# === Preflight: verify binaries ===
echo "=== Verifying binaries ==="
if ! run_remote_white "test -x ${WHITE_REPO_DIR}/rust/target/release/hcp-ringattn-rust"; then
    echo "ERROR: white binary not found" >&2
    exit 1
fi
if ! run_remote_pearl "test -x ${PEARL_REPO_DIR}/rust/target/release/hcp-ringattn-rust"; then
    echo "ERROR: pearl binary not found" >&2
    exit 1
fi

# === Generate prompt (on white) ===
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
run_remote_white "cd ${WHITE_REPO_DIR} && source .venv/bin/activate && python3 -c \"from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('${WHITE_MODEL_DIR}'); prompt=' '.join(['the']*${SEQ_LEN}); ids=tok.encode(prompt, add_special_tokens=False); print(tok.decode(ids[:${SEQ_LEN}]))\" > ${PROMPT_FILE}" 2>/dev/null || true

if [ ! -s "${PROMPT_FILE}" ]; then
    # Fallback: simple repetition
    python3 -c "print(' '.join(['the']*${SEQ_LEN}))" > "${PROMPT_FILE}"
fi

# Copy prompt to pearl
scp "${PROMPT_FILE}" "${PEARL_SSH}:~/hcp_prompt_${RUN_ID}.txt" >/dev/null 2>&1 || true

# === Launch Coordinator (white) ===
echo "=== Launching Coordinator (white) ==="
coordinator_cmd="cd ${WHITE_REPO_DIR} && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role coordinator \
    --model-dir ${WHITE_MODEL_DIR} \
    --prompt-file ~/hcp_prompt_${RUN_ID}.txt \
    --max-tokens ${MAX_NEW_TOKENS} \
    --num-domains 2 \
    --listen-addr 0.0.0.0:${COORD_PORT}"

run_remote_white "${coordinator_cmd}" >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_SSH_PID=$!
echo "Coordinator SSH PID: ${COORD_SSH_PID}"
sleep 3

# === Launch Worker 1 (pearl, domain 1) ===
echo "=== Launching Worker 1 (pearl, domain 1) ==="
pearl_worker_cmd="cd ${PEARL_REPO_DIR} && export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir ${PEARL_MODEL_DIR} \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${WHITE_ADDR}:${W0_PORT} \
    --coordinator-addr ${WHITE_ADDR}:${COORD_PORT} \
    --num-domains 2"

run_remote_pearl "${pearl_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
W1_SSH_PID=$!
echo "Worker 1 SSH PID: ${W1_SSH_PID}"
sleep 5

# === Launch Worker 0 (white, domain 0) ===
echo "=== Launching Worker 0 (white, domain 0) ==="
white_worker_cmd="cd ${WHITE_REPO_DIR} && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 0 \
    --model-dir ${WHITE_MODEL_DIR} \
    --listen-addr 0.0.0.0:${W0_PORT} \
    --next-peer-addr ${PEARL_ADDR}:${W1_PORT} \
    --coordinator-addr 127.0.0.1:${COORD_PORT} \
    --num-domains 2"

run_remote_white "${white_worker_cmd}" >"${REPORT_DIR}/worker0.log" 2>&1 &
W0_SSH_PID=$!
echo "Worker 0 SSH PID: ${W0_SSH_PID}"

# === Wait for inference ===
echo "Waiting for inference to complete..."
wait ${COORD_SSH_PID}
EXIT_CODE=$?
echo "Coordinator exit code: ${EXIT_CODE}"

echo "=== Generated text ==="
grep -A2 "generated:" "${REPORT_DIR}/coordinator.log" | tail -3 || tail -20 "${REPORT_DIR}/coordinator.log"

echo "=== Capacities ==="
grep "capacity:" "${REPORT_DIR}"/*.log || true

echo "=== Report: ${REPORT_DIR} ==="
exit ${EXIT_CODE}
