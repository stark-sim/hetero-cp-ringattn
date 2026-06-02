#!/usr/bin/env bash
set -euo pipefail

# 2-domain cross-node smoke: Mac MPS (domain 0) + pearl HIP (domain 1)
# white CUDA is temporarily offline.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# --- Env ---
export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

MAC_MODEL_DIR="${MAC_MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
PEARL_MODEL_DIR="${PEARL_MODEL_DIR:-models/Qwen2-0.5B}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

MAC_ADDR="${MAC_ADDR:-100.121.35.138}"
PEARL_ADDR="${PEARL_ADDR:-100.111.242.55}"
PEARL_SSH="${PEARL_SSH:-stark@pearl}"

SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"

COORD_PORT=29500
W0_PORT=29501
W1_PORT=29502

RUN_ID="2domain-mps-hip-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

# --- Cleanup trap ---
cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_PID:-}" ] && kill "${COORD_PID}" 2>/dev/null || true
    [ -n "${W0_PID:-}" ] && kill "${W0_PID}" 2>/dev/null || true
    [ -n "${W1_SSH_PID:-}" ] && kill "${W1_SSH_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Preflight: verify pearl binary exists ---
echo "=== Verifying pearl binary ==="
if ! ssh -o ConnectTimeout=10 "${PEARL_SSH}" "test -x ${PEARL_REPO_DIR}/rust/target/release/hcp-ringattn-rust"; then
    echo "ERROR: pearl binary not found. Run cargo build --release on pearl first." >&2
    exit 1
fi

# --- Generate prompt ---
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
cd "${REPO_ROOT}/rust"
cargo run --bin gen_prompt -- "${MAC_MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}" 2>&1 | tee "${REPORT_DIR}/gen_prompt.log"
cd "${REPO_ROOT}"

# --- Launch Coordinator (Mac) ---
echo "=== Launching Coordinator ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MAC_MODEL_DIR}" \
    --prompt-file "${PROMPT_FILE}" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --num-domains 2 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_PID=$!
echo "Coordinator PID: ${COORD_PID}"
sleep 2

# --- Launch Worker 0 (Mac, domain 0) ---
echo "=== Launching Worker 0 (Mac, domain 0) ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MAC_MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${PEARL_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    >"${REPORT_DIR}/worker0.log" 2>&1 &
W0_PID=$!
echo "Worker 0 PID: ${W0_PID}"

# --- Launch Worker 1 (pearl, domain 1) ---
echo "=== Launching Worker 1 (pearl, domain 1) ==="
pearl_cmd="cd ${PEARL_REPO_DIR} && export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir ${PEARL_MODEL_DIR} \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
    --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
    --num-domains 2"

ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "${PEARL_SSH}" "bash -lc $(printf '%q' "${pearl_cmd}")" >"${REPORT_DIR}/worker1.log" 2>&1 &
W1_SSH_PID=$!
echo "Worker 1 SSH PID: ${W1_SSH_PID}"

echo "Waiting 10s for remote workers to bind..."
sleep 10

echo "=== Waiting for inference ==="
wait ${COORD_PID}
EXIT_CODE=$?
echo "Coordinator exit code: ${EXIT_CODE}"

echo "=== Generated text ==="
grep -A2 "generated:" "${REPORT_DIR}/coordinator.log" | tail -3 || tail -20 "${REPORT_DIR}/coordinator.log"

echo "=== Capacities ==="
grep "capacity:" "${REPORT_DIR}"/*.log || true

echo "=== Report: ${REPORT_DIR} ==="
exit ${EXIT_CODE}
