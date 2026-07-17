#!/bin/bash
# Cross-node vLLM block-ring CP smoke: white CUDA (domain 0, vLLM 0.6.4)
# + pearl ROCm (domain 1, vLLM 0.23 V1).  Context-passing prefill + KV ring.
#
# Usage: bash scripts/run_cross_node_vllm_cp.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_USER="${WHITE_USER:-stark}"
WHITE_SSH="${WHITE_USER}@${WHITE_HOST}"
WHITE_REPO_DIR="${WHITE_REPO_DIR:-/home/stark/hetero-cp-ringattn}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_USER="${PEARL_USER:-stark}"
PEARL_SSH="${PEARL_USER}@${PEARL_HOST}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-/home/stark/hetero-cp-ringattn}"

WHITE_ADDR="${WHITE_ADDR:-${WHITE_HOST}}"
PEARL_ADDR="${PEARL_ADDR:-${PEARL_HOST}}"

WHITE_MODEL_DIR="${WHITE_MODEL_DIR:-/home/stark/models/Qwen2-0.5B}"
PEARL_MODEL_DIR="${PEARL_MODEL_DIR:-/home/stark/models/Qwen2-0.5B-1M}"

SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
CHUNK_A="${CHUNK_A:-32}"
CHUNK_B="${CHUNK_B:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

COORD_PORT=29500
W0_PORT=29501
W1_PORT=29502

RUN_ID="vllm-cp-cuda-hip-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() { printf "'"; printf "%s" "$1" | sed "s/'/'\\''/g"; printf "'"; }
run_white() { ssh -o ConnectTimeout=30 "${WHITE_SSH}" "bash -lc $(shell_quote "$1")"; }
run_pearl() { ssh -o ConnectTimeout=30 "${PEARL_SSH}" "bash -lc $(shell_quote "$1")"; }

echo "=== vLLM block-ring CP (CUDA+HIP) ==="
echo "RUN_ID=${RUN_ID}  SEQ_LEN=${SEQ_LEN} chunks=${CHUNK_A}+${CHUNK_B} block=${BLOCK_SIZE} max_new=${MAX_NEW_TOKENS}"
echo "Reports: ${REPORT_DIR}"

cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_SSH_PID:-}" ] && kill "${COORD_SSH_PID}" 2>/dev/null || true
    [ -n "${W0_SSH_PID:-}" ] && kill "${W0_SSH_PID}" 2>/dev/null || true
    [ -n "${W1_SSH_PID:-}" ] && kill "${W1_SSH_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# === Prompt ===
PROMPT_FILE="/tmp/hcp_cp_prompt_${RUN_ID}.txt"
if [ -n "${PROMPT_TEXT:-}" ]; then
    python3 -c "import sys; sys.stdout.write('${PROMPT_TEXT}')" > "${PROMPT_FILE}"
else
    python3 -c "import sys; sys.stdout.write(' '.join(['the']*int('${SEQ_LEN}')))" > "${PROMPT_FILE}"
fi
scp -o ConnectTimeout=30 "${PROMPT_FILE}" "${WHITE_SSH}:~/hcp_cp_prompt_${RUN_ID}.txt" >/dev/null 2>&1
scp -o ConnectTimeout=30 "${PROMPT_FILE}" "${PEARL_SSH}:~/hcp_cp_prompt_${RUN_ID}.txt" >/dev/null 2>&1

# === Coordinator (white, Rust) ===
echo "=== Launching Coordinator (white) ==="
coordinator_cmd="cd ${WHITE_REPO_DIR} && export HCP_TCH_DEVICE=cuda:0 && export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-} && \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role coordinator \
    --model-dir ${WHITE_MODEL_DIR} \
    --prompt-file ~/hcp_cp_prompt_${RUN_ID}.txt \
    --max-tokens ${MAX_NEW_TOKENS} \
    --num-domains 2 \
    --chunk-sizes ${CHUNK_A},${CHUNK_B} \
    --listen-addr 0.0.0.0:${COORD_PORT}"
run_white "${coordinator_cmd}" >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_SSH_PID=$!
echo "Coordinator SSH PID: ${COORD_SSH_PID}"
sleep 3

# === Worker 1 (pearl, domain 1) — listen first ===
echo "=== Launching Worker 1 (pearl, domain 1) ==="
pearl_worker_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-rocm && \
  SP=/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages && \
  export LD_LIBRARY_PATH=\$SP/torch/lib:\$SP/_rocm_sdk_core/lib:\$SP/_rocm_sdk_core/lib/host-math/lib:\$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:\$SP/_rocm_sdk_devel/lib:\$SP/_rocm_sdk_devel/lib/host-math/lib:\$SP/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-} && \
  python ${PEARL_REPO_DIR}/python/hcp_vllm_cp_worker.py \
    --model-dir ${PEARL_MODEL_DIR} --domain-id 1 --num-domains 2 \
    --coordinator-host ${WHITE_ADDR} --coordinator-port ${COORD_PORT} \
    --peer-listen-host 0.0.0.0 --peer-listen-port ${W1_PORT} \
    --next-peer-host ${WHITE_ADDR} --next-peer-port ${W0_PORT} \
    --block-size ${BLOCK_SIZE} --max-model-len 4096"
run_pearl "${pearl_worker_cmd}" >"${REPORT_DIR}/worker1.log" 2>&1 &
W1_SSH_PID=$!
echo "Worker 1 SSH PID: ${W1_SSH_PID}"
sleep 8

# === Worker 0 (white, domain 0) — connects to worker 1 ===
echo "=== Launching Worker 0 (white, domain 0) ==="
white_worker_cmd="cd ${WHITE_REPO_DIR} && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm && \
  python ${WHITE_REPO_DIR}/python/hcp_vllm_cp_worker.py \
    --model-dir ${WHITE_MODEL_DIR} --domain-id 0 --num-domains 2 \
    --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --peer-listen-host 0.0.0.0 --peer-listen-port ${W0_PORT} \
    --next-peer-host ${PEARL_ADDR} --next-peer-port ${W1_PORT} \
    --block-size ${BLOCK_SIZE} --max-model-len 4096"
run_white "${white_worker_cmd}" >"${REPORT_DIR}/worker0.log" 2>&1 &
W0_SSH_PID=$!
echo "Worker 0 SSH PID: ${W0_SSH_PID}"

echo "=== Waiting for coordinator to finish ==="
wait "${COORD_SSH_PID}" || true
echo "=== Coordinator log (tail) ==="
tail -40 "${REPORT_DIR}/coordinator.log" || true
echo "RUN_ID=${RUN_ID}"
