#!/bin/bash
# Phase-3 7a N=3 (no-Mac) driver — runs ON white.
# Topology: coordinator + worker 0 (white, RTX 4090 CUDA) + worker 1 (pearl,
# RX 9060 XT HIP) + worker 2 (laptop, RTX 4060 CUDA).
# Ring links: white -(LAN)-> pearl -(Tailscale)-> laptop -(Tailscale)-> white.
# Bench client hits white 127.0.0.1; the Mac only polls STATUS and fetches
# artifacts, so Mac-side network state cannot affect the run.
#
# Usage: phase3_7a_n3l_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/hcp-n3l-${RUN_ID}"
RESULT_DIR="${STATE_DIR}/bench"
REPO="${HOME}/hetero-cp-ringattn"
VLLM="${VLLM:-${HOME}/venv-bench/bin/vllm}"
MODEL_DIR="${HOME}/models/Qwen2-0.5B"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_REPO="${HOME}/hetero-cp-ringattn"
PEARL_MODEL="${HOME}/hetero-cp-ringattn/models/Qwen2-0.5B"
LAPTOP_SSH="${LAPTOP_SSH:-stark@100.96.154.1}"
LAPTOP_REPO="${HOME}/hetero-cp-ringattn"
LAPTOP_MODEL="${HOME}/models/Qwen2-0.5B"
WHITE_LAN="${WHITE_LAN:-192.168.8.172}"
PEARL_LAN="${PEARL_LAN:-192.168.8.176}"
WHITE_TS="${WHITE_TS:-100.118.253.68}"
LAPTOP_TS="${LAPTOP_TS:-100.96.154.1}"

COORD_PORT=29800
W0_PORT=29801
W1_PORT=29802
W2_PORT=29803
HTTP_PORT=8082

INPUT_LEN="${INPUT_LEN:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"

mkdir -p "${RESULT_DIR}"
log() { echo "[driver $(date +%H:%M:%S)] $*"; }

cleanup() {
    pkill -f 'hcp-ringattn-rust.*distributed-role' 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${LAPTOP_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    log "FAIL: $*"
    echo "FAIL: $*" > "${STATE_DIR}/STATUS"
    exit 1
}

# === Launch stack ===
cd "${REPO}" || fail "repo missing"

log "launching coordinator (white)"
setsid nohup env LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 3 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    --trace-jsonl "${STATE_DIR}/trace-n3l.jsonl" \
    >"${STATE_DIR}/coordinator-n3l.log" 2>&1 </dev/null &
sleep 2

log "launching worker 0 (white, CUDA)"
setsid nohup env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${PEARL_LAN}:${W1_PORT}" \
    --coordinator-addr "${WHITE_LAN}:${COORD_PORT}" \
    --num-domains 3 \
    >"${STATE_DIR}/worker0-white-n3l.log" 2>&1 </dev/null &
sleep 3

# ssh -n -f returns right after remote launch; remote setsid detaches the
# daemon so a later channel drop cannot SIGHUP it.
log "launching worker 1 (pearl, HIP) via ssh"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd '${PEARL_REPO}' && \
  setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir '${PEARL_MODEL}' \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${LAPTOP_TS}:${W2_PORT} \
    --coordinator-addr ${WHITE_TS}:${COORD_PORT} \
    --num-domains 3 \
    >'${STATE_DIR}/worker1-pearl-n3l.log' 2>&1 </dev/null" || fail "pearl worker launch ssh failed"

log "launching worker 2 (laptop, CUDA) via ssh"
ssh -n -f -o ConnectTimeout=15 "${LAPTOP_SSH}" "mkdir -p '${STATE_DIR}' && cd '${LAPTOP_REPO}' && \
  setsid env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 2 \
    --model-dir '${LAPTOP_MODEL}' \
    --listen-addr 0.0.0.0:${W2_PORT} \
    --next-peer-addr ${WHITE_TS}:${W0_PORT} \
    --coordinator-addr ${WHITE_TS}:${COORD_PORT} \
    --num-domains 3 \
    >'${STATE_DIR}/worker2-laptop-n3l.log' 2>&1 </dev/null" || fail "laptop worker launch ssh failed"

# === Wait for health (up to ~4 min) ===
log "waiting for 3 workers connected"
connected=0
for _ in $(seq 1 48); do
    sleep 5
    connected=$(curl -s --max-time 5 "http://127.0.0.1:${HTTP_PORT}/health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('workers_connected',0))" 2>/dev/null || echo 0)
    [ "${connected}" = "3" ] && break
done
[ "${connected}" = "3" ] || fail "workers_connected=${connected} after timeout"
log "healthy"

# === Bench ladder (client on white, server on white) ===
run_bench() { # label num_prompts rate max_concurrency
    local mc_arg=""
    [ "$4" -gt 0 ] && mc_arg="--max-concurrency $4"
    log "bench $1: num_prompts=$2 rate=$3 mc=$4"
    "${VLLM}" bench serve \
        --backend openai \
        --base-url "http://127.0.0.1:${HTTP_PORT}" \
        --endpoint /v1/completions \
        --model hcp-qwen2-0.5b \
        --tokenizer "${MODEL_DIR}" \
        --dataset-name random \
        --random-input-len "${INPUT_LEN}" \
        --random-output-len "${OUTPUT_LEN}" \
        --random-range-ratio 0.5 \
        --num-prompts "$2" \
        --request-rate "$3" \
        ${mc_arg} \
        --seed 42 \
        --save-result \
        --result-dir "${RESULT_DIR}" \
        --result-filename "$1.json" >"${STATE_DIR}/bench-$1.log" 2>&1
    log "bench $1 exit=$?"
}

run_bench n3l-l1 8 1 0
run_bench n3l-l2 8 inf 2
run_bench n3l-l3 16 inf 4

curl -s --max-time 10 "http://127.0.0.1:${HTTP_PORT}/metrics" > "${STATE_DIR}/metrics-n3l.json"
scp -q -o ConnectTimeout=15 "${PEARL_SSH}:${STATE_DIR}/worker1-pearl-n3l.log" "${STATE_DIR}/" 2>/dev/null || true
scp -q -o ConnectTimeout=15 "${LAPTOP_SSH}:${STATE_DIR}/worker2-laptop-n3l.log" "${STATE_DIR}/" 2>/dev/null || true

log "DONE"
echo "DONE" > "${STATE_DIR}/STATUS"
