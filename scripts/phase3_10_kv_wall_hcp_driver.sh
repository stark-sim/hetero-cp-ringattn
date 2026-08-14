#!/bin/bash
# Phase-3 10 KV-wall scan — HCP side driver (runs ON white).
# 3B N=2 ring with --max-batch-size 16; ladder: 30k-token prompts, mc sweep.
# Captures per-level bench json + coordinator admission log (accept/reject).
# Usage: phase3_10_kv_wall_hcp_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/kvwall-hcp-${RUN_ID}"
RESULT_DIR="${STATE_DIR}/bench"
VLLM_CLIENT="${HOME}/venv-bench/bin/vllm"
MODEL_W="${HOME}/models/Qwen2.5-3B-Instruct"
MODEL_P="${HOME}/models/Qwen2.5-3B-Instruct"
PEARL_SSH="${PEARL_SSH:-stark@192.168.100.2}"
WHITE_LAN="${WHITE_LAN:-192.168.100.1}"
PEARL_LAN="${PEARL_LAN:-192.168.100.2}"
INPUT_LEN="${INPUT_LEN:-30720}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
MAX_BATCH="${MAX_BATCH:-16}"
LEVELS="${LEVELS:-4 8 16 32}"

COORD_PORT=29800; W0_PORT=29801; W1_PORT=29802; HTTP_PORT=8082

mkdir -p "${RESULT_DIR}"
log() { echo "[kvwall-hcp $(date +%H:%M:%S)] $*"; }

cleanup() {
    pkill -f 'distributed-rol[e]' 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'distributed-rol[e]' || true" 2>/dev/null || true
}
trap cleanup EXIT

fail() { log "FAIL: $*"; echo "FAIL: $*" > "${STATE_DIR}/STATUS"; exit 1; }

cleanup
sleep 3

REPO="${HOME}/hetero-cp-ringattn"
cd "${REPO}" || fail "repo missing"

log "launching coordinator (max-batch-size ${MAX_BATCH})"
setsid nohup env LD_LIBRARY_PATH=${HOME}/libtorch/lib \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role coordinator --model-dir "${MODEL_W}" --num-domains 2 \
    --max-batch-size "${MAX_BATCH}" \
    --listen-addr 0.0.0.0:${COORD_PORT} --http-addr 0.0.0.0:${HTTP_PORT} \
    --trace-jsonl "${STATE_DIR}/trace.jsonl" >"${STATE_DIR}/coordinator.log" 2>&1 </dev/null &
sleep 2

log "launching worker0 (white CUDA)"
setsid nohup env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=${HOME}/libtorch/lib \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker --domain-id 0 --model-dir "${MODEL_W}" \
    --listen-addr 0.0.0.0:${W0_PORT} --next-peer-addr ${PEARL_LAN}:${W1_PORT} \
    --coordinator-addr ${WHITE_LAN}:${COORD_PORT} --num-domains 2 \
    >"${STATE_DIR}/worker0.log" 2>&1 </dev/null &
sleep 2

log "launching worker1 (pearl HIP)"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd ~/hetero-cp-ringattn && \
  setsid env LD_PRELOAD=${HOME}/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=${HOME}/libtorch/lib \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker --domain-id 1 --model-dir '${MODEL_P}' \
    --listen-addr 0.0.0.0:${W1_PORT} --next-peer-addr ${WHITE_LAN}:${W0_PORT} \
    --coordinator-addr ${WHITE_LAN}:${COORD_PORT} --num-domains 2 \
    >'${STATE_DIR}/worker1.log' 2>&1 </dev/null" || fail "pearl launch failed"

log "waiting for 2 workers"
ok=0
for _ in $(seq 1 48); do
    sleep 5
    n=$(curl -s --max-time 5 http://127.0.0.1:${HTTP_PORT}/health | python3 -c 'import json,sys; print(json.load(sys.stdin).get("workers_connected",0))' 2>/dev/null || echo 0)
    [ "${n}" = "2" ] && ok=1 && break
done
[ "${ok}" = 1 ] || fail "workers not connected"
grep -E "capacity=|max_batch_size" "${STATE_DIR}/coordinator.log" | head -5

for mc in ${LEVELS}; do
    log "level mc=${mc}: ${mc} prompts x ${INPUT_LEN} tokens"
    "${VLLM_CLIENT}" bench serve --backend openai --base-url http://127.0.0.1:${HTTP_PORT} \
        --endpoint /v1/completions --model hcp --tokenizer "${MODEL_W}" \
        --dataset-name random --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
        --random-range-ratio 0.05 --num-prompts "${mc}" --request-rate inf --max-concurrency "${mc}" \
        --seed 42 --save-result --result-dir "${RESULT_DIR}" --result-filename "hcp-mc${mc}.json" \
        >"${STATE_DIR}/bench-mc${mc}.log" 2>&1
    log "level mc=${mc} bench exit=$?"
    curl -s --max-time 5 http://127.0.0.1:${HTTP_PORT}/metrics >"${STATE_DIR}/metrics-mc${mc}.json" 2>/dev/null || true
done

# admission decisions are the wall signal on this side
grep "KV byte admission" "${STATE_DIR}/coordinator.log" > "${STATE_DIR}/admissions.log" 2>/dev/null || true
scp -q -o ConnectTimeout=15 "${PEARL_SSH}:${STATE_DIR}/worker1.log" "${STATE_DIR}/" 2>/dev/null || true

log "DONE"
echo "DONE" > "${STATE_DIR}/STATUS"
