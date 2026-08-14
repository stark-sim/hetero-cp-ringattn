#!/bin/bash
# Phase-3 9 PD driver — runs ON white (invoked by phase3_9_pd_interleaved_baseline.sh).
#
# vLLM PD-disaggregation stack, all on the white/pearl wired LAN:
#   prefill instance (white, CUDA, 192.168.100.1:8100) + decode instance
#   (pearl, ROCm via ssh from white, 192.168.100.2:8200) + disagg proxy
#   (white, 127.0.0.1:18000); vllm bench serve hits 127.0.0.1:18000.
# KV crosses enp10s0<->enp8s0 via NIXL/UCX (NixlConnector pull mode).
# The Mac only polls STATE_DIR/STATUS and fetches artifacts afterwards.
#
# Usage: phase3_9_pd_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/vllm-pd-${RUN_ID}"
RESULT_DIR="${STATE_DIR}/bench"
VLLM_CLIENT="${VLLM_CLIENT:-${HOME}/venv-bench/bin/vllm}"
VLLM_SERVE="${HOME}/miniconda3/envs/vllm-v1/bin"
MODEL_DIR="${HOME}/models/Qwen2-0.5B"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_VLLM="${HOME}/miniconda3/envs/vllm-rocm/bin"
PEARL_MODEL="${HOME}/hetero-cp-ringattn/models/Qwen2-0.5B"
WHITE_LAN="${WHITE_LAN:-192.168.100.1}"
PEARL_LAN="${PEARL_LAN:-192.168.100.2}"

PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=18000
INPUT_LEN="${INPUT_LEN:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"

mkdir -p "${RESULT_DIR}"
log() { echo "[pd-driver $(date +%H:%M:%S)] $*"; }

cleanup() {
    pkill -f 'vllm ser[v]e' 2>/dev/null || true
    pkill -f 'disagg_pro[x]y' 2>/dev/null || true
    pkill -f 'distributed-rol[e]' 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'vllm ser[v]e' || true; pkill -f 'distributed-rol[e]' || true" 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    log "FAIL: $*"
    echo "FAIL: $*" > "${STATE_DIR}/STATUS"
    exit 1
}

wait_vram_free_white() {
    for _ in $(seq 1 24); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used}" -lt 1000 ] && return 0
        sleep 5
    done
    return 1
}

# === Pre-clean (fresh stack per rep; also releases VRAM for interleaved HCP reps) ===
cleanup
sleep 3
wait_vram_free_white || fail "white VRAM not freed"
ssh -o ConnectTimeout=15 "${PEARL_SSH}" 'for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do u=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oE "Used Memory \(B\): [0-9]+" | grep -oE "[0-9]+" | head -1); [ -n "$u" ] && [ "$u" -lt 500000000 ] && exit 0; sleep 5; done; exit 1' \
    || fail "pearl VRAM not freed"

# === Launch prefill (white, CUDA) ===
log "launching prefill (white, CUDA)"
setsid nohup env PATH="${VLLM_SERVE}:$PATH" \
    VLLM_NIXL_SIDE_CHANNEL_HOST="${WHITE_LAN}" VLLM_NIXL_SIDE_CHANNEL_PORT=5555 \
    "${VLLM_SERVE}/vllm" serve "${MODEL_DIR}" \
    --host "${WHITE_LAN}" --port ${PREFILL_PORT} \
    --served-model-name qwen2-05b \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    >"${STATE_DIR}/prefill.log" 2>&1 </dev/null &

# === Launch decode (pearl, ROCm; enforce-eager: hipblaslt FULL cudagraph capture fails) ===
log "launching decode (pearl, ROCm) via ssh"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && \
  setsid env PATH=${PEARL_VLLM}:\$PATH \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${PEARL_LAN} VLLM_NIXL_SIDE_CHANNEL_PORT=5556 \
    ${PEARL_VLLM}/vllm serve ${PEARL_MODEL} \
    --host ${PEARL_LAN} --port ${DECODE_PORT} \
    --served-model-name qwen2-05b --enforce-eager \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
    >'${STATE_DIR}/decode.log' 2>&1 </dev/null" || fail "pearl decode launch ssh failed"

# === Wait for both instances (up to ~6 min each) ===
log "waiting for prefill+decode readiness"
for what in prefill decode; do
    if [ "${what}" = prefill ]; then url="http://${WHITE_LAN}:${PREFILL_PORT}/v1/models"; else url="http://${PEARL_LAN}:${DECODE_PORT}/v1/models"; fi
    ok=0
    for _ in $(seq 1 72); do
        sleep 5
        curl -s --max-time 5 "${url}" | grep -q qwen2-05b && ok=1 && break
    done
    [ "${ok}" = 1 ] || fail "${what} not ready after 6min (see ${STATE_DIR}/${what}.log)"
done
log "both instances ready"

# === Launch proxy ===
log "launching disagg proxy on 127.0.0.1:${PROXY_PORT}"
setsid nohup env PATH="${VLLM_SERVE}:$PATH" ADMIN_API_KEY=dummy \
    "${VLLM_SERVE}/python" "${HOME}/vllm/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py" \
    --model qwen2-05b \
    --prefill "${WHITE_LAN}:${PREFILL_PORT}" \
    --decode "${PEARL_LAN}:${DECODE_PORT}" \
    --port ${PROXY_PORT} >"${STATE_DIR}/proxy.log" 2>&1 </dev/null &
ok=0
for _ in $(seq 1 12); do
    sleep 5
    curl -s --max-time 5 "http://127.0.0.1:${PROXY_PORT}/status" | grep -q prefill_node_count && ok=1 && break
done
[ "${ok}" = 1 ] || fail "proxy not ready"

# === Bench ladder (identical params to the HCP N=2 ladder) ===
run_bench() { # label num_prompts rate max_concurrency
    local mc_arg=""
    [ "$4" -gt 0 ] && mc_arg="--max-concurrency $4"
    log "bench $1: num_prompts=$2 rate=$3 mc=$4"
    "${VLLM_CLIENT}" bench serve \
        --backend openai \
        --base-url "http://127.0.0.1:${PROXY_PORT}" \
        --endpoint /v1/completions \
        --model qwen2-05b \
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

run_bench pd-l1 8 1 0
run_bench pd-l2 8 inf 2
run_bench pd-l3 16 inf 4

scp -q -o ConnectTimeout=15 "${PEARL_SSH}:${STATE_DIR}/decode.log" "${STATE_DIR}/" 2>/dev/null || true

log "DONE"
echo "DONE" > "${STATE_DIR}/STATUS"
