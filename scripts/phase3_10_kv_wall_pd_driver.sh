#!/bin/bash
# Phase-3 10 KV-wall scan — vLLM PD side driver (runs ON white).
# 3B PD stack (prefill white CUDA + decode pearl ROCm, NixlConnector pull);
# ladder: 30k-token prompts, mc sweep; captures vllm:num_preemptions deltas
# per level plus the decode-side KV pool size from the startup log.
# Usage: phase3_10_kv_wall_pd_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/kvwall-pd-${RUN_ID}"
RESULT_DIR="${STATE_DIR}/bench"
VLLM_CLIENT="${HOME}/venv-bench/bin/vllm"
VLLM_SERVE="${HOME}/miniconda3/envs/vllm-v1/bin"
MODEL_W="${HOME}/models/Qwen2.5-3B-Instruct"
PEARL_SSH="${PEARL_SSH:-stark@192.168.100.2}"
PEARL_VLLM="${HOME}/miniconda3/envs/vllm-rocm/bin"
MODEL_P="${HOME}/models/Qwen2.5-3B-Instruct"
WHITE_LAN="${WHITE_LAN:-192.168.100.1}"
PEARL_LAN="${PEARL_LAN:-192.168.100.2}"
INPUT_LEN="${INPUT_LEN:-30720}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
LEVELS="${LEVELS:-4 8 16 32}"

PREFILL_PORT=8100; DECODE_PORT=8200; PROXY_PORT=18000

mkdir -p "${RESULT_DIR}"
log() { echo "[kvwall-pd $(date +%H:%M:%S)] $*"; }

cleanup() {
    pkill -f 'vllm ser[v]e' 2>/dev/null || true
    pkill -f 'disagg_pro[x]y' 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'vllm ser[v]e' || true" 2>/dev/null || true
}
trap cleanup EXIT

fail() { log "FAIL: $*"; echo "FAIL: $*" > "${STATE_DIR}/STATUS"; exit 1; }

cleanup
sleep 3

log "launching prefill (white, CUDA, 3B)"
setsid nohup env PATH="${VLLM_SERVE}:$PATH" \
    VLLM_NIXL_SIDE_CHANNEL_HOST="${WHITE_LAN}" VLLM_NIXL_SIDE_CHANNEL_PORT=5555 \
    "${VLLM_SERVE}/vllm" serve "${MODEL_W}" \
    --host "${WHITE_LAN}" --port ${PREFILL_PORT} \
    --served-model-name qwen25-3b \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    >"${STATE_DIR}/prefill.log" 2>&1 </dev/null &

log "launching decode (pearl, ROCm, 3B)"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && \
  setsid env PATH=${PEARL_VLLM}:\$PATH \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${PEARL_LAN} VLLM_NIXL_SIDE_CHANNEL_PORT=5556 \
    ${PEARL_VLLM}/vllm serve '${MODEL_P}' \
    --host ${PEARL_LAN} --port ${DECODE_PORT} \
    --served-model-name qwen25-3b --enforce-eager \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
    >'${STATE_DIR}/decode.log' 2>&1 </dev/null" || fail "pearl decode launch failed"

log "waiting for prefill+decode"
for what in prefill decode; do
    if [ "${what}" = prefill ]; then url="http://${WHITE_LAN}:${PREFILL_PORT}/v1/models"; else url="http://${PEARL_LAN}:${DECODE_PORT}/v1/models"; fi
    ok=0
    for _ in $(seq 1 72); do
        sleep 5
        curl -s --max-time 5 "${url}" | grep -q qwen25-3b && ok=1 && break
    done
    [ "${ok}" = 1 ] || fail "${what} not ready (see ${STATE_DIR}/${what}.log)"
done
log "both ready"

# KV pool sizes (ground truth for the wall position)
grep -h "GPU KV cache size" "${STATE_DIR}/prefill.log" | tail -1 > "${STATE_DIR}/kv-pool.txt" 2>/dev/null || true
ssh -o ConnectTimeout=15 "${PEARL_SSH}" "grep -h 'GPU KV cache size' '${STATE_DIR}/decode.log' | tail -1" >> "${STATE_DIR}/kv-pool.txt" 2>/dev/null || true
cat "${STATE_DIR}/kv-pool.txt" || true

log "launching proxy"
setsid nohup env PATH="${VLLM_SERVE}:$PATH" ADMIN_API_KEY=dummy \
    "${VLLM_SERVE}/python" "${HOME}/vllm/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py" \
    --model qwen25-3b --prefill "${WHITE_LAN}:${PREFILL_PORT}" --decode "${PEARL_LAN}:${DECODE_PORT}" \
    --port ${PROXY_PORT} >"${STATE_DIR}/proxy.log" 2>&1 </dev/null &
ok=0
for _ in $(seq 1 12); do
    sleep 5
    curl -s --max-time 5 "http://127.0.0.1:${PROXY_PORT}/status" | grep -q prefill_node_count && ok=1 && break
done
[ "${ok}" = 1 ] || fail "proxy not ready"

preemptions() {
    curl -s --max-time 5 "http://${PEARL_LAN}:${DECODE_PORT}/metrics" 2>/dev/null | grep -E '^vllm:num_preemptions' | awk '{s+=$2} END {print s+0}'
}

for mc in ${LEVELS}; do
    before=$(preemptions)
    log "level mc=${mc}: ${mc} prompts x ${INPUT_LEN} tokens (preemptions before=${before})"
    "${VLLM_CLIENT}" bench serve --backend openai --base-url http://127.0.0.1:${PROXY_PORT} \
        --endpoint /v1/completions --model qwen25-3b --tokenizer "${MODEL_W}" \
        --dataset-name random --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
        --random-range-ratio 0.05 --num-prompts "${mc}" --request-rate inf --max-concurrency "${mc}" \
        --seed 42 --save-result --result-dir "${RESULT_DIR}" --result-filename "pd-mc${mc}.json" \
        >"${STATE_DIR}/bench-mc${mc}.log" 2>&1
    rc=$?
    after=$(preemptions)
    log "level mc=${mc} bench exit=${rc} preemptions ${before} -> ${after}"
    echo "{\"mc\": ${mc}, \"preemptions_before\": ${before}, \"preemptions_after\": ${after}}" >> "${STATE_DIR}/preemptions.jsonl"
done

scp -q -o ConnectTimeout=15 "${PEARL_SSH}:${STATE_DIR}/decode.log" "${STATE_DIR}/" 2>/dev/null || true

log "DONE"
echo "DONE" > "${STATE_DIR}/STATUS"
