#!/bin/bash
# Phase-3 decode route comparison N=3 — Q-ring (legacy decode) vs SelfDriving ring
# (stationary continuation) on white+pearl+laptop. Runs ON white.
#
# Topology (reuses phase3_7a_n3l): white(CUDA 4090) -(2.5GbE LAN)-> pearl(HIP 9060XT)
#   -(Tailscale)-> laptop(CUDA 4060) -(Tailscale)-> white.
# Uses the coordinator --continuation-segment path: prefill -> 1 legacy decode
# (Q-ring) -> m stationary continuation (self-driving) -> remaining legacy decode.
# Workers set HCP_PERF_LOG so both routes emit JSONL timing events.
#
# Usage: decode_route_compare_n3_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/hcp-dec-route-n3-${RUN_ID}"
REPO="${HOME}/hetero-cp-ringattn"
MODEL_DIR="${HOME}/models/Qwen2-0.5B"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_REPO="${HOME}/hetero-cp-ringattn"
PEARL_MODEL="${HOME}/hetero-cp-ringattn/models/Qwen2-0.5B"
LAPTOP_SSH="${LAPTOP_SSH:-stark@100.96.154.1}"
LAPTOP_REPO="${HOME}/hetero-cp-ringattn"
LAPTOP_MODEL="${HOME}/models/Qwen2-0.5B"
WHITE_LAN="${WHITE_LAN:-192.168.100.1}"
PEARL_LAN="${PEARL_LAN:-192.168.100.2}"
WHITE_TS="${WHITE_TS:-100.118.253.68}"
LAPTOP_TS="${LAPTOP_TS:-100.96.154.1}"

COORD_PORT=29800
W0_PORT=29801
W1_PORT=29802
W2_PORT=29803
HTTP_PORT=8082

# Scenario knobs
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog near the river bank on a bright sunny morning in early spring when the flowers bloom}"
SEGMENT="${SEGMENT:-and}"
MAX_TOKENS="${MAX_TOKENS:-2}"
CHUNK_SIZES="${CHUNK_SIZES:-}"

mkdir -p "${STATE_DIR}"
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

T_START=$(date +%s)
cd "${REPO}" || fail "repo missing"

# === Tokenize prompt + segment exactly as coordinator will ===
log "tokenizing prompt/segment"
"${HOME}/venv-bench/bin/python" - <<PY > "${STATE_DIR}/token-ids.json" || fail "tokenization failed"
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${MODEL_DIR}")
p = tok("${PROMPT}", add_special_tokens=True)["input_ids"]
s = tok("${SEGMENT}", add_special_tokens=True)["input_ids"]
print(json.dumps({"prompt": p, "segment": s}))
PY
PROMPT_IDS=$(python3 -c "import json; print(','.join(map(str, json.load(open('${STATE_DIR}/token-ids.json'))['prompt'])))")
SEGMENT_IDS=$(python3 -c "import json; print(','.join(map(str, json.load(open('${STATE_DIR}/token-ids.json'))['segment'])))")
N_PROMPT=$(python3 -c "import json; print(len(json.load(open('${STATE_DIR}/token-ids.json'))['prompt']))")
N_SEGMENT=$(python3 -c "import json; print(len(json.load(open('${STATE_DIR}/token-ids.json'))['segment']))")
log "prompt ${N_PROMPT} tokens, segment ${N_SEGMENT} tokens"

# === Launch stack (n3l topology) ===
log "launching coordinator (white) with continuation-segment path"
setsid nohup env LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 3 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    --prompt-token-ids "${PROMPT_IDS}" \
    --continuation-segment "${SEGMENT_IDS}" \
    --continuation-request-id 75 \
    --max-tokens "${MAX_TOKENS}" \
    --trace-jsonl "${STATE_DIR}/trace.jsonl" \
    --export-logits-dir "${STATE_DIR}/export" \
    ${CHUNK_SIZES:+--chunk-sizes "${CHUNK_SIZES}"} \
    >"${STATE_DIR}/coordinator.log" 2>&1 </dev/null &
sleep 2

log "launching worker 0 (white, CUDA) with HCP_PERF_LOG"
setsid nohup env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    HCP_PERF_LOG="${STATE_DIR}/perf-white.jsonl" \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${PEARL_LAN}:${W1_PORT}" \
    --coordinator-addr "${WHITE_LAN}:${COORD_PORT}" \
    --num-domains 3 \
    >"${STATE_DIR}/worker0-white.log" 2>&1 </dev/null &
sleep 3

log "launching worker 1 (pearl, HIP) via ssh with HCP_PERF_LOG"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd '${PEARL_REPO}' && \
  setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  HCP_PERF_LOG='${STATE_DIR}/perf-pearl.jsonl' \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir '${PEARL_MODEL}' \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${LAPTOP_TS}:${W2_PORT} \
    --coordinator-addr ${WHITE_LAN}:${COORD_PORT} \
    --num-domains 3 \
    >'${STATE_DIR}/worker1-pearl.log' 2>&1 </dev/null" || fail "pearl worker launch ssh failed"

log "launching worker 2 (laptop, CUDA) via ssh with HCP_PERF_LOG"
ssh -n -f -o ConnectTimeout=15 "${LAPTOP_SSH}" "mkdir -p '${STATE_DIR}' && cd '${LAPTOP_REPO}' && \
  setsid env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  HCP_PERF_LOG='${STATE_DIR}/perf-laptop.jsonl' \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 2 \
    --model-dir '${LAPTOP_MODEL}' \
    --listen-addr 0.0.0.0:${W2_PORT} \
    --next-peer-addr ${WHITE_TS}:${W0_PORT} \
    --coordinator-addr ${WHITE_TS}:${COORD_PORT} \
    --num-domains 3 \
    >'${STATE_DIR}/worker2-laptop.log' 2>&1 </dev/null" || fail "laptop worker launch ssh failed"

# === Wait for completion (coordinator exits after continuation E2E) ===
log "waiting for coordinator to finish continuation E2E"
for _ in $(seq 1 80); do
    sleep 5
    if ! pgrep -f 'hcp-ringattn-rust.*distributed-role coordinator' >/dev/null 2>&1; then
        sleep 2
        break
    fi
done

T_END=$(date +%s)
log "run finished after $((T_END - T_START))s"
grep -E "experimental stationary continuation|continuation E2E done|continuation E2E failed|generated ids|continuation KV byte admission" "${STATE_DIR}/coordinator.log" | tail -10 > "${STATE_DIR}/summary.txt" || true
echo "OK" > "${STATE_DIR}/STATUS"
echo "DONE $((T_END - T_START))s"
