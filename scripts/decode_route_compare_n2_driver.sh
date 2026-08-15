#!/bin/bash
# Phase-3 decode route comparison N=2 — Q-ring (legacy decode) vs SelfDriving ring
# (stationary continuation) on white+pearl LAN. Runs ON white.
#
# Uses the coordinator --continuation-segment path, which internally runs:
#   prefill -> 1 legacy decode (Q-ring) -> m stationary continuation (self-driving)
#             -> remaining legacy decode steps
# Workers set HCP_PERF_LOG so both routes emit JSONL timing events:
#   ring_decode (Q-ring, per layer) and stationary_continuation (per segment).
# Mac polls STATE_DIR/STATUS and fetches artifacts.
#
# Usage: decode_route_compare_n2_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/hcp-dec-route-${RUN_ID}"
REPO="${HOME}/hetero-cp-ringattn"
MODEL_DIR="${HOME}/models/Qwen2-0.5B"
PEARL_MODEL="${HOME}/hetero-cp-ringattn/models/Qwen2-0.5B"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
WHITE_LAN="${WHITE_LAN:-192.168.100.1}"
PEARL_LAN="${PEARL_LAN:-192.168.100.2}"

COORD_PORT=29800
W0_PORT=29801
W1_PORT=29802
HTTP_PORT=8082

# Scenario knobs
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog near the river bank on a bright sunny morning}"
SEGMENT="${SEGMENT:-and the wind blows softly through the trees on a warm day in spring}"
MAX_TOKENS="${MAX_TOKENS:-4}"
CHUNK_SIZES="${CHUNK_SIZES:-}"

mkdir -p "${STATE_DIR}"
log() { echo "[driver $(date +%H:%M:%S)] $*"; }

cleanup() {
    pkill -f 'hcp-ringattn-rust.*distributed-role' 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
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
import json, os
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

# === Launch stack ===
log "launching coordinator (white) with continuation-segment path"
setsid nohup env LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 2 \
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
    --num-domains 2 \
    >"${STATE_DIR}/worker0-white.log" 2>&1 </dev/null &
sleep 3

log "launching worker 1 (pearl, HIP) via ssh with HCP_PERF_LOG"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd '${HOME}/hetero-cp-ringattn' && \
  setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  HCP_PERF_LOG='${STATE_DIR}/perf-pearl.jsonl' \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir '${PEARL_MODEL}' \
    --listen-addr 0.0.0.0:${W1_PORT} \
    --next-peer-addr ${WHITE_LAN}:${W0_PORT} \
    --coordinator-addr ${WHITE_LAN}:${COORD_PORT} \
    --num-domains 2 \
    >'${STATE_DIR}/worker1-pearl.log' 2>&1 </dev/null" || fail "pearl worker launch ssh failed"

# === Wait for completion (coordinator exits after continuation E2E) ===
log "waiting for coordinator to finish continuation E2E"
for _ in $(seq 1 60); do
    sleep 5
    if ! pgrep -f 'hcp-ringattn-rust.*distributed-role coordinator' >/dev/null 2>&1; then
        sleep 2
        break
    fi
done

T_END=$(date +%s)
log "run finished after $((T_END - T_START))s"
grep -E "experimental stationary continuation|continuation E2E done|continuation E2E failed|generated ids" "${STATE_DIR}/coordinator.log" | tail -10 > "${STATE_DIR}/summary.txt" || true
echo "OK" > "${STATE_DIR}/STATUS"
echo "DONE $((T_END - T_START))s"
