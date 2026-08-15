#!/bin/bash
# Phase-3 decode route comparison N=4 (2-host emulation) — white hosts domains 0,1;
# pearl hosts domains 2,3. Ring: 0->1 (white local) ->2 (cross) ->3 (pearl local) ->0 (cross).
# Runs the coordinator --continuation-segment path (prefill + 1 legacy decode + m stationary
# + remaining legacy decode) with HCP_PERF_LOG on each worker process.
#
# Usage: decode_route_compare_n4_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/hcp-dec-route-n4-${RUN_ID}"
REPO="${HOME}/hetero-cp-ringattn"
MODEL_DIR="${HOME}/models/Qwen2-0.5B"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_REPO="${HOME}/hetero-cp-ringattn"
PEARL_MODEL="${HOME}/hetero-cp-ringattn/models/Qwen2-0.5B"
WHITE_LAN="${WHITE_LAN:-192.168.100.1}"
PEARL_LAN="${PEARL_LAN:-192.168.100.2}"

COORD_PORT=29800
# white: d0 port 29801, d1 port 29802; pearl: d2 port 29803, d3 port 29804
W0_PORT=29801
W1_PORT=29802
W2_PORT=29803
W3_PORT=29804
HTTP_PORT=8082

PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog near the river bank on a bright sunny morning in early spring when the flowers bloom}"
SEGMENT="${SEGMENT:-and}"
MAX_TOKENS="${MAX_TOKENS:-2}"
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

# === Tokenize prompt + segment ===
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

# === Launch coordinator (white) ===
log "launching coordinator (white), num-domains 4"
setsid nohup env LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 4 \
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

# === White: single process hosting domains 0,1 ===
# Ring: d0->d1 (white local), d1->d2 (cross to pearl)
log "launching white multi-worker (domains 0,1)"
setsid nohup env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    HCP_PERF_LOG="${STATE_DIR}/perf-white.jsonl" \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --local-domain-ids 0,1 \
    --listen-addrs "${WHITE_LAN}:${W0_PORT},${WHITE_LAN}:${W1_PORT}" \
    --next-peer-addrs "${WHITE_LAN}:${W1_PORT},${PEARL_LAN}:${W2_PORT}" \
    --coordinator-addr "${WHITE_LAN}:${COORD_PORT}" \
    --num-domains 4 \
    --model-dir "${MODEL_DIR}" \
    >"${STATE_DIR}/worker0-white.log" 2>&1 </dev/null &
sleep 3

# === Pearl: single process hosting domains 2,3 ===
# Ring: d2->d3 (pearl local), d3->d0 (cross back to white)
log "launching pearl multi-worker (domains 2,3)"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd '${PEARL_REPO}' && \
  setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  HCP_PERF_LOG='${STATE_DIR}/perf-pearl.jsonl' \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --local-domain-ids 2,3 \
    --listen-addrs '${PEARL_LAN}:${W2_PORT},${PEARL_LAN}:${W3_PORT}' \
    --next-peer-addrs '${PEARL_LAN}:${W3_PORT},${WHITE_LAN}:${W0_PORT}' \
    --coordinator-addr '${WHITE_LAN}:${COORD_PORT}' \
    --num-domains 4 \
    --model-dir '${PEARL_MODEL}' \
    >'${STATE_DIR}/worker1-pearl.log' 2>&1 </dev/null" || fail "pearl worker launch ssh failed"

# === Wait for completion ===
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
grep -E "experimental stationary continuation|continuation E2E done|continuation E2E failed|generated ids|continuation KV byte admission|worker .* connected" "${STATE_DIR}/coordinator.log" | tail -15 > "${STATE_DIR}/summary.txt" || true
echo "OK" > "${STATE_DIR}/STATUS"
echo "DONE $((T_END - T_START))s"
