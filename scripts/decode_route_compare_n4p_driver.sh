#!/bin/bash
# Phase-3 decode route comparison N=4 — 4 independent worker PROCESSES across 2 hosts.
# white runs domain 0 and domain 2 (two separate --domain-id processes);
# pearl runs domain 1 and domain 3. Ring topology:
#   0(white) -> 1(pearl) -> 2(white) -> 3(pearl) -> 0(white)
# i.e. white -> pearl -> white -> pearl -> white (cross-host on every hop).
# Uses the coordinator --continuation-segment path (prefill + 1 legacy decode Q-ring
# + m stationary self-driving + remaining legacy decode) with HCP_PERF_LOG per process.
#
# Usage: decode_route_compare_n4p_driver.sh <RUN_ID>
set -uo pipefail

RUN_ID="${1:?RUN_ID required}"
STATE_DIR="/tmp/hcp-dec-route-n4p-${RUN_ID}"
REPO="${HOME}/hetero-cp-ringattn"
MODEL_DIR="${HOME}/models/Qwen2-0.5B"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_REPO="${HOME}/hetero-cp-ringattn"
PEARL_MODEL="${HOME}/hetero-cp-ringattn/models/Qwen2-0.5B"
WHITE_LAN="${WHITE_LAN:-192.168.100.1}"
PEARL_LAN="${PEARL_LAN:-192.168.100.2}"

COORD_PORT=29800
# 4 processes, 4 ports:
#   d0 white  :29801  -> d1 pearl  :29802
#   d1 pearl  :29802  -> d2 white  :29803
#   d2 white  :29803  -> d3 pearl  :29804
#   d3 pearl  :29804  -> d0 white  :29801
P0=29801
P1=29802
P2=29803
P3=29804
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

# === Launch coordinator (white), num-domains 4 ===
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

# === White: worker process for domain 0 (separate process) ===
log "launching white worker domain 0"
setsid nohup env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    HCP_PERF_LOG="${STATE_DIR}/perf-white-d0.jsonl" \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${P0}" \
    --next-peer-addr "${PEARL_LAN}:${P1}" \
    --coordinator-addr "${WHITE_LAN}:${COORD_PORT}" \
    --num-domains 4 \
    >"${STATE_DIR}/worker0-white-d0.log" 2>&1 </dev/null &
sleep 2

# === Pearl: worker process for domain 1 ===
log "launching pearl worker domain 1"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd '${PEARL_REPO}' && \
  setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  HCP_PERF_LOG='${STATE_DIR}/perf-pearl-d1.jsonl' \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 1 \
    --model-dir '${PEARL_MODEL}' \
    --listen-addr 0.0.0.0:${P1} \
    --next-peer-addr ${WHITE_LAN}:${P2} \
    --coordinator-addr ${WHITE_LAN}:${COORD_PORT} \
    --num-domains 4 \
    >'${STATE_DIR}/worker1-pearl-d1.log' 2>&1 </dev/null" || fail "pearl d1 launch failed"
sleep 2

# === White: worker process for domain 2 ===
log "launching white worker domain 2"
setsid nohup env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
    HCP_PERF_LOG="${STATE_DIR}/perf-white-d2.jsonl" \
    ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 2 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${P2}" \
    --next-peer-addr "${PEARL_LAN}:${P3}" \
    --coordinator-addr "${WHITE_LAN}:${COORD_PORT}" \
    --num-domains 4 \
    >"${STATE_DIR}/worker2-white-d2.log" 2>&1 </dev/null &
sleep 2

# === Pearl: worker process for domain 3 ===
log "launching pearl worker domain 3"
ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd '${PEARL_REPO}' && \
  setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
  HCP_PERF_LOG='${STATE_DIR}/perf-pearl-d3.jsonl' \
  ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker \
    --domain-id 3 \
    --model-dir '${PEARL_MODEL}' \
    --listen-addr 0.0.0.0:${P3} \
    --next-peer-addr ${WHITE_LAN}:${P0} \
    --coordinator-addr ${WHITE_LAN}:${COORD_PORT} \
    --num-domains 4 \
    >'${STATE_DIR}/worker3-pearl-d3.log' 2>&1 </dev/null" || fail "pearl d3 launch failed"

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
grep -E "experimental stationary continuation|continuation E2E done|continuation E2E failed|generated ids|continuation KV byte admission|worker .* connected" "${STATE_DIR}/coordinator.log" | tail -15 > "${STATE_DIR}/summary.txt" || true
echo "OK" > "${STATE_DIR}/STATUS"
echo "DONE $((T_END - T_START))s"
