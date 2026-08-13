#!/bin/bash
# Phase-3 8: continuation service-path E2E (HTTP session two-phase) on the
# N=2 white+pearl LAN.
#
# Proves the route-B stationary continuation on the real HTTP service path:
#   request 1 (keep_kv)  — normal prefill+decode, KV stays resident;
#   request 2 (append)   — stationary continuation on the frozen prefix KV,
#                          then legacy decode, same worker-side request id;
#   request 3 (negative) — append with unknown session_id must error out.
# The distributed prefill/continuation logits dumped by the coordinator
# (--export-logits-dir) are compared against the contiguous single-node
# golden (route_b_cross_node_smoke golden) with compare_route_b_dumps.py
# under the existing tie-aware tolerances.
#
# Topology: coordinator + worker 0 (CUDA) on white (LAN 192.168.8.172),
# worker 1 (HIP) on pearl (LAN 192.168.8.176). The Mac is control-only: it
# syncs/builds both remotes, launches the white-side driver half of this
# script with ssh -n -f + setsid, polls ${STATE_DIR}/STATUS, and fetches the
# artifacts afterwards (network drops cannot abort the run mid-flight).
#
# Usage (Mac):   bash scripts/test_phase3_8_continuation_e2e.sh
# Internal:      test_phase3_8_continuation_e2e.sh --driver <RUN_ID>   (runs ON white)
set -uo pipefail

# === Shared configuration ===
WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
WHITE_LAN="${WHITE_LAN:-192.168.8.172}"
PEARL_LAN="${PEARL_LAN:-192.168.8.176}"
REPO_DIR="${REPO_DIR:-hetero-cp-ringattn}"

COORD_PORT=29800
W0_PORT=29801
W1_PORT=29802
HTTP_PORT=8082

# Two-phase scenario. k1=2 => exactly one fed decode token before the append
# (the last sampled token is returned but never fed back), k2=4.
PREFIX="${PREFIX:-The quick brown fox jumps over the lazy dog near the river bank on a bright}"
SEGMENT="${SEGMENT:- sunny morning in early spring}"
K1=2
K2=4
SESSION_ID="s1"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\''/g"
    printf "'"
}

# =====================================================================
# White-side driver half (runs ON white, launched via setsid from the Mac).
# =====================================================================
if [ "${1:-}" = "--driver" ]; then
    RUN_ID="${2:?RUN_ID required}"
    STATE_DIR="/tmp/hcp-cont-e2e-${RUN_ID}"
    REPO="${HOME}/${REPO_DIR}"
    VENV_PYTHON="${VENV_PYTHON:-${HOME}/venv-bench/bin/python}"
    MODEL_DIR="${HOME}/models/Qwen2-0.5B"
    PEARL_MODEL="${HOME}/hetero-cp-ringattn/models/Qwen2-0.5B"

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

    # === Launch stack (identical roles/ports to the 7a N=2 driver) ===
    log "launching coordinator (white), export dir ${STATE_DIR}/export"
    setsid nohup env LD_LIBRARY_PATH=/home/stark/libtorch/lib \
        ./rust/target/release/hcp-ringattn-rust \
        --distributed-role coordinator \
        --model-dir "${MODEL_DIR}" \
        --num-domains 2 \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        --http-addr "0.0.0.0:${HTTP_PORT}" \
        --trace-jsonl "${STATE_DIR}/trace.jsonl" \
        --export-logits-dir "${STATE_DIR}/export" \
        --session-continuation-tokens 64 \
        >"${STATE_DIR}/coordinator.log" 2>&1 </dev/null &
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
        --num-domains 2 \
        >"${STATE_DIR}/worker0-white.log" 2>&1 </dev/null &
    sleep 3

    log "launching worker 1 (pearl, HIP) via ssh"
    # ssh -n -f returns immediately; remote-side redirection keeps the channel
    # independent of the daemon's lifetime (7a pattern).
    ssh -n -f -o ConnectTimeout=15 "${PEARL_SSH}" "mkdir -p '${STATE_DIR}' && cd '${HOME}/hetero-cp-ringattn' && \
      setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
      ./rust/target/release/hcp-ringattn-rust \
        --distributed-role worker \
        --domain-id 1 \
        --model-dir '${PEARL_MODEL}' \
        --listen-addr 0.0.0.0:${W1_PORT} \
        --next-peer-addr ${WHITE_LAN}:${W0_PORT} \
        --coordinator-addr ${WHITE_LAN}:${COORD_PORT} \
        --num-domains 2 \
        >'${STATE_DIR}/worker1-pearl.log' 2>&1 </dev/null" || fail "pearl worker launch ssh failed"

    # === Wait for health (up to ~4 min) ===
    log "waiting for 2 workers connected"
    connected=0
    for _ in $(seq 1 48); do
        sleep 5
        connected=$(curl -s --max-time 5 "http://127.0.0.1:${HTTP_PORT}/health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('workers_connected',0))" 2>/dev/null || echo 0)
        [ "${connected}" = "2" ] && break
    done
    [ "${connected}" = "2" ] || fail "workers_connected=${connected} after timeout"
    T_HEALTH=$(date +%s)
    log "healthy ($((T_HEALTH - T_START))s)"

    # === Tokenize prefix + segment exactly the way the coordinator will ===
    # Coordinator: tokenizer.encode(prompt, true) i.e. add_special_tokens=True,
    # each request independently. Mirror with HF add_special_tokens=True; the
    # HTTP usage.prompt_tokens assertions below cross-check the id counts.
    log "tokenizing prefix/segment"
    PREFIX="${PREFIX}" SEGMENT="${SEGMENT}" MODEL_DIR="${MODEL_DIR}" "${VENV_PYTHON}" - <<'PY' > "${STATE_DIR}/token-ids.json" || fail "tokenization failed"
import json, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
prefix_ids = tok(os.environ["PREFIX"], add_special_tokens=True)["input_ids"]
segment_ids = tok(os.environ["SEGMENT"], add_special_tokens=True)["input_ids"]
print(json.dumps({"prefix": prefix_ids, "segment": segment_ids}))
PY
    PROMPT_IDS=$(python3 -c "import json; print(','.join(map(str, json.load(open('${STATE_DIR}/token-ids.json'))['prefix'])))")
    SEGMENT_IDS=$(python3 -c "import json; print(','.join(map(str, json.load(open('${STATE_DIR}/token-ids.json'))['segment'])))")
    N_PROMPT=$(python3 -c "import json; print(len(json.load(open('${STATE_DIR}/token-ids.json'))['prefix']))")
    N_SEGMENT=$(python3 -c "import json; print(len(json.load(open('${STATE_DIR}/token-ids.json'))['segment']))")
    log "prefix ${N_PROMPT} tokens, segment ${N_SEGMENT} tokens"
    [ "${N_SEGMENT}" -ge 1 ] || fail "empty segment tokenization"

    # === Request 1: keep_kv prefix ===
    log "request 1: keep_kv prefix (max_tokens=${K1}, temperature=0)"
    PREFIX="${PREFIX}" SESSION_ID="${SESSION_ID}" K1="${K1}" python3 -c "
import json, os
print(json.dumps({'prompt': os.environ['PREFIX'], 'max_tokens': int(os.environ['K1']),
                  'temperature': 0, 'session_id': os.environ['SESSION_ID'], 'keep_kv': True}))
" > "${STATE_DIR}/request1.json"
    curl -s --max-time 300 -H 'Content-Type: application/json' \
        -d @"${STATE_DIR}/request1.json" \
        "http://127.0.0.1:${HTTP_PORT}/v1/completions" > "${STATE_DIR}/response1.json" \
        || fail "request 1 curl failed"
    T_REQ1=$(date +%s)

    # === Request 2: append segment on the frozen KV ===
    log "request 2: append segment (max_tokens=${K2}, temperature=0)"
    SEGMENT="${SEGMENT}" SESSION_ID="${SESSION_ID}" K2="${K2}" python3 -c "
import json, os
print(json.dumps({'prompt': os.environ['SEGMENT'], 'max_tokens': int(os.environ['K2']),
                  'temperature': 0, 'session_id': os.environ['SESSION_ID'], 'append': True}))
" > "${STATE_DIR}/request2.json"
    curl -s --max-time 300 -H 'Content-Type: application/json' \
        -d @"${STATE_DIR}/request2.json" \
        "http://127.0.0.1:${HTTP_PORT}/v1/completions" > "${STATE_DIR}/response2.json" \
        || fail "request 2 curl failed"
    T_REQ2=$(date +%s)

    # === Request 3 (negative): append with unknown session must error ===
    log "request 3 (negative): append with unknown session"
    curl -s --max-time 60 -H 'Content-Type: application/json' \
        -d '{"prompt":" rogue","max_tokens":2,"temperature":0,"session_id":"no-such-session","append":true}' \
        "http://127.0.0.1:${HTTP_PORT}/v1/completions" > "${STATE_DIR}/response3.json" \
        || fail "request 3 curl failed"

    curl -s --max-time 10 "http://127.0.0.1:${HTTP_PORT}/metrics" > "${STATE_DIR}/metrics.json"

    # === Response + metrics assertions ===
    STATE_DIR="${STATE_DIR}" N_PROMPT="${N_PROMPT}" N_SEGMENT="${N_SEGMENT}" K1="${K1}" K2="${K2}" python3 - <<'PY' || fail "response assertions failed"
import json, os, sys

state = os.environ["STATE_DIR"]
n_prompt, n_segment = int(os.environ["N_PROMPT"]), int(os.environ["N_SEGMENT"])
k1, k2 = int(os.environ["K1"]), int(os.environ["K2"])

def load(name):
    with open(os.path.join(state, name)) as f:
        return json.load(f)

r1, r2, r3 = load("response1.json"), load("response2.json"), load("response3.json")

def check_resp(r, label, expect_prompt_tokens, expect_completion):
    assert "id" in r, f"{label}: no id in {r}"
    idnum = int(r["id"].rsplit("-", 1)[1])
    ch = r["choices"][0]
    assert not ch["text"].startswith("[error:"), f"{label}: error result: {ch['text']}"
    assert ch["finish_reason"] == "length", f"{label}: finish_reason {ch['finish_reason']}"
    u = r["usage"]
    assert u["prompt_tokens"] == expect_prompt_tokens, \
        f"{label}: prompt_tokens {u['prompt_tokens']} != tokenized {expect_prompt_tokens}"
    assert u["completion_tokens"] == expect_completion, \
        f"{label}: completion_tokens {u['completion_tokens']} != {expect_completion}"
    return idnum

id1 = check_resp(r1, "phase1", n_prompt, k1)
id2 = check_resp(r2, "phase2", n_segment, k2)
print(f"phase1 request id={id1} phase2 request id={id2}")
with open(os.path.join(state, "request-ids.json"), "w") as f:
    json.dump({"phase1_id": id1, "phase2_id": id2}, f)

# Negative: unknown session must come back as an in-body error result.
ch3 = r3["choices"][0]
assert ch3["text"].startswith("[error:"), f"negative: expected error text, got {ch3['text']!r}"
assert ch3["finish_reason"] == "error", f"negative: finish_reason {ch3['finish_reason']}"
assert "session" in ch3["text"], f"negative: unexpected error {ch3['text']!r}"
print(f"negative append rejected as expected: {ch3['text']}")

m = load("metrics.json")
assert m["total_requests"] == 3, f"metrics total {m['total_requests']}"
assert m["completed_requests"] == 3, f"metrics completed {m['completed_requests']}"
assert m["failed_requests"] == 0, f"metrics failed {m['failed_requests']}"
assert m["active_requests"] == 0, f"metrics active {m['active_requests']}"
print("metrics: total=completed=3 failed=0 active=0")
PY

    PHASE1_ID=$(python3 -c "import json; print(json.load(open('${STATE_DIR}/request-ids.json'))['phase1_id'])")
    DECODE_STEPS=$((K1 - 1))
    log "phase-1 id=${PHASE1_ID} decode-steps=${DECODE_STEPS}"

    # === Golden contiguous reference on white (CUDA) ===
    log "running golden reference"
    env LD_LIBRARY_PATH=/home/stark/libtorch/lib \
        ./rust/target/release/route_b_cross_node_smoke golden \
        --device cuda \
        --model-dir "${MODEL_DIR}" \
        --prompt-token-ids "${PROMPT_IDS}" \
        --continuation-segment "${SEGMENT_IDS}" \
        --decode-steps "${DECODE_STEPS}" \
        --request-id "${PHASE1_ID}" \
        --out "${STATE_DIR}/golden" \
        > "${STATE_DIR}/golden.log" 2>&1 || fail "golden run failed (see golden.log)"
    T_GOLDEN=$(date +%s)

    # === Compare distributed service dump vs golden ===
    [ -f "${STATE_DIR}/export/request_${PHASE1_ID}/prefill_last_logits.f32le" ] \
        || fail "service export missing prefill_last_logits.f32le"
    [ -f "${STATE_DIR}/export/request_${PHASE1_ID}/continuation_last_logits.f32le" ] \
        || fail "service export missing continuation_last_logits.f32le"
    python3 "${REPO}/scripts/compare_route_b_dumps.py" \
        "${STATE_DIR}/export/request_${PHASE1_ID}" \
        "${STATE_DIR}/golden/request_${PHASE1_ID}" \
        > "${STATE_DIR}/compare.txt" 2>&1
    compare_rc=$?
    cat "${STATE_DIR}/compare.txt"
    [ "${compare_rc}" = "0" ] || fail "compare_route_b_dumps.py rc=${compare_rc}"
    grep -q "RESULT: PASS" "${STATE_DIR}/compare.txt" || fail "compare did not PASS"

    # === Trace assertions ===
    grep -q "stationary continuation done: request_id=${PHASE1_ID} " "${STATE_DIR}/coordinator.log" \
        || fail "coordinator log lacks the stationary continuation line"
    STATE_DIR="${STATE_DIR}" PHASE1_ID="${PHASE1_ID}" K1="${K1}" K2="${K2}" python3 - <<'PY' || fail "trace assertions failed"
import json, os

state = os.environ["STATE_DIR"]
phase1_id = int(os.environ["PHASE1_ID"])
k1, k2 = int(os.environ["K1"]), int(os.environ["K2"])
hops = 24 * (2 - 1)  # layers * (domains - 1)

records = [json.loads(l) for l in open(os.path.join(state, "trace.jsonl")) if l.strip()]
assert len(records) == 3, f"expected 3 trace records, got {len(records)}: {records}"

phase1, append, negative = records
# Phase 1 (keep_kv): KV retained — release must NOT appear on this record.
assert phase1["request_id"] == phase1_id, phase1
assert phase1["error"] is None, phase1
assert phase1["finish_reason"] == "length", phase1
assert phase1["reserved_bytes"] and any(b > 0 for b in phase1["reserved_bytes"]), phase1
assert phase1["released_bytes"] == [], f"keep_kv phase released KV: {phase1}"
assert phase1["decode_steps"] == k1 - 1, phase1
assert phase1["prefill_hops"] == hops, phase1
assert phase1["decode_hops"] == (k1 - 1) * hops, phase1
# Phase 2 (append): same worker-side request id; reservation released here.
assert append["request_id"] == phase1_id, append
assert append["error"] is None, append
assert append["finish_reason"] == "length", append
assert append["reserved_bytes"] == append["released_bytes"], \
    f"append reserved {append['reserved_bytes']} != released {append['released_bytes']}"
assert append["decode_steps"] == k2 - 1, append
assert append["decode_hops"] == (k2 - 1) * hops, append
# Negative append: rejected with an error record under its own job id.
assert negative["request_id"] != phase1_id, negative
assert negative["error"] is not None and "session" in negative["error"], negative
print("trace: phase1 retained (released=[]), append reserved==released, negative errored; hops ok")
PY
    T_END=$(date +%s)

    # === Collect artifacts ===
    scp -q -o ConnectTimeout=15 "${PEARL_SSH}:${STATE_DIR}/worker1-pearl.log" "${STATE_DIR}/" 2>/dev/null || true

    cat > "${STATE_DIR}/SUMMARY.md" <<EOF
# Phase-3 8 continuation service-path E2E (N=2 LAN) — ${RUN_ID}

Topology: coordinator + worker0 (CUDA) on white (${WHITE_LAN}), worker1 (HIP) on pearl (${PEARL_LAN}); Mac control-only.

Scenario:
- prefix  = "${PREFIX}"  (${N_PROMPT} tokens: ${PROMPT_IDS})
- segment = "${SEGMENT}" (${N_SEGMENT} tokens: ${SEGMENT_IDS})
- phase 1: keep_kv request id ${PHASE1_ID}, max_tokens=${K1}, temperature=0
- phase 2: append, max_tokens=${K2}, temperature=0, golden --decode-steps=${DECODE_STEPS}
- phase 3: negative append with unknown session (must error)

Results:
- compare_route_b_dumps.py: RESULT: PASS (see compare.txt)
- trace: phase-1 keep_kv record retains KV (released_bytes==[]); append record
  releases the same reservation (reserved==released); negative append errored;
  hop formulas hold (24 hops/prefill-or-continuation, 24/step decode).
- metrics: total=completed=3 failed=0 active=0.
- timings (s): health $((T_HEALTH - T_START)), req1 $((T_REQ1 - T_HEALTH)), req2 $((T_REQ2 - T_REQ1)), golden $((T_GOLDEN - T_REQ2)), compare+asserts $((T_END - T_GOLDEN)).

Note: phase-1 trace record intentionally shows released_bytes==[] (KV held for
the session); the release lands on the append record. The service dump exports
prefill_last_logits + continuation_last_logits only; decode_logits exists only
in the golden dump and is skipped by the comparator.
EOF

    log "DONE"
    echo "DONE" > "${STATE_DIR}/STATUS"
    exit 0
fi

# =====================================================================
# Mac orchestration half.
# =====================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="routeb-p3-continuation-e2e-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
STATE_DIR="/tmp/hcp-cont-e2e-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

run_remote() { # host cmd
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "$1" "bash -lc $(shell_quote "$2")"
}

echo "=== Phase-3 8: continuation service-path E2E (N=2 LAN) ==="
echo "RUN_ID=${RUN_ID}"
echo "Reports: ${REPORT_DIR}"

# === Preflight: sync + build white and pearl ===
build_cmd="cd $(shell_quote "${REPO_DIR}") && git checkout main 2>&1 | tail -1 && git pull --ff-only origin main 2>&1 | tail -1 && cd rust && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=/home/stark/libtorch LD_LIBRARY_PATH=/home/stark/libtorch/lib cargo build --features tch-backend --release 2>&1 | tail -2"
echo "=== Preflight: white build ==="
run_remote "${WHITE_SSH}" "${build_cmd}" 2>&1 | tail -3
echo "=== Preflight: pearl build ==="
run_remote "${PEARL_SSH}" "${build_cmd}" 2>&1 | tail -3

cleanup() {
    echo "=== Cleaning up ==="
    ssh -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# === Launch the white-side driver half ===
ssh -n -f -o ConnectTimeout=20 "${WHITE_SSH}" "mkdir -p ${STATE_DIR} && setsid bash $(shell_quote "${REPO_DIR}")/scripts/test_phase3_8_continuation_e2e.sh --driver ${RUN_ID} > ${STATE_DIR}/driver.log 2>&1 </dev/null"
echo "driver launched on white; polling STATUS (Mac network drops tolerated)..."

status=""
for i in $(seq 1 60); do
    sleep 30
    status=$(ssh -o ConnectTimeout=20 "${WHITE_SSH}" "cat ${STATE_DIR}/STATUS 2>/dev/null" 2>/dev/null || true)
    if [ -n "${status}" ]; then
        break
    fi
    echo "  ... still running (poll $i)"
done
if [ "${status}" != "DONE" ]; then
    echo "ERROR: driver finished with STATUS='${status}'" >&2
    ssh -o ConnectTimeout=20 "${WHITE_SSH}" "tail -40 ${STATE_DIR}/driver.log" 2>/dev/null || true
    exit 1
fi

echo "driver DONE; fetching artifacts..."
scp -rq -o ConnectTimeout=20 "${WHITE_SSH}:${STATE_DIR}/." "${REPORT_DIR}/"

echo ""
echo "=== PHASE-3 8 CONTINUATION E2E PASSED ==="
grep -E "RESULT|argmax" "${REPORT_DIR}/compare.txt" || true
echo "Reports: ${REPORT_DIR}"
