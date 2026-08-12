#!/bin/bash
# 6c.1 native service stability baseline: run the Rust HTTP service with the
# JSONL trace plane at concurrency 1/2/4 and unequal-length requests, and
# verify 0 errors, metric counters, reserved bytes, and the N/L hop formulas.
#
# This is an internal phase-2 stability baseline, not a vLLM performance claim.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
TRACE_FILE="${TRACE_FILE:-/tmp/hcp_phase2_6c1_trace.jsonl}"

COORD_PORT=29600
W0_PORT=29601
W1_PORT=29602
HTTP_PORT=8081

rm -f "${TRACE_FILE}"

echo "=== 6c.1 Native Service Stability Baseline ==="
echo "Building release binary..."
cd "${REPO_ROOT}/rust"
LIBTORCH="${LIBTORCH:-/Users/stark_sim/libtorch}" cargo build --features tch-backend --release 2>&1 | tail -2

cleanup() {
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
export HCP_TCH_DEVICE="${HCP_TCH_DEVICE:-mps}"

echo "=== Launching Coordinator (HTTP mode, trace=${TRACE_FILE}) ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --num-domains 2 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --http-addr "0.0.0.0:${HTTP_PORT}" \
    --trace-jsonl "${TRACE_FILE}" \
    > /tmp/hcp_6c1_coord.log 2>&1 &
COORD_PID=$!
sleep 2

echo "=== Launching Worker 0 ==="
"${BINARY}" --distributed-role worker \
    --domain-id 0 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "127.0.0.1:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    > /tmp/hcp_6c1_w0.log 2>&1 &
W0_PID=$!
sleep 2

echo "=== Launching Worker 1 ==="
"${BINARY}" --distributed-role worker \
    --domain-id 1 \
    --model-dir "${MODEL_DIR}" \
    --listen-addr "0.0.0.0:${W1_PORT}" \
    --next-peer-addr "127.0.0.1:${W0_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    > /tmp/hcp_6c1_w1.log 2>&1 &
W1_PID=$!
echo "Waiting 15s for workers to connect and load model..."
sleep 15

health=$(curl -s "http://localhost:${HTTP_PORT}/health")
echo "health: ${health}"

# request() runs one POST and appends its JSON body to $1, its request id to $2.
request() {
    local out="$1" ids_file="$2" prompt="$3" max_tokens="$4"
    curl -s -X POST "http://localhost:${HTTP_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": ${prompt}, \"max_tokens\": ${max_tokens}, \"temperature\": 0.0}" \
        >> "${out}"
    echo >> "${out}"
    python3 - "$out" "$ids_file" <<'PY'
import json, sys, os
out, ids_file = sys.argv[1], sys.argv[2]
ids = []
if os.path.exists(ids_file):
    with open(ids_file) as f:
        ids = [l.strip() for l in f if l.strip()]
# request id counter lives in ids_file; append next id.
ids.append(str(len(ids) + 1))
with open(ids_file, "w") as f:
    f.write("\n".join(ids) + "\n")
PY
}

verify_batch() {
    local label="$1" out="$2" expected="$3"
    local total completed failed
    total=$(python3 - "$out" <<'PY'
import json, sys
lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
print(len(lines))
PY
)
    # count non-error responses and extract texts
    python3 - "$out" "$label" <<'PY'
import json, sys
out, label = sys.argv[1], sys.argv[2]
texts, errors = [], 0
for line in open(out):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if "choices" in d and d["choices"]:
        texts.append(d["choices"][0]["text"])
    else:
        errors += 1
print(f"  [{label}] responses={len(texts)} errors={errors}")
if errors:
    sys.exit(1)
for t in texts:
    if t.startswith("[error:"):
        print(f"  [{label}] ERROR TEXT: {t!r}")
        sys.exit(1)
PY
    echo "  [${label}] expected=${expected} got=${total}"
    if [ "$total" -ne "$expected" ]; then
        echo "  [${label}] FAILED: expected ${expected} responses, got ${total}"
        exit 1
    fi
}

# --- Concurrency 1 (serial, unequal lengths) ---
echo ""
echo "=== Concurrency 1 (serial, unequal prompt lengths) ==="
OUT=/tmp/hcp_6c1_c1.jsonl; IDS=/tmp/hcp_6c1_c1.ids; rm -f "$OUT" "$IDS"
request "$OUT" "$IDS" '"short phrase"' 3
request "$OUT" "$IDS" '"The quick brown fox jumps over the lazy dog near the river bank, and then it pauses to think about the next move carefully, deciding to cross the bridge slowly."' 8
verify_batch "c1" "$OUT" 2

# --- Concurrency 2 (two simultaneous, unequal lengths) ---
echo ""
echo "=== Concurrency 2 (simultaneous, unequal lengths) ==="
OUT=/tmp/hcp_6c1_c2.jsonl; IDS=/tmp/hcp_6c1_c2.ids; rm -f "$OUT" "$IDS"
request "$OUT" "$IDS" '"Once upon a time in a distant land"' 6 &
P1=$!
request "$OUT" "$IDS" '"A single token"' 2 &
P2=$!
wait $P1; wait $P2
verify_batch "c2" "$OUT" 2

# --- Concurrency 4 (four simultaneous, unequal lengths) ---
echo ""
echo "=== Concurrency 4 (four simultaneous, unequal lengths) ==="
OUT=/tmp/hcp_6c1_c4.jsonl; IDS=/tmp/hcp_6c1_c4.ids; rm -f "$OUT" "$IDS"
request "$OUT" "$IDS" '"alpha beta"' 4 & P1=$!
request "$OUT" "$IDS" '"gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"' 10 & P2=$!
request "$OUT" "$IDS" '"one two"' 1 & P3=$!
request "$OUT" "$IDS" '"The answer to the ultimate question of life the universe and everything is"' 7 & P4=$!
wait $P1; wait $P2; wait $P3; wait $P4
verify_batch "c4" "$OUT" 4

# --- Metrics consistency ---
echo ""
echo "=== /metrics after all batches ==="
curl -s "http://localhost:${HTTP_PORT}/metrics" | python3 -m json.tool
metrics=$(curl -s "http://localhost:${HTTP_PORT}/metrics")
python3 - "$metrics" <<'PY'
import json, sys
m = json.loads(sys.argv[1])
expected_total = 8
print(f"  total={m['total_requests']} completed={m['completed_requests']} failed={m['failed_requests']} active={m['active_requests']}")
assert m["total_requests"] == expected_total, f"total {m['total_requests']} != {expected_total}"
assert m["failed_requests"] == 0, f"failed {m['failed_requests']} != 0"
assert m["active_requests"] == 0, f"active {m['active_requests']} != 0"
assert m["completed_requests"] == expected_total, f"completed {m['completed_requests']} != {expected_total}"
PY

# --- Trace validation ---
echo ""
echo "=== Trace validation (${TRACE_FILE}) ==="
python3 - "${TRACE_FILE}" <<'PY'
import json, sys
path = sys.argv[1]
records = [json.loads(l) for l in open(path) if l.strip()]
print(f"  trace records={len(records)}")
assert len(records) == 8, f"expected 8 trace records, got {len(records)}"
ids = sorted(r["request_id"] for r in records)
assert ids == list(range(1, 9)), f"request_ids {ids}"
for r in records:
    # Every request must have been accepted (not just enqueued) and completed.
    assert r["prefill_accepted_elapsed_ms"] > 0, f"request {r['request_id']} never accepted"
    assert r["completed_elapsed_ms"] > 0, f"request {r['request_id']} never completed"
    assert r["error"] is None, f"request {r['request_id']} error {r['error']}"
    assert len(r["reserved_bytes"]) == 2, f"request {r['request_id']} reserved {r['reserved_bytes']}"
    assert r["reserved_bytes"] == r["released_bytes"], f"request {r['request_id']} release mismatch"
    # N=2, L=24: prefill hops = 24*1 = 24; decode hops = decode_steps * 24.
    assert r["prefill_hops"] == 24, f"request {r['request_id']} prefill_hops {r['prefill_hops']}"
    assert r["decode_hops"] == r["decode_steps"] * 24, f"request {r['request_id']} decode_hops"
    assert r["finish_reason"] in ("stop", "length"), f"request {r['request_id']} finish {r['finish_reason']}"
    print(f"  req {r['request_id']}: prompt={r['prompt_tokens']} steps={r['decode_steps']} reserved={r['reserved_bytes']} hops={r['prefill_hops']}+{r['decode_hops']} finish={r['finish_reason']}")
PY

echo ""
echo "=== 6C.1 NATIVE SERVICE STABILITY BASELINE PASSED ==="
echo "0 errors across concurrency 1/2/4; metrics and trace records consistent."
