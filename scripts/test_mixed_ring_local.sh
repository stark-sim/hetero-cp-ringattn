#!/usr/bin/env bash
set -euo pipefail

# Mac loopback 混合 ring 测试：Rust worker (CPU) + Python worker (CPU, rust ring mode)
# gate: ring 输出与 1-domain golden 逐 token 一致 + logits 级比对。

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${REPO_ROOT}/models/Qwen2-0.5B"
SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROMPT_SEED="${PROMPT_SEED:-20261006}"

COORD_PORT=29920
W0_PORT=29921
W1_PORT=29922

RUN_ID="mixed-local-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

PIDS=()
cleanup() {
    echo "=== Cleanup ==="
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# --- prompt ---
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
PROMPT_PY="${PROMPT_PY:-$(/usr/local/bin/python3.11 -c 'import tokenizers' 2>/dev/null && echo /usr/local/bin/python3.11 || echo /Users/stark_sim/miniconda3/bin/python3.12)}"
"${PROMPT_PY}" "${REPO_ROOT}/scripts/gen_natural_prompt.py" \
    "${MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_SEED}" "${PROMPT_FILE}" | tee "${REPORT_DIR}/gen_prompt.log"
cp "${PROMPT_FILE}" "${REPORT_DIR}/prompt.txt"

start_coordinator() { # phase num_domains
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MODEL_DIR}" \
        --prompt-file "${PROMPT_FILE}" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --num-domains "$2" \
        --listen-addr "127.0.0.1:${COORD_PORT}" \
        --export-logits-dir "${REPORT_DIR}/logits_$1" \
        >"${REPORT_DIR}/coordinator_$1.log" 2>&1 &
    COORD_PID=$!
    PIDS+=("${COORD_PID}")
    sleep 2
}

# === golden: 1-domain Python worker (CPU) ===
# （Rust worker 的 setup_network 在 num_domains=1 时仍会等 ring 建链，故 golden 用 Python worker）
echo "=== Phase golden (Python CPU, 1 domain) ==="
start_coordinator golden 1
PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH /Users/stark_sim/miniconda3/bin/python3.12 \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 1 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W1_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W0_PORT} \
    --device cpu \
    >"${REPORT_DIR}/worker_golden.log" 2>&1 &
PIDS+=($!)
wait "${COORD_PID}" || true
echo "golden done"

# === ring: Rust (domain 0) + Python (domain 1, rust mode) ===
echo "=== Phase ring (Rust domain0 + Python domain1) ==="
start_coordinator ring 2
HCP_TCH_DEVICE=cpu "${BINARY}" --distributed-role worker \
    --domain-id 0 --model-dir "${MODEL_DIR}" \
    --listen-addr "127.0.0.1:${W0_PORT}" --next-peer-addr "127.0.0.1:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" --num-domains 2 \
    >"${REPORT_DIR}/worker0-rust_ring.log" 2>&1 &
PIDS+=($!)
PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH /Users/stark_sim/miniconda3/bin/python3.12 \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 1 --num-domains 2 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W1_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W0_PORT} \
    --device cpu --ring-mode rust \
    >"${REPORT_DIR}/worker1-python_ring.log" 2>&1 &
PIDS+=($!)

wait "${COORD_PID}" || true
echo "ring done"

echo ""
echo "=== Generated ==="
grep "generated:" "${REPORT_DIR}/coordinator_golden.log" || tail -5 "${REPORT_DIR}/coordinator_golden.log"
grep "generated:" "${REPORT_DIR}/coordinator_ring.log" || tail -5 "${REPORT_DIR}/coordinator_ring.log"

echo ""
G="$(grep 'generated:' "${REPORT_DIR}/coordinator_golden.log" | head -1 || true)"
R="$(grep 'generated:' "${REPORT_DIR}/coordinator_ring.log" | head -1 || true)"
if [ -z "$G" ] || [ -z "$R" ]; then
    echo "INCOMPLETE"; exit 2
elif [ "$G" = "$R" ]; then
    echo "TEXT MATCH"
else
    echo "TEXT DIFFER"; exit 1
fi
"${PROMPT_PY}" "${REPO_ROOT}/scripts/compare_logits_dir.py" "${REPORT_DIR}/logits_ring" "${REPORT_DIR}/logits_golden" | head -4
