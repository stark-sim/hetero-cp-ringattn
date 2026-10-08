#!/usr/bin/env bash
set -euo pipefail

# Mac loopback TP 抽象 worker ring 测试（N2 gate 1）：
#   golden : 1-domain 普通 Python worker (CPU)
#   ring   : domain 0 = TP 抽象 worker（gloo，2 本地 CPU 进程），
#            domain 1 = 普通 Python worker (CPU)
# gate: ring 输出与 golden 逐 token 一致 + logits 级比对。

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${REPO_ROOT}/models/Qwen2-0.5B"
PY="${PROMPT_PY:-/Users/stark_sim/miniconda3/bin/python3.12}"
DTYPE="${DTYPE:-float32}"
SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROMPT_SEED="${PROMPT_SEED:-20261006}"

COORD_PORT=29930
W0_PORT=29931   # TP 抽象 worker (rank 0) 的 ring 端口
W1_PORT=29932   # 普通 worker 的 ring 端口
TP_PORT=29611   # TP rank0<->rank1 的 gloo master 端口

RUN_ID="tp-ring-local-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

PIDS=()
cleanup() {
    echo "=== Cleanup ==="
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
"${PY}" "${REPO_ROOT}/scripts/gen_natural_prompt.py" \
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
echo "=== Phase golden (Python CPU, 1 domain) ==="
start_coordinator golden 1
PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${PY}" \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 1 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W1_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W0_PORT} \
    --device cpu --dtype "${DTYPE}" \
    >"${REPORT_DIR}/worker_golden.log" 2>&1 &
PIDS+=($!)
wait "${COORD_PID}" || true
echo "golden done"

# === ring: domain0 = TP 抽象 worker (gloo, 2 CPU 进程), domain1 = 普通 worker ===
echo "=== Phase ring (TP abstract domain0 + Python domain1) ==="
start_coordinator ring 2
PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${PY}" \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 2 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W0_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W1_PORT} \
    --device cpu --dtype "${DTYPE}" \
    --tp-size 2 --tp-backend gloo --tp-rank 0 --tp-master-port ${TP_PORT} \
    >"${REPORT_DIR}/worker0_tp_rank0.log" 2>&1 &
PIDS+=($!)
PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${PY}" \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 2 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W0_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W1_PORT} \
    --device cpu --dtype "${DTYPE}" \
    --tp-size 2 --tp-backend gloo --tp-rank 1 --tp-master-port ${TP_PORT} \
    >"${REPORT_DIR}/worker0_tp_rank1.log" 2>&1 &
PIDS+=($!)
PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${PY}" \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 1 --num-domains 2 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W1_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W0_PORT} \
    --device cpu --dtype "${DTYPE}" \
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
"${PY}" "${REPO_ROOT}/scripts/compare_logits_dir.py" "${REPORT_DIR}/logits_ring" "${REPORT_DIR}/logits_golden" | head -4
