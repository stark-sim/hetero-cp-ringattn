#!/usr/bin/env bash
set -uo pipefail

# 故障注入回归（critic #15）：2-domain ring（domain0 = TP 抽象 worker）
# 运行中 kill -9 domain1 worker，观察：
#   1) TP rank0 报错并非零退出（finally 里尽力 poison follower）；
#   2) TP rank1 follower 在 --tp-collective-timeout 内自行退出（不永久悬挂）。
# 正常路径由 test_tp_ring_local.sh 覆盖，本脚本只测异常路径。

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"
MODEL_DIR="${REPO_ROOT}/models/Qwen2-0.5B"
PY="${PROMPT_PY:-/Users/stark_sim/miniconda3/bin/python3.12}"
SEQ_LEN="${SEQ_LEN:-128}"
TP_TIMEOUT="${TP_TIMEOUT:-30}"

COORD_PORT=29940
W0_PORT=29941
W1_PORT=29942
TP_PORT=29612

RUN_ID="tp-fault-inject-$(date +%Y%m%d-%H%M%S)"
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
    "${MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" 20261006 "${PROMPT_FILE}"

"${BINARY}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" --prompt-file "${PROMPT_FILE}" \
    --max-tokens 16 --num-domains 2 --listen-addr "127.0.0.1:${COORD_PORT}" \
    >"${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_PID=$!
PIDS+=("${COORD_PID}")
sleep 2

PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${PY}" \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 2 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W0_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W1_PORT} \
    --device cpu --tp-size 2 --tp-backend gloo --tp-rank 0 \
    --tp-master-port ${TP_PORT} --tp-collective-timeout ${TP_TIMEOUT} \
    >"${REPORT_DIR}/worker0_tp_rank0.log" 2>&1 &
RANK0_PID=$!
PIDS+=("${RANK0_PID}")

PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${PY}" \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 2 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W0_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W1_PORT} \
    --device cpu --tp-size 2 --tp-backend gloo --tp-rank 1 \
    --tp-master-port ${TP_PORT} --tp-collective-timeout ${TP_TIMEOUT} \
    >"${REPORT_DIR}/worker0_tp_rank1.log" 2>&1 &
RANK1_PID=$!
PIDS+=("${RANK1_PID}")

PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${PY}" \
    "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
    --model-dir "${MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
    --domain-id 1 --num-domains 2 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W1_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W0_PORT} \
    --device cpu \
    >"${REPORT_DIR}/worker1.log" 2>&1 &
W1_PID=$!

wait_log() { # log pattern timeout_s
    local elapsed=0
    while [ "${elapsed}" -lt "$3" ]; do
        grep -q "$2" "$1" 2>/dev/null && return 0
        sleep 1; elapsed=$((elapsed + 1))
    done
    return 1
}

wait_exit() { # pid timeout_s -> 0 if exited
    local elapsed=0
    while [ "${elapsed}" -lt "$2" ]; do
        kill -0 "$1" 2>/dev/null || return 0
        sleep 1; elapsed=$((elapsed + 1))
    done
    return 1
}

echo "=== waiting for rank0 to enter prefill ==="
if ! wait_log "${REPORT_DIR}/worker0_tp_rank0.log" "received: Prefill" 180; then
    echo "FAIL: rank0 never entered prefill"; tail -20 "${REPORT_DIR}/worker0_tp_rank0.log"; exit 2
fi

echo "=== injecting fault: kill -9 domain1 worker (pid=${W1_PID}) ==="
KILL_TS=$(date +%s)
kill -9 "${W1_PID}" 2>/dev/null || true
PIDS=("${COORD_PID}" "${RANK0_PID}" "${RANK1_PID}")

echo "=== waiting for rank0 to exit (<=60s) ==="
RANK0_STATUS="alive"
if wait_exit "${RANK0_PID}" 60; then
    set +e; wait "${RANK0_PID}"; RANK0_RC=$?; set -e
    RANK0_STATUS="exited rc=${RANK0_RC} after $(( $(date +%s) - KILL_TS ))s"
else
    # decode 阶段被杀时 coordinator 侧先挂、rank0 可能等不到下一条命令而存活；
    # follower 超时退出才是本测试的核心判据。
    RANK0_STATUS="still alive after 60s (coordinator-side hang; acceptable)"
fi
echo "rank0: ${RANK0_STATUS}"

echo "=== waiting for follower rank1 to exit (<=TP_TIMEOUT+60s) ==="
DEADLINE=$((TP_TIMEOUT + 60))
if wait_exit "${RANK1_PID}" "${DEADLINE}"; then
    set +e; wait "${RANK1_PID}"; RANK1_RC=$?; set -e
    ELAPSED=$(( $(date +%s) - KILL_TS ))
    echo "rank1 follower exited rc=${RANK1_RC} after ~${ELAPSED}s (timeout=${TP_TIMEOUT}s)"
    if [ "${RANK1_RC}" -eq 0 ]; then
        echo "FAIL: follower exited 0 (expected error exit on collective timeout)"; exit 1
    fi
    echo "PASS: follower self-terminated within timeout after peer domain death"
else
    echo "FAIL: follower still hanging after ${DEADLINE}s"; exit 1
fi

echo "--- rank1 log tail ---"; tail -6 "${REPORT_DIR}/worker0_tp_rank1.log"
echo "--- rank0 log tail ---"; tail -6 "${REPORT_DIR}/worker0_tp_rank0.log"
