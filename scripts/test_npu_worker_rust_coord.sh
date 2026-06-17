#!/bin/bash
# End-to-end smoke: Python NPU vLLM worker <-> Rust coordinator over QUIC control plane.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/models/Qwen2-0.5B"
COORD_BIN="${REPO_ROOT}/rust/target/debug/hcp-ringattn-rust"
WORKER_PY="${REPO_ROOT}/python/hcp_vllm_quic_worker.py"
LISTEN_ADDR="127.0.0.1:26001"
REPORT_DIR="${REPO_ROOT}/reports/npu-worker-rust-coord"
mkdir -p "${REPORT_DIR}"

if [[ ! -x "${COORD_BIN}" ]]; then
    echo "[test] coordinator binary not found: ${COORD_BIN}"
    echo "[test] run: cd rust && cargo build --no-default-features --bin hcp-ringattn-rust"
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[test] model directory not found: ${MODEL_DIR}"
    exit 1
fi

# Clean up stale vLLM engine processes on NPU (would otherwise cause OOM)
echo "[test] cleaning up stale VLLM EngineCore processes..."
for pid in $(pgrep -f 'VLLMEngineCor' || true); do
    echo "[test] killing stale EngineCore pid=${pid}"
    kill -9 "${pid}" 2>/dev/null || true
done
sleep 2

echo "[test] starting Rust coordinator on ${LISTEN_ADDR}..."
"${COORD_BIN}" --distributed-role coordinator \
    --model-dir "${MODEL_DIR}" \
    --prompt "Hello world" \
    --max-tokens 3 \
    --num-domains 1 \
    --listen-addr "${LISTEN_ADDR}" \
    > "${REPORT_DIR}/coordinator.log" 2>&1 &
COORD_PID=$!

cleanup() {
    echo "[test] cleaning up..."
    kill "${COORD_PID}" 2>/dev/null || true
    kill "${WORKER_PID:-}" 2>/dev/null || true
    wait "${COORD_PID}" 2>/dev/null || true
    wait "${WORKER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2

echo "[test] starting Python NPU vLLM worker..."
PYTHONPATH="${REPO_ROOT}/python" python "${WORKER_PY}" \
    --model-dir "${MODEL_DIR}" \
    --coordinator-host 127.0.0.1 \
    --coordinator-port 26001 \
    --domain-id 0 \
    --num-domains 1 \
    --peer-listen-port 26091 \
    --next-peer-port 26092 \
    > "${REPORT_DIR}/worker.log" 2>&1 &
WORKER_PID=$!

echo "[test] waiting for generation..."
MAX_WAIT=120
for i in $(seq 1 "${MAX_WAIT}"); do
    if grep -q "generated:" "${REPORT_DIR}/coordinator.log"; then
        echo "[test] generation completed"
        break
    fi
    if ! kill -0 "${COORD_PID}" 2>/dev/null; then
        echo "[test] coordinator exited early"
        break
    fi
    if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
        echo "[test] worker exited early"
        break
    fi
    sleep 1
done

sleep 2

echo "===== COORDINATOR LOG ====="
cat "${REPORT_DIR}/coordinator.log"
echo "===== WORKER LOG (tail 80) ====="
tail -80 "${REPORT_DIR}/worker.log"

if grep -q "generated:" "${REPORT_DIR}/coordinator.log"; then
    GENERATED="$(grep "generated:" "${REPORT_DIR}/coordinator.log" | head -1)"
    echo ""
    echo "✅ Python NPU worker <-> Rust coordinator E2E PASSED"
    echo "   ${GENERATED}"
    exit 0
else
    echo ""
    echo "❌ Python NPU worker <-> Rust coordinator E2E FAILED"
    exit 1
fi
