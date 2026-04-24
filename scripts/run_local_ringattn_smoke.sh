#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/config/minimal_2domain_ring.json"

RUN_ID="${RUN_ID:-hcp-ringattn-smoke-local}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

echo "=== HCP RingAttn Local Smoke ==="
echo "RUN_ID=${RUN_ID}"
echo "CONFIG=${CONFIG_PATH}"

echo "[smoke] Configuring local build..."
cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/build"
cmake --build "${REPO_ROOT}/build" --target ringattn_coordinator_smoke -j4

echo "[smoke] Running C++ coordinator smoke..."
"${REPO_ROOT}/build/ringattn_coordinator_smoke" | tee "${REPORT_DIR}/cpp_smoke.log"

if [ "${SKIP_PYTHON_SMOKE:-0}" = "1" ]; then
    echo "[smoke] Skipping Python smoke because SKIP_PYTHON_SMOKE=1"
elif command -v python3 &> /dev/null && python3 -c "import numpy" >/dev/null 2>&1; then
    echo "[smoke] Running Python controller+worker placeholder..."
    python3 "${REPO_ROOT}/python/ringattn_worker.py" \
        --name domain-0 --port 26001 &
    WORKER0_PID=$!
    python3 "${REPO_ROOT}/python/ringattn_worker.py" \
        --name domain-1 --port 26002 &
    WORKER1_PID=$!

    sleep 1

    python3 "${REPO_ROOT}/python/ringattn_controller.py" \
        --config "${CONFIG_PATH}" \
        --report-path "${REPORT_DIR}/python_smoke.json" | tee "${REPORT_DIR}/python_smoke.log"

    kill "${WORKER0_PID}" "${WORKER1_PID}" || true
else
    echo "[smoke] Skipping Python smoke because python3 or numpy is unavailable"
fi

echo "=== HCP RingAttn Local Smoke Done ==="
echo "Reports: ${REPORT_DIR}"
