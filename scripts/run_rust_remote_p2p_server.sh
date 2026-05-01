#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-rust-remote-p2p-local}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
BIND_ADDR="${BIND_ADDR:-0.0.0.0:29172}"
mkdir -p "${REPORT_DIR}"

echo "=== HCP Rust Remote P2P Server ==="
echo "RUN_ID=${RUN_ID}"
echo "BIND_ADDR=${BIND_ADDR}"

cd "${REPO_ROOT}/rust"
CARGO_OFFLINE="${CARGO_OFFLINE:-1}"
if [ "${CARGO_OFFLINE}" = "1" ]; then
    CARGO_ARGS=(run)
else
    CARGO_ARGS=(run)
fi
set +e
cargo "${CARGO_ARGS[@]}" -- \
    --remote-p2p-role server \
    --bind "${BIND_ADDR}" \
    --report-path "${REPORT_DIR}/remote_p2p_server.json" 2>&1 |
    tee "${REPORT_DIR}/remote_p2p_server.log"
CARGO_STATUS="${PIPESTATUS[0]}"
set -e

if [ "${CARGO_STATUS}" -ne 0 ]; then
    exit "${CARGO_STATUS}"
fi

echo "=== HCP Rust Remote P2P Server Done ==="
echo "Reports: ${REPORT_DIR}"
