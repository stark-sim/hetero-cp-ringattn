#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-rust-remote-p2p-local}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
CONNECT_ADDR="${CONNECT_ADDR:-192.168.8.172:29172}"
mkdir -p "${REPORT_DIR}"

echo "=== HCP Rust Remote P2P Client ==="
echo "RUN_ID=${RUN_ID}"
echo "CONNECT_ADDR=${CONNECT_ADDR}"

cd "${REPO_ROOT}/rust"
CARGO_OFFLINE="${CARGO_OFFLINE:-1}"
if [ "${CARGO_OFFLINE}" = "1" ]; then
    CARGO_ARGS=(run --offline)
else
    CARGO_ARGS=(run)
fi
set +e
cargo "${CARGO_ARGS[@]}" -- \
    --remote-p2p-role client \
    --connect "${CONNECT_ADDR}" \
    --report-path "${REPORT_DIR}/remote_p2p_client.json" 2>&1 |
    tee "${REPORT_DIR}/remote_p2p_client.log"
CARGO_STATUS="${PIPESTATUS[0]}"
set -e

if [ "${CARGO_STATUS}" -ne 0 ]; then
    if [ "${CARGO_OFFLINE}" = "1" ] && grep -q "no matching package named" "${REPORT_DIR}/remote_p2p_client.log"; then
        echo "Cargo offline cache miss detected."
        echo "Run once with CARGO_OFFLINE=0, or prefetch with: cd ${REPO_ROOT}/rust && cargo fetch --locked"
    fi
    exit "${CARGO_STATUS}"
fi

echo "=== HCP Rust Remote P2P Client Done ==="
echo "Reports: ${REPORT_DIR}"
