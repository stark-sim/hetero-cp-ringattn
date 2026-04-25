#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-rust-remote-cp-node-local}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
NODE_INDEX="${NODE_INDEX:?NODE_INDEX is required: 0=mac-mps, 1=gpu-cuda}"
BIND_ADDR="${BIND_ADDR:?BIND_ADDR is required, e.g. 0.0.0.0:29174}"
CONNECT_ADDR="${CONNECT_ADDR:?CONNECT_ADDR is required, e.g. 192.168.8.172:29173}"
mkdir -p "${REPORT_DIR}"

echo "=== HCP Rust Remote CP Node ==="
echo "RUN_ID=${RUN_ID}"
echo "NODE_INDEX=${NODE_INDEX}"
echo "BIND_ADDR=${BIND_ADDR}"
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
    --remote-p2p-role cp-node \
    --node-index "${NODE_INDEX}" \
    --bind "${BIND_ADDR}" \
    --connect "${CONNECT_ADDR}" \
    --report-path "${REPORT_DIR}/remote_cp_node_${NODE_INDEX}.json" 2>&1 |
    tee "${REPORT_DIR}/remote_cp_node_${NODE_INDEX}.log"
CARGO_STATUS="${PIPESTATUS[0]}"
set -e

if [ "${CARGO_STATUS}" -ne 0 ]; then
    if [ "${CARGO_OFFLINE}" = "1" ] && grep -q "no matching package named" "${REPORT_DIR}/remote_cp_node_${NODE_INDEX}.log"; then
        echo "Cargo offline cache miss detected."
        echo "Run once with CARGO_OFFLINE=0, or prefetch with: cd ${REPO_ROOT}/rust && cargo fetch --locked"
    fi
    exit "${CARGO_STATUS}"
fi

echo "=== HCP Rust Remote CP Node Done ==="
echo "Reports: ${REPORT_DIR}"
