#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-rust-ringattn-smoke-local}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

echo "=== HCP Rust RingAttn Smoke ==="
echo "RUN_ID=${RUN_ID}"

cd "${REPO_ROOT}/rust"
if [ -n "${LIBTORCH:-}" ]; then
    export DYLD_LIBRARY_PATH="${LIBTORCH}/lib:${DYLD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}"
fi
CARGO_OFFLINE="${CARGO_OFFLINE:-1}"
FEATURES=""
if [ "${HCP_ENABLE_TORCH:-0}" = "1" ] || [ -n "${LIBTORCH:-}" ]; then
    FEATURES="tch-backend"
fi
if [ "${CARGO_OFFLINE}" = "1" ]; then
    if [ -n "${FEATURES}" ]; then
        CARGO_ARGS=(run --offline --features "${FEATURES}" --bin hcp-ringattn-rust)
    else
        CARGO_ARGS=(run --offline --bin hcp-ringattn-rust)
    fi
else
    if [ -n "${FEATURES}" ]; then
        CARGO_ARGS=(run --features "${FEATURES}" --bin hcp-ringattn-rust)
    else
        CARGO_ARGS=(run --bin hcp-ringattn-rust)
    fi
fi
set +e
cargo "${CARGO_ARGS[@]}" -- \
    --report-path "${REPORT_DIR}/rust_ringattn_correctness.json" 2>&1 |
    tee "${REPORT_DIR}/rust_ringattn_correctness.log"
CARGO_STATUS="${PIPESTATUS[0]}"
set -e

if [ "${CARGO_STATUS}" -ne 0 ]; then
    if [ "${CARGO_OFFLINE}" = "1" ] && grep -q "no matching package named" "${REPORT_DIR}/rust_ringattn_correctness.log"; then
        echo "Cargo offline cache miss detected."
        echo "Run once with CARGO_OFFLINE=0, or prefetch with: cd ${REPO_ROOT}/rust && cargo fetch --locked"
    fi
    exit "${CARGO_STATUS}"
fi

echo "=== HCP Rust RingAttn Smoke Done ==="
echo "Reports: ${REPORT_DIR}"
