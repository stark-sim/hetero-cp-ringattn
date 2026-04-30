#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LIBTORCH="${LIBTORCH:-/Users/stark_sim/libtorch}"
export DYLD_LIBRARY_PATH="${LIBTORCH}/lib:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}"

export HCP_TCH_DEVICE="${HCP_TCH_DEVICE:-cpu}"
export RUN_ID="${RUN_ID:-hcp-ringattn-tch-smoke-local}"

echo "=== tch-rs smoke ==="
echo "LIBTORCH=${LIBTORCH}"
echo "HCP_TCH_DEVICE=${HCP_TCH_DEVICE}"
echo "RUN_ID=${RUN_ID}"

cd "${REPO_ROOT}/rust"

cargo run --features tch-backend --bin tch_smoke
