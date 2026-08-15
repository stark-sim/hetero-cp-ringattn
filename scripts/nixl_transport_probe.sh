#!/usr/bin/env bash
# NIXL block transport probe (form-B S2) — register → transfer → poll on two
# CUDA/ROCm hosts. Remote verification for NixlBlockTransport that cannot run
# on the Mac (no UCX/CUDA/ROCm).
#
#   --build   compile with nixl-backend on the current host (NIXL_PREFIX points
#             at a build tree whose libnixl_capi.so is in the loader path).
#   --probe   (placeholder) minimal register→transfer→poll smoke.
set -euo pipefail
cd "$(dirname "$0")/.."

NIXL_PREFIX="${NIXL_PREFIX:-/home/stark/build/nixl-1.4.0}"
NIXL_LIB_PATH="${NIXL_LIB_PATH:-${NIXL_PREFIX}/build/src/bindings:${NIXL_PREFIX}/build/src/core:${NIXL_PREFIX}/build/src/infra}"

mode="${1:---build}"
case "$mode" in
  --build)
    echo "[nixl-probe] building with nixl-backend (NIXL_PREFIX=$NIXL_PREFIX)"
    NIXL_PREFIX="$NIXL_PREFIX" \
      LD_LIBRARY_PATH="$NIXL_LIB_PATH:${LD_LIBRARY_PATH:-}" \
      PATH="/home/stark/.cargo/bin:$PATH" \
      LIBTORCH="/home/stark/libtorch" \
      cargo build --features tch-backend,nixl-backend --lib
    echo "[nixl-probe] build OK"
    ;;
  --probe)
    echo "[nixl-probe] probe not yet wired (S2 remote smoke)."
    echo "  Once a probe binary exists: cargo run --features tch-backend,nixl-backend --bin nixl-probe -- <peer-agent> <local-block-bytes>"
    exit 0
    ;;
  *)
    echo "usage: $0 [--build|--probe]" >&2
    exit 2
    ;;
esac
