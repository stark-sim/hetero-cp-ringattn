#!/usr/bin/env bash
# NIXL block transport probe (form-B S2) — register → metadata smoke on CUDA/ROCm.
# Remote verification for NixlBlockTransport (cannot run on Mac: no UCX/CUDA/ROCm).
#
# Usage on pearl (ROCm):
#   scripts/nixl_transport_probe.sh --build
#   scripts/nixl_transport_probe.sh --run
#
# Environment (pearl, ROCm):
#   NIXL_PREFIX     /home/stark/build/nixl-1.4.0 (source build tree)
#   LD_PRELOAD      /home/stark/libtorch/lib/libtorch_hip.so (HIP device)
#   NIXL_PLUGIN_DIR <build>/src/plugins/ucx  (UCX plugin .so dir)
set -euo pipefail
cd "$(dirname "$0")/.."

NIXL_PREFIX="${NIXL_PREFIX:-/home/stark/build/nixl-1.4.0}"
NIXL_BUILD="$NIXL_PREFIX/build/src"
NIXL_LD="$NIXL_BUILD/bindings:$NIXL_BUILD/core:$NIXL_BUILD/infra:$NIXL_BUILD/utils/serdes:$NIXL_BUILD/utils/stream:$NIXL_BUILD/utils/common:$NIXL_BUILD/plugins/ucx"
PLUGIN_DIR="${NIXL_PLUGIN_DIR:-$NIXL_BUILD/plugins/ucx}"
HIP_PRELOAD="${HCP_HIP_PRELOAD:-/home/stark/libtorch/lib/libtorch_hip.so}"

mode="${1:---run}"
case "$mode" in
  --build)
    echo "[nixl-probe] build with nixl-backend (NIXL_PREFIX=$NIXL_PREFIX)"
    PATH="/home/stark/.cargo/bin:$PATH" \
      LIBTORCH="/home/stark/libtorch" \
      LIBRARY_PATH="$NIXL_LD:/home/stark/libtorch/lib:${LIBRARY_PATH:-}" \
      LD_LIBRARY_PATH="$NIXL_LD:/home/stark/libtorch/lib:${LD_LIBRARY_PATH:-}" \
      cargo build --features tch-backend,nixl-backend --bin nixl-probe
    echo "[nixl-probe] build OK"
    ;;
  --run)
    echo "[nixl-probe] run register->metadata smoke on this host"
    LD_PRELOAD="$HIP_PRELOAD" \
      LD_LIBRARY_PATH="$NIXL_LD:/home/stark/libtorch/lib:${LD_LIBRARY_PATH:-}" \
      NIXL_PLUGIN_DIR="$PLUGIN_DIR" \
      HCP_TCH_DEVICE=cuda:0 \
      ./rust/target/debug/nixl-probe
    echo "[nixl-probe] run OK"
    ;;
  *)
    echo "usage: $0 [--build|--run]" >&2
    exit 2
    ;;
esac
