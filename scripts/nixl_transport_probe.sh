#!/usr/bin/env bash
# NIXL block transport probe (form-B S2) — register → metadata smoke on CUDA/ROCm.
# Remote verification for NixlBlockTransport (cannot run on Mac: no UCX/CUDA/ROCm).
#
# Usage (auto-detects host white/pearl):
#   scripts/nixl_transport_probe.sh --build
#   scripts/nixl_transport_probe.sh --run
#
# Hosts:
#   white (CUDA): NIXL libs from the nixl_cu13 pip wheel in the vllm-v1 conda env.
#     stub-api dlopens libnixl_capi.so at runtime, so LD_LIBRARY_PATH points at the
#     wheel's .nixl_cu13.mesonpy.libs. No LD_PRELOAD (tch links CUDA libtorch directly).
#   pearl (ROCm): NIXL libs from the /home/stark/build/nixl-1.4.0 source build tree.
#     Needs LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so (HIP device).
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="$(hostname -s)"
LIBTORCH="${LIBTORCH:-/home/stark/libtorch}"

case "$HOST" in
  white)
    WHEEL_DIR="${NIXL_WHEEL_DIR:-$HOME/miniconda3/envs/vllm-v1/lib/python3.11/site-packages}"
    NIXL_LD="$WHEEL_DIR/.nixl_cu13.mesonpy.libs"
    PLUGIN_DIR="${NIXL_PLUGIN_DIR:-$WHEEL_DIR/.nixl_cu13.mesonpy.libs/plugins}"
    HIP_PRELOAD=""
    LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-21/lib}"
    ;;
  laptop)
    # RTX 4060 Laptop (CUDA), libtorch CUDA 13, clang 18 — same conda-wheel
    # nixl_cu13 layout as white, no LD_PRELOAD.
    WHEEL_DIR="${NIXL_WHEEL_DIR:-$HOME/miniconda3/envs/vllm-v1/lib/python3.11/site-packages}"
    NIXL_LD="$WHEEL_DIR/.nixl_cu13.mesonpy.libs"
    PLUGIN_DIR="${NIXL_PLUGIN_DIR:-$WHEEL_DIR/.nixl_cu13.mesonpy.libs/plugins}"
    HIP_PRELOAD=""
    LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}"
    ;;
  pearl)
    NIXL_PREFIX="${NIXL_PREFIX:-/home/stark/build/nixl-1.4.0}"
    NIXL_BUILD="$NIXL_PREFIX/build/src"
    NIXL_LD="$NIXL_BUILD/bindings:$NIXL_BUILD/core:$NIXL_BUILD/infra:$NIXL_BUILD/utils/serdes:$NIXL_BUILD/utils/stream:$NIXL_BUILD/utils/common:$NIXL_BUILD/plugins/ucx"
    PLUGIN_DIR="${NIXL_PLUGIN_DIR:-$NIXL_BUILD/plugins/ucx}"
    HIP_PRELOAD="${HCP_HIP_PRELOAD:-/home/stark/libtorch/lib/libtorch_hip.so}"
    LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}"
    ;;
  *)
    echo "unknown host '$HOST'; set NIXL_LD/NIXL_PLUGIN_DIR/LIBCLANG_PATH explicitly" >&2
    exit 2
    ;;
esac

mode="${1:---run}"
case "$mode" in
  --build)
    echo "[nixl-probe] build with nixl-backend on $HOST (NIXL_LD=$NIXL_LD)"
    PATH="/home/stark/.cargo/bin:$PATH" \
      LIBTORCH="$LIBTORCH" \
      LIBCLANG_PATH="$LIBCLANG_PATH" \
      LIBRARY_PATH="$NIXL_LD:$LIBTORCH/lib:${LIBRARY_PATH:-}" \
      LD_LIBRARY_PATH="$NIXL_LD:$LIBTORCH/lib:${LD_LIBRARY_PATH:-}" \
      cargo build --manifest-path rust/Cargo.toml --features tch-backend,nixl-backend --bin nixl-probe
    echo "[nixl-probe] build OK"
    ;;
  --run)
    echo "[nixl-probe] run register->metadata smoke on $HOST"
    if [ -n "$HIP_PRELOAD" ]; then
      LD_PRELOAD="$HIP_PRELOAD" \
        LD_LIBRARY_PATH="$NIXL_LD:$LIBTORCH/lib:${LD_LIBRARY_PATH:-}" \
        NIXL_PLUGIN_DIR="$PLUGIN_DIR" \
        HCP_TCH_DEVICE=cuda:0 \
        ./rust/target/debug/nixl-probe
    else
      LD_LIBRARY_PATH="$NIXL_LD:$LIBTORCH/lib:${LD_LIBRARY_PATH:-}" \
        NIXL_PLUGIN_DIR="$PLUGIN_DIR" \
        HCP_TCH_DEVICE=cuda:0 \
        ./rust/target/debug/nixl-probe
    fi
    echo "[nixl-probe] run OK"
    ;;
  *)
    echo "usage: $0 [--build|--run]" >&2
    exit 2
    ;;
esac
