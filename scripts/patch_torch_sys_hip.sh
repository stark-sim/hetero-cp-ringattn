#!/usr/bin/env bash
# Patch torch-sys 0.24.0 for ROCm/HIP device support.
# tch-rs 0.24.0 hardcodes at::kCUDA for all GPU devices, but ROCm builds
# only register kernels under at::kHIP. This patch adds hasHIP() detection.
#
# Usage: bash scripts/patch_torch_sys_hip.sh

set -euo pipefail

SYS_DIR=$(find ~/.cargo/registry/src -name "torch-sys-0.24.0" -type d | head -1)
if [ -z "$SYS_DIR" ]; then
    echo "ERROR: torch-sys-0.24.0 not found in cargo registry cache."
    echo "Run 'cargo check --features tch-backend' first to download it."
    exit 1
fi

FILE="$SYS_DIR/libtch/torch_api.cpp"

if grep -q "hasHIP()" "$FILE" 2>/dev/null; then
    echo "torch-sys already patched for HIP."
    exit 0
fi

sed -i \
    's/return at::Device(at::kCUDA, \/\*index=\*\/d);/if (at::globalContext().hasHIP()) return at::Device(at::kHIP, \/*index=*\/d);\n  return at::Device(at::kCUDA, \/*index=*\/d);/' \
    "$FILE"

echo "torch-sys patched for HIP: $FILE"
