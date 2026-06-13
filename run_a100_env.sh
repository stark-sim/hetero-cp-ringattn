#!/bin/bash
# A100 4x environment for hetero-cp-ringattn
# Source this before any cargo/build/run command on the A100 host.

export LIBTORCH=/root/libtorch
export LIBTORCH_INCLUDE=/root/libtorch/include
export LIBTORCH_LIB=/root/libtorch/lib

# Collect all extracted NVIDIA wheel lib directories
THIRD_PARTY_LIBS="/root/hetero-cp-ringattn/third_party_libs/extracted"
NVIDIA_LIBS=""
if [ -d "$THIRD_PARTY_LIBS/nvidia" ]; then
    for libdir in "$THIRD_PARTY_LIBS"/nvidia/*/lib; do
        if [ -d "$libdir" ]; then
            NVIDIA_LIBS="${NVIDIA_LIBS}:${libdir}"
        fi
    done
fi

export LD_LIBRARY_PATH="/root/libtorch/lib:/usr/local/cuda/lib64${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cuda:0}"

# Verify cargo is available
if ! command -v cargo >/dev/null 2>&1; then
    echo "[WARN] cargo not found in PATH. Make sure Rust is installed at ~/.cargo/bin" >&2
fi
