#!/bin/bash
# One-shot environment provisioning for laptop as HCP N=3 third node.
# Runs detached on laptop; writes /tmp/laptop_hcp_setup.status when done.
# Rust via rsproxy.cn mirror (direct static.rust-lang.org stalled at KB/s).
set -uo pipefail
exec > /tmp/laptop_hcp_setup.log 2>&1
echo "[setup $(date +%H:%M:%S)] start"

fail() { echo "[setup] FAIL: $*"; echo "FAIL: $*" > /tmp/laptop_hcp_setup.status; exit 1; }

# 1. Rust toolchain
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    export RUSTUP_DIST_SERVER=https://rsproxy.cn
    export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
    curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh -o /tmp/rustup-init.sh || fail "rustup-init.sh download"
    sh /tmp/rustup-init.sh -y --default-toolchain stable || fail "rustup-init"
fi
"$HOME/.cargo/bin/rustc" --version || fail "rustc missing after install"
echo "[setup] rust ok: $("$HOME/.cargo/bin/rustc" --version)"

# 2. libtorch 2.11.0+cu130 (skip if already pushed from white)
if [ ! -f "$HOME/libtorch/build-version" ]; then
    cd /tmp
    if curl -sSL --speed-limit 102400 --speed-time 60 -o libtorch.zip \
        'https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.11.0%2Bcu130.zip'; then
        unzip -q -o libtorch.zip -d "$HOME" && rm -f libtorch.zip
        echo "[setup] libtorch ok (pytorch.org)"
    else
        rm -f libtorch.zip
        echo "[setup] libtorch download too slow/failed; leaving marker for white-side push"
        echo "FAIL: libtorch-download" > /tmp/laptop_hcp_setup.status
        exit 1
    fi
fi
cat "$HOME/libtorch/build-version" || fail "libtorch missing"
echo "[setup] libtorch ok: $(cat "$HOME/libtorch/build-version")"

# 3. Repo sync + build (model is pushed separately from white)
cd "$HOME/hetero-cp-ringattn" || fail "repo missing"
git checkout main && git pull --ff-only origin main || fail "git sync"
cd rust
PATH="$HOME/.cargo/bin:$PATH" LIBTORCH="$HOME/libtorch" LD_LIBRARY_PATH="$HOME/libtorch/lib" \
    cargo build --features tch-backend --release || fail "cargo build"
ls target/release/hcp-ringattn-rust > /dev/null || fail "binary missing"

echo "DONE" > /tmp/laptop_hcp_setup.status
echo "[setup $(date +%H:%M:%S)] DONE"
