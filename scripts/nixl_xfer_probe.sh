#!/usr/bin/env bash
# NIXL cross-machine block transfer probe (form-B S3a) — host-side orchestration.
#
# Runs the symmetric bidirectional transfer between white (CUDA) and pearl
# (ROCm). Each node registers a src + dest block and writes the peer's dest
# block, so one run proves BOTH CUDA->ROCm and ROCm->CUDA.
#
# The script moves metadata/desc/done files between the two hosts as a throwaway
# probe channel (NOT the HCP side-channel; that is S3b). Final byte-for-byte
# comparison: white's dest == pearl's src, pearl's dest == white's src.
#
# Usage (from the Mac, which can SSH to both hosts):
#   scripts/nixl_xfer_probe.sh --build   # build nixl-xfer-probe on both hosts
#   scripts/nixl_xfer_probe.sh --run     # run the cross-machine transfer
set -euo pipefail

SSH_USER="${SSH_USER:-stark}"
WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
PEARL_HOST="${PEARL_HOST:-100.111.242.55}"

REPO=/home/stark/hetero-cp-ringattn
PROBE="$REPO/rust/target/debug/nixl-xfer-probe"
LIBTORCH=/home/stark/libtorch
WD=/tmp/nixl_xfer

# white (CUDA): NIXL from the nixl_cu13 pip wheel (vllm-v1 conda env), no preload.
WHITE_WHEEL=/home/stark/miniconda3/envs/vllm-v1/lib/python3.11/site-packages
WHITE_NIXL_LD="$WHITE_WHEEL/.nixl_cu13.mesonpy.libs"
WHITE_PLUGIN_DIR="$WHITE_WHEEL/.nixl_cu13.mesonpy.libs/plugins"
WHITE_PRELOAD=""
WHITE_LIBCLANG=/usr/lib/llvm-21/lib

# pearl (ROCm): NIXL from the source build tree, needs libtorch_hip.so preload.
PEARL_BUILD=/home/stark/build/nixl-1.4.0/build/src
PEARL_NIXL_LD="$PEARL_BUILD/bindings:$PEARL_BUILD/core:$PEARL_BUILD/infra:$PEARL_BUILD/utils/serdes:$PEARL_BUILD/utils/stream:$PEARL_BUILD/utils/common:$PEARL_BUILD/plugins/ucx"
PEARL_PLUGIN_DIR="$PEARL_BUILD/plugins/ucx"
PEARL_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so
PEARL_LIBCLANG=/usr/lib/llvm-18/lib

LOCAL_TMP="$(mktemp -d /tmp/nixl_xfer_local.XXXXXX)"
trap 'rm -rf "$LOCAL_TMP"' EXIT

ssh_() { ssh -o BatchMode=yes "$SSH_USER@$1" "${@:2}"; }

wait_remote_file() {
  local host="$1" path="$2" timeout="${3:-300}"
  local deadline=$(( $(date +%s) + timeout ))
  while ! ssh_ "$host" "test -f '$path'"; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "[xfer] TIMEOUT waiting for $host:$path" >&2
      return 1
    fi
    sleep 2
  done
}

build_host() {
  local host="$1" nixl_ld="$2" libclang="$3"
  echo "[xfer] build nixl-xfer-probe on $host"
  ssh_ "$host" "cd $REPO && PATH=/home/stark/.cargo/bin:\$PATH \
    LIBTORCH=$LIBTORCH LIBCLANG_PATH=$libclang \
    LIBRARY_PATH=$nixl_ld:$LIBTORCH/lib:\$LIBRARY_PATH \
    LD_LIBRARY_PATH=$nixl_ld:$LIBTORCH/lib:\$LD_LIBRARY_PATH \
    cargo build --manifest-path rust/Cargo.toml --features tch-backend,nixl-backend --bin nixl-xfer-probe"
  echo "[xfer] build OK on $host"
}

run_probe() {
  local host="$1" agent="$2" seed="$3" nixl_ld="$4" plugin_dir="$5" preload="$6"
  local tag
  tag="$(echo "$agent" | sed 's/hcp-xfer-//')"   # white / pearl
  local prefix
  prefix="cd $REPO && env LD_LIBRARY_PATH=$nixl_ld:$LIBTORCH/lib NIXL_PLUGIN_DIR=$plugin_dir"
  # UCX_TLS=tcp: force TCP transport and exclude cuda_ipc/rocm GPU-direct,
  # which has no cross-vendor (CUDA<->ROCm) remote protocol ("cannot find
  # remote protocol for put(multi) from cuda/GPU0 to rocm").
  prefix="$prefix NIXL_TELEMETRY_ENABLE=1 NIXL_TELEMETRY_DIR=$WD/tel_$agent HCP_TCH_DEVICE=cuda:0 UCX_TLS=tcp"
  if [ -n "$preload" ]; then
    prefix="$prefix LD_PRELOAD=$preload"
  fi
  # ssh -f backgrounds the SSH itself so the orchestrator returns immediately
  # while the probe keeps running on the remote host (avoids the SSH-blocking
  # behavior of a command ending in `&` with no follow-up command).
  ssh -f -o BatchMode=yes "$SSH_USER@$host" "$prefix $PROBE \
    --agent $agent --seed $seed \
    --md-out $WD/${tag}_md --md-in $WD/${tag}_peer_md \
    --desc-out $WD/${tag}_desc --desc-in $WD/${tag}_peer_desc \
    --src-dump-out $WD/${tag}_src.bin --dest-dump-out $WD/${tag}_dest.bin \
    --done-out $WD/${tag}_done --done-in $WD/${tag}_peer_done \
    > $WD/${tag}.log 2>&1"
}

compare_bin() {
  local a="$1" b="$2" label="$3"
  echo "[xfer] compare $label"
  python3 - "$a" "$b" <<'PY'
import struct, sys
a = open(sys.argv[1], 'rb').read()
b = open(sys.argv[2], 'rb').read()
assert len(a) == len(b) == 96, f"expected 96 bytes, got {len(a)}/{len(b)}"
va = struct.unpack('<24f', a)
vb = struct.unpack('<24f', b)
diffs = [abs(x - y) for x, y in zip(va, vb)]
md = max(diffs)
print(f"  {sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]}: max|diff|={md}")
print(f"  dest[:5]={[round(v,4) for v in vb[:5]]}")
print(f"  src[:5] ={[round(v,4) for v in va[:5]]}")
if md == 0.0:
    print(f"  {label}: PASS (byte-identical)")
else:
    print(f"  {label}: FAIL")
    sys.exit(1)
PY
}

mode="${1:---run}"
case "$mode" in
  --build)
    build_host "$WHITE_HOST" "$WHITE_NIXL_LD" "$WHITE_LIBCLANG"
    build_host "$PEARL_HOST" "$PEARL_NIXL_LD" "$PEARL_LIBCLANG"
    ;;
  --run)
    echo "[xfer] clean remote work dirs + kill residual probes"
    # [n]ixl regex trick: the pattern matches the probe's argv but NOT the
    # pkill shell's own argv (which contains the literal "[n]ixl-xfer-probe").
    for h in "$WHITE_HOST" "$PEARL_HOST"; do ssh_ "$h" "pkill -f '[n]ixl-xfer-probe' 2>/dev/null; true"; done
    sleep 2
    ssh_ "$WHITE_HOST" "rm -rf $WD && mkdir -p $WD/tel_hcp-xfer-white"
    ssh_ "$PEARL_HOST" "rm -rf $WD && mkdir -p $WD/tel_hcp-xfer-pearl"

    echo "[xfer] start white (CUDA, seed=0) + pearl (ROCm, seed=100)"
    run_probe "$WHITE_HOST" hcp-xfer-white 0   "$WHITE_NIXL_LD" "$WHITE_PLUGIN_DIR" "$WHITE_PRELOAD"
    run_probe "$PEARL_HOST" hcp-xfer-pearl 100 "$PEARL_NIXL_LD" "$PEARL_PLUGIN_DIR" "$PEARL_PRELOAD"

    echo "[xfer] wait for both nodes to export md/desc/src"
    for f in white_md white_desc white_src.bin; do wait_remote_file "$WHITE_HOST" "$WD/$f"; done
    for f in pearl_md pearl_desc pearl_src.bin; do wait_remote_file "$PEARL_HOST" "$WD/$f"; done

    echo "[xfer] exchange md + desc (via local host as relay)"
    scp -q "$SSH_USER@$WHITE_HOST:$WD/white_md"   "$LOCAL_TMP/white_md"
    scp -q "$SSH_USER@$WHITE_HOST:$WD/white_desc" "$LOCAL_TMP/white_desc"
    scp -q "$SSH_USER@$PEARL_HOST:$WD/pearl_md"   "$LOCAL_TMP/pearl_md"
    scp -q "$SSH_USER@$PEARL_HOST:$WD/pearl_desc" "$LOCAL_TMP/pearl_desc"
    # white's peer = pearl's md/desc; pearl's peer = white's md/desc
    # write to .tmp then atomically mv so the probe never reads a half-written
    # file (scp creates the target before finishing; wait_for_file would see it).
    scp -q "$LOCAL_TMP/pearl_md"   "$SSH_USER@$WHITE_HOST:$WD/white_peer_md.tmp"
    scp -q "$LOCAL_TMP/pearl_desc" "$SSH_USER@$WHITE_HOST:$WD/white_peer_desc.tmp"
    scp -q "$LOCAL_TMP/white_md"   "$SSH_USER@$PEARL_HOST:$WD/pearl_peer_md.tmp"
    scp -q "$LOCAL_TMP/white_desc" "$SSH_USER@$PEARL_HOST:$WD/pearl_peer_desc.tmp"
    ssh_ "$WHITE_HOST" "mv $WD/white_peer_md.tmp $WD/white_peer_md; mv $WD/white_peer_desc.tmp $WD/white_peer_desc"
    ssh_ "$PEARL_HOST" "mv $WD/pearl_peer_md.tmp $WD/pearl_peer_md; mv $WD/pearl_peer_desc.tmp $WD/pearl_peer_desc"

    echo "[xfer] wait for both nodes to finish their transfer (done signals)"
    wait_remote_file "$WHITE_HOST" "$WD/white_done"
    wait_remote_file "$PEARL_HOST" "$WD/pearl_done"

    echo "[xfer] exchange done signals"
    scp -q "$SSH_USER@$WHITE_HOST:$WD/white_done" "$SSH_USER@$PEARL_HOST:$WD/pearl_peer_done.tmp"
    scp -q "$SSH_USER@$PEARL_HOST:$WD/pearl_done" "$SSH_USER@$WHITE_HOST:$WD/white_peer_done.tmp"
    ssh_ "$WHITE_HOST" "mv $WD/white_peer_done.tmp $WD/white_peer_done"
    ssh_ "$PEARL_HOST" "mv $WD/pearl_peer_done.tmp $WD/pearl_peer_done"

    echo "[xfer] wait for both nodes to dump their dest blocks"
    wait_remote_file "$WHITE_HOST" "$WD/white_dest.bin"
    wait_remote_file "$PEARL_HOST" "$WD/pearl_dest.bin"

    echo "[xfer] pull dumps + src for comparison"
    scp -q "$SSH_USER@$WHITE_HOST:$WD/white_dest.bin" "$SSH_USER@$WHITE_HOST:$WD/white_src.bin" "$LOCAL_TMP/"
    scp -q "$SSH_USER@$PEARL_HOST:$WD/pearl_dest.bin" "$SSH_USER@$PEARL_HOST:$WD/pearl_src.bin" "$LOCAL_TMP/"

    echo "[xfer] ===== comparison ====="
    # white's dest should now hold pearl's src; pearl's dest should hold white's src.
    compare_bin "$LOCAL_TMP/pearl_src.bin" "$LOCAL_TMP/white_dest.bin" "white.dest == pearl.src (ROCm->CUDA)"
    compare_bin "$LOCAL_TMP/white_src.bin" "$LOCAL_TMP/pearl_dest.bin" "pearl.dest == white.src (CUDA->ROCm)"

    echo "[xfer] ===== node logs ====="
    echo "--- white ---"; ssh_ "$WHITE_HOST" "cat $WD/white.log"
    echo "--- pearl ---"; ssh_ "$PEARL_HOST" "cat $WD/pearl.log"
    echo "[xfer] CROSS-MACHINE TRANSFER: PASS"
    ;;
  *)
    echo "usage: $0 [--build|--run]" >&2
    exit 2
    ;;
esac
