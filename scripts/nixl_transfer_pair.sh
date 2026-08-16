#!/usr/bin/env bash
# NIXL pairwise cross-machine transfer probe (S4-1) — parameterized host pair.
#
# Reuses nixl-xfer-probe (bidirectional register->transfer->poll->dump) against
# any host pair among white/pearl/laptop, so S4-1 can validate the N=3 edges:
#   white<->laptop (CUDA<->CUDA, same-vendor)
#   pearl<->laptop (ROCm<->CUDA, cross-vendor)
# (white<->pearl is already covered by scripts/nixl_xfer_probe.sh in S3a.)
#
# Usage (from the Mac):
#   scripts/nixl_transfer_pair.sh --host-a white --host-b laptop [--seq 64] [--device cpu|cuda] [--no-tcp]
set -euo pipefail

SSH_USER="${SSH_USER:-stark}"
LIBTORCH=/home/stark/libtorch
REPO=/home/stark/hetero-cp-ringattn
PROBE="$REPO/rust/target/debug/nixl-xfer-probe"

HOST_A=""; HOST_B=""; SEQ=64; DEVICE=cpu; FORCE_TCP=1; MODE=run
while [ $# -gt 0 ]; do
  case "$1" in
    --host-a) HOST_A="$2"; shift 2 ;;
    --host-b) HOST_B="$2"; shift 2 ;;
    --seq) SEQ="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --no-tcp) FORCE_TCP=0; shift ;;
    --build|--run) MODE="$1"; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
if [ -z "$HOST_A" ] || [ -z "$HOST_B" ]; then
  echo "usage: $0 --host-a <white|pearl|laptop> --host-b <white|pearl|laptop> [--seq N] [--device cpu|cuda] [--no-tcp]" >&2
  exit 2
fi

# host -> (ssh ip, nixl_ld, plugin_dir, preload, libclang, ucx_net_device)
# ucx_net_device pins UCX to the WiFi iface shared by all three nodes (192.168.8.x),
# otherwise white's wired enp10s0 (192.168.100.1) is picked and laptop cannot reach it.
host_env() {
  case "$1" in
    white)
      W=/home/stark/miniconda3/envs/vllm-v1/lib/python3.11/site-packages
      echo "100.118.253.68|$W/.nixl_cu13.mesonpy.libs|$W/.nixl_cu13.mesonpy.libs/plugins||/usr/lib/llvm-21/lib|wlp11s0"
      ;;
    laptop)
      W=/home/stark/miniconda3/envs/vllm-v1/lib/python3.11/site-packages
      echo "100.96.154.1|$W/.nixl_cu13.mesonpy.libs|$W/.nixl_cu13.mesonpy.libs/plugins||/usr/lib/llvm-18/lib|wlp3s0"
      ;;
    pearl)
      B=/home/stark/build/nixl-1.4.0/build/src
      echo "100.111.242.55|$B/bindings:$B/core:$B/infra:$B/utils/serdes:$B/utils/stream:$B/utils/common:$B/plugins/ucx|$B/plugins/ucx|/home/stark/libtorch/lib/libtorch_hip.so|/usr/lib/llvm-18/lib|wlo1"
      ;;
    *) echo "unknown host $1" >&2; exit 2 ;;
  esac
}

A_IP=""; A_LD=""; A_PLUGIN=""; A_PRELOAD=""; A_CLANG=""; A_WIFI=""
B_IP=""; B_LD=""; B_PLUGIN=""; B_PRELOAD=""; B_CLANG=""; B_WIFI=""
IFS='|' read -r A_IP A_LD A_PLUGIN A_PRELOAD A_CLANG A_WIFI <<< "$(host_env $HOST_A)"
IFS='|' read -r B_IP B_LD B_PLUGIN B_PRELOAD B_CLANG B_WIFI <<< "$(host_env $HOST_B)"

ssh_() { ssh -o BatchMode=yes "$SSH_USER@$1" "${@:2}"; }

build_host() {
  local host="$1" ip="$2" ld="$3" clang="$4"
  echo "[pair] build on $host"
  ssh_ "$ip" "cd $REPO && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=$LIBTORCH LIBCLANG_PATH=$clang \
    LIBRARY_PATH=$ld:$LIBTORCH/lib:\${LIBRARY_PATH:-} LD_LIBRARY_PATH=$ld:$LIBTORCH/lib:\${LD_LIBRARY_PATH:-} \
    cargo build --manifest-path rust/Cargo.toml --features tch-backend,nixl-backend --bin nixl-xfer-probe"
}

run_probe() {
  local ip="$1" agent="$2" seed="$3" ld="$4" plugin="$5" preload="$6" wifi="$7"
  local tag; tag="$(echo "$agent" | sed 's/hcp-xfer-//')"
  local prefix
  prefix="cd $REPO && env LD_LIBRARY_PATH=$ld:$LIBTORCH/lib NIXL_PLUGIN_DIR=$plugin NIXL_TELEMETRY_ENABLE=1 NIXL_TELEMETRY_DIR=$WD/tel_$agent HCP_TCH_DEVICE=cuda:0 UCX_NET_DEVICES=$wifi"
  if [ "$FORCE_TCP" = "1" ]; then prefix="$prefix UCX_TLS=tcp"; fi
  if [ -n "$preload" ]; then prefix="$prefix LD_PRELOAD=$preload"; fi
  ssh -f -o BatchMode=yes "$SSH_USER@$ip" "$prefix $PROBE \
    --agent $agent --seed $seed --seq $SEQ --device $DEVICE \
    --md-out $WD/${tag}_md --md-in $WD/${tag}_peer_md \
    --desc-out $WD/${tag}_desc --desc-in $WD/${tag}_peer_desc \
    --src-dump-out $WD/${tag}_src.bin --dest-dump-out $WD/${tag}_dest.bin \
    --done-out $WD/${tag}_done --done-in $WD/${tag}_peer_done \
    > $WD/${tag}.log 2>&1"
}

compare_bin() {
  local a="$1" b="$2" label="$3"
  python3 - "$a" "$b" "$label" <<'PY'
import struct, sys
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read(); label = sys.argv[3]
assert len(a) == len(b) and len(a) > 0 and len(a) % 4 == 0, f"expected equal f32-aligned dumps, got {len(a)}/{len(b)}"
va = struct.unpack(f'<{len(a)//4}f', a); vb = struct.unpack(f'<{len(b)//4}f', b)
md = max(abs(x-y) for x,y in zip(va,vb))
print(f"  {label}: max|diff|={md} ({len(a)} bytes)")
sys.exit(0 if md == 0.0 else 1)
PY
}

wait_remote_file() {
  local ip="$1" path="$2" timeout="${3:-180}"
  local deadline=$(( $(date +%s) + timeout ))
  while ! ssh_ "$ip" "test -f '$path'"; do
    [ "$(date +%s)" -gt "$deadline" ] && { echo "[pair] TIMEOUT $ip:$path" >&2; return 1; }
    sleep 2
  done
}

mode="$MODE"
case "$mode" in
  --build)
    build_host "$HOST_A" "$A_IP" "$A_LD" "$A_CLANG"
    build_host "$HOST_B" "$B_IP" "$B_LD" "$B_CLANG"
    ;;
  --run)
    WD=/tmp/nixl_pair
    LOCAL_TMP="$(mktemp -d /tmp/nixl_pair_local.XXXXXX)"
    trap 'rm -rf "$LOCAL_TMP"' EXIT
    echo "[pair] ${HOST_A} <-> ${HOST_B} (device=$DEVICE, seq=$SEQ, force_tcp=$FORCE_TCP)"
    for ip in "$A_IP" "$B_IP"; do ssh_ "$ip" "pkill -f '[n]ixl-xfer-probe' 2>/dev/null; true; rm -rf $WD; mkdir -p $WD/tel_hcp-xfer-a $WD/tel_hcp-xfer-b"; done
    sleep 1

    run_probe "$A_IP" hcp-xfer-a 0   "$A_LD" "$A_PLUGIN" "$A_PRELOAD" "$A_WIFI"
    run_probe "$B_IP" hcp-xfer-b 100 "$B_LD" "$B_PLUGIN" "$B_PRELOAD" "$B_WIFI"

    for f in a_md a_desc a_src.bin; do wait_remote_file "$A_IP" "$WD/$f"; done
    for f in b_md b_desc b_src.bin; do wait_remote_file "$B_IP" "$WD/$f"; done

    scp -q "$SSH_USER@$A_IP:$WD/a_md" "$SSH_USER@$A_IP:$WD/a_desc" "$LOCAL_TMP/"
    scp -q "$SSH_USER@$B_IP:$WD/b_md" "$SSH_USER@$B_IP:$WD/b_desc" "$LOCAL_TMP/"
    scp -q "$LOCAL_TMP/b_md"   "$SSH_USER@$A_IP:$WD/a_peer_md.tmp"
    scp -q "$LOCAL_TMP/b_desc" "$SSH_USER@$A_IP:$WD/a_peer_desc.tmp"
    scp -q "$LOCAL_TMP/a_md"   "$SSH_USER@$B_IP:$WD/b_peer_md.tmp"
    scp -q "$LOCAL_TMP/a_desc" "$SSH_USER@$B_IP:$WD/b_peer_desc.tmp"
    ssh_ "$A_IP" "mv $WD/a_peer_md.tmp $WD/a_peer_md; mv $WD/a_peer_desc.tmp $WD/a_peer_desc"
    ssh_ "$B_IP" "mv $WD/b_peer_md.tmp $WD/b_peer_md; mv $WD/b_peer_desc.tmp $WD/b_peer_desc"

    wait_remote_file "$A_IP" "$WD/a_done"
    wait_remote_file "$B_IP" "$WD/b_done"
    scp -q "$SSH_USER@$A_IP:$WD/a_done" "$SSH_USER@$B_IP:$WD/b_peer_done.tmp"
    scp -q "$SSH_USER@$B_IP:$WD/b_done" "$SSH_USER@$A_IP:$WD/a_peer_done.tmp"
    ssh_ "$A_IP" "mv $WD/a_peer_done.tmp $WD/a_peer_done"
    ssh_ "$B_IP" "mv $WD/b_peer_done.tmp $WD/b_peer_done"

    wait_remote_file "$A_IP" "$WD/a_dest.bin"
    wait_remote_file "$B_IP" "$WD/b_dest.bin"
    scp -q "$SSH_USER@$A_IP:$WD/a_dest.bin" "$SSH_USER@$B_IP:$WD/b_src.bin" "$LOCAL_TMP/"
    scp -q "$SSH_USER@$B_IP:$WD/b_dest.bin" "$SSH_USER@$A_IP:$WD/a_src.bin" "$LOCAL_TMP/"

    echo "[pair] ===== comparison ====="
    compare_bin "$LOCAL_TMP/b_src.bin" "$LOCAL_TMP/a_dest.bin" "${HOST_A}.dest == ${HOST_B}.src"
    compare_bin "$LOCAL_TMP/a_src.bin" "$LOCAL_TMP/b_dest.bin" "${HOST_B}.dest == ${HOST_A}.src"
    echo "[pair] ${HOST_A} <-> ${HOST_B}: PASS"
    ;;
  *) echo "usage: $0 [--build|--run] --host-a A --host-b B [--seq N] [--device cpu|cuda] [--no-tcp]" >&2; exit 2 ;;
esac
