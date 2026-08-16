#!/usr/bin/env bash
# NIXL ring probe (S4-2a): hop-by-hop KV circulation on the 3-node ring.
#
# Ring: white -> pearl -> laptop -> white. Each node forwards its current block
# to its successor and receives from its predecessor, N-1=2 rounds. The script
# drives desc exchange (successor direction) and per-round done sync
# (predecessor direction) via files — the throwaway probe channel (production
# uses the coordinator control plane from S3b).
#
# Success check: after 2 rounds, white.recv == pearl.orig, pearl.recv ==
# laptop.orig, laptop.recv == white.orig (the last received predecessor KV).
set -euo pipefail

SSH_USER="${SSH_USER:-stark}"
LIBTORCH=/home/stark/libtorch
REPO=/home/stark/hetero-cp-ringattn
PROBE="$REPO/rust/target/debug/nixl-ring-probe"

# host -> ssh ip
ip_of() { case "$1" in white) echo 100.118.253.68 ;; pearl) echo 100.111.242.55 ;; laptop) echo 100.96.154.1 ;; esac; }
# host -> nixl_ld|plugin_dir|preload|libclang|wifi
env_of() {
  case "$1" in
    white)
      W=/home/stark/miniconda3/envs/vllm-v1/lib/python3.11/site-packages
      echo "$W/.nixl_cu13.mesonpy.libs|$W/.nixl_cu13.mesonpy.libs/plugins||/usr/lib/llvm-21/lib|wlp11s0" ;;
    laptop)
      W=/home/stark/miniconda3/envs/vllm-v1/lib/python3.11/site-packages
      echo "$W/.nixl_cu13.mesonpy.libs|$W/.nixl_cu13.mesonpy.libs/plugins||/usr/lib/llvm-18/lib|wlp3s0" ;;
    pearl)
      B=/home/stark/build/nixl-1.4.0/build/src
      echo "$B/bindings:$B/core:$B/infra:$B/utils/serdes:$B/utils/stream:$B/utils/common:$B/plugins/ucx|$B/plugins/ucx|/home/stark/libtorch/lib/libtorch_hip.so|/usr/lib/llvm-18/lib|wlo1" ;;
  esac
}

ssh_() { ssh -o BatchMode=yes "$SSH_USER@$1" "${@:2}"; }

run_probe() {
  local host="$1" seed="$2" seq="$3"
  local ip; ip="$(ip_of $host)"
  local ld plugin preload clang wifi
  IFS='|' read -r ld plugin preload clang wifi <<< "$(env_of $host)"
  local prefix
  prefix="cd $REPO && env LD_LIBRARY_PATH=$ld:$LIBTORCH/lib NIXL_PLUGIN_DIR=$plugin NIXL_TELEMETRY_ENABLE=1 NIXL_TELEMETRY_DIR=$WD/tel_$host HCP_TCH_DEVICE=cuda:0 UCX_TLS=tcp UCX_NET_DEVICES=$wifi"
  if [ -n "$preload" ]; then prefix="$prefix LD_PRELOAD=$preload"; fi
  ssh -f -o BatchMode=yes "$SSH_USER@$ip" "$prefix $PROBE \
    --agent hcp-ring-$host --seed $seed --seq $seq --rounds 2 \
    --md-out $WD/${host}_md --md-in $WD/${host}_peer_md \
    --desc-out $WD/${host}_desc --desc-in $WD/${host}_peer_desc \
    --done-out $WD/${host}_done --done-in $WD/${host}_peer_done \
    --dump-out $WD/${host}_recv.bin \
    > $WD/${host}.log 2>&1"
}

wait_remote_file() {
  local ip="$1" path="$2" timeout="${3:-180}"
  local deadline=$(( $(date +%s) + timeout ))
  while ! ssh_ "$ip" "test -f '$path'"; do
    [ "$(date +%s)" -gt "$deadline" ] && { echo "[ring] TIMEOUT $ip:$path" >&2; return 1; }
    sleep 2
  done
}

# successor of host on the ring white->pearl->laptop->white
succ_of() { case "$1" in white) echo pearl ;; pearl) echo laptop ;; laptop) echo white ;; esac; }
# predecessor of host
pred_of() { case "$1" in white) echo laptop ;; pearl) echo white ;; laptop) echo pearl ;; esac; }

mode="${1:---run}"
case "$mode" in
  --build)
    for h in white pearl laptop; do
      ip="$(ip_of $h)"; IFS='|' read -r ld _ _ clang _ <<< "$(env_of $h)"
      echo "[ring] build on $h"
      ssh_ "$ip" "cd $REPO && PATH=/home/stark/.cargo/bin:\$PATH LIBTORCH=$LIBTORCH LIBCLANG_PATH=$clang \
        LIBRARY_PATH=$ld:$LIBTORCH/lib:\${LIBRARY_PATH:-} LD_LIBRARY_PATH=$ld:$LIBTORCH/lib:\${LD_LIBRARY_PATH:-} \
        cargo build --manifest-path rust/Cargo.toml --features tch-backend,nixl-backend --bin nixl-ring-probe"
    done
    ;;
  --run)
    WD=/tmp/nixl_ring
    LOCAL="$(mktemp -d /tmp/nixl_ring_local.XXXXXX)"
    trap 'rm -rf "$LOCAL"' EXIT
    SEQ="${SEQ:-64}"
    echo "[ring] 3-node ring white -> pearl -> laptop -> white (seq=$SEQ, rounds=2)"
    for h in white pearl laptop; do
      ip="$(ip_of $h)"; ssh_ "$ip" "pkill -f '[n]ixl-ring-probe' 2>/dev/null; true; rm -rf $WD; mkdir -p $WD/tel_$h"
    done
    sleep 1

    run_probe white 0   "$SEQ"
    run_probe pearl 100 "$SEQ"
    run_probe laptop 200 "$SEQ"

    for h in white pearl laptop; do
      ip="$(ip_of $h)"
      for f in ${h}_md ${h}_desc; do wait_remote_file "$ip" "$WD/$f"; done
    done

    # desc + md exchange (successor direction): succ's md/desc -> pred's peer_*
    echo "[ring] exchange md + desc (successor -> predecessor)"
    for h in white pearl laptop; do
      s="$(succ_of $h)"
      ip="$(ip_of $h)"; sip="$(ip_of $s)"
      scp -q "$SSH_USER@$sip:$WD/${s}_md"   "$LOCAL/${s}_md"
      scp -q "$SSH_USER@$sip:$WD/${s}_desc" "$LOCAL/${s}_desc"
      scp -q "$LOCAL/${s}_md"   "$SSH_USER@$ip:$WD/${h}_peer_md.tmp"
      scp -q "$LOCAL/${s}_desc" "$SSH_USER@$ip:$WD/${h}_peer_desc.tmp"
      ssh_ "$ip" "mv $WD/${h}_peer_md.tmp $WD/${h}_peer_md; mv $WD/${h}_peer_desc.tmp $WD/${h}_peer_desc"
    done

    # two rounds of done sync (predecessor direction): pred's done -> succ's peer_done.
    # Per-round file names (done.0/done.1) avoid any stale-file race across rounds.
    for round in 0 1; do
      echo "[ring] round $round done sync"
      for h in white pearl laptop; do ip="$(ip_of $h)"; wait_remote_file "$ip" "$WD/${h}_done.$round"; done
      for h in white pearl laptop; do
        p="$(pred_of $h)"
        ip="$(ip_of $h)"; pip="$(ip_of $p)"
        scp -q "$SSH_USER@$pip:$WD/${p}_done.$round" "$SSH_USER@$ip:$WD/${h}_peer_done.$round.tmp"
        ssh_ "$ip" "mv $WD/${h}_peer_done.$round.tmp $WD/${h}_peer_done.$round"
      done
    done

    for h in white pearl laptop; do ip="$(ip_of $h)"; wait_remote_file "$ip" "$WD/${h}_recv.bin"; done
    for h in white pearl laptop; do ip="$(ip_of $h)"; scp -q "$SSH_USER@$ip:$WD/${h}_recv.bin" "$LOCAL/${h}_recv.bin"; done

    echo "[ring] ===== comparison (recv == predecessor's original) ====="
    python3 - "$LOCAL" "$SEQ" <<'PY'
import struct, sys, os
d = sys.argv[1]; seq = int(sys.argv[2]); n = 128 * seq  # 1*2*seq*64 f32
def load(h):
    b = open(os.path.join(d, f"{h}_recv.bin"),'rb').read()
    assert len(b) == n*4, f"{h} recv {len(b)} != {n*4}"
    return struct.unpack(f'<{n}f', b)
def expected(seed):
    return [float(i) + seed for i in range(n)]
# after 2 rounds: white.recv == pearl.orig(100), pearl.recv == laptop.orig(200), laptop.recv == white.orig(0)
checks = [("white", 100.0), ("pearl", 200.0), ("laptop", 0.0)]
ok = True
for h, seed in checks:
    got = load(h); exp = expected(seed)
    md = max(abs(g - e) for g, e in zip(got, exp))
    status = "PASS" if md == 0.0 else "FAIL"
    if md != 0.0: ok = False
    print(f"  {h}.recv == seed {seed}: max|diff|={md} -> {status}")
print("[ring] RING CIRCULATION:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
    ;;
  *) echo "usage: $0 [--build|--run] [SEQ=64]" >&2; exit 2 ;;
esac
