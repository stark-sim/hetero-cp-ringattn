#!/usr/bin/env bash
# NIXL S3b side-channel probe: exchange NIXL block-transport metadata over the
# coordinator control plane (no separate side-channel port).
#
# Topology: coordinator on white (domain order 0,1); worker 0 on white (CUDA),
# worker 1 on pearl (ROCm). Each worker builds with nixl-backend and registers
# a probe block in WorkerRuntime::new; the coordinator's --nixl-exchange runs
# NixlExchange -> collect NixlMetadata -> broadcast NixlPeers, then shuts down.
#
# Success = coordinator prints "exchanged NIXL metadata across 2 workers" and
# each worker prints "loaded NIXL metadata from domain N (agent hcp-worker-N)",
# proving load_remote_md established the cross-vendor UCX connection via the
# control plane.
set -euo pipefail

SSH_USER="${SSH_USER:-stark}"
WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
MODEL_DIR="${MODEL_DIR:-/home/stark/models/Qwen2-0.5B}"

REPO=/home/stark/hetero-cp-ringattn
BIN="$REPO/rust/target/debug/hcp-ringattn-rust"
LIBTORCH=/home/stark/libtorch

WHITE_WHEEL=/home/stark/miniconda3/envs/vllm-v1/lib/python3.11/site-packages
WHITE_NIXL_LD="$WHITE_WHEEL/.nixl_cu13.mesonpy.libs"
WHITE_PLUGIN_DIR="$WHITE_WHEEL/.nixl_cu13.mesonpy.libs/plugins"
WHITE_PRELOAD=""
WHITE_LIBCLANG=/usr/lib/llvm-21/lib

PEARL_BUILD=/home/stark/build/nixl-1.4.0/build/src
PEARL_NIXL_LD="$PEARL_BUILD/bindings:$PEARL_BUILD/core:$PEARL_BUILD/infra:$PEARL_BUILD/utils/serdes:$PEARL_BUILD/utils/stream:$PEARL_BUILD/utils/common:$PEARL_BUILD/plugins/ucx"
PEARL_PLUGIN_DIR="$PEARL_BUILD/plugins/ucx"
PEARL_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so
PEARL_LIBCLANG=/usr/lib/llvm-18/lib

COORD_PORT=29510
W0_PORT=29511
W1_PORT=29512

ssh_() { ssh -o BatchMode=yes "$SSH_USER@$1" "${@:2}"; }

build_host() {
  local host="$1" nixl_ld="$2" libclang="$3"
  echo "[s3b] build hcp-ringattn-rust (nixl-backend) on $host"
  ssh_ "$host" "cd $REPO && PATH=/home/stark/.cargo/bin:\$PATH \
    LIBTORCH=$LIBTORCH LIBCLANG_PATH=$libclang \
    LIBRARY_PATH=$nixl_ld:$LIBTORCH/lib:\${LIBRARY_PATH:-} \
    LD_LIBRARY_PATH=$nixl_ld:$LIBTORCH/lib:\${LD_LIBRARY_PATH:-} \
    cargo build --features tch-backend,nixl-backend --bin hcp-ringattn-rust"
  echo "[s3b] build OK on $host"
}

mode="${1:---run}"
case "$mode" in
  --build)
    build_host "$WHITE_HOST" "$WHITE_NIXL_LD" "$WHITE_LIBCLANG"
    build_host "$PEARL_HOST" "$PEARL_NIXL_LD" "$PEARL_LIBCLANG"
    ;;
  --run)
    WD=/tmp/nixl_s3b
    echo "[s3b] clean + kill residual"
    for h in "$WHITE_HOST" "$PEARL_HOST"; do ssh_ "$h" "pkill -f '[h]cp-ringattn-rust' 2>/dev/null; true"; done
    sleep 1

    echo "[s3b] launch coordinator (white, --nixl-exchange)"
    ssh_ "$WHITE_HOST" "cd $REPO && env LD_LIBRARY_PATH=$LIBTORCH/lib HCP_TCH_DEVICE=cuda:0 nohup $BIN \
      --distributed-role coordinator --model-dir $MODEL_DIR --num-domains 2 \
      --listen-addr 0.0.0.0:$COORD_PORT --nixl-exchange \
      > $WD/coordinator.log 2>&1 < /dev/null &"

    echo "[s3b] launch worker 1 (pearl, domain 1)"
    ssh_ "$PEARL_HOST" "cd $REPO && env LD_LIBRARY_PATH=$PEARL_NIXL_LD:$LIBTORCH/lib NIXL_PLUGIN_DIR=$PEARL_PLUGIN_DIR UCX_TLS=tcp HCP_TCH_DEVICE=cuda:0 LD_PRELOAD=$PEARL_PRELOAD nohup $BIN \
      --distributed-role worker --domain-id 1 --model-dir $MODEL_DIR \
      --listen-addr 0.0.0.0:$W1_PORT --next-peer-addr $WHITE_HOST:$W0_PORT \
      --coordinator-addr $WHITE_HOST:$COORD_PORT --num-domains 2 \
      > $WD/worker1.log 2>&1 < /dev/null &"

    echo "[s3b] launch worker 0 (white, domain 0)"
    ssh_ "$WHITE_HOST" "cd $REPO && env LD_LIBRARY_PATH=$WHITE_NIXL_LD:$LIBTORCH/lib NIXL_PLUGIN_DIR=$WHITE_PLUGIN_DIR UCX_TLS=tcp HCP_TCH_DEVICE=cuda:0 nohup $BIN \
      --distributed-role worker --domain-id 0 --model-dir $MODEL_DIR \
      --listen-addr 0.0.0.0:$W0_PORT --next-peer-addr $PEARL_HOST:$W1_PORT \
      --coordinator-addr 127.0.0.1:$COORD_PORT --num-domains 2 \
      > $WD/worker0.log 2>&1 < /dev/null &"

    echo "[s3b] waiting for coordinator to finish the exchange (up to 180s)"
    deadline=$(( $(date +%s) + 180 ))
    while ! ssh_ "$WHITE_HOST" "grep -q 'NIXL metadata exchange done' $WD/coordinator.log 2>/dev/null || grep -q 'NIXL metadata exchange failed' $WD/coordinator.log 2>/dev/null"; do
      if [ "$(date +%s)" -gt "$deadline" ]; then echo "[s3b] TIMEOUT waiting for coordinator" >&2; break; fi
      sleep 2
    done

    echo "[s3b] ===== coordinator log ====="
    ssh_ "$WHITE_HOST" "cat $WD/coordinator.log" || true
    echo "[s3b] ===== worker 0 (white) log ====="
    ssh_ "$WHITE_HOST" "grep -E 'NIXL|loaded NIXL|reported NIXL|handshake sent' $WD/worker0.log" || true
    echo "[s3b] ===== worker 1 (pearl) log ====="
    ssh_ "$PEARL_HOST" "grep -E 'NIXL|loaded NIXL|reported NIXL|handshake sent' $WD/worker1.log" || true

    # Success = both workers loaded the peer's metadata via the control plane.
    w0_ok=$(ssh_ "$WHITE_HOST" "grep -c 'loaded NIXL metadata from domain 1' $WD/worker0.log" || echo 0)
    w1_ok=$(ssh_ "$PEARL_HOST" "grep -c 'loaded NIXL metadata from domain 0' $WD/worker1.log" || echo 0)
    echo "[s3b] worker0 loaded peer metadata: ${w0_ok}; worker1 loaded peer metadata: ${w1_ok}"
    if [ "${w0_ok}" -ge 1 ] && [ "${w1_ok}" -ge 1 ]; then
      echo "[s3b] SIDE-CHANNEL EXCHANGE: PASS"
    else
      echo "[s3b] SIDE-CHANNEL EXCHANGE: FAIL" >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [--build|--run]" >&2
    exit 2
    ;;
esac
