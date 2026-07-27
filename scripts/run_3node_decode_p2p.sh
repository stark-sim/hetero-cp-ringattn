#!/bin/bash
# 3-node P2P decode Q-ring validation: laptop A + white B relay + pearl C owner.
#
# Topology (two-phase unified ring, N=3):
#   prefill (HTTP KV store, neighbor accumulate-forward):
#     laptop (100.96.154.1, RTX 4060 Laptop CUDA, vllm-v1)  = A producer (c0)
#     white  (100.118.253.68, RTX 4090 CUDA, vllm-v1)       = B relay (c1)
#     pearl  (100.111.242.55, RX 9060 XT gfx1200, vllm-rocm) = C owner (c2)
#   decode (P2P TCP Q-ring, HCP_RING_DECODE_TRANSPORT=ring):
#     ring order C(pearl, idx0) -> A(laptop, idx1) -> B(white, idx2) -> C
#     Q + accumulator (O, LSE) hop-by-hop; growth KV piggybacked to its
#     round-robin assignee (append-then-serve).  No collective, no HTTP in
#     decode.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LAPTOP_HOST="${LAPTOP_HOST:-100.96.154.1}"
LAPTOP_SSH="${LAPTOP_USER:-stark}@${LAPTOP_HOST}"
LAPTOP_PLUGIN_REPO="${LAPTOP_PLUGIN_REPO:-/home/stark/hcp-vllm-plugin}"

WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_SSH="${WHITE_USER:-stark}@${WHITE_HOST}"
WHITE_PLUGIN_REPO="${WHITE_PLUGIN_REPO:-/home/stark/hcp-vllm-plugin}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_SSH="${PEARL_USER:-stark}@${PEARL_HOST}"
PEARL_PLUGIN_REPO="${PEARL_PLUGIN_REPO:-/home/stark/hcp-vllm-plugin}"

SERVE_PORT="${SERVE_PORT:-8901}"
DECODE_PORT_A="${DECODE_PORT_A:-8951}"
DECODE_PORT_B="${DECODE_PORT_B:-8952}"
DECODE_PORT_C="${DECODE_PORT_C:-8950}"
RING_ORDER="${PEARL_HOST}:${DECODE_PORT_C},${LAPTOP_HOST}:${DECODE_PORT_A},${WHITE_HOST}:${DECODE_PORT_B}"

TOTAL="${TOTAL:-1536}"
SPLIT0="${SPLIT0:-512}"
SPLIT1="${SPLIT1:-512}"
DECODE="${DECODE:-8}"
RUN_ID="p2p3n-$(date +%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/ring-decode-p2p-3node-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() { printf "'"; printf "%s" "$1" | sed "s/'/'\\''/g"; printf "'"; }
run_laptop() { ssh -o ConnectTimeout=20 "${LAPTOP_SSH}" "bash -lc $(shell_quote "$1")"; }
run_white()  { ssh -o ConnectTimeout=20 "${WHITE_SSH}"  "bash -lc $(shell_quote "$1")"; }
run_pearl()  { ssh -o ConnectTimeout=20 "${PEARL_SSH}"  "bash -lc $(shell_quote "$1")"; }

echo "=== HCP 3-node P2P decode Q-ring: C(pearl)->A(laptop)->B(white)->C ==="
echo "RUN_ID=${RUN_ID}  total=${TOTAL} chunks=${SPLIT0}+${SPLIT1}+$((TOTAL-SPLIT0-SPLIT1)) decode=${DECODE}"
echo "ring order: ${RING_ORDER}"
exec > >(tee -a "${REPORT_DIR}/driver.log") 2>&1

# === Pre-checks ===
echo "=== pre-checks ==="
LAP_HEAD="$(run_laptop "git -C ${LAPTOP_PLUGIN_REPO} rev-parse HEAD")"
WHITE_HEAD="$(run_white "git -C ${WHITE_PLUGIN_REPO} rev-parse HEAD")"
PEARL_HEAD="$(run_pearl "git -C ${PEARL_PLUGIN_REPO} rev-parse HEAD")"
echo "plugin HEAD laptop=${LAP_HEAD} white=${WHITE_HEAD} pearl=${PEARL_HEAD}"
if [ "${LAP_HEAD}" != "${WHITE_HEAD}" ] || [ "${LAP_HEAD}" != "${PEARL_HEAD}" ]; then
  echo "FATAL: plugin repo HEAD mismatch across nodes; git pull first" >&2
  exit 2
fi
run_laptop "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
run_white  "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
run_pearl  "rocm-smi --showmeminfo vram 2>/dev/null | grep -i used || true"
# Ring TCP pairwise reachability will be exercised at startup; ports must be free.
run_laptop "ss -tln | grep -E \":(8901|${DECODE_PORT_A})\" && exit 1 || true"
run_white  "ss -tln | grep -E \":(8901|${DECODE_PORT_B})\" && exit 1 || true"
run_pearl  "ss -tln | grep -E \":(8901|${DECODE_PORT_C})\" && exit 1 || true"
run_laptop "pgrep -f '[v]alidate_ring_decode_p2p' && exit 1 || true"
run_white  "pgrep -f '[v]alidate_ring_decode_p2p' && exit 1 || true"
run_pearl  "pgrep -f '[v]alidate_ring_decode_p2p' && exit 1 || true"
echo "pre-checks OK (plugin HEAD=${LAP_HEAD})"

STORE_A="/tmp/hcp_p2p3n_store_a_${RUN_ID}"
STORE_B="/tmp/hcp_p2p3n_store_b_${RUN_ID}"
STORE_C="/tmp/hcp_p2p3n_store_c_${RUN_ID}"
DONE_FILE="/tmp/hcp_p2p3n_done_${RUN_ID}"
SCRIPT="validate_ring_decode_p2p.py"

graceful_ab_shutdown() {
  # Signal A/B via the done file, then WAIT for them to print their
  # RingDecodeNode decode-phase stats and exit gracefully (they poll every
  # 2s); pkill is only the straggler fallback.  Killing first would lose
  # the decode-phase evidence from the archive (Reviewer WARN, p2p3n-235241).
  run_laptop "touch ${DONE_FILE}" >/dev/null 2>&1 || true
  run_white  "touch ${DONE_FILE}" >/dev/null 2>&1 || true
  for i in $(seq 1 20); do
    local a_alive=false b_alive=false
    kill -0 "${A_PID:-0}" 2>/dev/null && a_alive=true
    kill -0 "${B_PID:-0}" 2>/dev/null && b_alive=true
    $a_alive || $b_alive || break
    sleep 3
  done
}

cleanup() {
  graceful_ab_shutdown
  run_laptop "pkill -f validate_ring_decode_p2p || true" >/dev/null 2>&1 || true
  run_white  "pkill -f validate_ring_decode_p2p || true" >/dev/null 2>&1 || true
  run_pearl  "pkill -f validate_ring_decode_p2p || true" >/dev/null 2>&1 || true
  wait "${A_PID:-0}" 2>/dev/null || true
  wait "${B_PID:-0}" 2>/dev/null || true
}
trap cleanup EXIT

wait_ready_http() { # $1=host $2=chunk $3=ssh-pid $4=name
  local ready=0
  for i in $(seq 1 150); do
    if curl -sf -o /dev/null "http://$1:${SERVE_PORT}/${RUN_ID}/$2/_READY"; then
      echo "$4 KV ready after $((i * 5))s"
      return 0
    fi
    if ! kill -0 "$3" 2>/dev/null; then
      echo "FATAL: $4 ssh exited early; see report log" >&2
      return 1
    fi
    sleep 5
  done
  echo "FATAL: $4 never became ready" >&2
  return 1
}

RING_ENV="HCP_RING_DECODE_RING=1 HCP_RING_DECODE_TRANSPORT=ring"

# === A: producer on laptop (c0, ring idx 1) ===
echo "=== A: producer on laptop (c0=${SPLIT0}, ring idx 1, TCP :${DECODE_PORT_A}) ==="
a_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1 && \
  ${RING_ENV} python ${LAPTOP_PLUGIN_REPO}/${SCRIPT} --mode producer \
    --total ${TOTAL} --split0 ${SPLIT0} --split1 ${SPLIT1} --run-id ${RUN_ID} \
    --port-a ${SERVE_PORT} --decode-port-a ${DECODE_PORT_A} \
    --ring-order ${RING_ORDER} \
    --store-a ${STORE_A} --store-b /tmp/u1_${RUN_ID} --store-c /tmp/u2_${RUN_ID} \
    --done-file ${DONE_FILE} --gpu-mem-ab 0.4 --hold-secs 900"
run_laptop "${a_cmd}" >"${REPORT_DIR}/producer_a.log" 2>&1 &
A_PID=$!
echo "A ssh pid=${A_PID}"
wait_ready_http "${LAPTOP_HOST}" "c0" "${A_PID}" "A(laptop)" || { tail -20 "${REPORT_DIR}/producer_a.log" >&2 || true; exit 1; }

# === B: relay on white (c1, ring idx 2; consumes c0 from laptop) ===
echo "=== B: relay on white (c1=${SPLIT1}, ring idx 2, TCP :${DECODE_PORT_B}) ==="
b_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1 && \
  ${RING_ENV} python ${WHITE_PLUGIN_REPO}/${SCRIPT} --mode relay \
    --total ${TOTAL} --split0 ${SPLIT0} --split1 ${SPLIT1} --run-id ${RUN_ID} \
    --url-a http://${LAPTOP_HOST}:${SERVE_PORT} --port-b ${SERVE_PORT} \
    --decode-port-b ${DECODE_PORT_B} --ring-order ${RING_ORDER} \
    --store-a /tmp/u1_${RUN_ID} --store-b ${STORE_B} --store-c /tmp/u2_${RUN_ID} \
    --done-file ${DONE_FILE} --gpu-mem-ab 0.18 --hold-secs 900"
run_white "${b_cmd}" >"${REPORT_DIR}/relay_b.log" 2>&1 &
B_PID=$!
echo "B ssh pid=${B_PID}"
wait_ready_http "${WHITE_HOST}" "c1" "${B_PID}" "B(white)" || { tail -20 "${REPORT_DIR}/relay_b.log" >&2 || true; exit 1; }

# === C: owner on pearl (c2, ring idx 0; stages the ACCUMULATED prefix from
# its physical predecessor white only — neighbor accumulate-forward keeps
# every node's connections ring-adjacent even for N>3 topologies where not
# all devices are directly reachable) ===
echo "=== C: owner on pearl (c2=$((TOTAL-SPLIT0-SPLIT1)), ring idx 0, TCP :${DECODE_PORT_C}; prefix via white only) ==="
SP=/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages
c_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-rocm && \
  export LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_devel/lib/host-math/lib:${SP}/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-} && \
  ${RING_ENV} python ${PEARL_PLUGIN_REPO}/${SCRIPT} --mode consumer \
    --total ${TOTAL} --split0 ${SPLIT0} --split1 ${SPLIT1} --run-id ${RUN_ID} \
    --url-a http://${WHITE_HOST}:${SERVE_PORT} --url-b http://${WHITE_HOST}:${SERVE_PORT} \
    --decode-port-c ${DECODE_PORT_C} --ring-order ${RING_ORDER} \
    --store-a /tmp/u1_${RUN_ID} --store-b /tmp/u2_${RUN_ID} --store-c ${STORE_C} \
    --done-file ${DONE_FILE} --decode ${DECODE} --gpu-mem-c 0.35"
set +e
run_pearl "${c_cmd}" >"${REPORT_DIR}/consumer_c.log" 2>&1
CONS_RC=$?
set -e

# Graceful A/B shutdown FIRST so their decode-phase RingDecodeNode stats
# reach the archive before we gate on them.
graceful_ab_shutdown

echo "=== consumer log (tail) ==="
tail -30 "${REPORT_DIR}/consumer_c.log" || true
echo "=== relay log (tail) ==="
tail -6 "${REPORT_DIR}/relay_b.log" || true
echo "=== producer log (tail) ==="
tail -6 "${REPORT_DIR}/producer_a.log" || true

# Gate: A/B decode-phase stats must be in the archive (self-sufficient
# evidence — the verdict must not rest on C-side checks alone).
STATS_OK=0
grep -q "RingDecodeNode" "${REPORT_DIR}/producer_a.log" && \
grep -q "RingDecodeNode" "${REPORT_DIR}/relay_b.log" && STATS_OK=1
if [ "${STATS_OK}" != 1 ]; then
  echo "FATAL: A/B RingDecodeNode decode-phase stats missing from archive" >&2
fi

echo "RUN_ID=${RUN_ID}  report=${REPORT_DIR}"
if [ "${CONS_RC}" = 0 ] && [ "${STATS_OK}" = 1 ]; then
  echo "=== VERDICT: PASS — 3-node heterogeneous P2P decode Q-ring validated ==="
  exit 0
else
  echo "=== VERDICT: FAIL (consumer exit=${CONS_RC}, stats_ok=${STATS_OK}) ==="
  exit 1
fi
