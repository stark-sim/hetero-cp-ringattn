#!/bin/bash
# 3-node true HCP ring (generic-N) validation: laptop A + white B relay + pearl C.
#
# Topology (generic-N memory-splitting ring, N=3):
#   laptop (100.96.154.1, RTX 4060 Laptop CUDA, vllm-v1 env)  = A producer (c0):
#       prefills chunk c0 only, serves KV store over HTTP (0.0.0.0).
#   white  (100.118.253.68, RTX 4090 CUDA, vllm-v1 env)       = B relay (c1):
#       marks c0 external, stages c0 transiently from laptop over HTTP,
#       prefills c1, saves c1 KV to its own store, serves it over HTTP.
#       (consumer-side of c0 + producer-side of c1 in ONE instance; ready
#       state cascades: B's _READY only after laptop's.)
#   pearl  (100.111.242.55, RX 9060 XT gfx1200 ROCm, vllm-rocm env) = C consumer (c2):
#       full prompt; marks c0+c1 external, stages BOTH prefix chunks from TWO
#       different peers (plural chunk_ids/peer_urls), ring backend cats them
#       into one contiguous peer KV segment for a single non-causal peer pass.
#       Runs an inline single-node reference and checks tokens + memsplit.
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
TOTAL="${TOTAL:-1536}"
SPLIT0="${SPLIT0:-512}"
SPLIT1="${SPLIT1:-512}"
DECODE="${DECODE:-4}"
RUN_ID="ring3-$(date +%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/ring3node-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() { printf "'"; printf "%s" "$1" | sed "s/'/'\\''/g"; printf "'"; }
run_laptop() { ssh -o ConnectTimeout=20 "${LAPTOP_SSH}" "bash -lc $(shell_quote "$1")"; }
run_white()  { ssh -o ConnectTimeout=20 "${WHITE_SSH}"  "bash -lc $(shell_quote "$1")"; }
run_pearl()  { ssh -o ConnectTimeout=20 "${PEARL_SSH}"  "bash -lc $(shell_quote "$1")"; }

echo "=== HCP 3-node true ring: laptop A(CUDA) + white B relay(CUDA) + pearl C(ROCm) ==="
echo "RUN_ID=${RUN_ID}  total=${TOTAL} chunks=${SPLIT0}+${SPLIT1}+$((TOTAL-SPLIT0-SPLIT1)) decode=${DECODE} port=${SERVE_PORT}"
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
run_laptop "/home/stark/miniconda3/envs/vllm-v1/bin/python -c \"import hcp_vllm_plugin.ring_connector\"" \
  || { echo "FATAL: plugin not importable on laptop" >&2; exit 2; }
run_white "/home/stark/miniconda3/envs/vllm-v1/bin/python -c \"import hcp_vllm_plugin.ring_connector\"" \
  || { echo "FATAL: plugin not importable on white" >&2; exit 2; }
run_laptop "pgrep -f '[v]alidate_ring_relay.py' && exit 1 || true"
run_white  "pgrep -f '[v]alidate_ring_relay.py' && exit 1 || true"
run_pearl  "pgrep -f '[v]alidate_ring_relay.py' && exit 1 || true"
run_laptop "curl -sf -o /dev/null http://127.0.0.1:${SERVE_PORT}/ && exit 1 || true"
run_white  "curl -sf -o /dev/null http://127.0.0.1:${SERVE_PORT}/ && exit 1 || true"
echo "pre-checks OK (plugin HEAD=${LAP_HEAD})"

STORE_A="/tmp/hcp_3node_store_a_${RUN_ID}"
STORE_B="/tmp/hcp_3node_store_b_${RUN_ID}"
STORE_C="/tmp/hcp_3node_store_c_${RUN_ID}"
DONE_FILE="/tmp/hcp_3node_done_${RUN_ID}"
SCRIPT="validate_ring_relay.py"

cleanup() {
  run_laptop "touch ${DONE_FILE}; pkill -f validate_ring_relay.py || true" >/dev/null 2>&1 || true
  run_white  "touch ${DONE_FILE}; pkill -f validate_ring_relay.py || true" >/dev/null 2>&1 || true
  run_pearl  "pkill -f validate_ring_relay.py || true" >/dev/null 2>&1 || true
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

# === A: producer on laptop (c0) ===
echo "=== A: producer on laptop (c0=${SPLIT0}) ==="
a_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1 && \
  python ${LAPTOP_PLUGIN_REPO}/${SCRIPT} --mode producer \
    --total ${TOTAL} --split0 ${SPLIT0} --split1 ${SPLIT1} --run-id ${RUN_ID} \
    --port-a ${SERVE_PORT} --store-a ${STORE_A} --store-b /tmp/u1_${RUN_ID} --store-c /tmp/u2_${RUN_ID} \
    --done-file ${DONE_FILE} --gpu-mem-ab 0.4 --hold-secs 900"
run_laptop "${a_cmd}" >"${REPORT_DIR}/producer_a.log" 2>&1 &
A_PID=$!
echo "A ssh pid=${A_PID}"
wait_ready_http "${LAPTOP_HOST}" "c0" "${A_PID}" "A(laptop)" || { tail -20 "${REPORT_DIR}/producer_a.log" >&2 || true; exit 1; }

# === B: relay on white (c1, consumes c0 from laptop) ===
echo "=== B: relay on white (c1=${SPLIT1}, prefix c0 from laptop) ==="
b_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1 && \
  python ${WHITE_PLUGIN_REPO}/${SCRIPT} --mode relay \
    --total ${TOTAL} --split0 ${SPLIT0} --split1 ${SPLIT1} --run-id ${RUN_ID} \
    --url-a http://${LAPTOP_HOST}:${SERVE_PORT} --port-b ${SERVE_PORT} \
    --store-a /tmp/u1_${RUN_ID} --store-b ${STORE_B} --store-c /tmp/u2_${RUN_ID} \
    --done-file ${DONE_FILE} --gpu-mem-ab 0.18 --hold-secs 900"
run_white "${b_cmd}" >"${REPORT_DIR}/relay_b.log" 2>&1 &
B_PID=$!
echo "B ssh pid=${B_PID}"
wait_ready_http "${WHITE_HOST}" "c1" "${B_PID}" "B(white)" || { tail -20 "${REPORT_DIR}/relay_b.log" >&2 || true; exit 1; }

# === C: consumer on pearl (c2, stages c0 from laptop + c1 from white) ===
echo "=== C: consumer on pearl (c2=$((TOTAL-SPLIT0-SPLIT1)), peers laptop+white) ==="
SP=/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages
c_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-rocm && \
  export LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_devel/lib/host-math/lib:${SP}/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-} && \
  python ${PEARL_PLUGIN_REPO}/${SCRIPT} --mode consumer \
    --total ${TOTAL} --split0 ${SPLIT0} --split1 ${SPLIT1} --run-id ${RUN_ID} \
    --url-a http://${LAPTOP_HOST}:${SERVE_PORT} --url-b http://${WHITE_HOST}:${SERVE_PORT} \
    --store-a /tmp/u1_${RUN_ID} --store-b /tmp/u2_${RUN_ID} --store-c ${STORE_C} \
    --done-file ${DONE_FILE} --decode ${DECODE} --gpu-mem-c 0.35"
set +e
run_pearl "${c_cmd}" >"${REPORT_DIR}/consumer_c.log" 2>&1
CONS_RC=$?
set -e

echo "=== consumer log (tail) ==="
tail -30 "${REPORT_DIR}/consumer_c.log" || true
echo "=== relay log (tail) ==="
tail -6 "${REPORT_DIR}/relay_b.log" || true
echo "=== producer log (tail) ==="
tail -6 "${REPORT_DIR}/producer_a.log" || true

echo "RUN_ID=${RUN_ID}  report=${REPORT_DIR}"
if [ "${CONS_RC}" = 0 ]; then
  echo "=== VERDICT: PASS — 3-node heterogeneous true ring validated ==="
else
  echo "=== VERDICT: FAIL (consumer exit=${CONS_RC}) ==="
fi
exit "${CONS_RC}"
