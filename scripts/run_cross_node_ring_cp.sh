#!/bin/bash
# Cross-node HCP ring-KV split-CP validation (memory-splitting, NOT full-KV copy).
#
# Topology:
#   white (100.118.253.68, RTX 4090 CUDA, vllm-v1 env)  = producer:
#       vLLM CUSTOM backend (HcpRingAttentionBackend) + HcpRingKvConnector
#       prefills chunk A only, saves per-layer K/V, serves store over HTTP
#       (bound to 0.0.0.0).  Its paged pool holds ONLY chunk-A KV.
#   pearl (100.111.242.55, RX 9060 XT gfx1200 ROCm, vllm-rocm env) = consumer:
#       vLLM CUSTOM backend + HcpRingKvConnector gets the FULL prompt; the
#       scheduler marks chunk A external (global RoPE positions, no recompute),
#       the worker fetches chunk-A KV over HTTP into the ring backend's
#       TRANSIENT PEER_KV_STAGING (never into pearl's paged pool), and the
#       online-softmax merge reproduces full attention.  Consumer also runs an
#       inline single-node reference and checks:
#         1. greedy tokens match the reference,
#         2. WRITE_TRACK: zero chunk-A pool slots written on pearl
#            (memory-splitting evidence: chunk-A KV never enters pearl's pool).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_SSH="${WHITE_USER:-stark}@${WHITE_HOST}"
WHITE_REPO="${WHITE_REPO:-/home/stark/hetero-cp-ringattn}"
WHITE_PLUGIN_REPO="${WHITE_PLUGIN_REPO:-/home/stark/hcp-vllm-plugin}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_SSH="${PEARL_USER:-stark}@${PEARL_HOST}"
PEARL_REPO="${PEARL_REPO:-/home/stark/hetero-cp-ringattn}"
PEARL_PLUGIN_REPO="${PEARL_PLUGIN_REPO:-/home/stark/hcp-vllm-plugin}"

SERVE_PORT="${SERVE_PORT:-8901}"
TOTAL="${TOTAL:-2048}"
SPLIT="${SPLIT:-1024}"
DECODE="${DECODE:-4}"
RUN_ID="ringx-$(date +%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/ring-cross-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() { printf "'"; printf "%s" "$1" | sed "s/'/'\\''/g"; printf "'"; }
run_white() { ssh -o ConnectTimeout=20 "${WHITE_SSH}" "bash -lc $(shell_quote "$1")"; }
run_pearl() { ssh -o ConnectTimeout=20 "${PEARL_SSH}" "bash -lc $(shell_quote "$1")"; }

echo "=== HCP ring split-CP cross-node validation (white CUDA producer + pearl ROCm consumer) ==="
echo "RUN_ID=${RUN_ID}  total=${TOTAL} split=${SPLIT} decode=${DECODE} port=${SERVE_PORT}"
# Tee driver stdout/stderr into the report dir as audit evidence.
exec > >(tee -a "${REPORT_DIR}/driver.log") 2>&1

# === Pre-checks (per AGENTS.md hardware discipline) ===
echo "=== pre-checks ==="
LOCAL_HEAD="$(git rev-parse HEAD)"
WHITE_HEAD="$(run_white "git -C ${WHITE_REPO} rev-parse HEAD")"
PEARL_HEAD="$(run_pearl "git -C ${PEARL_REPO} rev-parse HEAD")"
echo "HEAD local=${LOCAL_HEAD} white=${WHITE_HEAD} pearl=${PEARL_HEAD}"
if [ "${WHITE_HEAD}" != "${LOCAL_HEAD}" ] || [ "${PEARL_HEAD}" != "${LOCAL_HEAD}" ]; then
  echo "FATAL: repo HEAD mismatch; run git pull on both nodes first" >&2
  exit 2
fi
run_white "/home/stark/miniconda3/envs/vllm-v1/bin/python -c \"import hcp_vllm_plugin.ring_connector, hcp_vllm_plugin.ring_backend\"" \
  || { echo "FATAL: hcp_vllm_plugin not importable on white" >&2; exit 2; }
run_white "curl -sf -o /dev/null http://127.0.0.1:${SERVE_PORT}/ && exit 1 || true" # port must be free
run_white "pgrep -f '[v]alidate_ring_connector.py' && exit 1 || true"
run_pearl "pgrep -f '[v]alidate_ring_connector.py' && exit 1 || true"
echo "pre-checks OK"

WHITE_STORE="/tmp/hcp_ring_store_producer_${RUN_ID}"
PEARL_STORE="/tmp/hcp_ring_store_consumer_${RUN_ID}"
DONE_FILE="/tmp/hcp_ring_conn_done_${RUN_ID}"
SCRIPT="validate_ring_connector.py"

cleanup() {
  run_white "touch ${DONE_FILE}; pkill -f validate_ring_connector.py || true" >/dev/null 2>&1 || true
  wait "${PROD_PID:-0}" 2>/dev/null || true
}
trap cleanup EXIT

# === Launch producer on white ===
echo "=== launching producer on white (CUDA) ==="
prod_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1 && \
  python ${WHITE_PLUGIN_REPO}/${SCRIPT} --mode producer \
    --total ${TOTAL} --split ${SPLIT} --run-id ${RUN_ID} --chunk-id chunk0 \
    --port ${SERVE_PORT} --producer-store ${WHITE_STORE} \
    --consumer-store /tmp/unused_${RUN_ID} --done-file ${DONE_FILE} \
    --producer-gpu-mem 0.25 --hold-secs 600"
run_white "${prod_cmd}" >"${REPORT_DIR}/producer.log" 2>&1 &
PROD_PID=$!
echo "producer ssh pid=${PROD_PID}"

# === Wait until producer's KV store is reachable over HTTP from THIS mac ===
echo "=== waiting for producer KV store (_READY over HTTP) ==="
ready=0
for i in $(seq 1 120); do
  if curl -sf -o /dev/null "http://${WHITE_HOST}:${SERVE_PORT}/${RUN_ID}/chunk0/_READY"; then
    echo "producer KV ready after $((i * 5))s"
    ready=1
    break
  fi
  if ! kill -0 "${PROD_PID}" 2>/dev/null; then
    echo "FATAL: producer ssh exited early; see ${REPORT_DIR}/producer.log" >&2
    tail -20 "${REPORT_DIR}/producer.log" >&2 || true
    exit 1
  fi
  sleep 5
done
[ "${ready}" = 1 ] || { echo "FATAL: producer never became ready" >&2; exit 1; }

# === Launch consumer on pearl ===
echo "=== launching consumer on pearl (ROCm) ==="
SP=/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages
cons_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-rocm && \
  export LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_devel/lib/host-math/lib:${SP}/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-} && \
  python ${PEARL_PLUGIN_REPO}/${SCRIPT} --mode consumer \
    --total ${TOTAL} --split ${SPLIT} --run-id ${RUN_ID} --chunk-id chunk0 \
    --port ${SERVE_PORT} --producer-store /tmp/unused_${RUN_ID} \
    --consumer-store ${PEARL_STORE} --done-file ${DONE_FILE} \
    --peer-url http://${WHITE_HOST}:${SERVE_PORT} \
    --decode ${DECODE} --gpu-mem 0.3"
set +e
run_pearl "${cons_cmd}" >"${REPORT_DIR}/consumer.log" 2>&1
CONS_RC=$?
set -e

echo "=== consumer log (tail) ==="
tail -30 "${REPORT_DIR}/consumer.log" || true
echo "=== producer log (tail) ==="
tail -8 "${REPORT_DIR}/producer.log" || true

echo "RUN_ID=${RUN_ID}  report=${REPORT_DIR}"
if [ "${CONS_RC}" = 0 ]; then
  echo "=== VERDICT: PASS — cross-node heterogeneous ring split-CP validated ==="
else
  echo "=== VERDICT: FAIL (consumer exit=${CONS_RC}) ==="
fi
exit "${CONS_RC}"
