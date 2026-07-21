#!/bin/bash
# Cross-node MULTI-REQUEST HCP ring split-CP validation (continuous batching
# on the CP path, heterogeneous CUDA<->ROCm).
#
#   white (100.118.253.68, RTX 4090 CUDA, vllm-v1 env)  = producer:
#       prefills chunk A of TWO different prompts (chunk keys c0 / c1),
#       serves per-layer KV over HTTP (0.0.0.0).
#   pearl (100.111.242.55, RX 9060 XT ROCm, vllm-rocm env) = consumer:
#       ONE generate call with BOTH full prompts (max_num_seqs=2); each
#       request carries its own peer chunk via kv_transfer_params.hcp_ring.
#       Checks: tokens match reference, 2 chunks staged concurrently,
#       CP path ran inside a continuous batch, zero chunk-A pool writes,
#       staged KV freed after finish.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_SSH="${WHITE_USER:-stark}@${WHITE_HOST}"
WHITE_REPO="${WHITE_REPO:-/home/stark/hetero-cp-ringattn}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_SSH="${PEARL_USER:-stark}@${PEARL_HOST}"
PEARL_REPO="${PEARL_REPO:-/home/stark/hetero-cp-ringattn}"

SERVE_PORT="${SERVE_PORT:-8902}"
TOTAL="${TOTAL:-1024}"
SPLIT="${SPLIT:-512}"
DECODE="${DECODE:-4}"
RUN_ID="ringconc-$(date +%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/ring-conc-cross-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() { printf "'"; printf "%s" "$1" | sed "s/'/'\\''/g"; printf "'"; }
run_white() { ssh -o ConnectTimeout=20 "${WHITE_SSH}" "bash -lc $(shell_quote "$1")"; }
run_pearl() { ssh -o ConnectTimeout=20 "${PEARL_SSH}" "bash -lc $(shell_quote "$1")"; }

echo "=== HCP ring split-CP cross-node MULTI-REQUEST validation ==="
echo "RUN_ID=${RUN_ID}  total=${TOTAL} split=${SPLIT} decode=${DECODE} port=${SERVE_PORT}"
exec > >(tee -a "${REPORT_DIR}/driver.log") 2>&1

# === Pre-checks ===
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
run_white "pgrep -f '[v]alidate_ring_concurrent.py' && exit 1 || true"
run_pearl "pgrep -f '[v]alidate_ring_concurrent.py' && exit 1 || true"
echo "pre-checks OK"

WHITE_STORE="/tmp/hcp_ring_conc_producer_${RUN_ID}"
PEARL_STORE="/tmp/hcp_ring_conc_consumer_${RUN_ID}"
DONE_FILE="/tmp/hcp_ring_conc_done_${RUN_ID}"
SCRIPT="hcp_vllm_plugin/validate_ring_concurrent.py"

cleanup() {
  run_white "touch ${DONE_FILE}; pkill -f '[v]alidate_ring_concurrent.py' || true" >/dev/null 2>&1 || true
  wait "${PROD_PID:-0}" 2>/dev/null || true
}
trap cleanup EXIT

# === Launch producer on white ===
echo "=== launching producer on white (CUDA, 2 chunks c0/c1) ==="
prod_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1 && \
  python ${WHITE_REPO}/${SCRIPT} --mode producer \
    --total ${TOTAL} --split ${SPLIT} --run-id ${RUN_ID} \
    --port ${SERVE_PORT} --producer-store ${WHITE_STORE} \
    --consumer-store /tmp/unused_${RUN_ID} --done-file ${DONE_FILE} \
    --producer-gpu-mem 0.25 --hold-secs 600"
run_white "${prod_cmd}" >"${REPORT_DIR}/producer.log" 2>&1 &
PROD_PID=$!
echo "producer ssh pid=${PROD_PID}"

# === Wait for BOTH chunk _READY markers ===
echo "=== waiting for producer KV store (c0 + c1 _READY over HTTP) ==="
ready=0
for i in $(seq 1 120); do
  if curl -sf -o /dev/null "http://${WHITE_HOST}:${SERVE_PORT}/${RUN_ID}/c0/_READY" \
     && curl -sf -o /dev/null "http://${WHITE_HOST}:${SERVE_PORT}/${RUN_ID}/c1/_READY"; then
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
echo "=== launching consumer on pearl (ROCm, 2 concurrent requests) ==="
SP=/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages
cons_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-rocm && \
  export LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_devel/lib/host-math/lib:${SP}/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-} && \
  python ${PEARL_REPO}/${SCRIPT} --mode consumer \
    --total ${TOTAL} --split ${SPLIT} --run-id ${RUN_ID} \
    --port ${SERVE_PORT} --producer-store /tmp/unused_${RUN_ID} \
    --consumer-store ${PEARL_STORE} --done-file ${DONE_FILE} \
    --peer-url http://${WHITE_HOST}:${SERVE_PORT} \
    --decode ${DECODE} --gpu-mem 0.3"
set +e
run_pearl "${cons_cmd}" >"${REPORT_DIR}/consumer.log" 2>&1
CONS_RC=$?
set -e

echo "=== consumer log (key lines) ==="
grep -E '\[memsplit\]|\[batch\]|\[lifecycle\]|tokens match|multi-req|verdict|\[ref\]|\[consumer\]' "${REPORT_DIR}/consumer.log" || tail -20 "${REPORT_DIR}/consumer.log"
echo "=== producer log (tail) ==="
tail -6 "${REPORT_DIR}/producer.log" || true

echo "RUN_ID=${RUN_ID}  report=${REPORT_DIR}"
if [ "${CONS_RC}" = 0 ]; then
  echo "=== VERDICT: PASS — cross-node heterogeneous multi-request ring split-CP validated ==="
else
  echo "=== VERDICT: FAIL (consumer exit=${CONS_RC}) ==="
fi
exit "${CONS_RC}"
