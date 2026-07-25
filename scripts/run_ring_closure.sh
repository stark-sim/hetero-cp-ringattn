#!/bin/bash
# 3-node ring-CLOSURE validation: unified ring role + neighbor accumulate-forward
# + rotated request placement, closing the ring at the workload level.
#
# Physical ring: laptop(100.96.154.1, node0) -> white(100.118.253.68, node1)
#                -> pearl(100.111.242.55, node2) -> laptop.
# Every node runs ONE engine with ring_role="ring" and pulls its accumulated
# prefix from its physical predecessor ONLY (laptop<-pearl, white<-laptop,
# pearl<-white); staged chunks are re-served (READY markers after staging).
# Workload: 3 concurrent requests, request j's chunk p on node (j+p)%3, so
# each node is position-0 producer for one request, position-1 relay for
# another, position-2 consumer for a third — producer N's consumer is
# literally (N+1)%N.
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
CHUNK_LEN="${CHUNK_LEN:-512}"
DECODE="${DECODE:-4}"
RUN_ID="ringc-$(date +%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/ring-closure-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() { printf "'"; printf "%s" "$1" | sed "s/'/'\\''/g"; printf "'"; }
run_laptop() { ssh -o ConnectTimeout=20 "${LAPTOP_SSH}" "bash -lc $(shell_quote "$1")"; }
run_white()  { ssh -o ConnectTimeout=20 "${WHITE_SSH}"  "bash -lc $(shell_quote "$1")"; }
run_pearl()  { ssh -o ConnectTimeout=20 "${PEARL_SSH}"  "bash -lc $(shell_quote "$1")"; }

echo "=== HCP ring closure: laptop(0) -> white(1) -> pearl(2) -> laptop ==="
echo "RUN_ID=${RUN_ID}  total=${TOTAL} chunk_len=${CHUNK_LEN} decode=${DECODE} port=${SERVE_PORT}"
exec > >(tee -a "${REPORT_DIR}/driver.log") 2>&1

echo "=== pre-checks ==="
LAP_HEAD="$(run_laptop "git -C ${LAPTOP_PLUGIN_REPO} rev-parse HEAD")"
WHITE_HEAD="$(run_white "git -C ${WHITE_PLUGIN_REPO} rev-parse HEAD")"
PEARL_HEAD="$(run_pearl "git -C ${PEARL_PLUGIN_REPO} rev-parse HEAD")"
echo "plugin HEAD laptop=${LAP_HEAD} white=${WHITE_HEAD} pearl=${PEARL_HEAD}"
if [ "${LAP_HEAD}" != "${WHITE_HEAD}" ] || [ "${LAP_HEAD}" != "${PEARL_HEAD}" ]; then
  echo "FATAL: plugin repo HEAD mismatch across nodes; git pull first" >&2
  exit 2
fi
run_laptop "pgrep -f '[v]alidate_ring_closure.py' && exit 1 || true"
run_white  "pgrep -f '[v]alidate_ring_closure.py' && exit 1 || true"
run_pearl  "pgrep -f '[v]alidate_ring_closure.py' && exit 1 || true"
run_laptop "curl -sf -o /dev/null http://127.0.0.1:${SERVE_PORT}/ && exit 1 || true"
run_white  "curl -sf -o /dev/null http://127.0.0.1:${SERVE_PORT}/ && exit 1 || true"
run_pearl  "curl -sf -o /dev/null http://127.0.0.1:${SERVE_PORT}/ && exit 1 || true"
echo "pre-checks OK (plugin HEAD=${LAP_HEAD})"

PEARL_SP=/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages
pearl_env="source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-rocm && \
  export LD_LIBRARY_PATH=${PEARL_SP}/torch/lib:${PEARL_SP}/_rocm_sdk_core/lib:${PEARL_SP}/_rocm_sdk_core/lib/host-math/lib:${PEARL_SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${PEARL_SP}/_rocm_sdk_devel/lib:${PEARL_SP}/_rocm_sdk_devel/lib/host-math/lib:${PEARL_SP}/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-}"
conda_v1="source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1"

# === Step 0: references on pearl ===
echo "=== step 0: single-node references on pearl ==="
run_pearl "cd /tmp && ${pearl_env} && python ${PEARL_PLUGIN_REPO}/validate_ring_closure.py --mode ref \
  --total ${TOTAL} --nreq 3 --decode ${DECODE} --gpu-mem-ref 0.3 \
  --ref-file /tmp/hcp_closure_refs_${RUN_ID}.json" >"${REPORT_DIR}/refs.log" 2>&1
scp -q "${PEARL_SSH}:/tmp/hcp_closure_refs_${RUN_ID}.json" "${REPORT_DIR}/refs.json"
echo "refs: $(cat "${REPORT_DIR}/refs.json")"

cleanup() {
  run_laptop "touch /tmp/hcp_closure_done_${RUN_ID}; pkill -f validate_ring_closure.py || true" >/dev/null 2>&1 || true
  run_white  "touch /tmp/hcp_closure_done_${RUN_ID}; pkill -f validate_ring_closure.py || true" >/dev/null 2>&1 || true
  run_pearl  "touch /tmp/hcp_closure_done_${RUN_ID}; pkill -f validate_ring_closure.py || true" >/dev/null 2>&1 || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

start_node() { # $1=index $2=host-role(laptop|white|pearl) $3=pred-host
  local k=$1 role=$2 pred=$3
  local env_cmd repo gpu
  case "${role}" in
    laptop) env_cmd="${conda_v1}"; repo="${LAPTOP_PLUGIN_REPO}"; gpu=0.4 ;;
    white)  env_cmd="${conda_v1}"; repo="${WHITE_PLUGIN_REPO}"; gpu=0.18 ;;
    pearl)  env_cmd="${pearl_env}"; repo="${PEARL_PLUGIN_REPO}"; gpu=0.3 ;;
  esac
  local cmd="cd /tmp && ${env_cmd} && python ${repo}/validate_ring_closure.py --mode ringnode \
    --node-index ${k} --nnodes 3 --nreq 3 --total ${TOTAL} --chunk-len ${CHUNK_LEN} \
    --decode ${DECODE} --run-id ${RUN_ID} --port ${SERVE_PORT} \
    --pred-url http://${pred}:${SERVE_PORT} --store /tmp/hcp_closure_store_${k}_${RUN_ID} \
    --tokens-file /tmp/hcp_closure_tokens_${k}_${RUN_ID}.json \
    --done-file /tmp/hcp_closure_done_${RUN_ID} --gpu-mem ${gpu} --hold-secs 1200"
  "run_${role}" "${cmd}" >"${REPORT_DIR}/node${k}.log" 2>&1 &
  echo "node${k}(${role}) ssh pid=$! pred=http://${pred}:${SERVE_PORT}"
}

# === Start the three ring engines concurrently (requests self-cascade) ===
echo "=== starting ring engines ==="
start_node 0 laptop "${PEARL_HOST}"
start_node 1 white  "${LAPTOP_HOST}"
start_node 2 pearl  "${WHITE_HOST}"

# === Wait for RINGNODE_DONE on all three ===
declare -a NODE_SSH=("${LAPTOP_SSH}" "${WHITE_SSH}" "${PEARL_SSH}")
finished=0
for i in $(seq 1 240); do
  finished=0
  for k in 0 1 2; do
    if ssh -o ConnectTimeout=10 "${NODE_SSH[$k]}" "grep -q RINGNODE_DONE /tmp/ring_closure_node${k}.log 2>/dev/null" 2>/dev/null; then
      finished=$((finished + 1))
    fi
  done
  [ "${finished}" = 3 ] && break
  # Note: ringnode logs live in each node's LOCAL /tmp only if we wrote them there;
  # our ssh wrapper writes them to REPORT_DIR locally, so check those instead.
  finished=0
  for k in 0 1 2; do
    grep -q RINGNODE_DONE "${REPORT_DIR}/node${k}.log" 2>/dev/null && finished=$((finished + 1)) || true
  done
  [ "${finished}" = 3 ] && break
  sleep 10
done
echo "nodes finished: ${finished}/3 after wait loop"
touch_done() { :; }
run_laptop "touch /tmp/hcp_closure_done_${RUN_ID}" || true
run_white  "touch /tmp/hcp_closure_done_${RUN_ID}" || true
run_pearl  "touch /tmp/hcp_closure_done_${RUN_ID}" || true
[ "${finished}" = 3 ] || { echo "FATAL: not all nodes finished; see node logs" >&2; exit 1; }

# === Compare closure tokens with references ===
echo "=== results ==="
for k in 0 1 2; do
  scp -q "${NODE_SSH[$k]}:/tmp/hcp_closure_tokens_${k}_${RUN_ID}.json" "${REPORT_DIR}/tokens_node${k}.json"
done
python3 - "${REPORT_DIR}" << 'PYEOF'
import json, sys
d = sys.argv[1]
refs = json.load(open(f"{d}/refs.json"))
ok = True
for k in range(3):
    j = (k - 2) % 3
    got = json.load(open(f"{d}/tokens_node{k}.json")).get(str(j))
    exp = refs[str(j)]
    match = got == exp
    ok &= match
    print(f"node{k} req{j} (position 2): tokens={got} ref={exp} match={match}")
print("TOKENS_ALL_MATCH" if ok else "TOKENS_MISMATCH")
sys.exit(0 if ok else 1)
PYEOF
TOK_RC=$?

echo "=== node stats ==="
# NOTE: the global WRITE_TRACK overlap probe false-positives under block
# recycling (freed blocks reused as later requests' prefix regions).  The
# memory-splitting gate is slots_written accounting: each node must have
# written approximately its OWN chunks' tokens only.
EXPECTED=$((3 * CHUNK_LEN + DECODE + 64))
SLOTS_OK=1
for k in 0 1 2; do
  echo "--- node${k} ---"
  grep -E "\[memsplit\]|\[staging\]|\[batch\]|\[impl\]" "${REPORT_DIR}/node${k}.log" || true
  SW=$(grep "slots_written=" "${REPORT_DIR}/node${k}.log" | tail -1 | sed "s/.*slots_written=\([0-9]*\).*/\1/")
  echo "node${k} slots_written=${SW:-0} (expected <= ${EXPECTED})"
  [ -n "${SW}" ] && [ "${SW}" -le "${EXPECTED}" ] || SLOTS_OK=0
done

echo "RUN_ID=${RUN_ID}  report=${REPORT_DIR}"
if [ "${TOK_RC}" = 0 ] && [ "${SLOTS_OK}" = 1 ]; then
  echo "=== VERDICT: PASS — ring closed at workload level (rotated 3 requests) ==="
  exit 0
else
  echo "=== VERDICT: FAIL (tokens_rc=${TOK_RC} slots_ok=${SLOTS_OK}) ==="
  exit 1
fi
