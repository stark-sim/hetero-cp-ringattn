#!/usr/bin/env bash
set -euo pipefail

# NPU 2-domain smoke: Mac Rust coordinator + AtomGit container dual-Ascend910 workers.
#   Phase 1 (golden): 1 domain  — worker0 on npu:0
#   Phase 2 (ring):   2 domains — worker0 on npu:0 + worker1 on npu:1, KV ring over
#                     container loopback; control plane over tailscale (DERP relay).
# Correctness gate: generated text of phase 2 must exactly match phase 1
# (distributed path samples deterministically via argmax).

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

SSH_CFG="${HOME}/.atomgitdevenv/.ssh/config"
NPU_HOST="devenvc_wsa4w.9d5a4a35b7ae4f94b09636ec713325cc.atomgit.0"
NPU_PY="/opt/buildtools/Python-3.11.4/bin/python3"
NPU_HCP_DIR="/home/developer/hcp"
NPU_MODEL_DIR="${NPU_MODEL_DIR:-/home/developer/models/Qwen2-0.5B}"
MAC_MODEL_DIR="${MAC_MODEL_DIR:-${REPO_ROOT}/models/Qwen2-0.5B}"
MAC_TAILNET_IP="${MAC_TAILNET_IP:-100.121.35.138}"

SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"

COORD_PORT=29700
W0_PORT=29701
W1_PORT=29702

RUN_ID="npu-2domain-atomgit-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

COORD_PID=""
cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_PID}" ] && kill "${COORD_PID}" 2>/dev/null || true
    npu_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
}
trap cleanup EXIT INT TERM

npu_ssh() {
    ssh -F "${SSH_CFG}" -o BatchMode=yes -o ConnectTimeout=20 "${NPU_HOST}" "$1"
}

# Remote worker launcher. Args: domain_id device peer_port next_peer_port num_domains log_name
npu_start_worker() {
    local domain_id="$1" device="$2" peer_port="$3" next_port="$4" num_domains="$5" log_name="$6"
    npu_ssh "
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:\${LD_LIBRARY_PATH:-}
cd ${NPU_HCP_DIR}
nohup ${NPU_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${NPU_MODEL_DIR} \
    --coordinator-host ${MAC_TAILNET_IP} \
    --coordinator-port ${COORD_PORT} \
    --domain-id ${domain_id} \
    --num-domains ${num_domains} \
    --peer-listen-host 127.0.0.1 \
    --peer-listen-port ${peer_port} \
    --next-peer-host 127.0.0.1 \
    --next-peer-port ${next_port} \
    --device ${device} \
    > ${NPU_HCP_DIR}/logs/${log_name} 2>&1 < /dev/null &
echo started worker domain=${domain_id} device=${device} pid=\$!
"
}

# Wait until remote worker log shows handshake (model loaded + connected to coordinator)
npu_wait_handshake() {
    local log_name="$1" timeout_s="$2" elapsed=0
    while [ "${elapsed}" -lt "${timeout_s}" ]; do
        if npu_ssh "grep -q 'handshake sent' ${NPU_HCP_DIR}/logs/${log_name} 2>/dev/null"; then
            return 0
        fi
        if npu_ssh "grep -qE 'Traceback|Error|error' ${NPU_HCP_DIR}/logs/${log_name} 2>/dev/null"; then
            echo "ERROR: remote worker ${log_name} failed:" >&2
            npu_ssh "tail -30 ${NPU_HCP_DIR}/logs/${log_name}" >&2 || true
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: timeout waiting for handshake in ${log_name}" >&2
    return 1
}

# --- Preflight ---
echo "=== Preflight ==="
[ -x "${BINARY}" ] || { echo "ERROR: ${BINARY} missing" >&2; exit 1; }
npu_ssh "mkdir -p ${NPU_HCP_DIR}/logs && test -f ${NPU_MODEL_DIR}/model.safetensors && echo npu model ok"

# --- Generate prompt ---
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
(cd "${REPO_ROOT}/rust" && cargo run --bin gen_prompt -- "${MAC_MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}") 2>&1 | tee "${REPORT_DIR}/gen_prompt.log"

run_phase() {
    local phase="$1" num_domains="$2"
    echo ""
    echo "=== Phase ${phase}: num-domains=${num_domains} ==="

    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MAC_MODEL_DIR}" \
        --prompt-file "${PROMPT_FILE}" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --num-domains "${num_domains}" \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        >"${REPORT_DIR}/coordinator_${phase}.log" 2>&1 &
    COORD_PID=$!
    echo "Coordinator PID: ${COORD_PID}"
    sleep 2

    npu_start_worker 0 npu:0 ${W0_PORT} ${W1_PORT} "${num_domains}" "worker0_${phase}.log"
    if [ "${num_domains}" = "2" ]; then
        npu_start_worker 1 npu:1 ${W1_PORT} ${W0_PORT} "${num_domains}" "worker1_${phase}.log"
    fi

    echo "Waiting for worker handshake (model load on NPU takes a while)..."
    npu_wait_handshake "worker0_${phase}.log" 600
    if [ "${num_domains}" = "2" ]; then
        npu_wait_handshake "worker1_${phase}.log" 600
    fi

    echo "=== Waiting for inference (coordinator) ==="
    set +e
    wait "${COORD_PID}"
    local exit_code=$?
    set -e
    COORD_PID=""
    echo "Coordinator exit code: ${exit_code}"

    npu_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
    npu_ssh "cat ${NPU_HCP_DIR}/logs/worker0_${phase}.log" > "${REPORT_DIR}/worker0_${phase}.log" || true
    if [ "${num_domains}" = "2" ]; then
        npu_ssh "cat ${NPU_HCP_DIR}/logs/worker1_${phase}.log" > "${REPORT_DIR}/worker1_${phase}.log" || true
    fi
    return "${exit_code}"
}

GOLDEN_EXIT=0
run_phase golden 1 || GOLDEN_EXIT=$?
RING_EXIT=0
run_phase ring 2 || RING_EXIT=$?

echo ""
echo "=== Golden (1-domain) generated ==="
grep "generated:" "${REPORT_DIR}/coordinator_golden.log" || tail -20 "${REPORT_DIR}/coordinator_golden.log"
echo "=== Ring (2-domain) generated ==="
grep "generated:" "${REPORT_DIR}/coordinator_ring.log" || tail -20 "${REPORT_DIR}/coordinator_ring.log"

echo ""
echo "=== Capacities ==="
grep -h "capacity" "${REPORT_DIR}"/worker*_*.log 2>/dev/null || true

echo ""
GOLDEN_TEXT="$(grep 'generated:' "${REPORT_DIR}/coordinator_golden.log" | head -1 || true)"
RING_TEXT="$(grep 'generated:' "${REPORT_DIR}/coordinator_ring.log" | head -1 || true)"
echo "=== Verdict ==="
if [ -z "${GOLDEN_TEXT}" ] || [ -z "${RING_TEXT}" ]; then
    echo "INCOMPLETE: missing generated output (golden_exit=${GOLDEN_EXIT} ring_exit=${RING_EXIT})"
    exit 2
elif [ "${GOLDEN_TEXT}" = "${RING_TEXT}" ]; then
    echo "PASS: 2-domain NPU ring output matches 1-domain NPU golden exactly"
    exit 0
else
    echo "FAIL: outputs differ"
    exit 1
fi
