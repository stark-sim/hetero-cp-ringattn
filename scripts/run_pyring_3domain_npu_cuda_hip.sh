#!/usr/bin/env bash
set -euo pipefail

# 3-domain heterogeneous correctness smoke, ALL workers on the Python
# transformers backend (route C step B):
#   domain 0: AtomGit container Ascend910 npu:0   (torch_npu, transformers 4.45.2)
#   domain 1: white RTX 4090 cuda:0               (torch 2.13.0+cu130, transformers 5.15.0)
#   domain 2: pearl RX 9060 XT cuda:0 (ROCm gfx1200, torch 2.13.0a0+rocm7.13, transformers 5.12.1)
#   coordinator: Mac Rust binary (tailnet)
#
# Ring edges: container -> white (tailscale DERP), white -> pearl (2.5GbE LAN),
#             pearl -> container (tailscale DERP).
#
# Gates:
#   1) per-backend 1-domain golden on each of npu/cuda/hip (also measures
#      cross-backend numeric determinism of argmax decode);
#   2) 3-domain ring output must match the golden of the LAST domain's backend
#      (the domain that recomputes final logits after KV exchange).

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

SSH_CFG="${HOME}/.atomgitdevenv/.ssh/config"
NPU_HOST="devenvc_wsa4w.9d5a4a35b7ae4f94b09636ec713325cc.atomgit.0"
NPU_PY="/opt/buildtools/Python-3.11.4/bin/python3"
NPU_HCP_DIR="/home/developer/hcp"
NPU_MODEL_DIR="${NPU_MODEL_DIR:-/home/developer/models/Qwen2-0.5B}"
NPU_TS="100.91.253.114"

WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
WHITE_TS="100.118.253.68"
WHITE_PY="/home/stark/venv-bench/bin/python"
WHITE_HCP_DIR="/home/stark/hcp-python"
WHITE_MODEL="/home/stark/hetero-cp-ringattn/models/Qwen2-0.5B"

PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_LAN="192.168.100.2"
PEARL_PY="/home/stark/miniconda3/envs/vllm-rocm/bin/python"
PEARL_HCP_DIR="/home/stark/hcp-python"
PEARL_MODEL="/home/stark/hetero-cp-ringattn/models/Qwen2-0.5B"

MAC_MODEL_DIR="${MAC_MODEL_DIR:-${REPO_ROOT}/models/Qwen2-0.5B}"
MAC_TS="${MAC_TS:-100.121.35.138}"

SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"

COORD_PORT=29910
W0_PORT=29911
W1_PORT=29912
W2_PORT=29913

RUN_ID="pyring3-npu-cuda-hip-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

COORD_PID=""
cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_PID}" ] && kill "${COORD_PID}" 2>/dev/null || true
    npu_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
    white_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
    pearl_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
}
trap cleanup EXIT INT TERM

npu_ssh()   { ssh -F "${SSH_CFG}" -o BatchMode=yes -o ConnectTimeout=20 "${NPU_HOST}" "$1"; }
white_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${WHITE_SSH}" "$1"; }
pearl_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${PEARL_SSH}" "$1"; }

# --- Remote worker launchers (log goes to remote file, fetched at phase end) ---

npu_start_worker() { # domain device listen_port next_host next_port num_domains log
    npu_ssh "
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:\${LD_LIBRARY_PATH:-}
cd ${NPU_HCP_DIR}
nohup ${NPU_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${NPU_MODEL_DIR} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id $1 --num-domains $6 \
    --peer-listen-host 0.0.0.0 --peer-listen-port $3 \
    --next-peer-host $4 --next-peer-port $5 \
    --device $2 > ${NPU_HCP_DIR}/logs/$7 2>&1 < /dev/null &
echo started npu worker domain=$1 pid=\$!
"
}

white_start_worker() { # domain device listen_port next_host next_port num_domains log
    white_ssh "
export PYTHONUNBUFFERED=1
cd ${WHITE_HCP_DIR}
nohup ${WHITE_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${WHITE_MODEL} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id $1 --num-domains $6 \
    --peer-listen-host 0.0.0.0 --peer-listen-port $3 \
    --next-peer-host $4 --next-peer-port $5 \
    --device $2 > ${WHITE_HCP_DIR}/logs/$7 2>&1 < /dev/null &
echo started white worker domain=$1 pid=\$!
"
}

pearl_start_worker() { # domain device listen_port next_host next_port num_domains log
    pearl_ssh "
export PYTHONUNBUFFERED=1
cd ${PEARL_HCP_DIR}
nohup ${PEARL_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${PEARL_MODEL} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id $1 --num-domains $6 \
    --peer-listen-host 0.0.0.0 --peer-listen-port $3 \
    --next-peer-host $4 --next-peer-port $5 \
    --device $2 > ${PEARL_HCP_DIR}/logs/$7 2>&1 < /dev/null &
echo started pearl worker domain=$1 pid=\$!
"
}

wait_remote_handshake() { # ssh_fn log timeout_s
    local fn="$1" log="$2" timeout_s="$3" elapsed=0
    while [ "${elapsed}" -lt "${timeout_s}" ]; do
        if ${fn} "grep -q 'handshake sent' ${log} 2>/dev/null"; then return 0; fi
        if ${fn} "grep -qE 'Traceback' ${log} 2>/dev/null"; then
            echo "ERROR: remote worker failed (${log}):" >&2
            ${fn} "tail -30 ${log}" >&2 || true
            return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "ERROR: timeout waiting for handshake (${log})" >&2
    return 1
}

# --- Preflight ---
echo "=== Preflight ==="
[ -x "${BINARY}" ] || { echo "ERROR: ${BINARY} missing" >&2; exit 1; }
npu_ssh "mkdir -p ${NPU_HCP_DIR}/logs && test -f ${NPU_MODEL_DIR}/model.safetensors && echo npu ok"
white_ssh "mkdir -p ${WHITE_HCP_DIR}/logs && test -f ${WHITE_MODEL}/model.safetensors && echo white ok"
pearl_ssh "mkdir -p ${PEARL_HCP_DIR}/logs && test -f ${PEARL_MODEL}/model.safetensors && echo pearl ok"

# --- Generate prompt ---
echo "=== Generating prompt (${SEQ_LEN} tokens) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
(cd "${REPO_ROOT}/rust" && cargo run --bin gen_prompt -- "${MAC_MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_FILE}") 2>&1 | tail -1 | tee "${REPORT_DIR}/gen_prompt.log"

start_coordinator() { # phase num_domains
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MAC_MODEL_DIR}" \
        --prompt-file "${PROMPT_FILE}" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --num-domains "$2" \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        >"${REPORT_DIR}/coordinator_$1.log" 2>&1 &
    COORD_PID=$!
    echo "Coordinator PID: ${COORD_PID} (phase=$1)"
    sleep 2
}

finish_phase() { # phase
    echo "=== Waiting for coordinator (phase=$1) ==="
    set +e
    wait "${COORD_PID}"
    local exit_code=$?
    set -e
    COORD_PID=""
    echo "Coordinator exit code: ${exit_code}"
    npu_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
    white_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
    pearl_ssh 'pkill -f hcp_transformers_quic_worker || true' || true
    return "${exit_code}"
}

# === Per-backend 1-domain goldens ===
run_golden() { # name  start_fn  device  logpath_fn
    local name="$1" device="$2"
    echo ""
    echo "=== Phase golden-${name}: 1-domain on ${device} ==="
    start_coordinator "golden-${name}" 1
    case "${name}" in
        npu)   npu_start_worker 0 "${device}" ${W0_PORT} 127.0.0.1 ${W1_PORT} 1 "w_golden-npu.log"
               wait_remote_handshake npu_ssh "${NPU_HCP_DIR}/logs/w_golden-npu.log" 600 ;;
        cuda)  white_start_worker 0 "${device}" ${W1_PORT} 127.0.0.1 ${W2_PORT} 1 "w_golden-cuda.log"
               wait_remote_handshake white_ssh "${WHITE_HCP_DIR}/logs/w_golden-cuda.log" 600 ;;
        hip)   pearl_start_worker 0 "${device}" ${W2_PORT} 127.0.0.1 ${W0_PORT} 1 "w_golden-hip.log"
               wait_remote_handshake pearl_ssh "${PEARL_HCP_DIR}/logs/w_golden-hip.log" 600 ;;
    esac
    local rc=0
    finish_phase "golden-${name}" || rc=$?
    return "${rc}"
}

GOLDEN_FAILURES=""
run_golden npu  npu:0  || GOLDEN_FAILURES="${GOLDEN_FAILURES} npu"
run_golden cuda cuda:0 || GOLDEN_FAILURES="${GOLDEN_FAILURES} cuda"
run_golden hip  cuda:0 || GOLDEN_FAILURES="${GOLDEN_FAILURES} hip"

# === 3-domain ring ===
echo ""
echo "=== Phase ring: 3-domain (npu:0 + white cuda + pearl hip) ==="
start_coordinator ring 3
npu_start_worker   0 npu:0  ${W0_PORT} "${WHITE_TS}"  ${W1_PORT} 3 "w0_ring.log"
white_start_worker 1 cuda:0 ${W1_PORT} "${PEARL_LAN}" ${W2_PORT} 3 "w1_ring.log"
pearl_start_worker 2 cuda:0 ${W2_PORT} "${NPU_TS}"    ${W0_PORT} 3 "w2_ring.log"
wait_remote_handshake npu_ssh   "${NPU_HCP_DIR}/logs/w0_ring.log" 600
wait_remote_handshake white_ssh "${WHITE_HCP_DIR}/logs/w1_ring.log" 600
wait_remote_handshake pearl_ssh "${PEARL_HCP_DIR}/logs/w2_ring.log" 600
RING_EXIT=0
finish_phase ring || RING_EXIT=$?

# === Fetch worker logs ===
npu_ssh   "cat ${NPU_HCP_DIR}/logs/w_golden-npu.log"  > "${REPORT_DIR}/worker-golden-npu.log"  2>/dev/null || true
white_ssh "cat ${WHITE_HCP_DIR}/logs/w_golden-cuda.log" > "${REPORT_DIR}/worker-golden-cuda.log" 2>/dev/null || true
pearl_ssh "cat ${PEARL_HCP_DIR}/logs/w_golden-hip.log"  > "${REPORT_DIR}/worker-golden-hip.log"  2>/dev/null || true
npu_ssh   "cat ${NPU_HCP_DIR}/logs/w0_ring.log"   > "${REPORT_DIR}/worker0-npu_ring.log"   2>/dev/null || true
white_ssh "cat ${WHITE_HCP_DIR}/logs/w1_ring.log" > "${REPORT_DIR}/worker1-cuda_ring.log" 2>/dev/null || true
pearl_ssh "cat ${PEARL_HCP_DIR}/logs/w2_ring.log" > "${REPORT_DIR}/worker2-hip_ring.log" 2>/dev/null || true

echo ""
echo "=== Generated outputs ==="
for p in golden-npu golden-cuda golden-hip ring; do
    printf '%-14s ' "${p}:"
    grep "generated:" "${REPORT_DIR}/coordinator_${p}.log" || echo "(none)"
done

echo ""
echo "=== Capacities (ring) ==="
grep -h "capacity" "${REPORT_DIR}"/worker*_ring.log "${REPORT_DIR}/coordinator_ring.log" 2>/dev/null | sort -u || true

echo ""
echo "=== Verdict ==="
FAIL=0
if [ -n "${GOLDEN_FAILURES}" ]; then
    echo "GOLDEN FAILURES:${GOLDEN_FAILURES}"
    FAIL=1
fi
G_NPU="$(grep 'generated:' "${REPORT_DIR}/coordinator_golden-npu.log" | head -1 || true)"
G_CUDA="$(grep 'generated:' "${REPORT_DIR}/coordinator_golden-cuda.log" | head -1 || true)"
G_HIP="$(grep 'generated:' "${REPORT_DIR}/coordinator_golden-hip.log" | head -1 || true)"
G_RING="$(grep 'generated:' "${REPORT_DIR}/coordinator_ring.log" | head -1 || true)"
[ -z "${G_RING}" ] && { echo "INCOMPLETE: no ring output (exit=${RING_EXIT})"; exit 2; }
# Ring final logits come from the last domain (pearl/HIP). Primary gate:
if [ "${G_RING}" = "${G_HIP}" ]; then
    echo "PASS: 3-domain ring matches HIP (last-domain backend) golden exactly"
else
    echo "FAIL: ring output differs from HIP golden"
    FAIL=1
fi
# Cross-backend determinism is informational:
if [ "${G_NPU}" = "${G_CUDA}" ] && [ "${G_CUDA}" = "${G_HIP}" ]; then
    echo "INFO: all three backend goldens identical (cross-backend argmax determinism holds for this prompt)"
else
    echo "INFO: backend goldens differ (npu/cuda/hip) — cross-backend numerics diverge on argmax"
fi
exit ${FAIL}
