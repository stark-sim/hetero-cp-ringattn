#!/usr/bin/env bash
set -euo pipefail

# 4-domain heterogeneous correctness smoke with a CROSS-MACHINE NCCL TP
# abstract worker (D1b 线):
#   domain 0: 抽象 worker = laptop RTX 4060 (rank0, 协议面) + white RTX 4090
#             (rank1, 纯计算 follower)，NCCL over tailscale0
#   domain 1: AtomGit container Ascend910 npu:0 (单芯普通 worker)
#   domain 2: pearl RX 9060 XT (ROCm)
#   domain 3: Mac MPS
#   coordinator: Mac Rust binary (tailnet)
#
# Ring edges: d0 laptop-rank0 -> d1 container (DERP 100.91.253.114),
#             d1 -> d2 pearl (DERP 100.111.242.55),
#             d2 -> d3 mac (tailnet 100.121.35.138),
#             d3 -> d0 laptop (tailnet 100.96.154.1；LAN 192.168.8.109 实测
#             从 Mac 侧 no route to host，WiFi 客户端隔离或网段漂移，退回 tailnet)。
# TP bootstrap: NCCL master = laptop tailnet 100.96.154.1:29601。
#
# 前置条件（当前阻塞项）：两端 NCCL 小版本必须兼容。laptop=2.28.9 /
# white=2.29.7 的 bootstrap wire 不兼容（"Message truncated: received 176
# bytes instead of 172"，scripts/nccl_smoke_2node.py 可复现），本脚本在版本
# 对齐前会在 TP init 阶段超时。
#
# Gates: 1) per-backend 1-domain golden (npu/cuda/hip/mps);
#        2) 4-domain ring output must match the golden of the LAST domain (MPS);
#        3) ring vs 四 golden 的 logits max|Δ| 报告。

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

SSH_CFG="${HOME}/.atomgitdevenv/.ssh/config"
NPU_HOST="devenvc_wsa4w.9d5a4a35b7ae4f94b09636ec713325cc.atomgit.0"
NPU_PY="/opt/buildtools/Python-3.11.4/bin/python3"
NPU_HCP_DIR="/home/developer/hcp"
NPU_MODEL_DIR="${NPU_MODEL_DIR:-/home/developer/models/Qwen2-0.5B}"
NPU_TS="100.91.253.114"

LAPTOP_SSH="${LAPTOP_SSH:-stark@100.96.154.1}"
LAPTOP_TS="100.96.154.1"
LAPTOP_LAN="192.168.8.109"
LAPTOP_PY="/home/stark/venv-nccl229/bin/python"
LAPTOP_DIR="/home/stark/hetero-cp-ringattn"
LAPTOP_MODEL="${LAPTOP_DIR}/models/Qwen2-0.5B"

WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
WHITE_TS="100.118.253.68"
WHITE_PY="/home/stark/venv-bench/bin/python"
WHITE_HCP_DIR="/home/stark/hcp-python"
WHITE_MODEL="/home/stark/hetero-cp-ringattn/models/Qwen2-0.5B"

PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_TS="100.111.242.55"
PEARL_PY="/home/stark/miniconda3/envs/vllm-rocm/bin/python"
PEARL_HCP_DIR="/home/stark/hcp-python"
PEARL_MODEL="/home/stark/hetero-cp-ringattn/models/Qwen2-0.5B"

MAC_PY="${MAC_PY:-/Users/stark_sim/miniconda3/bin/python3.12}"
MAC_MODEL_DIR="${MAC_MODEL_DIR:-${REPO_ROOT}/models/Qwen2-0.5B}"
MAC_TS="${MAC_TS:-100.121.35.138}"

SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROMPT_SEED="${PROMPT_SEED:-20261006}"

COORD_PORT=29920
W0_PORT=29921
W1_PORT=29922
W2_PORT=29923
W3_PORT=29924
TP_PORT=29601   # laptop<->white NCCL master

NCCL_ENV="NCCL_SOCKET_IFNAME=tailscale0 NCCL_IB_DISABLE=1 HCP_TP_TIMING=1"

RUN_ID="pyring4-nccl-abstract-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

COORD_PID=""
MAC_WORKER_PID=""
cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_PID}" ] && kill "${COORD_PID}" 2>/dev/null || true
    [ -n "${MAC_WORKER_PID}" ] && kill "${MAC_WORKER_PID}" 2>/dev/null || true
    laptop_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    white_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    pearl_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    npu_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
}
trap cleanup EXIT INT TERM

npu_ssh()   { ssh -F "${SSH_CFG}" -o BatchMode=yes -o ConnectTimeout=20 "${NPU_HOST}" "$1"; }
laptop_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${LAPTOP_SSH}" "$1"; }
white_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${WHITE_SSH}" "$1"; }
pearl_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${PEARL_SSH}" "$1"; }

# --- Worker launchers ---

# d0 抽象 worker：laptop rank0（协议面 + ring 监听）
laptop_start_tp_rank0() { # num_domains log
    laptop_ssh "
export PYTHONUNBUFFERED=1 ${NCCL_ENV}
cd ${LAPTOP_DIR}
mkdir -p logs
nohup ${LAPTOP_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${LAPTOP_MODEL} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains $1 \
    --peer-listen-host 0.0.0.0 --peer-listen-port ${W0_PORT} \
    --next-peer-host ${NPU_TS} --next-peer-port ${W1_PORT} \
    --device cuda \
    --tp-size 2 --tp-backend nccl --tp-rank 0 \
    --tp-master-addr ${LAPTOP_TS} --tp-master-port ${TP_PORT} \
    > logs/$2 2>&1 < /dev/null &
echo started laptop tp rank0 pid=\$!
"
}

# d0 抽象 worker：white rank1（纯计算 follower，不连 coordinator/ring）
white_start_tp_rank1() { # num_domains log
    white_ssh "
export PYTHONUNBUFFERED=1 ${NCCL_ENV}
cd ${WHITE_HCP_DIR}
nohup ${WHITE_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${WHITE_MODEL} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains $1 \
    --device cuda \
    --tp-size 2 --tp-backend nccl --tp-rank 1 \
    --tp-master-addr ${LAPTOP_TS} --tp-master-port ${TP_PORT} \
    > logs/$2 2>&1 < /dev/null &
echo started white tp rank1 follower pid=\$!
"
}

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

mac_start_worker() { # domain listen_port next_host next_port num_domains log
    PYTHONUNBUFFERED=1 env -u DYLD_LIBRARY_PATH "${MAC_PY}" \
        "${REPO_ROOT}/python/hcp_transformers_quic_worker.py" \
        --model-dir "${MAC_MODEL_DIR}" --coordinator-host 127.0.0.1 --coordinator-port ${COORD_PORT} \
        --domain-id "$1" --num-domains "$5" \
        --peer-listen-host 0.0.0.0 --peer-listen-port "$2" \
        --next-peer-host "$3" --next-peer-port "$4" \
        --device mps > "${REPORT_DIR}/$6" 2>&1 &
    MAC_WORKER_PID=$!
    echo "started mac worker domain=$1 pid=${MAC_WORKER_PID}"
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

wait_remote_ready() { # ssh_fn log pattern timeout_s
    local fn="$1" log="$2" pattern="$3" timeout_s="$4" elapsed=0
    while [ "${elapsed}" -lt "${timeout_s}" ]; do
        if ${fn} "grep -q '${pattern}' ${log} 2>/dev/null"; then return 0; fi
        if ${fn} "grep -qE 'Traceback' ${log} 2>/dev/null"; then
            echo "ERROR: remote process failed (${log}):" >&2
            ${fn} "tail -30 ${log}" >&2 || true
            return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "ERROR: timeout waiting for '${pattern}' (${log})" >&2
    return 1
}

wait_local_handshake() { # log timeout_s
    local log="$1" timeout_s="$2" elapsed=0
    while [ "${elapsed}" -lt "${timeout_s}" ]; do
        if grep -q 'handshake sent' "${log}" 2>/dev/null; then return 0; fi
        if grep -qE 'Traceback' "${log}" 2>/dev/null; then
            echo "ERROR: mac worker failed (${log}):" >&2
            tail -30 "${log}" >&2 || true
            return 1
        fi
        sleep 2; elapsed=$((elapsed + 2))
    done
    echo "ERROR: timeout waiting for handshake (${log})" >&2
    return 1
}

# --- Preflight ---
echo "=== Preflight ==="
[ -x "${BINARY}" ] || { echo "ERROR: ${BINARY} missing" >&2; exit 1; }
npu_ssh "mkdir -p ${NPU_HCP_DIR}/logs && test -f ${NPU_MODEL_DIR}/model.safetensors && echo npu ok"
laptop_ssh "mkdir -p ${LAPTOP_DIR}/logs && test -f ${LAPTOP_MODEL}/model.safetensors && echo laptop ok"
white_ssh "mkdir -p ${WHITE_HCP_DIR}/logs && test -f ${WHITE_MODEL}/model.safetensors && echo white ok"
pearl_ssh "mkdir -p ${PEARL_HCP_DIR}/logs && test -f ${PEARL_MODEL}/model.safetensors && echo pearl ok"

# --- Generate prompt ---
echo "=== Generating prompt (~${SEQ_LEN} tokens, seed=${PROMPT_SEED}) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
PROMPT_PY="${PROMPT_PY:-$(/usr/local/bin/python3.11 -c 'import tokenizers' 2>/dev/null && echo /usr/local/bin/python3.11 || echo /Users/stark_sim/miniconda3/bin/python3.12)}"
"${PROMPT_PY}" "${REPO_ROOT}/scripts/gen_natural_prompt.py" \
    "${MAC_MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_SEED}" "${PROMPT_FILE}" \
    | tee "${REPORT_DIR}/gen_prompt.log"
cp "${PROMPT_FILE}" "${REPORT_DIR}/prompt.txt"

start_coordinator() { # phase num_domains
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${MAC_MODEL_DIR}" \
        --prompt-file "${PROMPT_FILE}" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --num-domains "$2" \
        --listen-addr "0.0.0.0:${COORD_PORT}" \
        --export-logits-dir "${REPORT_DIR}/logits_$1" \
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
    if [ -n "${MAC_WORKER_PID}" ]; then
        kill "${MAC_WORKER_PID}" 2>/dev/null || true
        wait "${MAC_WORKER_PID}" 2>/dev/null || true
        MAC_WORKER_PID=""
    fi
    laptop_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    white_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    pearl_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    npu_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    return "${exit_code}"
}

# === Per-backend 1-domain goldens ===
run_golden() { # name device
    local name="$1" device="$2"
    echo ""
    echo "=== Phase golden-${name}: 1-domain on ${device} ==="
    start_coordinator "golden-${name}" 1
    case "${name}" in
        npu)   npu_start_worker 0 "${device}" ${W1_PORT} 127.0.0.1 ${W2_PORT} 1 "w_golden-npu.log"
               wait_remote_handshake npu_ssh "${NPU_HCP_DIR}/logs/w_golden-npu.log" 600 ;;
        cuda)  white_start_worker 0 "${device}" ${W1_PORT} 127.0.0.1 ${W2_PORT} 1 "w_golden-cuda.log"
               wait_remote_handshake white_ssh "${WHITE_HCP_DIR}/logs/w_golden-cuda.log" 600 ;;
        hip)   pearl_start_worker 0 "${device}" ${W2_PORT} 127.0.0.1 ${W1_PORT} 1 "w_golden-hip.log"
               wait_remote_handshake pearl_ssh "${PEARL_HCP_DIR}/logs/w_golden-hip.log" 600 ;;
        mps)   mac_start_worker 0 ${W3_PORT} 127.0.0.1 ${W1_PORT} 1 "worker-golden-mps.log"
               wait_local_handshake "${REPORT_DIR}/worker-golden-mps.log" 300 ;;
    esac
    local rc=0
    finish_phase "golden-${name}" || rc=$?
    return "${rc}"
}

GOLDEN_FAILURES=""
run_golden npu  npu:0  || GOLDEN_FAILURES="${GOLDEN_FAILURES} npu"
run_golden cuda cuda:0 || GOLDEN_FAILURES="${GOLDEN_FAILURES} cuda"
run_golden hip  cuda:0 || GOLDEN_FAILURES="${GOLDEN_FAILURES} hip"
run_golden mps  mps    || GOLDEN_FAILURES="${GOLDEN_FAILURES} mps"

# === 4-domain ring：d0 = 跨机 NCCL 抽象 worker（laptop 4060 + white 4090）===
echo ""
echo "=== Phase ring: 4-domain (abstract[4060+4090] + npu + hip + mps) ==="
start_coordinator ring 4
# rank1 follower 先起（它只是阻塞等 NCCL bootstrap），rank0 后起
white_start_tp_rank1 4 "w0r1_ring.log"
laptop_start_tp_rank0 4 "w0_ring.log"
npu_start_worker   1 npu:0  ${W1_PORT} "${PEARL_TS}"   ${W2_PORT} 4 "w1_ring.log"
pearl_start_worker 2 cuda:0 ${W2_PORT} "${MAC_TS}"     ${W3_PORT} 4 "w2_ring.log"
mac_start_worker   3         ${W3_PORT} "${LAPTOP_TS}" ${W0_PORT} 4 "worker3-mps_ring.log"
wait_remote_ready     white_ssh  "${WHITE_HCP_DIR}/logs/w0r1_ring.log" "follower rank 1. ready" 600
wait_remote_handshake laptop_ssh "${LAPTOP_DIR}/logs/w0_ring.log" 600
wait_remote_handshake npu_ssh    "${NPU_HCP_DIR}/logs/w1_ring.log" 600
wait_remote_handshake pearl_ssh  "${PEARL_HCP_DIR}/logs/w2_ring.log" 600
wait_local_handshake "${REPORT_DIR}/worker3-mps_ring.log" 300
RING_EXIT=0
finish_phase ring || RING_EXIT=$?

# === Fetch worker logs ===
npu_ssh    "cat ${NPU_HCP_DIR}/logs/w_golden-npu.log"    > "${REPORT_DIR}/worker-golden-npu.log"  2>/dev/null || true
white_ssh  "cat ${WHITE_HCP_DIR}/logs/w_golden-cuda.log" > "${REPORT_DIR}/worker-golden-cuda.log" 2>/dev/null || true
pearl_ssh  "cat ${PEARL_HCP_DIR}/logs/w_golden-hip.log"  > "${REPORT_DIR}/worker-golden-hip.log"  2>/dev/null || true
laptop_ssh "cat ${LAPTOP_DIR}/logs/w0_ring.log"          > "${REPORT_DIR}/worker0-tp_rank0_ring.log" 2>/dev/null || true
white_ssh  "cat ${WHITE_HCP_DIR}/logs/w0r1_ring.log"     > "${REPORT_DIR}/worker0-tp_rank1_ring.log" 2>/dev/null || true
npu_ssh    "cat ${NPU_HCP_DIR}/logs/w1_ring.log"         > "${REPORT_DIR}/worker1-npu_ring.log"   2>/dev/null || true
pearl_ssh  "cat ${PEARL_HCP_DIR}/logs/w2_ring.log"       > "${REPORT_DIR}/worker2-hip_ring.log"   2>/dev/null || true

echo ""
echo "=== Generated outputs ==="
for p in golden-npu golden-cuda golden-hip golden-mps ring; do
    printf '%-14s ' "${p}:"
    grep "generated:" "${REPORT_DIR}/coordinator_${p}.log" || echo "(none)"
done

echo ""
echo "=== Capacities (ring) ==="
grep -h "capacity" "${REPORT_DIR}"/worker*_ring.log "${REPORT_DIR}/coordinator_ring.log" 2>/dev/null | sort -u || true

echo ""
echo "=== Logits diff: ring vs goldens ==="
for g in golden-npu golden-cuda golden-hip golden-mps; do
    if [ -d "${REPORT_DIR}/logits_ring" ] && [ -d "${REPORT_DIR}/logits_${g}" ]; then
        echo "--- ring vs ${g} ---"
        "${PROMPT_PY}" "${REPO_ROOT}/scripts/compare_logits_dir.py" \
            "${REPORT_DIR}/logits_ring" "${REPORT_DIR}/logits_${g}" 2>/dev/null | head -1 || true
    fi
done

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
G_MPS="$(grep 'generated:' "${REPORT_DIR}/coordinator_golden-mps.log" | head -1 || true)"
G_RING="$(grep 'generated:' "${REPORT_DIR}/coordinator_ring.log" | head -1 || true)"
[ -z "${G_RING}" ] && { echo "INCOMPLETE: no ring output (exit=${RING_EXIT})"; exit 2; }
# Ring final logits come from the last domain (mac/MPS). Primary gate:
if [ "${G_RING}" = "${G_MPS}" ]; then
    echo "PASS: 4-domain ring (with cross-machine NCCL abstract worker) matches MPS golden exactly"
else
    echo "FAIL: ring output differs from MPS golden"
    FAIL=1
fi
if [ "${G_NPU}" = "${G_CUDA}" ] && [ "${G_CUDA}" = "${G_HIP}" ] && [ "${G_HIP}" = "${G_MPS}" ]; then
    echo "INFO: all four backend goldens identical"
else
    echo "INFO: backend goldens differ (npu/cuda/hip/mps) — cross-backend numerics diverge on argmax"
fi
exit ${FAIL}
