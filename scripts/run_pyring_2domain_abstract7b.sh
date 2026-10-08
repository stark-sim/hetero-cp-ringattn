#!/usr/bin/env bash
set -euo pipefail

# 节点①：Qwen2.5-7B (bf16) 双逻辑 worker CP 正确性验证
#   d0 = 抽象 worker：容器双芯 Ascend910 npu:0+npu:1（HCCL TP=2）
#   d1 = white RTX 4090 单卡（普通 worker，无 TP）
#   coordinator: Mac Rust binary (tailnet)
#   ring: d0(container rank0) -> d1(white) -> d0，均走 tailnet/DERP
#
# Gates:
#   1) golden-npu（容器单芯 bf16）+ golden-cuda（white bf16）各跑一次 1-domain；
#   2) 2-domain ring 输出与 d1（末域 white CUDA）golden 逐 token 一致；
#   3) ring vs 两 golden 的 logits max|Δ|（bf16 量级，预期 1e-2 上下）；
#   4) capacity/chunk 分配证据（NPU 对 ~119GB vs white ~21GB，chunk 应不均分）。

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}"

SSH_CFG="${HOME}/.atomgitdevenv/.ssh/config"
NPU_HOST="devenvc_wsa4w.9d5a4a35b7ae4f94b09636ec713325cc.atomgit.0"
NPU_PY="/opt/buildtools/Python-3.11.4/bin/python3"
NPU_HCP_DIR="/home/developer/hcp"
NPU_MODEL_DIR="${NPU_MODEL_DIR:-/home/developer/models/Qwen2.5-7B}"
NPU_TS="100.91.253.114"

WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
WHITE_TS="100.118.253.68"
WHITE_PY="/home/stark/venv-bench/bin/python"
WHITE_HCP_DIR="/home/stark/hcp-python"
WHITE_MODEL="${WHITE_MODEL:-/home/stark/hetero-cp-ringattn/models/Qwen2.5-7B}"

MAC_TS="${MAC_TS:-100.121.35.138}"
DTYPE="${DTYPE:-bfloat16}"

SEQ_LEN="${SEQ_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROMPT_SEED="${PROMPT_SEED:-20261008}"

COORD_PORT=29930
W0_PORT=29931   # 容器 TP rank0 的 ring 端口
W1_PORT=29932   # white 的 ring 端口
TP_PORT=29621   # 容器内 TP rank0<->rank1 的 hccl master 端口

RUN_ID="pyring2-abstract7b-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

COORD_PID=""
cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_PID}" ] && kill "${COORD_PID}" 2>/dev/null || true
    npu_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    white_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
}
trap cleanup EXIT INT TERM

npu_ssh()   { ssh -F "${SSH_CFG}" -o BatchMode=yes -o ConnectTimeout=20 "${NPU_HOST}" "$1"; }
white_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${WHITE_SSH}" "$1"; }

# 容器 TP 抽象 worker：rank 0（协议面）+ rank 1（纯计算 follower）
npu_start_tp_worker() { # listen_port next_host next_port num_domains log0 log1
    npu_ssh "
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:\${LD_LIBRARY_PATH:-}
cd ${NPU_HCP_DIR}
nohup ${NPU_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${NPU_MODEL_DIR} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains $4 \
    --peer-listen-host 0.0.0.0 --peer-listen-port $1 \
    --next-peer-host $2 --next-peer-port $3 \
    --device npu --dtype ${DTYPE} \
    --tp-size 2 --tp-backend hccl --tp-rank 0 --tp-master-addr 127.0.0.1 --tp-master-port ${TP_PORT} \
    > ${NPU_HCP_DIR}/logs/$5 2>&1 < /dev/null &
echo started npu tp rank0 pid=\$!
nohup ${NPU_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${NPU_MODEL_DIR} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains $4 \
    --peer-listen-host 0.0.0.0 --peer-listen-port $1 \
    --next-peer-host $2 --next-peer-port $3 \
    --device npu --dtype ${DTYPE} \
    --tp-size 2 --tp-backend hccl --tp-rank 1 --tp-master-addr 127.0.0.1 --tp-master-port ${TP_PORT} \
    > ${NPU_HCP_DIR}/logs/$6 2>&1 < /dev/null &
echo started npu tp rank1 pid=\$!
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
    --device $2 --dtype ${DTYPE} > ${NPU_HCP_DIR}/logs/$7 2>&1 < /dev/null &
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
    --device $2 --dtype ${DTYPE} > ${WHITE_HCP_DIR}/logs/$7 2>&1 < /dev/null &
echo started white worker domain=$1 pid=\$!
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

# --- Preflight ---
echo "=== Preflight ==="
[ -x "${BINARY}" ] || { echo "ERROR: ${BINARY} missing" >&2; exit 1; }
[ -f "${REPO_ROOT}/models/Qwen2.5-7B/tokenizer.json" ] || { echo "ERROR: mac tokenizer missing" >&2; exit 1; }
npu_ssh "mkdir -p ${NPU_HCP_DIR}/logs && test -f ${NPU_MODEL_DIR}/model.safetensors.index.json && echo npu ok"
white_ssh "mkdir -p ${WHITE_HCP_DIR}/logs && test -f ${WHITE_MODEL}/model.safetensors.index.json && echo white ok"

# --- Generate prompt ---
echo "=== Generating prompt (~${SEQ_LEN} tokens, seed=${PROMPT_SEED}) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
PROMPT_PY="${PROMPT_PY:-$(/usr/local/bin/python3.11 -c 'import tokenizers' 2>/dev/null && echo /usr/local/bin/python3.11 || echo /Users/stark_sim/miniconda3/bin/python3.12)}"
"${PROMPT_PY}" "${REPO_ROOT}/scripts/gen_natural_prompt.py" \
    "${REPO_ROOT}/models/Qwen2.5-7B/tokenizer.json" "${SEQ_LEN}" "${PROMPT_SEED}" "${PROMPT_FILE}" \
    | tee "${REPORT_DIR}/gen_prompt.log"
cp "${PROMPT_FILE}" "${REPORT_DIR}/prompt.txt"

start_coordinator() { # phase num_domains
    "${BINARY}" --distributed-role coordinator \
        --model-dir "${REPO_ROOT}/models/Qwen2.5-7B" \
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
    npu_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    white_ssh 'pkill -f hcp_transformers_quic_worke[r] || true' || true
    return "${exit_code}"
}

# === Per-backend 1-domain goldens（bf16；7B 加载慢，handshake 超时放宽）===
run_golden() { # name device
    local name="$1" device="$2"
    echo ""
    echo "=== Phase golden-${name}: 1-domain ${device} ${DTYPE} ==="
    start_coordinator "golden-${name}" 1
    case "${name}" in
        npu)   npu_start_worker 0 "${device}" ${W0_PORT} 127.0.0.1 ${W1_PORT} 1 "w_golden-npu.log"
               wait_remote_handshake npu_ssh "${NPU_HCP_DIR}/logs/w_golden-npu.log" 900 ;;
        cuda)  white_start_worker 0 "${device}" ${W1_PORT} 127.0.0.1 ${W0_PORT} 1 "w_golden-cuda.log"
               wait_remote_handshake white_ssh "${WHITE_HCP_DIR}/logs/w_golden-cuda.log" 900 ;;
    esac
    local rc=0
    finish_phase "golden-${name}" || rc=$?
    return "${rc}"
}

GOLDEN_FAILURES=""
run_golden npu  npu:0  || GOLDEN_FAILURES="${GOLDEN_FAILURES} npu"
run_golden cuda cuda:0 || GOLDEN_FAILURES="${GOLDEN_FAILURES} cuda"

# === 2-domain ring：d0 = 容器双芯 TP 抽象 worker，d1 = white 4090 ===
echo ""
echo "=== Phase ring: 2-domain (container TP npu:0+1 + white cuda) bf16 ==="
start_coordinator ring 2
npu_start_tp_worker ${W0_PORT} "${WHITE_TS}" ${W1_PORT} 2 "w0_ring.log" "w0r1_ring.log"
white_start_worker 1 cuda:0 ${W1_PORT} "${NPU_TS}" ${W0_PORT} 2 "w1_ring.log"
wait_remote_handshake npu_ssh   "${NPU_HCP_DIR}/logs/w0_ring.log" 900
wait_remote_ready     npu_ssh   "${NPU_HCP_DIR}/logs/w0r1_ring.log" "follower rank 1. ready" 900
wait_remote_handshake white_ssh "${WHITE_HCP_DIR}/logs/w1_ring.log" 900
RING_EXIT=0
finish_phase ring || RING_EXIT=$?

# === Fetch worker logs ===
npu_ssh   "cat ${NPU_HCP_DIR}/logs/w_golden-npu.log"    > "${REPORT_DIR}/worker-golden-npu.log"   2>/dev/null || true
white_ssh "cat ${WHITE_HCP_DIR}/logs/w_golden-cuda.log" > "${REPORT_DIR}/worker-golden-cuda.log"  2>/dev/null || true
npu_ssh   "cat ${NPU_HCP_DIR}/logs/w0_ring.log"         > "${REPORT_DIR}/worker0-tp_rank0_ring.log" 2>/dev/null || true
npu_ssh   "cat ${NPU_HCP_DIR}/logs/w0r1_ring.log"       > "${REPORT_DIR}/worker0-tp_rank1_ring.log" 2>/dev/null || true
white_ssh "cat ${WHITE_HCP_DIR}/logs/w1_ring.log"       > "${REPORT_DIR}/worker1-cuda_ring.log"   2>/dev/null || true

echo ""
echo "=== Generated outputs ==="
for p in golden-npu golden-cuda ring; do
    printf '%-14s ' "${p}:"
    grep "generated:" "${REPORT_DIR}/coordinator_${p}.log" || echo "(none)"
done

echo ""
echo "=== Capacities & chunk assignment (ring) ==="
grep -h "capacity" "${REPORT_DIR}"/worker*_ring.log "${REPORT_DIR}/coordinator_ring.log" 2>/dev/null | sort -u || true
grep -h "prefill done" "${REPORT_DIR}/worker0-tp_rank1_ring.log" 2>/dev/null || true

echo ""
echo "=== Logits diff: ring vs goldens (bf16 容差) ==="
for g in golden-npu golden-cuda; do
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
G_RING="$(grep 'generated:' "${REPORT_DIR}/coordinator_ring.log" | head -1 || true)"
[ -z "${G_RING}" ] && { echo "INCOMPLETE: no ring output (exit=${RING_EXIT})"; exit 2; }
# Ring final logits come from the last domain (white/CUDA). Primary gate:
if [ "${G_RING}" = "${G_CUDA}" ]; then
    echo "PASS: 2-domain ring (TP abstract 2xNPU + white 4090) matches CUDA (last-domain) golden exactly"
else
    echo "FAIL: ring output differs from CUDA golden"
    FAIL=1
fi
if [ "${G_NPU}" = "${G_CUDA}" ]; then
    echo "INFO: npu/cuda goldens identical"
else
    echo "INFO: npu/cuda goldens differ — cross-backend bf16 numerics diverge on argmax"
fi
exit ${FAIL}
