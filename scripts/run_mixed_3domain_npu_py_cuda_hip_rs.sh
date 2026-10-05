#!/usr/bin/env bash
set -euo pipefail

# 远程混合 ring：Python(NPU) + Rust(CUDA) + Rust(HIP) 3-domain
#   domain 0: AtomGit 容器 Ascend910 npu:0，Python worker --ring-mode rust
#   domain 1: white RTX 4090，Rust worker（HCP_TCH_DEVICE=cuda:0）
#   domain 2: pearl RX 9060 XT，Rust worker（LD_PRELOAD libtorch_hip + cuda:0）
#   coordinator: Mac（tailnet）
#
# ring 边：container->white（DERP）、white->pearl（LAN 2.5GbE）、pearl->container（DERP）。
# 注意数值域：Rust worker 按模型 config 跑 bf16，Python worker 跑 fp32；
# ring 的 decode logits（worker0=NPU 产出）经 ring packet 在 Rust 侧 bf16 合并，
# 因此 ring vs fp32 golden 的 max|Δlogit| 预期为 bf16 量级（~1e-2..1e0），
# gate = argmax/文本一致，logits diff 仅作记录。

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
WHITE_REPO="/home/stark/hetero-cp-ringattn"
WHITE_MODEL="${WHITE_REPO}/models/Qwen2-0.5B"

PEARL_SSH="${PEARL_SSH:-stark@100.111.242.55}"
PEARL_LAN="192.168.100.2"
PEARL_REPO="/home/stark/hetero-cp-ringattn"
PEARL_MODEL="${PEARL_REPO}/models/Qwen2-0.5B"

MAC_MODEL_DIR="${MAC_MODEL_DIR:-${REPO_ROOT}/models/Qwen2-0.5B}"
MAC_TS="${MAC_TS:-100.121.35.138}"

SEQ_LEN="${SEQ_LEN:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROMPT_SEED="${PROMPT_SEED:-20261005}"

COORD_PORT=29930
W0_PORT=29931
W1_PORT=29932
W2_PORT=29933

RUN_ID="mixed3-npuPy-cudaRs-hipRs-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

BINARY="${REPO_ROOT}/rust/target/release/hcp-ringattn-rust"

COORD_PID=""
cleanup() {
    echo "=== Cleanup ==="
    [ -n "${COORD_PID}" ] && kill "${COORD_PID}" 2>/dev/null || true
    npu_ssh 'pkill -f "hcp_transformers_quic_worke[r]" || true' || true
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

npu_ssh()   { ssh -F "${SSH_CFG}" -o BatchMode=yes -o ConnectTimeout=20 "${NPU_HOST}" "$1"; }

# --- Preflight ---
echo "=== Preflight ==="
[ -x "${BINARY}" ] || { echo "ERROR: ${BINARY} missing" >&2; exit 1; }
npu_ssh "mkdir -p ${NPU_HCP_DIR}/logs && test -f ${NPU_MODEL_DIR}/model.safetensors && echo npu ok"
ssh -o BatchMode=yes -o ConnectTimeout=15 "${WHITE_SSH}" \
    "test -x ${WHITE_REPO}/rust/target/release/hcp-ringattn-rust && echo white ok"
ssh -o BatchMode=yes -o ConnectTimeout=15 "${PEARL_SSH}" \
    "test -x ${PEARL_REPO}/rust/target/release/hcp-ringattn-rust && echo pearl ok"

# --- Prompt ---
echo "=== Generating prompt (~${SEQ_LEN} tokens, seed=${PROMPT_SEED}) ==="
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
PROMPT_PY="${PROMPT_PY:-$(/usr/local/bin/python3.11 -c 'import tokenizers' 2>/dev/null && echo /usr/local/bin/python3.11 || echo /Users/stark_sim/miniconda3/bin/python3.12)}"
"${PROMPT_PY}" "${REPO_ROOT}/scripts/gen_natural_prompt.py" \
    "${MAC_MODEL_DIR}/tokenizer.json" "${SEQ_LEN}" "${PROMPT_SEED}" "${PROMPT_FILE}" | tee "${REPORT_DIR}/gen_prompt.log"
cp "${PROMPT_FILE}" "${REPORT_DIR}/prompt.txt"

# === golden: 1-domain NPU Python ===
echo ""
echo "=== Phase golden-npu (1 domain) ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MAC_MODEL_DIR}" --prompt-file "${PROMPT_FILE}" \
    --max-tokens "${MAX_NEW_TOKENS}" --num-domains 1 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --export-logits-dir "${REPORT_DIR}/logits_golden-npu" \
    >"${REPORT_DIR}/coordinator_golden-npu.log" 2>&1 &
COORD_PID=$!
sleep 2
npu_ssh "
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:\${LD_LIBRARY_PATH:-}
cd ${NPU_HCP_DIR}
nohup ${NPU_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${NPU_MODEL_DIR} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 1 \
    --peer-listen-host 127.0.0.1 --peer-listen-port ${W0_PORT} \
    --next-peer-host 127.0.0.1 --next-peer-port ${W1_PORT} \
    --device npu:0 > ${NPU_HCP_DIR}/logs/w_golden.log 2>&1 < /dev/null &
echo started golden worker pid=\$!
"
set +e
wait "${COORD_PID}"; G_EXIT=$?
set -e
COORD_PID=""
npu_ssh 'pkill -f "hcp_transformers_quic_worke[r]" || true' || true
echo "golden exit: ${G_EXIT}"

# === ring: Python(NPU d0) + Rust(white d1) + Rust(pearl d2) ===
echo ""
echo "=== Phase ring: mixed 3-domain ==="
"${BINARY}" --distributed-role coordinator \
    --model-dir "${MAC_MODEL_DIR}" --prompt-file "${PROMPT_FILE}" \
    --max-tokens "${MAX_NEW_TOKENS}" --num-domains 3 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    --export-logits-dir "${REPORT_DIR}/logits_ring" \
    >"${REPORT_DIR}/coordinator_ring.log" 2>&1 &
COORD_PID=$!
sleep 2

npu_ssh "
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:\${LD_LIBRARY_PATH:-}
cd ${NPU_HCP_DIR}
nohup ${NPU_PY} python/hcp_transformers_quic_worker.py \
    --model-dir ${NPU_MODEL_DIR} --coordinator-host ${MAC_TS} --coordinator-port ${COORD_PORT} \
    --domain-id 0 --num-domains 3 \
    --peer-listen-host 0.0.0.0 --peer-listen-port ${W0_PORT} \
    --next-peer-host ${WHITE_TS} --next-peer-port ${W1_PORT} \
    --device npu:0 --ring-mode rust > ${NPU_HCP_DIR}/logs/w0_ring.log 2>&1 < /dev/null &
echo started npu worker pid=\$!
"

ssh -n -f -o BatchMode=yes -o ConnectTimeout=15 "${WHITE_SSH}" \
    "cd ${WHITE_REPO} && setsid env HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
     ./rust/target/release/hcp-ringattn-rust \
     --distributed-role worker --domain-id 1 --model-dir ${WHITE_MODEL} \
     --listen-addr 0.0.0.0:${W1_PORT} --next-peer-addr ${PEARL_LAN}:${W2_PORT} \
     --coordinator-addr ${MAC_TS}:${COORD_PORT} --num-domains 3 \
     > /tmp/hcp_mixed_w1.log 2>&1 < /dev/null"
echo "started white rust worker"

ssh -n -f -o BatchMode=yes -o ConnectTimeout=15 "${PEARL_SSH}" \
    "cd ${PEARL_REPO} && setsid env LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so HCP_TCH_DEVICE=cuda:0 LD_LIBRARY_PATH=/home/stark/libtorch/lib \
     ./rust/target/release/hcp-ringattn-rust \
     --distributed-role worker --domain-id 2 --model-dir ${PEARL_MODEL} \
     --listen-addr 0.0.0.0:${W2_PORT} --next-peer-addr ${NPU_TS}:${W0_PORT} \
     --coordinator-addr ${MAC_TS}:${COORD_PORT} --num-domains 3 \
     > /tmp/hcp_mixed_w2.log 2>&1 < /dev/null"
echo "started pearl rust worker"

set +e
wait "${COORD_PID}"; RING_EXIT=$?
set -e
COORD_PID=""
echo "ring exit: ${RING_EXIT}"

npu_ssh 'pkill -f "hcp_transformers_quic_worke[r]" || true' || true
ssh -o BatchMode=yes "${WHITE_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true
ssh -o BatchMode=yes "${PEARL_SSH}" "pkill -f 'hcp-ringattn-rust.*distributed-role' || true" 2>/dev/null || true

npu_ssh "cat ${NPU_HCP_DIR}/logs/w0_ring.log" > "${REPORT_DIR}/worker0-npu-py_ring.log" 2>/dev/null || true
scp -o BatchMode=yes "${WHITE_SSH}:/tmp/hcp_mixed_w1.log" "${REPORT_DIR}/worker1-cuda-rs_ring.log" >/dev/null 2>&1 || true
scp -o BatchMode=yes "${PEARL_SSH}:/tmp/hcp_mixed_w2.log" "${REPORT_DIR}/worker2-hip-rs_ring.log" >/dev/null 2>&1 || true

echo ""
echo "=== Generated ==="
printf '%-14s ' "golden-npu:"; grep "generated:" "${REPORT_DIR}/coordinator_golden-npu.log" || echo "(none)"
printf '%-14s ' "ring:"; grep "generated:" "${REPORT_DIR}/coordinator_ring.log" || echo "(none)"

echo ""
G="$(grep 'generated:' "${REPORT_DIR}/coordinator_golden-npu.log" | head -1 || true)"
R="$(grep 'generated:' "${REPORT_DIR}/coordinator_ring.log" | head -1 || true)"
echo "=== Verdict ==="
if [ -z "$G" ] || [ -z "$R" ]; then
    echo "INCOMPLETE (golden_exit=${G_EXIT} ring_exit=${RING_EXIT})"; exit 2
elif [ "$G" = "$R" ]; then
    echo "PASS: mixed ring (Python NPU + Rust CUDA + Rust HIP) matches NPU golden exactly"
else
    echo "FAIL: outputs differ"; exit 1
fi
"${PROMPT_PY}" "${REPO_ROOT}/scripts/compare_logits_dir.py" "${REPORT_DIR}/logits_ring" "${REPORT_DIR}/logits_golden-npu" | head -3 || true
echo "(注：Rust 侧 bf16 数值域，logits diff 预期 ~1e-2..1e0 量级)"
