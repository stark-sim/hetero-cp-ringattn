#!/usr/bin/env bash
set -euo pipefail

# 节点②：单机最小 PD 闭环（white 4090 单机上跑）
#   P 侧：HCP TransformersBackend 1-domain prefill（Qwen2.5-7B bf16）
#         + golden decode 16 token + KV 导出（hcp-kv-v1 文件格式）
#   D 侧：vLLM 0.31 + HcpFileConnector 从文件载入 KV，跳过 prefill，greedy 16 token
#   gate：vLLM 输出 token ids 与 HCP golden 逐 token 一致
#
# 产物：reports/pd-loop-<ts>/（kv/ meta.json + safetensors、两侧日志、verdict）

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

WHITE_SSH="${WHITE_SSH:-stark@100.118.253.68}"
WHITE_HCP_DIR="/home/stark/hcp-python"
WHITE_PLUGIN_DIR="${WHITE_PLUGIN_DIR:-/home/stark/hcp-vllm-plugin}"
WHITE_VLLM_PY="${WHITE_VLLM_PY:-/home/stark/venv-vllm-pd/bin/python}"
WHITE_BENCH_PY="/home/stark/venv-bench/bin/python"
MODEL_DIR="${MODEL_DIR:-/home/stark/hetero-cp-ringattn/models/Qwen2.5-7B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SEQ_LEN="${SEQ_LEN:-128}"
PROMPT_SEED="${PROMPT_SEED:-20261008}"

RUN_ID="pd-loop-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"
echo "Report dir: ${REPORT_DIR}"

KV_DIR="/tmp/hcp_pd_kv_${RUN_ID}"
PROMPT_FILE="/tmp/hcp_pd_prompt_${RUN_ID}.txt"

white_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "${WHITE_SSH}" "$1"; }

cleanup() {
    white_ssh "rm -rf ${KV_DIR} ${PROMPT_FILE}" || true
}
trap cleanup EXIT INT TERM

# --- prompt：Mac 本地生成（7B tokenizer 已在 repo models/），scp 到 white ---
echo "=== Generating prompt (~${SEQ_LEN} tokens, seed=${PROMPT_SEED}) ==="
PROMPT_PY="${PROMPT_PY:-$(/usr/local/bin/python3.11 -c 'import tokenizers' 2>/dev/null && echo /usr/local/bin/python3.11 || echo /Users/stark_sim/miniconda3/bin/python3.12)}"
"${PROMPT_PY}" "${REPO_ROOT}/scripts/gen_natural_prompt.py" \
    "${REPO_ROOT}/models/Qwen2.5-7B/tokenizer.json" "${SEQ_LEN}" "${PROMPT_SEED}" "${PROMPT_FILE}"
cp "${PROMPT_FILE}" "${REPORT_DIR}/prompt.txt"
scp -o BatchMode=yes "${PROMPT_FILE}" "${WHITE_SSH}:${PROMPT_FILE}"

# --- P 侧：HCP prefill + golden decode + KV 导出 ---
echo "=== Phase P: HCP prefill + golden decode + KV export (bf16) ==="
white_ssh "cd ${WHITE_HCP_DIR} && PYTHONUNBUFFERED=1 ${WHITE_BENCH_PY} python/hcp_pd_export.py \
    --model-dir ${MODEL_DIR} --prompt-file ${PROMPT_FILE} \
    --out-dir ${KV_DIR} --max-new-tokens ${MAX_NEW_TOKENS} \
    --device cuda --dtype bfloat16" > "${REPORT_DIR}/export.log" 2>&1 || {
    echo "ERROR: HCP export failed:"; tail -20 "${REPORT_DIR}/export.log"; exit 1
}
grep "pd-export" "${REPORT_DIR}/export.log"

# --- D 侧：vLLM consumer（connector 在插件仓 hcp_vllm_plugin.file_connector） ---
echo "=== Phase D: vLLM consumer (HcpFileConnector, skip prefill) ==="
white_ssh "cd ${WHITE_PLUGIN_DIR} && PYTHONUNBUFFERED=1 ${WHITE_VLLM_PY} pd_consume.py \
    --model-dir ${MODEL_DIR} --kv-dir ${KV_DIR} \
    --max-new-tokens ${MAX_NEW_TOKENS}" > "${REPORT_DIR}/consume.log" 2>&1
D_RC=$?
grep "pd-consume" "${REPORT_DIR}/consume.log" || tail -30 "${REPORT_DIR}/consume.log"

echo ""
echo "=== Verdict ==="
if [ "${D_RC}" -eq 0 ] && grep -q "MATCH=True" "${REPORT_DIR}/consume.log"; then
    echo "PASS: vLLM decode matches HCP golden token-by-token (PD loop closed)"
    exit 0
else
    echo "FAIL: see ${REPORT_DIR}/consume.log"
    exit 1
fi
