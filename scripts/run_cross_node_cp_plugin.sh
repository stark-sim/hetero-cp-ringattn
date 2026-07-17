#!/bin/bash
# Cross-node HcpCpConnector plugin validation: producer on white (CUDA, V1 vLLM)
# computes chunk A and serves KV over HTTP; consumer on pearl (ROCm, V1 vLLM)
# pulls chunk A KV as external prefix and computes chunk B.  The consumer also
# computes a single-node reference inline and compares.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_SSH="${WHITE_USER:-stark}@${WHITE_HOST}"
WHITE_REPO="${WHITE_REPO:-/home/stark/hetero-cp-ringattn}"
WHITE_PYENV="${WHITE_PYENV:-/home/stark/miniconda3/envs/vllm-v1}"
WHITE_MODEL="${WHITE_MODEL:-/home/stark/models/Qwen2-0.5B-1M}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_SSH="${PEARL_USER:-stark}@${PEARL_HOST}"
PEARL_REPO="${PEARL_REPO:-/home/stark/hetero-cp-ringattn}"
PEARL_MODEL="${PEARL_MODEL:-/home/stark/models/Qwen2-0.5B-1M}"

SERVE_PORT="${SERVE_PORT:-8899}"
CHUNK_A="${CHUNK_A:-32}"
DECODE="${DECODE:-4}"
RUN_ID="cpplug-$(date +%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/cp-plugin-${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() { printf "'"; printf "%s" "$1" | sed "s/'/'\\''/g"; printf "'"; }
run_white() { ssh -o ConnectTimeout=20 "${WHITE_SSH}" "bash -lc $(shell_quote "$1")"; }
run_pearl() { ssh -o ConnectTimeout=20 "${PEARL_SSH}" "bash -lc $(shell_quote "$1")"; }

echo "=== HcpCpConnector cross-node plugin validation (white CUDA + pearl ROCm) ==="
echo "RUN_ID=${RUN_ID}  chunk_a=${CHUNK_A} decode=${DECODE} port=${SERVE_PORT}"

# === Tokenize the 64-token varied prompt and split into chunk A / full ===
TOK_DIR="/tmp/cpplug_${RUN_ID}"
mkdir -p "${TOK_DIR}"
python3 - <<'PY'
from transformers import AutoTokenizer
import os
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/Qwen2-0.5B"))
text = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
        "lima mike november oscar papa quebec romeo sierra tango uniform "
        "victor whiskey xray yankee zulu apple banana cherry dragon eagle "
        "falcon grape hotel igloo jungle koala lemon mango ninja orange panda qu")
ids = tok.encode(text, add_special_tokens=False)
assert len(ids) == 64, len(ids)
d = os.environ["TOK_DIR"]
open(os.path.join(d, "full.txt"), "w").write(" ".join(map(str, ids)))
open(os.path.join(d, "chunkA.txt"), "w").write(" ".join(map(str, ids[:32])))
print("tokenized 64 tokens")
PY
export TOK_DIR
scp -q "${TOK_DIR}/full.txt" "${TOK_DIR}/chunkA.txt" "${WHITE_SSH}:${TOK_DIR}/" 2>/dev/null || run_white "mkdir -p ${TOK_DIR}"
scp -q "${TOK_DIR}/full.txt" "${TOK_DIR}/chunkA.txt" "${WHITE_SSH}:${TOK_DIR}/"
scp -q "${TOK_DIR}/full.txt" "${PEARL_SSH}:${TOK_DIR}/" 2>/dev/null || run_pearl "mkdir -p ${TOK_DIR}"
scp -q "${TOK_DIR}/full.txt" "${PEARL_SSH}:${TOK_DIR}/"

# === Launch producer on white (holds to serve KV) ===
echo "=== Launching producer on white ==="
prod_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-v1 && \
  python ${WHITE_REPO}/scripts/cp_producer.py \
    --model-dir ${WHITE_MODEL} --chunk-file ${TOK_DIR}/chunkA.txt \
    --serve-port ${SERVE_PORT} --shared-path /tmp/hcp_cp_${RUN_ID} --run-id ${RUN_ID} \
    --gpu-mem 0.35 --hold-secs 300"
run_white "${prod_cmd}" >"${REPORT_DIR}/producer.log" 2>&1 &
PROD_PID=$!
echo "producer ssh pid=${PROD_PID}"

# Wait for producer to be serving KV (_READY reachable)
echo "=== Waiting for producer KV server ==="
for i in $(seq 1 60); do
  if curl -sf -o /dev/null "http://${WHITE_HOST}:${SERVE_PORT}/${RUN_ID}/chunk0/_READY"; then
    echo "producer KV ready after ${i} checks"
    break
  fi
  sleep 3
done

# === Launch consumer on pearl ===
echo "=== Launching consumer on pearl ==="
SP=/home/stark/miniconda3/envs/vllm-rocm/lib/python3.11/site-packages
cons_cmd="cd /tmp && source /home/stark/miniconda3/etc/profile.d/conda.sh && conda activate vllm-rocm && \
  export LD_LIBRARY_PATH=${SP}/torch/lib:${SP}/_rocm_sdk_core/lib:${SP}/_rocm_sdk_core/lib/host-math/lib:${SP}/_rocm_sdk_core/lib/rocm_sysdeps/lib:${SP}/_rocm_sdk_devel/lib:${SP}/_rocm_sdk_devel/lib/host-math/lib:${SP}/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH:-} && \
  python ${PEARL_REPO}/scripts/cp_consumer.py \
    --model-dir ${PEARL_MODEL} --full-file ${TOK_DIR}/full.txt --prefix-len ${CHUNK_A} \
    --peer-url http://${WHITE_HOST}:${SERVE_PORT} --shared-path /tmp/hcp_cp_${RUN_ID} \
    --run-id ${RUN_ID} --gpu-mem 0.35 --decode ${DECODE}"
run_pearl "${cons_cmd}" >"${REPORT_DIR}/consumer.log" 2>&1 || true

echo "=== consumer log (tail) ==="
tail -25 "${REPORT_DIR}/consumer.log" || true
echo "=== producer log (tail) ==="
tail -8 "${REPORT_DIR}/producer.log" || true
kill "${PROD_PID}" 2>/dev/null || true
run_white "pkill -9 -f cp_producer.py || true" || true
echo "RUN_ID=${RUN_ID}  report=${REPORT_DIR}"
