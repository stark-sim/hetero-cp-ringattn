#!/bin/bash
# Validate BF16 correctness on white (CUDA) and pearl (HIP)
# Run this script on each host separately, then compare results locally

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
HOSTNAME="${HOSTNAME:-$(hostname)}"
DEVICE="${HCP_TORCH_DEVICE:-cpu}"
MODEL_DIR="${MODEL_DIR:-models/Qwen2-0.5B}"
PROMPT="${PROMPT:-Hello, world!}"
MAX_TOKENS="${MAX_TOKENS:-1}"
RUN_ID="${RUN_ID:-bf16-hetero-$(hostname)-$(date +%Y%m%d-%H%M%S)}"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"

mkdir -p "${REPORT_DIR}"

echo "=== BF16 Hetero Validation: ${HOSTNAME} ==="
echo "RUN_ID=${RUN_ID}"
echo "DEVICE=${DEVICE}"
echo "MODEL=${MODEL_DIR}"
echo "PROMPT='${PROMPT}'"
echo ""

cd "${REPO_ROOT}/rust"

# Setup environment
if [ -n "${LIBTORCH:-}" ]; then
    export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}"
fi

# Build if needed
echo "[1/4] Building Rust binary..."
export PATH="${HOME}/.cargo/bin:${PATH}"
cargo build --release --features tch-backend --bin hcp-ringattn-rust 2>&1 | tail -3

# Run BF16 inference and export logits
echo ""
echo "[2/4] Running BF16 inference..."
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE="${DEVICE}" \
    "${REPO_ROOT}/rust/target/release/hcp-ringattn-rust" \
    --infer-model-dir "${MODEL_DIR}" \
    --infer-prompt "${PROMPT}" \
    --infer-max-tokens "${MAX_TOKENS}" \
    --infer-temperature 0.0 \
    --export-logits "${REPORT_DIR}" \
    2>&1 | tee "${REPORT_DIR}/inference.log"

# Also run Python BF16 baseline for comparison
echo ""
echo "[3/4] Running Python BF16 baseline..."
source "${REPO_ROOT}/.venv/bin/activate" 2>/dev/null || true
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import struct
import numpy as np

model = AutoModelForCausalLM.from_pretrained('${MODEL_DIR}', torch_dtype=torch.bfloat16, device_map='cpu')
tokenizer = AutoTokenizer.from_pretrained('${MODEL_DIR}')
model.eval()

inputs = tokenizer('${PROMPT}', return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].to(torch.float32).numpy()

# Save in same format as Rust
vocab_size = len(logits)
with open('${REPORT_DIR}/python_bf16_logits.bin', 'wb') as f:
    f.write(struct.pack('<Q', vocab_size))
    f.write(struct.pack('<Q', 1))  # 1 chunk
    f.write(logits.astype(np.float32).tobytes())

# Print top-5 tokens
probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
top5 = torch.topk(probs, 5)
print('Python BF16 Top-5:')
for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
    token = tokenizer.decode([idx.item()])
    print(f'  {i+1}. token={idx.item()} -> {token!r} (prob={prob.item():.4f})')
" 2>&1 | tee -a "${REPORT_DIR}/inference.log"

# Compare Rust vs Python
echo ""
echo "[4/4] Comparing Rust vs Python BF16 logits..."
python3 "${REPO_ROOT}/scripts/compare_logits.py" \
    "${REPORT_DIR}/python_bf16_logits.bin" \
    "${REPORT_DIR}/logits.bin" \
    --atol 0.01 \
    --verbose \
    2>&1 | tee "${REPORT_DIR}/comparison.log"

echo ""
echo "=== Validation Complete ==="
echo "Report: ${REPORT_DIR}"
