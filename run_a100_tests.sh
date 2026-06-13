#!/bin/bash
set -e
cd /root/hetero-cp-ringattn
source run_a100_env.sh

MODEL_DIR=/root/hetero-cp-ringattn/models/Qwen2.5-7B-Instruct
PROMPT="The quick brown fox jumps over the lazy dog."
MAX_TOKENS=10
TEMP=0.0

echo "=== A100 single-node 7B validation ==="
HCP_TORCH_DEVICE=cuda:0 ./rust/target/release/hcp-ringattn-rust \
  --infer-model-dir "$MODEL_DIR" \
  --infer-prompt "$PROMPT" \
  --infer-max-tokens $MAX_TOKENS \
  --infer-temperature $TEMP \
  --infer-num-domains 1

echo ""
echo "=== A100 4-domain distributed 7B validation ==="
bash run_a100_4domain.sh
