#!/bin/bash
set -e
cd /root/hetero-cp-ringattn
source run_a100_env.sh

MODEL_DIR=/root/hetero-cp-ringattn/models/Qwen2.5-7B-Instruct
PROMPT="The quick brown fox jumps over the lazy dog."
MAX_TOKENS=10
TEMP=0.0

echo "=== A100 single-node 7B validation ==="
echo "device=cuda:0"
echo "start_time=$(date -Iseconds)"
echo "log=/tmp/a100_single_node.log"

HCP_TORCH_DEVICE=cuda:0 ./rust/target/release/hcp-ringattn-rust \
  --infer-model-dir "$MODEL_DIR" \
  --infer-prompt "$PROMPT" \
  --infer-max-tokens $MAX_TOKENS \
  --infer-temperature $TEMP \
  --infer-num-domains 1 2>&1 | tee /tmp/a100_single_node.log

echo "=== finished at $(date -Iseconds) ==="
