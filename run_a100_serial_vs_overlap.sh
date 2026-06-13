#!/bin/bash
set -e
cd /root/hetero-cp-ringattn
source run_a100_env.sh

MODEL_DIR=/root/hetero-cp-ringattn/models/Qwen2.5-7B-Instruct
PROMPTS_FILE="${1:-/tmp/long_prompt_32k.txt}"
MAX_TOKENS="${2:-5}"

echo "=== A100 Serial vs Pipeline Overlap A/B ==="
echo "prompts_file=$PROMPTS_FILE"
echo "max_tokens=$MAX_TOKENS"
echo "start_time=$(date -Iseconds)"

# --- Serial baseline ---
echo ""
echo "[SERIAL] HCP_DISABLE_OVERLAP=1 start=$(date -Iseconds)"
export HCP_DISABLE_OVERLAP=1
export HCP_MICRO_KV_BLOCK_SIZE=0
bash run_a100_long_context.sh "$PROMPTS_FILE" "$MAX_TOKENS" 2>&1 | tee /tmp/a100_ab_serial.log
cp /tmp/a100_lc_coord.log /tmp/a100_ab_serial_coord.log
for i in 0 1 2 3; do cp /tmp/a100_lc_w${i}.log /tmp/a100_ab_serial_w${i}.log; done
echo "[SERIAL] end=$(date -Iseconds)"

# --- Pipeline overlap ---
echo ""
echo "[PIPELINE] HCP_DISABLE_OVERLAP=0 start=$(date -Iseconds)"
export HCP_DISABLE_OVERLAP=0
export HCP_MICRO_KV_BLOCK_SIZE=0
bash run_a100_long_context.sh "$PROMPTS_FILE" "$MAX_TOKENS" 2>&1 | tee /tmp/a100_ab_pipeline.log
cp /tmp/a100_lc_coord.log /tmp/a100_ab_pipeline_coord.log
for i in 0 1 2 3; do cp /tmp/a100_lc_w${i}.log /tmp/a100_ab_pipeline_w${i}.log; done
echo "[PIPELINE] end=$(date -Iseconds)"

echo ""
echo "=== A/B complete at $(date -Iseconds) ==="
echo "Serial log:    /tmp/a100_ab_serial.log"
echo "Pipeline log:  /tmp/a100_ab_pipeline.log"
