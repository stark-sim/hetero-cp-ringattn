#!/bin/bash
set -e
cd /root/hetero-cp-ringattn
source run_a100_env.sh

MODEL_DIR=/root/hetero-cp-ringattn/models/Qwen2.5-7B-Instruct
PROMPTS_FILE="${1:-/tmp/long_prompts.txt}"
MAX_TOKENS="${2:-5}"

# Pipeline overlap enabled by default; set HCP_DISABLE_OVERLAP=1 externally for serial baseline
export HCP_DISABLE_OVERLAP="${HCP_DISABLE_OVERLAP:-0}"
export HCP_MICRO_KV_BLOCK_SIZE="${HCP_MICRO_KV_BLOCK_SIZE:-0}"

echo "=== A100 long-context test ==="
echo "prompts_file=$PROMPTS_FILE"
echo "max_tokens=$MAX_TOKENS"
echo "HCP_DISABLE_OVERLAP=$HCP_DISABLE_OVERLAP"
echo "HCP_MICRO_KV_BLOCK_SIZE=$HCP_MICRO_KV_BLOCK_SIZE"
echo "start_time=$(date -Iseconds)"

pkill -f hcp-ringattn-rust 2>/dev/null || true
sleep 2

rm -f /tmp/a100_lc_coord.pid
nohup ./rust/target/release/hcp-ringattn-rust --distributed-role coordinator \
  --listen-addr 0.0.0.0:9000 --num-domains 4 \
  --worker-addrs 127.0.0.1:9100,127.0.0.1:9101,127.0.0.1:9102,127.0.0.1:9103 \
  --model-dir "$MODEL_DIR" --prompts-file "$PROMPTS_FILE" \
  --max-tokens $MAX_TOKENS --temperature 0.0 > /tmp/a100_lc_coord.log 2>&1 &
COORD_PID=$!
echo $COORD_PID > /tmp/a100_lc_coord.pid
echo "Coordinator PID=$COORD_PID"
sleep 3

for domain in 0 1 2 3; do
  port=$((9100 + domain))
  next_port=$((9100 + (domain + 1) % 4))
  HCP_TORCH_DEVICE=cuda:$domain nohup ./rust/target/release/hcp-ringattn-rust \
    --distributed-role worker --domain-id $domain \
    --model-dir "$MODEL_DIR" --listen-addr 0.0.0.0:$port \
    --next-peer-addr 127.0.0.1:$next_port \
    --coordinator-addr 127.0.0.1:9000 --num-domains 4 \
    > /tmp/a100_lc_w${domain}.log 2>&1 &
  echo "Worker $domain started on cuda:$domain, PID=$!"
  sleep 1
done

echo "Waiting for coordinator $COORD_PID to finish..."
wait $COORD_PID
EXIT_CODE=$?
echo "Coordinator exited with code $EXIT_CODE at $(date -Iseconds)"
cat /tmp/a100_lc_coord.log
exit $EXIT_CODE
