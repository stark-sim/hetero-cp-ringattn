#!/bin/bash
# 551-token cross-node hetero test (Mac MPS + white RTX 4090)
set -euo pipefail

REPORT_DIR="/Users/stark_sim/VSCodeProjects/hetero-cp-ringattn/reports/cross-node-white-551-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$REPORT_DIR"

cd /Users/stark_sim/VSCodeProjects/hetero-cp-ringattn

BINARY="./rust/target/release/hcp-ringattn-rust"
MODEL_DIR="/Users/stark_sim/VSCodeProjects/hetero-cp-ringattn/models/Qwen2-0.5B"
PROMPT_FILE="/tmp/hcp_prompt_551.txt"

COORD_PORT=29500
W0_PORT=29501
W1_PORT=29502

MAC_ADDR=100.64.0.95
GPU_ADDR=100.64.0.2

# Cleanup
pkill -f 'hcp-ringattn-rust.*distributed-role' 2>/dev/null || true
ssh -o ConnectTimeout=5 stark@white "pkill -f 'hcp-ringattn-rust.*distributed-role' 2>/dev/null || true" 2>/dev/null || true
sleep 1

# Launch Coordinator
DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}" "$BINARY" --distributed-role coordinator \
    --model-dir "$MODEL_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --max-tokens 5 \
    --num-domains 2 \
    --listen-addr "0.0.0.0:${COORD_PORT}" \
    >"$REPORT_DIR/coordinator.log" 2>&1 &
COORD_PID=$!
sleep 2

# Launch Worker 1 (GPU, domain 1)
ssh -o ServerAliveInterval=60 stark@white "bash -lc 'cd hetero-cp-ringattn && HCP_TCH_DEVICE=cuda:0 PATH=/home/stark/.cargo/bin:\$PATH ./rust/target/release/hcp-ringattn-rust \
  --distributed-role worker \
  --domain-id 1 \
  --model-dir ~/models/Qwen2-0.5B \
  --listen-addr 0.0.0.0:${W1_PORT} \
  --next-peer-addr ${MAC_ADDR}:${W0_PORT} \
  --coordinator-addr ${MAC_ADDR}:${COORD_PORT} \
  --num-domains 2'" >"$REPORT_DIR/worker1.log" 2>&1 &
sleep 5

# Launch Worker 0 (Mac, domain 0)
DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib:${DYLD_LIBRARY_PATH:-}" HCP_TCH_DEVICE=mps "$BINARY" --distributed-role worker \
    --domain-id 0 \
    --model-dir "$MODEL_DIR" \
    --listen-addr "0.0.0.0:${W0_PORT}" \
    --next-peer-addr "${GPU_ADDR}:${W1_PORT}" \
    --coordinator-addr "127.0.0.1:${COORD_PORT}" \
    --num-domains 2 \
    >"$REPORT_DIR/worker0.log" 2>&1 &

# Wait for coordinator
wait "$COORD_PID" || true

# Cleanup
pkill -f 'hcp-ringattn-rust.*distributed-role' 2>/dev/null || true
ssh -o ConnectTimeout=5 stark@white "pkill -f 'hcp-ringattn-rust.*distributed-role' 2>/dev/null || true" 2>/dev/null || true

# Output summary
echo "=== SUMMARY ==="
echo "Report: $REPORT_DIR"
echo "--- Coordinator ---"
cat "$REPORT_DIR/coordinator.log"
echo "--- Worker 0 tail ---"
tail -15 "$REPORT_DIR/worker0.log"
echo "--- Worker 1 tail ---"
tail -15 "$REPORT_DIR/worker1.log"
