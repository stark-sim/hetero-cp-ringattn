#!/bin/bash
set -euo pipefail
export PATH="/home/user/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
export LIBTORCH="$HOME/libtorch"
export LD_LIBRARY_PATH="$HOME/libtorch/lib:${LD_LIBRARY_PATH:-}"
BINARY="$HOME/hetero-cp-ringattn/rust/target/release/hcp-ringattn-rust"
MODEL="/home/user/models/qwen2-0.5b"

pkill -f "hcp-ringattn-rust --distributed-role" 2>/dev/null || true
sleep 1

HCP_TORCH_DEVICE=cuda:1 "$BINARY" --distributed-role worker \
    --domain-id 0 --seq-offset 0 --model-dir "$MODEL" \
    --listen-addr 127.0.0.1:9100 --next-peer-addr 127.0.0.1:9101 \
    --coordinator-addr 127.0.0.1:9000 --num-domains 2 \
    > /tmp/worker0_16k.log 2>&1 &

HCP_TORCH_DEVICE=cuda:2 "$BINARY" --distributed-role worker \
    --domain-id 1 --seq-offset 8192 --model-dir "$MODEL" \
    --listen-addr 127.0.0.1:9101 --next-peer-addr 127.0.0.1:9100 \
    --coordinator-addr 127.0.0.1:9000 --num-domains 2 \
    > /tmp/worker1_16k.log 2>&1 &

sleep 2

time "$BINARY" --distributed-role coordinator \
    --model-dir "$MODEL" \
    --prompt-file /tmp/prompt_16k.txt \
    --max-tokens 4 --temperature 0.0 --num-domains 2 \
    --listen-addr 127.0.0.1:9000 --chunk-sizes 8192,8192 \
    > /tmp/coord_16k.log 2>&1

echo "=== COORD ==="
cat /tmp/coord_16k.log | tail -10
echo "=== W0 ==="
cat /tmp/worker0_16k.log | tail -5
echo "=== W1 ==="
cat /tmp/worker1_16k.log | tail -5
