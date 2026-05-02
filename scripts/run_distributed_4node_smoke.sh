#!/bin/env bash
# Local 4-node distributed inference smoke test.
# Runs 1 coordinator + 4 workers on loopback.
# Supports --capacity-aware and long prompt files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_DIR/rust"

MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model dir not found: $MODEL_DIR"
    exit 1
fi

# Ports
COORD_LISTEN="127.0.0.1:9000"
W0_LISTEN="127.0.0.1:9100"
W1_LISTEN="127.0.0.1:9101"
W2_LISTEN="127.0.0.1:9102"
W3_LISTEN="127.0.0.1:9103"

# Build first
echo "=== Building Rust binary ==="
cd "$RUST_DIR"
export LIBTORCH="${LIBTORCH:-/Users/stark_sim/libtorch}"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}"
cargo build --features tch-backend --release

BINARY="$RUST_DIR/target/release/hcp-ringattn-rust"

# Prompt
PROMPT="${PROMPT:-The answer to life, the universe, and everything is}"
PROMPT_FILE="${PROMPT_FILE:-}"
MAX_TOKENS="${MAX_TOKENS:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DOMAINS=4

# Coordinator flags
COORD_FLAGS=""
if [[ "${CAPACITY_AWARE:-}" == "1" ]]; then
    COORD_FLAGS="$COORD_FLAGS --capacity-aware"
fi

# Cleanup function
cleanup() {
    echo "=== Cleaning up background jobs ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Launching 4 workers ==="

# Worker 0
HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cpu}" "$BINARY" --distributed-role worker \
    --domain-id 0 \
    --seq-offset 0 \
    --model-dir "$MODEL_DIR" \
    --listen-addr "$W0_LISTEN" \
    --next-peer-addr "$W1_LISTEN" \
    --coordinator-addr "$COORD_LISTEN" \
    --num-domains $NUM_DOMAINS \
    > "$PROJECT_DIR/reports/worker0.log" 2>&1 &
echo "Worker 0 launched"

# Worker 1
HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cpu}" "$BINARY" --distributed-role worker \
    --domain-id 1 \
    --seq-offset 0 \
    --model-dir "$MODEL_DIR" \
    --listen-addr "$W1_LISTEN" \
    --next-peer-addr "$W2_LISTEN" \
    --coordinator-addr "$COORD_LISTEN" \
    --num-domains $NUM_DOMAINS \
    > "$PROJECT_DIR/reports/worker1.log" 2>&1 &
echo "Worker 1 launched"

# Worker 2
HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cpu}" "$BINARY" --distributed-role worker \
    --domain-id 2 \
    --seq-offset 0 \
    --model-dir "$MODEL_DIR" \
    --listen-addr "$W2_LISTEN" \
    --next-peer-addr "$W3_LISTEN" \
    --coordinator-addr "$COORD_LISTEN" \
    --num-domains $NUM_DOMAINS \
    > "$PROJECT_DIR/reports/worker2.log" 2>&1 &
echo "Worker 2 launched"

# Worker 3
HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cpu}" "$BINARY" --distributed-role worker \
    --domain-id 3 \
    --seq-offset 0 \
    --model-dir "$MODEL_DIR" \
    --listen-addr "$W3_LISTEN" \
    --next-peer-addr "$W0_LISTEN" \
    --coordinator-addr "$COORD_LISTEN" \
    --num-domains $NUM_DOMAINS \
    > "$PROJECT_DIR/reports/worker3.log" 2>&1 &
echo "Worker 3 launched"

# Give workers a moment to start listening
sleep 4

echo "=== Launching coordinator ==="
if [[ -n "$PROMPT_FILE" ]]; then
    "$BINARY" --distributed-role coordinator \
        --model-dir "$MODEL_DIR" \
        --prompt-file "$PROMPT_FILE" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --num-domains $NUM_DOMAINS \
        --listen-addr "$COORD_LISTEN" \
        $COORD_FLAGS \
        > "$PROJECT_DIR/reports/coordinator.log" 2>&1
else
    "$BINARY" --distributed-role coordinator \
        --model-dir "$MODEL_DIR" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --num-domains $NUM_DOMAINS \
        --listen-addr "$COORD_LISTEN" \
        $COORD_FLAGS \
        > "$PROJECT_DIR/reports/coordinator.log" 2>&1
fi

echo "=== Coordinator finished ==="
echo "=== Logs ==="
echo "--- Coordinator ---"
cat "$PROJECT_DIR/reports/coordinator.log"
echo "--- Worker 0 ---"
cat "$PROJECT_DIR/reports/worker0.log"
echo "--- Worker 1 ---"
cat "$PROJECT_DIR/reports/worker1.log"
echo "--- Worker 2 ---"
cat "$PROJECT_DIR/reports/worker2.log"
echo "--- Worker 3 ---"
cat "$PROJECT_DIR/reports/worker3.log"
