#!/usr/bin/env bash
# Local 2-node distributed inference smoke test.
# Runs 1 coordinator + 2 workers on loopback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_DIR/rust"

MODEL_DIR="${MODEL_DIR:-/Users/stark_sim/models/qwen2-0.5b}"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model dir not found: $MODEL_DIR"
    echo "Set MODEL_DIR env var or place Qwen2-0.5B next to project dir."
    exit 1
fi

# Ports
COORD_LISTEN="127.0.0.1:9000"
W0_LISTEN="127.0.0.1:9100"
W1_LISTEN="127.0.0.1:9101"

# Build first
echo "=== Building Rust binary ==="
cd "$RUST_DIR"
export LIBTORCH="${LIBTORCH:-/Users/stark_sim/libtorch}"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}"
cargo build --features tch-backend --release

BINARY="$RUST_DIR/target/release/hcp-ringattn-rust"

# Prompt
PROMPT="${PROMPT:-The answer to life, the universe, and everything is}"
MAX_TOKENS="${MAX_TOKENS:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# Cleanup function
cleanup() {
    echo "=== Cleaning up background jobs ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Launching workers ==="

# Worker 0
HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cpu}" "$BINARY" --distributed-role worker \
    --domain-id 0 \
    --seq-offset 0 \
    --model-dir "$MODEL_DIR" \
    --listen-addr "$W0_LISTEN" \
    --peer-addr "$W1_LISTEN" \
    --coordinator-addr "$COORD_LISTEN" \
    --num-domains 2 \
    > "$PROJECT_DIR/reports/worker0.log" 2>&1 &
W0_PID=$!
echo "Worker 0 PID: $W0_PID"

# Worker 1
HCP_TORCH_DEVICE="${HCP_TORCH_DEVICE:-cpu}" "$BINARY" --distributed-role worker \
    --domain-id 1 \
    --seq-offset 6 \
    --model-dir "$MODEL_DIR" \
    --listen-addr "$W1_LISTEN" \
    --peer-addr "$W0_LISTEN" \
    --coordinator-addr "$COORD_LISTEN" \
    --num-domains 2 \
    > "$PROJECT_DIR/reports/worker1.log" 2>&1 &
W1_PID=$!
echo "Worker 1 PID: $W1_PID"

# Give workers a moment to start listening
sleep 2

echo "=== Launching coordinator ==="
"$BINARY" --distributed-role coordinator \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --num-domains 2 \
    --listen-addr "$COORD_LISTEN" \
    > "$PROJECT_DIR/reports/coordinator.log" 2>&1

echo "=== Coordinator finished ==="
echo "=== Logs ==="
echo "--- Coordinator ---"
cat "$PROJECT_DIR/reports/coordinator.log"
echo "--- Worker 0 ---"
cat "$PROJECT_DIR/reports/worker0.log"
echo "--- Worker 1 ---"
cat "$PROJECT_DIR/reports/worker1.log"
