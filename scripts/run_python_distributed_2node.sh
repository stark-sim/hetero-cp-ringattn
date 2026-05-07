#!/usr/bin/env bash
# Cross-machine Python distributed inference: Mac (coordinator + worker 0) + GPU (worker 1)
#
# Usage:
#   bash scripts/run_python_distributed_2node.sh
#
# Environment:
#   MAC_ADDR     - Mac VPN address (default: auto-detect from 100.64.0.x)
#   GPU_ADDR     - GPU VPN address (default: 100.64.0.2)
#   GPU_USER     - GPU SSH user (default: stark)
#   MODEL_DIR    - Model path (default: models/Qwen2-0.5B)
#   PROMPT       - Prompt text (default: "Hello world")
#   MAX_TOKENS   - Max decode tokens (default: 3)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- Configuration ---
MAC_ADDR="${MAC_ADDR:-$(ifconfig | grep 'inet 100\.64\.' | awk '{print $2}' | head -n 1)}"
GPU_ADDR="${GPU_ADDR:-100.64.0.2}"
GPU_USER="${GPU_USER:-stark}"
MODEL_DIR="${MODEL_DIR:-models/Qwen2-0.5B}"
PROMPT="${PROMPT:-Hello world}"
MAX_TOKENS="${MAX_TOKENS:-3}"

COORD_PORT=26001
PEER_PORT0=26091
PEER_PORT1=26092

# Abs path for remote reference
MODEL_DIR_ABS="$(cd "$PROJECT_ROOT" && realpath "$MODEL_DIR")"

echo "=== Python Distributed 2-Node Smoke ==="
echo "Mac address:  $MAC_ADDR"
echo "GPU address:  $GPU_ADDR"
echo "Model dir:    $MODEL_DIR_ABS"
echo "Prompt:       $PROMPT"
echo "Max tokens:   $MAX_TOKENS"
echo ""

# --- Kill existing processes on exit ---
cleanup() {
    echo "[cleanup] stopping local processes..."
    kill ${COORD_PID:-} ${WORKER0_PID:-} 2>/dev/null || true
    ssh "${GPU_USER}@${GPU_ADDR}" "bash -lc 'pkill -f hcp_vllm_quic_worker'" 2>/dev/null || true
}
trap cleanup EXIT

# --- Start Coordinator (local Mac) ---
echo "[local] starting coordinator on 0.0.0.0:$COORD_PORT..."
DYLD_LIBRARY_PATH="/Users/stark_sim/libtorch/lib" \
cargo run --manifest-path "$PROJECT_ROOT/rust/Cargo.toml" \
    --features tch-backend --bin hcp-ringattn-rust -- \
    --distributed-role coordinator \
    --model-dir "$MODEL_DIR_ABS" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --num-domains 2 \
    --listen-addr "0.0.0.0:$COORD_PORT" \
    >"$PROJECT_ROOT/reports/coordinator.log" 2>&1 &
COORD_PID=$!
sleep 4

# --- Pre-flight: remote network check ---
echo "[remote] checking network access on $GPU_ADDR..."
NET_OK=$(ssh "${GPU_USER}@${GPU_ADDR}" "bash -lc 'curl -s -o /dev/null -w %{http_code} --max-time 5 https://huggingface.co || echo 000'" 2>/dev/null || echo "000")
if [ "$NET_OK" != "200" ]; then
    echo "WARNING: Remote GPU cannot reach HuggingFace (http_code=$NET_OK)."
    echo "         Model download will fail. Ensure remote has internet access"
    echo "         or pre-download the model to ~/hetero-cp-ringattn/models/Qwen2-0.5B"
fi

# --- Start Worker 1 on remote GPU via SSH ---
echo "[remote] starting worker 1 on $GPU_ADDR..."
ssh "${GPU_USER}@${GPU_ADDR}" "bash -lc '
    source /home/stark/miniconda3/etc/profile.d/conda.sh
    conda activate vllm
    cd ~/hetero-cp-ringattn
    python python/hcp_vllm_quic_worker.py \
        --model-dir Qwen/Qwen2-0.5B \
        --coordinator-host $MAC_ADDR \
        --coordinator-port $COORD_PORT \
        --domain-id 1 \
        --peer-listen-host 0.0.0.0 \
        --peer-listen-port $PEER_PORT1 \
        --next-peer-host $MAC_ADDR \
        --next-peer-port $PEER_PORT0 \
        > reports/worker1.log 2>&1 &
    echo worker1_pid=\$!
'" &
REMOTE_SSH_PID=$!
sleep 5

# --- Start Worker 0 on local Mac ---
echo "[local] starting worker 0 (TransformersBackend)..."
python python/hcp_transformers_quic_worker.py \
    --model-dir "$MODEL_DIR_ABS" \
    --coordinator-host 127.0.0.1 \
    --coordinator-port $COORD_PORT \
    --domain-id 0 \
    --num-domains 2 \
    --peer-listen-host 0.0.0.0 \
    --peer-listen-port $PEER_PORT0 \
    --next-peer-host "$GPU_ADDR" \
    --next-peer-port $PEER_PORT1 \
    >"$PROJECT_ROOT/reports/worker0.log" 2>&1 &
WORKER0_PID=$!

# --- Wait for completion ---
echo "[test] waiting for coordinator (PID $COORD_PID)..."
wait $COORD_PID
COORD_EXIT=$?

# Fetch remote worker log
ssh "${GPU_USER}@${GPU_ADDR}" "bash -lc 'cat ~/hetero-cp-ringattn/reports/worker1.log 2>/dev/null || echo no remote log'" > "$PROJECT_ROOT/reports/worker1.log.fetch" 2>/dev/null || true

echo ""
echo "=== Coordinator log ==="
cat "$PROJECT_ROOT/reports/coordinator.log" || true

echo ""
echo "=== Worker 0 log ==="
cat "$PROJECT_ROOT/reports/worker0.log" || true

echo ""
ssh "${GPU_USER}@${GPU_ADDR}" "bash -lc 'cat ~/hetero-cp-ringattn/reports/worker1.log 2>/dev/null || echo no remote log'" || true

if [ "$COORD_EXIT" -eq 0 ]; then
    echo ""
    echo "✅ Cross-machine Python distributed 2-node test PASSED"
else
    echo ""
    echo "❌ coordinator exited with code $COORD_EXIT"
    exit 1
fi
