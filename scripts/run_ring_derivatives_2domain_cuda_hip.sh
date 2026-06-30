#!/bin/bash
# Cross-node 2-domain Ring Attention derivatives comparison:
# white (RTX 4090 CUDA) + pearl (RX 9060 XT HIP)
# Tests Vanilla / Striped / ZigZag scheduling strategies on real hardware.
#
# Usage:
#   bash scripts/run_ring_derivatives_2domain_cuda_hip.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# === Configuration ===
WHITE_HOST="${WHITE_HOST:-100.118.253.68}"
WHITE_USER="${WHITE_USER:-stark}"
WHITE_SSH="${WHITE_USER}@${WHITE_HOST}"
WHITE_REPO_DIR="${WHITE_REPO_DIR:-hetero-cp-ringattn}"

PEARL_HOST="${PEARL_HOST:-100.111.242.55}"
PEARL_USER="${PEARL_USER:-stark}"
PEARL_SSH="${PEARL_USER}@${PEARL_HOST}"
PEARL_REPO_DIR="${PEARL_REPO_DIR:-hetero-cp-ringattn}"

WHITE_ADDR="${WHITE_ADDR:-${WHITE_HOST}}"
PEARL_ADDR="${PEARL_ADDR:-${PEARL_HOST}}"

MODEL_DIR="${MODEL_DIR:-/home/stark/models/Qwen2-0.5B-1M}"

SEQ_LEN="${SEQ_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"
STRATEGIES="${STRATEGIES:-vanilla striped zigzag}"
CHUNK_SIZES="${CHUNK_SIZES:-}"
# If CHUNK_SIZES is set, pass --chunk-sizes to the coordinator to override
# the default capacity-aware split.  Use e.g. CHUNK_SIZES=2048,2048 for 1:1.
CHUNK_SIZES_ARG=""
if [ -n "${CHUNK_SIZES}" ]; then
    CHUNK_SIZES_ARG="--chunk-sizes ${CHUNK_SIZES}"
fi

BASE_PORT="${BASE_PORT:-29600}"
RUN_ID="ring-derivatives-$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

echo "=== Ring Attention Derivatives: white CUDA + pearl HIP ==="
echo "RUN_ID=${RUN_ID}"
echo "WHITE=${WHITE_ADDR} (CUDA) | PEARL=${PEARL_ADDR} (HIP)"
echo "MODEL=${MODEL_DIR}"
echo "SEQ_LEN=${SEQ_LEN}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "STRATEGIES=${STRATEGIES}"
echo "Reports: ${REPORT_DIR}"

# === Helpers ===
# Upload a local script to /tmp on the remote host and run it synchronously.
remote_run() {
    local ssh_target="$1"
    local local_script="$2"
    local remote_name="$3"
    scp -o ConnectTimeout=30 -o BatchMode=yes "${local_script}" "${ssh_target}:/tmp/${remote_name}" >/dev/null
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 -o BatchMode=yes "${ssh_target}" "bash /tmp/${remote_name}"
}

# Upload a local script to /tmp on the remote host and run it in the background
# via nohup.  stdout/stderr go to remote_log.
remote_nohup() {
    local ssh_target="$1"
    local local_script="$2"
    local remote_name="$3"
    local remote_log="$4"
    scp -o ConnectTimeout=30 -o BatchMode=yes "${local_script}" "${ssh_target}:/tmp/${remote_name}" >/dev/null
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 -o BatchMode=yes -f "${ssh_target}" \
        "nohup bash /tmp/${remote_name} > $(printf '%q' "${remote_log}") 2>&1 &"
}

# === Preflight: build binaries on both hosts ===
echo "=== Building release binaries on white ==="
cat > "${REPORT_DIR}/build_white.sh" <<EOF
export PATH="\$HOME/.cargo/bin:\$PATH"
export LIBTORCH="\$HOME/libtorch"
export LD_LIBRARY_PATH="\$HOME/libtorch/lib:\${LD_LIBRARY_PATH:-}"
cd ${WHITE_REPO_DIR}
git pull --ff-only
cd rust
cargo build --release --features tch-backend
EOF
remote_run "${WHITE_SSH}" "${REPORT_DIR}/build_white.sh" "build_white_${RUN_ID}.sh" | tee "${REPORT_DIR}/build_white.log"

echo "=== Building release binaries on pearl ==="
cat > "${REPORT_DIR}/build_pearl.sh" <<EOF
export PATH="\$HOME/.cargo/bin:\$PATH"
export LIBTORCH="\$HOME/libtorch"
export LD_LIBRARY_PATH="\$HOME/libtorch/lib:\${LD_LIBRARY_PATH:-}"
cd ${PEARL_REPO_DIR}
git pull --ff-only
cd rust
cargo build --release --features tch-backend
EOF
remote_run "${PEARL_SSH}" "${REPORT_DIR}/build_pearl.sh" "build_pearl_${RUN_ID}.sh" | tee "${REPORT_DIR}/build_pearl.log"

# === Generate prompt ===
PROMPT_FILE="/tmp/hcp_prompt_${RUN_ID}.txt"
python3 -c "import sys; tok=sys.argv[1]; n=int(sys.argv[2]); print(' '.join([tok]*n))" "the" "${SEQ_LEN}" > "${PROMPT_FILE}"
scp -o ConnectTimeout=30 "${PROMPT_FILE}" "${WHITE_SSH}:~/hcp_prompt_${RUN_ID}.txt" >/dev/null 2>&1 || true
scp -o ConnectTimeout=30 "${PROMPT_FILE}" "${PEARL_SSH}:~/hcp_prompt_${RUN_ID}.txt" >/dev/null 2>&1 || true

# === Run each strategy ===
for STRATEGY in ${STRATEGIES}; do
    STRAT_REPORT="${REPORT_DIR}/${STRATEGY}"
    mkdir -p "${STRAT_REPORT}"
    COORD_PORT=$((BASE_PORT))
    W0_PORT=$((BASE_PORT + 1))
    W1_PORT=$((BASE_PORT + 2))
    BASE_PORT=$((BASE_PORT + 10))

    echo ""
    echo "=== Strategy: ${STRATEGY} (coord=${COORD_PORT}, w0=${W0_PORT}, w1=${W1_PORT}) ==="

    # Cleanup any previous processes
    cat > "${REPORT_DIR}/cleanup.sh" <<EOF
pkill -f 'hcp-ringattn-rust --distributed-role' || true
EOF
    remote_run "${WHITE_SSH}" "${REPORT_DIR}/cleanup.sh" "cleanup_${RUN_ID}.sh" || true
    remote_run "${PEARL_SSH}" "${REPORT_DIR}/cleanup.sh" "cleanup_${RUN_ID}.sh" || true
    sleep 3

    # Worker 0 (white, domain 0) - background
    cat > "${STRAT_REPORT}/worker0.sh" <<EOF
export PATH="\$HOME/.cargo/bin:\$PATH"
export LIBTORCH="\$HOME/libtorch"
cd ${WHITE_REPO_DIR}
export HCP_TCH_DEVICE=cuda:0
export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-}
export HCP_PERF_LOG=/tmp/perf_${RUN_ID}_${STRATEGY}_w0.jsonl
./rust/target/release/hcp-ringattn-rust \
  --distributed-role worker \
  --domain-id 0 \
  --model-dir ${MODEL_DIR} \
  --listen-addr 0.0.0.0:${W0_PORT} \
  --next-peer-addr ${PEARL_ADDR}:${W1_PORT} \
  --coordinator-addr 127.0.0.1:${COORD_PORT} \
  --num-domains 2
EOF
    remote_nohup "${WHITE_SSH}" "${STRAT_REPORT}/worker0.sh" "worker0_${RUN_ID}_${STRATEGY}.sh" "/tmp/hcp_w0_${RUN_ID}_${STRATEGY}.log"

    # Worker 1 (pearl, domain 1) - background
    cat > "${STRAT_REPORT}/worker1.sh" <<EOF
export PATH="\$HOME/.cargo/bin:\$PATH"
export LIBTORCH="\$HOME/libtorch"
cd ${PEARL_REPO_DIR}
export LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so
export HCP_TCH_DEVICE=cuda:0
export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-}
export HCP_PERF_LOG=/tmp/perf_${RUN_ID}_${STRATEGY}_w1.jsonl
./rust/target/release/hcp-ringattn-rust \
  --distributed-role worker \
  --domain-id 1 \
  --model-dir ${MODEL_DIR} \
  --listen-addr 0.0.0.0:${W1_PORT} \
  --next-peer-addr ${WHITE_ADDR}:${W0_PORT} \
  --coordinator-addr ${WHITE_ADDR}:${COORD_PORT} \
  --num-domains 2
EOF
    remote_nohup "${PEARL_SSH}" "${STRAT_REPORT}/worker1.sh" "worker1_${RUN_ID}_${STRATEGY}.sh" "/tmp/hcp_w1_${RUN_ID}_${STRATEGY}.log"

    sleep 5

    # Coordinator (white) - foreground, we wait for it
    cat > "${STRAT_REPORT}/coordinator.sh" <<EOF
export PATH="\$HOME/.cargo/bin:\$PATH"
export LIBTORCH="\$HOME/libtorch"
cd ${WHITE_REPO_DIR}
export HCP_TCH_DEVICE=cuda:0
export LD_LIBRARY_PATH=/home/stark/libtorch/lib:\${LD_LIBRARY_PATH:-}
./rust/target/release/hcp-ringattn-rust \
  --distributed-role coordinator \
  --model-dir ${MODEL_DIR} \
  --prompt-file ~/hcp_prompt_${RUN_ID}.txt \
  --max-tokens ${MAX_NEW_TOKENS} \
  --num-domains 2 \
  --listen-addr 0.0.0.0:${COORD_PORT} \
  --ring-strategy ${STRATEGY} ${CHUNK_SIZES_ARG}
EOF
    remote_run "${WHITE_SSH}" "${STRAT_REPORT}/coordinator.sh" "coordinator_${RUN_ID}_${STRATEGY}.sh" >"${STRAT_REPORT}/coordinator.log" 2>&1 || true

    # Fetch worker logs
    scp -o ConnectTimeout=30 "${WHITE_SSH}:/tmp/hcp_w0_${RUN_ID}_${STRATEGY}.log" "${STRAT_REPORT}/worker0.log" >/dev/null 2>&1 || true
    scp -o ConnectTimeout=30 "${PEARL_SSH}:/tmp/hcp_w1_${RUN_ID}_${STRATEGY}.log" "${STRAT_REPORT}/worker1.log" >/dev/null 2>&1 || true
    scp -o ConnectTimeout=30 "${WHITE_SSH}:/tmp/perf_${RUN_ID}_${STRATEGY}_w0.jsonl" "${STRAT_REPORT}/perf_w0.jsonl" >/dev/null 2>&1 || true
    scp -o ConnectTimeout=30 "${PEARL_SSH}:/tmp/perf_${RUN_ID}_${STRATEGY}_w1.jsonl" "${STRAT_REPORT}/perf_w1.jsonl" >/dev/null 2>&1 || true

    # Capture summary
    {
        echo "--- ${STRATEGY} output ---"
        grep -E "(generated|capacity|Prefill done|Decode done|Output)" "${STRAT_REPORT}/coordinator.log" | tail -5 || true
        grep -E "(capacity|Prefill done|Decode done)" "${STRAT_REPORT}/worker0.log" | tail -5 || true
        grep -E "(capacity|Prefill done|Decode done)" "${STRAT_REPORT}/worker1.log" | tail -5 || true
        echo ""
    } >> "${REPORT_DIR}/summary.txt"

    # Cleanup
    remote_run "${WHITE_SSH}" "${REPORT_DIR}/cleanup.sh" "cleanup_${RUN_ID}.sh" || true
    remote_run "${PEARL_SSH}" "${REPORT_DIR}/cleanup.sh" "cleanup_${RUN_ID}.sh" || true
    sleep 3
done

echo ""
echo "=== All strategies completed ==="
cat "${REPORT_DIR}/summary.txt"
echo "Reports: ${REPORT_DIR}"
