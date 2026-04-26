#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-rust-remote-cp-3node-local}"
GPU_HOST="${GPU_HOST:-192.168.8.172}"
GPU_USER="${GPU_USER:-stark}"
GPU_SSH="${GPU_SSH:-${GPU_USER}@${GPU_HOST}}"
GPU_REPO_DIR="${GPU_REPO_DIR:-hetero-cp-ringattn}"
PORT_BASE="${PORT_BASE:-29250}"
GPU_PORT="${GPU_PORT:-${PORT_BASE}}"
NODE0_PORT="${NODE0_PORT:-$((PORT_BASE + 1))}"
NODE2_PORT="${NODE2_PORT:-$((PORT_BASE + 2))}"
LOCAL_CARGO_OFFLINE="${LOCAL_CARGO_OFFLINE:-0}"
REMOTE_CARGO_OFFLINE="${REMOTE_CARGO_OFFLINE:-0}"
LOCAL_TORCH_DEVICE="${LOCAL_TORCH_DEVICE:-mps}"
GPU_TORCH_DEVICE="${GPU_TORCH_DEVICE:-cuda:0}"
HCP_ENABLE_TORCH="${HCP_ENABLE_TORCH:-1}"
LOCAL_PREFLIGHT_BUILD="${LOCAL_PREFLIGHT_BUILD:-1}"
REMOTE_PREFLIGHT_BUILD="${REMOTE_PREFLIGHT_BUILD:-1}"
REMOTE_CARGO_PATH="${REMOTE_CARGO_PATH:-/home/stark/.cargo/bin}"
REMOTE_LIBTORCH="${REMOTE_LIBTORCH:-/home/stark/libtorch}"
REMOTE_LIBTORCH_INCLUDE="${REMOTE_LIBTORCH_INCLUDE:-${REMOTE_LIBTORCH}/include}"
REMOTE_LIBTORCH_LIB="${REMOTE_LIBTORCH_LIB:-${REMOTE_LIBTORCH}/lib}"

if [ -z "${MAC_192_ADDR:-}" ]; then
    MAC_192_ADDR="$(ifconfig | awk '/inet 192\.168\.8\./ { print $2; exit }')"
fi
if [ -z "${MAC_192_ADDR}" ]; then
    echo "Could not find a local 192.168.8.x address. Set MAC_192_ADDR explicitly." >&2
    exit 1
fi

REPORT_DIR="${REPO_ROOT}/reports/${RUN_ID}"
mkdir -p "${REPORT_DIR}"

shell_quote() {
    printf "'"
    printf "%s" "$1" | sed "s/'/'\\\\''/g"
    printf "'"
}

cargo_build_args() {
    if [ "$1" = "1" ]; then
        printf "build --offline"
    else
        printf "build"
    fi
}

remote_env_exports() {
    local cargo_path_q libtorch_q libtorch_include_q libtorch_lib_q
    cargo_path_q="$(shell_quote "${REMOTE_CARGO_PATH}")"
    libtorch_q="$(shell_quote "${REMOTE_LIBTORCH}")"
    libtorch_include_q="$(shell_quote "${REMOTE_LIBTORCH_INCLUDE}")"
    libtorch_lib_q="$(shell_quote "${REMOTE_LIBTORCH_LIB}")"
    printf "export PATH=%s:\$PATH; " "${cargo_path_q}"
    printf "export LIBTORCH=%s; " "${libtorch_q}"
    printf "export LIBTORCH_INCLUDE=%s; " "${libtorch_include_q}"
    printf "export LIBTORCH_LIB=%s; " "${libtorch_lib_q}"
    printf "export LD_LIBRARY_PATH=%s:\${LD_LIBRARY_PATH:-}; " "${libtorch_lib_q}"
}

run_remote() {
    local command="$1"
    ssh "${GPU_SSH}" "bash -lc $(shell_quote "${command}")"
}

echo "=== HCP Rust Remote CP 3-Node Smoke ==="
echo "RUN_ID=${RUN_ID}"
echo "MAC_192_ADDR=${MAC_192_ADDR}"
echo "GPU_HOST=${GPU_HOST}"
echo "node0: bind=0.0.0.0:${NODE0_PORT} connect=${GPU_HOST}:${GPU_PORT} device=${LOCAL_TORCH_DEVICE}"
echo "node1: bind=0.0.0.0:${GPU_PORT} connect=${MAC_192_ADDR}:${NODE2_PORT} device=${GPU_TORCH_DEVICE}"
echo "node2: bind=0.0.0.0:${NODE2_PORT} connect=${MAC_192_ADDR}:${NODE0_PORT} device=${LOCAL_TORCH_DEVICE}"
echo "Reports: ${REPORT_DIR}"

if [ "${LOCAL_PREFLIGHT_BUILD}" = "1" ]; then
    echo "Preflight local cargo build"
    (
        cd "${REPO_ROOT}/rust"
        cargo $(cargo_build_args "${LOCAL_CARGO_OFFLINE}")
    )
fi

if [ "${REMOTE_PREFLIGHT_BUILD}" = "1" ]; then
    echo "Preflight remote git pull and cargo build"
    remote_repo_q="$(shell_quote "${GPU_REPO_DIR}")"
    remote_build_args="$(cargo_build_args "${REMOTE_CARGO_OFFLINE}")"
    remote_command="$(remote_env_exports) cd ${remote_repo_q} && git pull --ff-only && cd rust && cargo ${remote_build_args}"
    run_remote "${remote_command}" 2>&1 | tee "${REPORT_DIR}/remote_preflight.log"
else
    echo "Preflight remote git pull"
    remote_repo_q="$(shell_quote "${GPU_REPO_DIR}")"
    remote_command="$(remote_env_exports) cd ${remote_repo_q} && git pull --ff-only"
    run_remote "${remote_command}" 2>&1 | tee "${REPORT_DIR}/remote_preflight.log"
fi

pids=()
names=()

start_local_node() {
    local node_index="$1"
    local bind_port="$2"
    local connect_addr="$3"
    local log_path="${REPORT_DIR}/launch_node_${node_index}.log"
    (
        cd "${REPO_ROOT}"
        RUN_ID="${RUN_ID}" \
            HCP_REMOTE_CP_DOMAINS=3 \
            NODE_INDEX="${node_index}" \
            BIND_ADDR="0.0.0.0:${bind_port}" \
            CONNECT_ADDR="${connect_addr}" \
            CARGO_OFFLINE="${LOCAL_CARGO_OFFLINE}" \
            HCP_ENABLE_TORCH="${HCP_ENABLE_TORCH}" \
            HCP_TORCH_DEVICE="${LOCAL_TORCH_DEVICE}" \
            bash scripts/run_rust_remote_cp_node.sh
    ) >"${log_path}" 2>&1 &
    local pid="$!"
    pids+=("${pid}")
    names+=("node${node_index}")
    echo "Started node${node_index} pid=${pid} log=${log_path}"
}

start_remote_node() {
    local log_path="${REPORT_DIR}/launch_node_1.log"
    local remote_repo_q remote_run_id_q remote_bind_q remote_connect_q remote_cargo_offline_q
    local remote_enable_torch_q remote_torch_device_q remote_command
    remote_repo_q="$(shell_quote "${GPU_REPO_DIR}")"
    remote_run_id_q="$(shell_quote "${RUN_ID}")"
    remote_bind_q="$(shell_quote "0.0.0.0:${GPU_PORT}")"
    remote_connect_q="$(shell_quote "${MAC_192_ADDR}:${NODE2_PORT}")"
    remote_cargo_offline_q="$(shell_quote "${REMOTE_CARGO_OFFLINE}")"
    remote_enable_torch_q="$(shell_quote "${HCP_ENABLE_TORCH}")"
    remote_torch_device_q="$(shell_quote "${GPU_TORCH_DEVICE}")"
    remote_command="$(remote_env_exports) cd ${remote_repo_q} && RUN_ID=${remote_run_id_q} HCP_REMOTE_CP_DOMAINS=3 NODE_INDEX=1 BIND_ADDR=${remote_bind_q} CONNECT_ADDR=${remote_connect_q} CARGO_OFFLINE=${remote_cargo_offline_q} HCP_ENABLE_TORCH=${remote_enable_torch_q} HCP_TORCH_DEVICE=${remote_torch_device_q} bash scripts/run_rust_remote_cp_node.sh"
    run_remote "${remote_command}" >"${log_path}" 2>&1 &
    local pid="$!"
    pids+=("${pid}")
    names+=("node1")
    echo "Started node1 pid=${pid} log=${log_path}"
}

cleanup() {
    local status="$1"
    if [ "${status}" -ne 0 ]; then
        for pid in "${pids[@]}"; do
            kill "${pid}" >/dev/null 2>&1 || true
        done
    fi
}

trap 'status=$?; cleanup "${status}"; exit "${status}"' INT TERM EXIT

start_local_node 0 "${NODE0_PORT}" "${GPU_HOST}:${GPU_PORT}"
start_local_node 2 "${NODE2_PORT}" "${MAC_192_ADDR}:${NODE0_PORT}"
start_remote_node

overall_status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "${names[$index]} exited successfully"
    else
        node_status="$?"
        echo "${names[$index]} failed with status ${node_status}" >&2
        overall_status=1
    fi
done

trap - INT TERM EXIT

echo "=== HCP Rust Remote CP 3-Node Summary ==="
for node_index in 0 1 2; do
    log_path="${REPORT_DIR}/launch_node_${node_index}.log"
    if [ -f "${log_path}" ]; then
        grep -E "\\[rust-remote-cp-node\\]" "${log_path}" || true
    fi
done

if [ "${overall_status}" -ne 0 ]; then
    echo "=== HCP Rust Remote CP 3-Node Failed ===" >&2
    exit "${overall_status}"
fi

echo "=== HCP Rust Remote CP 3-Node Done ==="
