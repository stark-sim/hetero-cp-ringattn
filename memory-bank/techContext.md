# 技术上下文

## 技术栈

### Core

- C++17
- CMake 3.16+
- Rust 2021
- Cargo
- Python 3
- NumPy，用于 Python correctness / kernel stub 原型
- PyTorch / libtorch 2.11.0 可通过独立 libtorch 或 Python 安装提供的 headers/libs 接入 C++ ATen bridge

### Styling

- 不适用。本项目不是前端 UI 仓库。

### State & Data

- C++ core 使用 `Status`、`TensorDType`、`BoundaryTensor`、`RingAttnConfig` 等轻量结构体。
- Python 原型使用 NumPy array 表达 attention 输入和 online softmax state。
- 实验输出写入 `reports/<RUN_ID>/`；`reports/**/*.json` / `reports/**/*.log` 默认视为生成产物并被 git 忽略。

### Testing

- 当前主要验证入口是 smoke 脚本：
  - C++ coordinator smoke
  - Rust Ring Attention correctness smoke
  - Rust -> C++ runtime bridge smoke
  - 可选 Rust -> C++ ATen/libtorch smoke
  - Rust CP update-driven ATen/libtorch block compute smoke
  - Rust CP payload-backed ATen/libtorch block compute smoke
  - Rust `DomainModelState` unit tests for Q/K/V state ownership and K/V block slicing
  - Rust remote P2P pair smoke
  - Python controller / worker placeholder smoke 仅作历史占位
- 后续应增加 online softmax correctness script / report。

### Dev Tools

- `cmake`
- `cmake --build`
- `rustc`
- `cargo`
- `bash`
- `python3`

## 开发环境

### 前置条件

- C++17 编译器。
- CMake >= 3.16。
- Rust / Cargo。
- Python 3。
- Python smoke 需要 NumPy。
- 可选 PyTorch C++ bridge 优先使用独立 libtorch：设置 `LIBTORCH`，或显式设置 `LIBTORCH_INCLUDE` / `LIBTORCH_LIB`。
- 如果没有独立 libtorch，才 fallback 到 Python 环境中的 `torch.utils.cpp_extension` 发现 include/lib path。
- 当前验证环境：`torch==2.11.0`、`torchvision==0.26.0`、`torchaudio==2.11.0`。

### 设置与验证命令

```bash
cmake -S . -B build
cmake --build build --target ringattn_coordinator_smoke -j4
./build/ringattn_coordinator_smoke
```

```bash
bash scripts/run_local_ringattn_smoke.sh
```

如果当前环境不允许本地端口绑定，或只想验证 C++ skeleton：

```bash
SKIP_PYTHON_SMOKE=1 bash scripts/run_local_ringattn_smoke.sh
```

Rust + C++ smoke：

```bash
bash scripts/run_rust_ringattn_smoke.sh
```

Rust + C++ + ATen/libtorch smoke：

```bash
HCP_ENABLE_TORCH=1 bash scripts/run_rust_ringattn_smoke.sh
```

Rust remote P2P pair smoke：

```bash
# 远端 GPU 节点
PATH=/home/stark/.cargo/bin:$PATH \
  RUN_ID=rust-remote-p2p-<timestamp> \
  BIND_ADDR=0.0.0.0:29172 \
  CARGO_OFFLINE=0 \
  bash scripts/run_rust_remote_p2p_server.sh
```

```bash
# 本机 client
RUN_ID=rust-remote-p2p-<timestamp> \
  CONNECT_ADDR=192.168.8.172:29172 \
  CARGO_OFFLINE=0 \
  bash scripts/run_rust_remote_p2p_client.sh
```

Rust remote CP dual-role node smoke：

```bash
# 本机 Mac node
RUN_ID=rust-remote-cp-node-<timestamp> \
  NODE_INDEX=0 \
  BIND_ADDR=0.0.0.0:29176 \
  CONNECT_ADDR=192.168.8.172:29175 \
  CARGO_OFFLINE=0 \
  HCP_ENABLE_TORCH=1 \
  HCP_TORCH_DEVICE=mps \
  bash scripts/run_rust_remote_cp_node.sh
```

```bash
# 远端 GPU node
PATH=/home/stark/.cargo/bin:$PATH \
  LIBTORCH=/home/stark/libtorch \
  LIBTORCH_INCLUDE=/home/stark/libtorch/include \
  LIBTORCH_LIB=/home/stark/libtorch/lib \
  LD_LIBRARY_PATH=/home/stark/libtorch/lib:$LD_LIBRARY_PATH \
  RUN_ID=rust-remote-cp-node-<timestamp> \
  NODE_INDEX=1 \
  BIND_ADDR=0.0.0.0:29175 \
  CONNECT_ADDR=<MAC_192_ADDR>:29176 \
  CARGO_OFFLINE=0 \
  HCP_ENABLE_TORCH=1 \
  HCP_TORCH_DEVICE=cuda:0 \
  bash scripts/run_rust_remote_cp_node.sh
```

Rust remote CP 3-node unified smoke：

```bash
RUN_ID=rust-remote-cp-3node-<timestamp> \
  PORT_BASE=29285 \
  bash scripts/run_rust_remote_cp_3node_smoke.sh
```

该脚本会自动发现当前 Mac `192.168.8.x` 地址，在 GPU host `192.168.8.172` 上执行 `git pull --ff-only` 和 cargo preflight build，然后统一启动本机 node0/node2 与远端 CUDA node1。默认本机节点使用 MPS，GPU 节点使用 CUDA。

### 环境变量

- `RUN_ID`：覆盖 smoke report 目录名，默认 `hcp-ringattn-smoke-local`。
- `SKIP_PYTHON_SMOKE=1`：跳过 Python worker/controller smoke。
- `RUN_PYTHON_CORRECTNESS=1`：显式运行 Python correctness 历史对照。
- `CARGO_OFFLINE=1`：Rust smoke 默认离线构建，避免 cargo registry 网络依赖。
- `HCP_ENABLE_TORCH=1`：启用 C++ ATen/libtorch bridge。
- `HCP_TORCH_DEVICE=cpu|mps|cuda|cuda:N`：选择 ATen smoke 设备；成功码分别为 CPU=1、MPS=2、CUDA=3。
- `BIND_ADDR`：remote P2P server 监听地址，双机 smoke 使用 `0.0.0.0:29172` 或 GPU 子网地址。
- `CONNECT_ADDR`：remote P2P client 连接地址，当前 GPU host 为 `192.168.8.172:29172`。
- `NODE_INDEX`：remote CP node index；当前 `0=mac-mps`，`1=gpu-cuda`。
- `HCP_REMOTE_CP_DOMAINS=2|3`：remote CP node 拓扑大小，默认 2；设置为 3 时为 `mac-mps -> gpu-cuda -> mac-mps-2 -> mac-mps`。
- `PORT_BASE`：`run_rust_remote_cp_3node_smoke.sh` 使用的三节点端口基准，默认 `29250`；GPU node1 使用 `PORT_BASE`，node0 使用 `PORT_BASE+1`，node2 使用 `PORT_BASE+2`。
- `MAC_192_ADDR`：覆盖统一 launcher 自动发现的 Mac `192.168.8.x` 地址。
- `GPU_HOST` / `GPU_USER` / `GPU_REPO_DIR`：覆盖统一 launcher 的远端 GPU 地址、SSH 用户和远端仓库目录；默认分别为 `192.168.8.172`、`stark`、`hetero-cp-ringattn`。
- `LOCAL_CARGO_OFFLINE` / `REMOTE_CARGO_OFFLINE`：统一 launcher 的本机/远端 cargo offline 开关，默认均为 `0`，避免远端 cache miss 干扰 smoke。
- remote CP smoke 前应先用 `ifconfig | rg 'inet 192\\.168\\.8\\.'` 确认当前 Mac `192.168.8.x` 地址，并把 GPU 侧 `CONNECT_ADDR=<MAC_192_ADDR>:...` 更新为当前值；2026-04-26 已出现从 `192.168.8.204` 变为 `192.168.8.239` 的情况。
- 本机 Mac hardware smoke 使用 `HCP_TORCH_DEVICE=mps` 并越过普通沙箱；CPU smoke 只用于编译/链接 fallback。
- 启用 `HCP_ENABLE_TORCH=1` 后，Rust smoke 要求 torch bridge 成功；CLI summary 中 `torch_status=pass` 且设备成功码匹配才算硬件 smoke 通过。
- `torch_block_update_status=pass` 表示 `cp_ring_node_runtime.compute_updates()` 已驱动同等次数的 C++ ATen attention block compute；MPS 成功码为 2，CUDA 成功码为 3。
- `torch_payload_block_status=pass` 表示 `RingAttnMessage.payload` 中的 captured K/V blocks 已逐块驱动 C++ ATen attention block compute；MPS 成功码为 2，CUDA 成功码为 3。
- `torch_payload_online_status=pass` 表示 captured K/V block 流已在 C++ ATen 中逐 block 维护 online softmax state，并与 full attention CPU reference 对比；MPS 成功码为 2，CUDA 成功码为 3。
- `torch_payload_chunk_status=pass` 表示 captured K/V block 流已在 C++ ATen 中对小尺寸 Q chunk 维护 online softmax output，并与 full attention CPU reference 对比；MPS 成功码为 2，CUDA 成功码为 3。
- remote CP node 启用 `HCP_ENABLE_TORCH=1` 后也会执行 payload-backed compute；当前双机期望每个 node `torch_payload_blocks=8/8`。
- 3-node remote CP smoke 期望每个 node `messages_sent=8 messages_received=8 compute_updates=12 torch_payload_blocks=12/12 torch_payload_online_blocks=12/12 torch_payload_chunk_blocks=12/12 torch_query_chunk_blocks=12/12 torch_query_output_blocks=12/12`；正式验证优先使用 `scripts/run_rust_remote_cp_3node_smoke.sh` 统一启动。
- torch bridge 失败时 CLI summary 后会打印压缩 `torch_message`；完整信息写入 JSON report。
- CUDA 请求下 `torch_code=-5` 表示当前 libtorch 进程无 CUDA backend，通常是 CPU-only libtorch 或 `libtorch_cuda` / `c10_cuda` 没有被链接/加载。
- Linux CUDA libtorch 构建需要保留 `libtorch_cuda` / `c10_cuda` 动态依赖；build script 在检测到这两个库时会用同一个 linker group 传入 `--push-state,--no-as-needed,-ltorch_cuda,-lc10_cuda,--pop-state`。
- 远端非交互 SSH 默认 PATH 不包含 `/home/stark/.cargo/bin`，也不会自动加载 libtorch 环境；通过 SSH 启动 CUDA Rust smoke 时显式设置 `PATH=/home/stark/.cargo/bin:$PATH`、`LIBTORCH=/home/stark/libtorch`、`LIBTORCH_INCLUDE=/home/stark/libtorch/include`、`LIBTORCH_LIB=/home/stark/libtorch/lib`、`LD_LIBRARY_PATH=/home/stark/libtorch/lib:$LD_LIBRARY_PATH`。

## 项目结构

```text
include/hcp_ringattn/core/  # 独立公共类型、协议、runtime 抽象
src/                        # NoOp runtime 与 C++ coordinator smoke
python/                     # controller、worker、online softmax 原型
rust/                       # Rust correctness model、report、C++ bridge build
config/                     # 最小 ring 配置
scripts/                    # 本地 smoke 入口
docs/                       # 设计、验证、路线图、产品论证
reports/                    # 实验报告输出目录
```

## Import Aliases

- 无 TypeScript / Python package alias 配置。
- C++ include root 为仓库内 `include/`。

## 构建与部署

- 本地构建通过 CMake 完成。
- 当前没有 CI/CD 或 Docker 配置。
- `build/` 和 `reports/*` 被 `.gitignore` 忽略，`reports/.gitkeep` 保留目录。
