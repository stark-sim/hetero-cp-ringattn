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
- 实验输出写入 `reports/<RUN_ID>/`。

### Testing

- 当前主要验证入口是 smoke 脚本：
  - C++ coordinator smoke
  - Rust Ring Attention correctness smoke
  - Rust -> C++ runtime bridge smoke
  - 可选 Rust -> C++ ATen/libtorch smoke
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

### 环境变量

- `RUN_ID`：覆盖 smoke report 目录名，默认 `hcp-ringattn-smoke-local`。
- `SKIP_PYTHON_SMOKE=1`：跳过 Python worker/controller smoke。
- `RUN_PYTHON_CORRECTNESS=1`：显式运行 Python correctness 历史对照。
- `CARGO_OFFLINE=1`：Rust smoke 默认离线构建，避免 cargo registry 网络依赖。
- `HCP_ENABLE_TORCH=1`：启用 C++ ATen/libtorch bridge。
- `HCP_TORCH_DEVICE=cpu|mps|cuda|cuda:N`：选择 ATen smoke 设备；成功码分别为 CPU=1、MPS=2、CUDA=3。
- 本机 Mac hardware smoke 使用 `HCP_TORCH_DEVICE=mps` 并越过普通沙箱；CPU smoke 只用于编译/链接 fallback。
- 启用 `HCP_ENABLE_TORCH=1` 后，Rust smoke 要求 torch bridge 成功；CLI summary 中 `torch_status=pass` 且设备成功码匹配才算硬件 smoke 通过。
- torch bridge 失败时 CLI summary 后会打印压缩 `torch_message`；完整信息写入 JSON report。
- CUDA 请求下 `torch_code=-5` 表示当前 libtorch 进程无 CUDA backend，通常是 CPU-only libtorch 或 `libtorch_cuda` / `c10_cuda` 没有被链接/加载。
- Linux CUDA libtorch 构建需要保留 `libtorch_cuda` / `c10_cuda` 动态依赖；build script 在检测到这两个库时会用同一个 linker group 传入 `--push-state,--no-as-needed,-ltorch_cuda,-lc10_cuda,--pop-state`。

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
