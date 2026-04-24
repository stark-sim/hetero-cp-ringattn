# Rust + C++ Ring Attention Path

## 目标

后续 HCP correctness / protocol 原型优先迁移到 Rust，减少 Python 在核心验证路径中的占比。C++ 继续承载已有 core/runtime skeleton，Rust 负责更安全地组织 correctness model、report、后续 protocol schema 与 transport smoke。

## 当前实现

`rust/` crate 提供一个 Rust Ring Attention correctness model：

- 纯 Rust `Tensor3`，默认不依赖 Python。
- 按 domain 不均分 `seq_chunk_len`。
- 按 source domain 不同 `block_size` 遍历 K/V block。
- 使用 online softmax state 与 full attention reference 对照。
- 输出 JSON report。
- 通过 C ABI 调用 `src/rust_bridge.cc`，证明 Rust binary 已链接并调用 C++ core/runtime skeleton。

运行：

```bash
bash scripts/run_rust_ringattn_smoke.sh
```

## PyTorch / libtorch 路线

Rust 社区常用的 PyTorch 绑定是 `tch-rs`，它是 PyTorch C++ API/libtorch 的 Rust wrapper。`tch-rs` 不要求 Python；默认是 system-wide libtorch，也可以通过 `LIBTORCH` 指向手动安装的 libtorch。`LIBTORCH_USE_PYTORCH=1` 只是复用 Python torch 的可选路径。

当前本机 cargo 缓存没有 `tch` / `torch-sys`，因此默认路径没有引入该依赖，避免构建时触发网络下载。

详细接入策略见 `docs/TCH_RS_USAGE_PLAN.md`。

本仓先提供 C++ libtorch bridge：

- 默认关闭，不影响纯 Rust smoke。
- 设置 `HCP_ENABLE_TORCH=1` 后，`rust/build.rs` 会优先使用 `LIBTORCH` / `LIBTORCH_INCLUDE` / `LIBTORCH_LIB` 指向的独立 libtorch；如果这些变量缺失，才 fallback 到 Python torch path discovery。
- 这条路径验证的是 Rust -> C ABI -> C++ -> libtorch。

尝试方式：

```bash
HCP_ENABLE_TORCH=1 bash scripts/run_rust_ringattn_smoke.sh
```

CPU / MPS / CUDA 选择：

```bash
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=cpu bash scripts/run_rust_ringattn_smoke.sh
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps bash scripts/run_rust_ringattn_smoke.sh
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=cuda:0 bash scripts/run_rust_ringattn_smoke.sh
```

`HCP_TORCH_DEVICE=mps` 必须在非沙箱/授权进程中运行，否则 Metal device 不可见。
本机 Mac 的 libtorch hardware smoke 应直接使用 MPS，并越过普通沙箱运行；CPU smoke 只作为编译/链接 fallback，不作为有意义的本机硬件验证结论。
`HCP_TORCH_DEVICE=cuda` / `cuda:N` 必须使用 CUDA 版 libtorch；`rust/build.rs` 会在 libtorch 库目录中发现 `torch_cuda` / `c10_cuda` 时自动追加链接。

远端 GPU 机器首次运行如果 Cargo cache 没有依赖，默认 `CARGO_OFFLINE=1` 会失败。先执行一次：

```bash
cd rust && cargo fetch --locked
```

或直接对 smoke 放开一次网络：

```bash
CARGO_OFFLINE=0 HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=cuda:0 bash scripts/run_rust_ringattn_smoke.sh
```

当前本机验证结论：

- Python PyTorch 已升级到 `2.11.0`，目前仅作为快速验证来源。
- `torchvision` 已升级到 `0.26.0`。
- `torchaudio` 已升级到 `2.11.0`。
- 独立 libtorch 已安装在 `/Users/stark_sim/libtorch`，`build-version=2.11.0`。
- Rust build script 已优先使用 `LIBTORCH` / `LIBTORCH_INCLUDE` / `LIBTORCH_LIB`；只有这些变量不存在时才 fallback 到 Python torch path discovery。
- `<torch/torch.h>` 在升级前的当前 Apple Clang / libc++ 组合下曾触发 `std::is_arithmetic` specialization 相关编译错误；短期仍避免把全量 header 作为默认路径。
- 改用更窄的 `<ATen/ATen.h>` 后，`HCP_ENABLE_TORCH=1` 可以编译并执行 C++ tensor smoke。
- `HCP_TORCH_DEVICE=cpu` 时 report 中 `torch_bridge.status_code=1`。
- `HCP_TORCH_DEVICE=mps` 时 report 中 `torch_bridge.status_code=2`，这才表示实际跑到 MPS。
- `HCP_TORCH_DEVICE=cuda` / `cuda:N` 时 report 中 `torch_bridge.status_code=3`，这才表示实际跑到 CUDA。
- `HCP_ENABLE_TORCH=1` 时，torch bridge 不再只是附加信息；请求设备没有拿到对应成功码会使整体 smoke 失败，并在 CLI summary 中打印 `torch_status`、`torch_device`、`torch_code`。
- 因此短期推荐路线是 Rust -> C ABI -> C++ ATen/libtorch，而不是马上把完整 `torch/torch.h` 或 `tch-rs` 作为必需路径。
- `torch.backends.mps.is_built()` 为 true；沙箱进程看不到 Metal device，非沙箱进程 MPS 可用。

如果后续允许下载 Rust crate，再考虑加入 feature-gated `tch` 后端：

```text
rust correctness model
  ├── backend=pure-rust
  ├── backend=cxx-libtorch
  └── backend=tch-rs
```

## 设计取舍

- 不直接把 `tch` 作为默认依赖：避免网络、libtorch 版本、C++ ABI 问题阻塞基础 correctness。
- 先保留纯 Rust correctness：保证任何有 Rust toolchain 的环境都能跑。
- C++ bridge 用 C ABI：避免 Rust 直接绑定 C++ class layout，也避免过早引入复杂 bindgen / cxx 依赖。
- libtorch smoke 放在 C++ shim：符合 PyTorch 原生 C++ API 的支持路径，也便于后续接入真实 tensor kernel。
- `tch-rs` 仍是主要参考；当前 upstream `tch` crate 版本为 `0.24.0`，对应 PyTorch/libtorch 2.11 路线，后续可作为 feature-gated backend 接入，并优先绑定独立 libtorch 而不是 Python torch。
