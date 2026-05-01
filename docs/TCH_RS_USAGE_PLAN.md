# tch-rs Usage Plan

## 结论

`tch-rs` 可以作为 HCP Rust 路线的 PyTorch tensor backend。更准确地说，它绑定的是 PyTorch C++ API，也就是 `libtorch`，并不要求运行 Python。

本仓应优先走独立 `libtorch` 路线：手动安装 / 解压 libtorch 2.11.0，并通过 `LIBTORCH`、`LIBTORCH_INCLUDE`、`LIBTORCH_LIB` 指向它。`LIBTORCH_USE_PYTORCH=1` 只作为 fallback 或本机快速验证路径。

推荐分层：

```text
Rust HCP model / protocol
├── pure-rust backend        # 默认，离线、无 libtorch 依赖，保证 correctness 可跑
├── cxx-aten backend         # 当前已验证，Rust -> C ABI -> C++ -> ATen/libtorch
└── tch backend              # 下一阶段，feature-gated，直接 Rust 调 PyTorch tensor API
```

## 版本与环境

当前官方资料显示：

- `tch` 最新版本是 `0.24.0`。
- `tch 0.24.0` 需要 libtorch / PyTorch `2.11.0`。
- 默认路径是 system-wide libtorch。
- 可以通过 `LIBTORCH=/path/to/libtorch` 指向手动安装的 libtorch。
- 也可以通过 `LIBTORCH_INCLUDE` 和 `LIBTORCH_LIB` 分别指定 header 与 library 目录。
- `LIBTORCH_USE_PYTORCH=1` 会调用 active Python interpreter 查询 `torch` 包信息；这不是默认路径。
- `tch::Device` 支持 `Cpu`、`Cuda`、`Mps`、`Vulkan`。

本机当前状态：

- `torch==2.11.0`
- `torchvision==0.26.0`
- `torchaudio==2.11.0`
- 非沙箱进程下 MPS 可用：`torch.backends.mps.is_available() == true`
- 默认沙箱进程下 Metal device 不可见；MPS smoke 必须用非沙箱/授权命令运行。
- `tch` crate 当前不在本机 cargo cache 中，直接接入需要 cargo registry/network。
- 独立 system-wide libtorch 已安装在 `/Users/stark_sim/libtorch`，`build-version=2.11.0`。
- 当前环境变量：`LIBTORCH=/Users/stark_sim/libtorch`、`LIBTORCH_INCLUDE=/Users/stark_sim/libtorch/include`、`LIBTORCH_LIB=/Users/stark_sim/libtorch/lib`。

## 为什么不能直接全量切换到 tch

1. `tch` 会引入 `torch-sys`，编译和链接成本明显高于当前纯 Rust correctness。
2. `tch` 依赖 libtorch 版本严格匹配；本仓默认验证不应被外部 Python/torch 环境阻塞，因此应固定独立 libtorch 路径。
3. 环境默认在线（rsproxy-sparse 可达），cargo 命令正常在线执行。
4. HCP 仍需要明确自己的 protocol / report / transport schema，不能把模型逻辑完全耦合进 PyTorch binding。

## 应该怎么用好 tch

### 1. 作为可选 tensor backend

在 `rust/Cargo.toml` 中后续可采用：

```toml
[features]
default = []
tch-backend = ["dep:tch"]

[dependencies]
tch = { version = "0.24.0", optional = true, default-features = false }
```

运行时：

```bash
LIBTORCH=/opt/libtorch-2.11.0 cargo run --features tch-backend --bin tch_ringattn_smoke
```

如果要跑 MPS：

```bash
LIBTORCH=/opt/libtorch-2.11.0 HCP_TCH_DEVICE=mps cargo run --features tch-backend --bin tch_ringattn_smoke
```

MPS 命令必须在非沙箱进程中运行。

当前 C++ ATen bridge 也支持同样的设备显式选择：

```bash
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=cpu bash scripts/run_rust_ringattn_smoke.sh
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps bash scripts/run_rust_ringattn_smoke.sh
```

其中 `torch_bridge.status_code=1` 表示 CPU，`torch_bridge.status_code=2` 表示 MPS。

快速验证或过渡期可以用：

```bash
LIBTORCH_USE_PYTORCH=1 cargo run --features tch-backend --bin tch_ringattn_smoke
```

但这条路径会重新引入 Python 环境耦合，不应作为项目长期默认。

### 2. 先实现一个小 smoke，而不是直接改主模型

第一步只做：

- 创建 `Tensor`。
- 在 `Device::Cpu` 上跑 `matmul` / `softmax`。
- 在非沙箱进程上验证 `Device::Mps`。
- 输出 report：device、dtype、shape、max_abs_err、mean_abs_err。

只有这个 smoke 稳定后，再迁移 Ring Attention block update。

### 3. 用 tch 承担重 tensor op，保留 HCP 控制逻辑在 Rust

tch 适合承载：

- `Q @ K^T`
- block-wise `softmax`
- `P @ V`
- dtype / device movement
- 后续 safetensors 权重加载

HCP Rust 自己保留：

- domain config
- ring source order
- block traversal
- online softmax state lifecycle
- report schema
- protocol / transport state machine

这样不会把 HCP 的低边界协议语义埋进 PyTorch API 调用里。

### 4. 设备选择策略

建议顺序：

1. `HCP_TCH_DEVICE=cpu`：默认，可在沙箱/CI 跑。
2. `HCP_TCH_DEVICE=mps`：只在非沙箱 macOS 本机跑。
3. 未来 `cuda:N`：远端 CUDA domain。

不要使用 `Device::cuda_if_available()` 作为 HCP 默认选择，因为本仓需要明确记录每个 domain 的 device 类型。

## 接入步骤

1. 保持当前 pure-rust correctness 和 C++ ATen bridge 不变。
2. 在 cargo registry 可用时，引入 optional `tch = "0.24.0"`。
3. 新增 `rust/src/bin/tch_smoke.rs`，只做 tensor/device smoke。
4. 新增 `scripts/run_tch_ringattn_smoke.sh`，默认 CPU，MPS 需要非沙箱。
5. 新增结构化 report 到 `reports/<RUN_ID>/tch_smoke.json`。
6. 验证后单独 commit。
7. 再新增 `TchTensorBackend`，实现与 pure-rust backend 相同的 correctness case。

## 参考资料

- `tch-rs` GitHub: https://github.com/LaurentMazare/tch-rs
- `tch 0.24.0` docs.rs: https://docs.rs/tch/latest/tch/
- `Device` enum: https://docs.rs/tch/latest/tch/enum.Device.html
- PyTorch install: https://pytorch.org/get-started/locally/
