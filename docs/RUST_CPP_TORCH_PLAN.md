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
- `torch_bridge` 验证基础张量创建和 matmul 是否实际落在请求设备。
- `torch_attention_bridge` 验证真实 attention block compute：C++ ATen 在请求设备上计算 `softmax(QK^T / sqrt(d))V`，再搬回 CPU 与 CPU reference 对比误差。
- `torch_block_update_bridge` 将 `cp_ring_node_runtime` 统计出的 compute update 数量传入 C++ ATen bridge，并在请求设备上循环执行同等次数的 attention block compute。
- `torch_payload_block_bridge` 将 `RingAttnMessage.payload` 中的 float32 K/V bytes 传入 C++ ATen bridge，在请求设备上对每个 captured CP block 执行 payload-backed attention compute。
- `torch_payload_online_bridge` 在请求设备上按 K/V payload block 流维护 online softmax state。
- `torch_payload_chunk_bridge` 在请求设备上对小尺寸 Q chunk 输出 `[query, head, dim]` attention chunk。
- `torch_query_chunk_bridge` 从 Rust/domain-side 生成显式 float32 Q chunk payload，并与 captured K/V payload blocks 一起传入 C++ ATen bridge；C++ 不再为该路径内部构造 Q。

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
- `torch_attention_bridge.status_code` 使用同一套设备成功码；`torch_attention_status=pass` 表示 attention compute 也在目标设备上执行并通过 CPU reference tolerance。
- `torch_block_update_bridge.status_code` 使用同一套设备成功码；`torch_block_update_status=pass` 表示 CP ring smoke 产生的 block update 数量已驱动同等次数的 device-side attention block compute。
- `torch_payload_block_bridge.status_code` 使用同一套设备成功码；`torch_payload_block_status=pass` 表示 CP runtime 捕获的 K/V payload blocks 已逐块驱动 device-side attention compute。
- `HCP_ENABLE_TORCH=1` 时，torch bridge 不再只是附加信息；请求设备没有拿到对应成功码会使整体 smoke 失败，并在 CLI summary 中打印 `torch_status`、`torch_device`、`torch_code`。
- `HCP_ENABLE_TORCH=1` 时，`torch_attention_bridge` 也参与整体 smoke 成败；CLI summary 会打印 `torch_attention_status` 和 `torch_attention_code`。
- `HCP_ENABLE_TORCH=1` 时，`torch_block_update_bridge` 也参与整体 smoke 成败；CLI summary 会打印 `torch_block_update_status`、`torch_block_update_code`、`torch_block_updates`。
- `HCP_ENABLE_TORCH=1` 时，`torch_payload_block_bridge` 也参与整体 smoke 成败；CLI summary 会打印 `torch_payload_block_status`、`torch_payload_block_code`、`torch_payload_blocks=<processed>/<requested>`。
- torch bridge 失败时 CLI 会打印压缩后的 `torch_message`；完整异常仍保存在 `reports/<RUN_ID>/rust_ringattn_correctness.json`。
- `torch_code=-5` 表示 `cuda` / `cuda:N` 设备名有效，但当前进程中的 libtorch 没有 CUDA backend；优先检查 `LIBTORCH` / `LIBTORCH_LIB` 是否指向 CUDA-enabled libtorch，以及是否链接/加载了 `libtorch_cuda`、`c10_cuda`。
- Linux 下 CUDA 版 libtorch 需要保留 `libtorch_cuda` / `c10_cuda` 这类 registration library；`rust/build.rs` 会在检测到二者时用同一个 linker group 传入 `--push-state,--no-as-needed,-ltorch_cuda,-lc10_cuda,--pop-state`，防止链接器或 rustc 参数重排把它们优化掉。远端 `ldd rust/target/debug/hcp-ringattn-rust | grep -E 'torch|c10|cuda'` 应能看到 `libtorch_cuda.so` 和 `libc10_cuda.so`。
- 远端 CUDA 验证已通过：`torch_status=pass torch_device=cuda:0 torch_code=3`，且 `ldd` 显示 `libtorch_cuda.so` / `libc10_cuda.so`。
- 远端 CUDA attention compute 验证已通过：显式传入 `LIBTORCH*` 和 `LD_LIBRARY_PATH` 后，`torch_attention_status=pass torch_attention_code=3`。
- CP block update device-side compute 验证已通过：本机非沙箱 MPS 显示 `torch_block_update_status=pass torch_block_update_code=2 torch_block_updates=30`，远端 CUDA 显示 `torch_block_update_status=pass torch_block_update_code=3 torch_block_updates=30`。
- CP payload-backed device-side compute 验证已通过：本机非沙箱 MPS 显示 `torch_payload_block_status=pass torch_payload_block_code=2 torch_payload_blocks=30/30`，远端 CUDA 显示 `torch_payload_block_status=pass torch_payload_block_code=3 torch_payload_blocks=30/30`。
- 双机 remote CP node payload-backed compute 验证已通过：Mac node 显示 `torch_payload_block_status=pass torch_payload_block_code=2 torch_payload_blocks=8/8`，GPU node 显示 `torch_payload_block_status=pass torch_payload_block_code=3 torch_payload_blocks=8/8`。
- 3-node remote CP forwarding + payload-backed compute 验证已通过：Mac node0 / GPU node1 / Mac node2 均显示 `sent=8 received=8 compute_updates=12`，MPS nodes 显示 `torch_payload_block_code=2 torch_payload_blocks=12/12`，CUDA node 显示 `torch_payload_block_code=3 torch_payload_blocks=12/12`。
- Payload online softmax state 验证已通过：本机 MPS 和远端 CUDA 主 smoke 均显示 `torch_payload_online_blocks=30/30`；3-node remote CP 中 MPS nodes 和 CUDA node 均显示 `torch_payload_online_blocks=12/12`。
- Payload chunk output 验证已通过：本机 MPS 和远端 CUDA 主 smoke 均显示 `torch_payload_chunk_blocks=30/30`；3-node remote CP 中 MPS nodes 和 CUDA node 均显示 `torch_payload_chunk_blocks=12/12`。
- Query chunk payload 验证已通过：本机非沙箱 MPS 主 smoke 显示 `torch_query_chunk_status=pass torch_query_chunk_code=2 torch_query_chunk_blocks=30/30`，远端 CUDA 主 smoke 显示 `torch_query_chunk_status=pass torch_query_chunk_code=3 torch_query_chunk_blocks=30/30`。
- 当前 query chunk bridge 已消费 Rust/domain-side 显式 Q payload 和 `RingAttnMessage` 的 K/V bytes；Q 仍是 deterministic smoke tensor，尚未升级为真实模型 activation / weight lifecycle。
- 远端非交互 SSH 不会自动加载 `/home/stark/.cargo/bin` 或 libtorch 环境；通过 SSH 运行 CUDA smoke 时应显式传入 `PATH=/home/stark/.cargo/bin:$PATH LIBTORCH=/home/stark/libtorch LIBTORCH_INCLUDE=/home/stark/libtorch/include LIBTORCH_LIB=/home/stark/libtorch/lib LD_LIBRARY_PATH=/home/stark/libtorch/lib:$LD_LIBRARY_PATH`。
- 因此短期推荐路线是 Rust -> C ABI -> C++ ATen/libtorch，而不是马上把完整 `torch/torch.h` 或 `tch-rs` 作为必需路径。
- `torch.backends.mps.is_built()` 为 true；沙箱进程看不到 Metal device，非沙箱进程 MPS 可用。

如果后续允许下载 Rust crate，再考虑加入 feature-gated `tch` 后端：

```text
rust correctness model
  ├── backend=pure-rust
  ├── protocol=local-p2p-queue
  ├── backend=cxx-libtorch
  └── backend=tch-rs
```

## 设计取舍

- 不直接把 `tch` 作为默认依赖：避免网络、libtorch 版本、C++ ABI 问题阻塞基础 correctness。
- 先保留纯 Rust correctness：保证任何有 Rust toolchain 的环境都能跑。
- C++ bridge 用 C ABI：避免 Rust 直接绑定 C++ class layout，也避免过早引入复杂 bindgen / cxx 依赖。
- libtorch smoke 放在 C++ shim：符合 PyTorch 原生 C++ API 的支持路径，也便于后续接入真实 tensor kernel。
- `tch-rs` 仍是主要参考；当前 upstream `tch` crate 版本为 `0.24.0`，对应 PyTorch/libtorch 2.11 路线，后续可作为 feature-gated backend 接入，并优先绑定独立 libtorch 而不是 Python torch。
