# 进展

## 已完成功能

- [x] [2026-04-24] 仓库已从 `honolulu` 抽离为 standalone HCP Ring Attention 研究仓。
- [x] [2026-04-24] 已完成独立 C++ core skeleton：`Status`、`TensorDType`、`BoundaryTensor`、protocol、runtime 抽象。
- [x] [2026-04-24] 已实现最小 `NoOpRingAttnRuntime`。
- [x] [2026-04-24] 已实现 C++ coordinator smoke。
- [x] [2026-04-24] 已提供 Python controller / worker / kernel stub 占位。
- [x] [2026-04-24] 已完成核心 docs：README、product thesis、HLPP vs HCP、history、design、validation plan、roadmap。
- [x] [2026-04-24] `SKIP_PYTHON_SMOKE=1 bash scripts/run_local_ringattn_smoke.sh` 通过，C++ coordinator smoke OK。
- [x] [2026-04-24] 已创建 Basic memory bank 和 Codex `AGENTS.md` 协议文件。
- [x] [2026-04-24] 已建立 NumPy Ring Attention correctness model，包含 ring source order、per-source block traversal、online softmax state update、full attention reference 对照。
- [x] [2026-04-24] correctness model 默认 3 个 case 全部通过：2-domain uneven chunks、3-domain uneven blocks、4-domain tail blocks。
- [x] [2026-04-24] 本地 C++ smoke 在 `SKIP_PYTHON_SMOKE=1` 下通过；Python correctness 已降为显式 opt-in。
- [x] [2026-04-24] 已新增 Rust crate，实现纯 Rust Ring Attention correctness model。
- [x] [2026-04-24] 已新增 Rust -> C ABI -> C++ runtime bridge，并成功调用 C++ `NoOpRingAttnRuntime`。
- [x] [2026-04-24] 已验证 `HCP_ENABLE_TORCH=1` 下 Rust -> C++ ATen/libtorch smoke 可编译执行，report 中 `torch_bridge.compiled=true`。
- [x] [2026-04-24] `cargo clippy --offline -- -D warnings` 通过。
- [x] [2026-04-24] 已将当前 miniconda Python 环境中的 PyTorch 升级到 stable `torch==2.11.0`，并同步安装 `torchvision==0.26.0`、`torchaudio==2.11.0`。
- [x] [2026-04-24] 已完成 `tch-rs` 正式接入方案文档，建议作为 optional backend 接入，默认 pure-rust 路径保持可离线验证。
- [x] [2026-04-24] 已确认独立 libtorch 2.11.0 安装在 `/Users/stark_sim/libtorch`，并更新 Rust build script 优先使用 system-wide libtorch。
- [x] [2026-04-24] 已验证 C++ ATen/libtorch bridge 可显式跑 CPU 与 MPS；MPS report 为 `requested_device=mps`、`status_code=2`、`message=ok`。
- [x] [2026-04-25] Rust -> C++ ATen/libtorch bridge 已增加 `cuda` / `cuda:N` 设备解析和 CUDA 库自动链接发现，CUDA 成功码约定为 `torch_bridge.status_code=3`。
- [x] [2026-04-25] Rust smoke 脚本在 `CARGO_OFFLINE=1` 且依赖 cache miss 时会明确提示 `CARGO_OFFLINE=0` 或 `cargo fetch --locked`。

## 进行中

- [ ] M2：Rust online softmax correctness report 与 tolerance policy 扩展。
- [ ] M3：`RingAttnMessage` schema、serialization / deserialization、本地 send/recv smoke。
- [ ] M4：heterogeneous runtime stubs 与配置 / 环境纪律。
- [ ] M5：2-domain remote heterogeneous smoke。
- [ ] M6：memory / bandwidth scaling notes 与 context-length growth argument。
- [ ] GPU 端 CUDA 版 libtorch smoke：`HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=cuda:0 bash scripts/run_rust_ringattn_smoke.sh`。

## 已知问题

- [2026-04-24] 完整 Python smoke 在当前沙箱下无法绑定本地端口。
- [2026-04-24] `ringattn_controller.py` 存在 `bytes` JSON 序列化问题。
- [2026-04-24] `ringattn_kernel_stub.py` 已有 correctness JSON report 入口，但还没有整理成正式 M2 report 文档。
- [2026-04-24] `tch` crate 未在本机 cargo cache 中；system-wide libtorch 已就绪，剩余阻塞是 cargo registry/network 拉取 `tch` / `torch-sys`。
- [2026-04-24] MPS 排查结论：沙箱进程隐藏 Metal device，非沙箱进程下 PyTorch 2.11.0 的 MPS 可用。
- [2026-04-25] GPU 远端默认 `CARGO_OFFLINE=1` 时可能因 cargo cache 缺 `serde_json` 等基础依赖失败；这不是 CUDA smoke 结果，需要先在线 fetch 或放开一次 `CARGO_OFFLINE=0`。

## 里程碑

| 里程碑 | 状态 | 目标日期 |
|--------|------|----------|
| M0: 独立化完成 | 已完成 | [2026-04-24] |
| M1: 问题定义固定 | 已完成 | [2026-04-24] |
| M2: 数学闭环 | 进行中 | 待定 |
| M3: 协议闭环 | 未开始 | 待定 |
| M4: 异构 runtime 闭环 | 未开始 | 待定 |
| M5: 远端闭环 | 未开始 | 待定 |
| M6: 扩展性论证 | 未开始 | 待定 |
