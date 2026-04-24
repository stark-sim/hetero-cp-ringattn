# 当前上下文

## 当前焦点

[2026-04-24] 当前主线切到 Rust + C++：Rust 负责 Ring Attention correctness / report / 后续协议模型，C++ 继续承载已有 core/runtime skeleton，并通过 C ABI 与 Rust binary 集成。Python 只保留历史原型，不再作为优先实现路径。

## 近期变化

- [2026-04-24] 初始化 HCP standalone repo。
- [2026-04-24] 文档已覆盖产品论证、HCP/HLPP 边界、历史经验、设计、验证计划和路线图。
- [2026-04-24] C++ core 已具备独立 `Status`、`TensorDType` / `BoundaryTensor`、`RingAttnProtocol` / `RingAttnRuntime`。
- [2026-04-24] 已存在最小 `NoOp` runtime、C++ coordinator smoke、Python controller / worker / kernel stub。
- [2026-04-24] 本次创建 `memory-bank/` Basic profile，并为 Codex 创建 `AGENTS.md` 协议文件。
- [2026-04-24] 已将 `ringattn_kernel_stub.py` 扩展为 NumPy Ring Attention correctness model，覆盖不均分 domain / block size，并输出 JSON report。
- [2026-04-24] 已新增 `docs/RINGATTN_MODEL.md`，记录 correctness model 的边界、数据流和验证入口。
- [2026-04-24] `scripts/run_local_ringattn_smoke.sh` 默认只跑 C++ smoke；Python correctness 仅在 `RUN_PYTHON_CORRECTNESS=1` 时作为历史对照运行。
- [2026-04-24] 新增 `rust/` crate，实现纯 Rust Ring Attention correctness model，并通过 C ABI 调用 C++ `NoOpRingAttnRuntime`。
- [2026-04-24] 新增 `src/rust_bridge.cc` 和 `scripts/run_rust_ringattn_smoke.sh`，Rust smoke 已通过 3/3 correctness cases，C++ bridge 返回 3 domains。
- [2026-04-24] 参考 `tch-rs` 路线后验证本机 PyTorch/libtorch：已将当前 miniconda Python 环境升级到 `torch==2.11.0`、`torchvision==0.26.0`、`torchaudio==2.11.0`。
- [2026-04-24] `HCP_ENABLE_TORCH=1` 可通过 C++ ATen bridge 编译并执行 tensor smoke；Rust report 中 `torch_bridge.compiled=true`。
- [2026-04-24] 新增 `docs/TCH_RS_USAGE_PLAN.md`，明确 `tch-rs` 在本仓中应作为 feature-gated tensor backend 使用，而不是替代默认 pure-rust correctness 路径。
- [2026-04-24] 独立 libtorch 2.11.0 已安装在 `/Users/stark_sim/libtorch`，环境变量 `LIBTORCH` / `LIBTORCH_INCLUDE` / `LIBTORCH_LIB` 已配置。
- [2026-04-24] Rust build script 已改为优先使用独立 libtorch 环境变量；只有缺失时才 fallback 到 Python torch path discovery。
- [2026-04-24] C++ ATen bridge 已支持 `HCP_TORCH_DEVICE=cpu|mps`；report 中 `status_code=1` 表示 CPU，`status_code=2` 表示 MPS。
- [2026-04-24] 非沙箱运行 `HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps bash scripts/run_rust_ringattn_smoke.sh` 已通过，report 显示 `requested_device=mps`、`status_code=2`、`message=ok`。

## 活跃决策

- HCP 与 HLPP 保持边界清晰：本仓只做 intra-layer / low-boundary Ring Attention。
- 跨异构 domain 坚持 P2P，不把 collective 作为主通信假设。
- 当前优先级是 correctness -> protocol -> transport -> remote heterogeneous deployment -> scaling argument。
- 每个实验阶段都应产生结构化 report，而不是只保留日志或口头结论。
- Rust + C++ 是后续核心工程路径；`tch-rs` 作为 PyTorch Rust 绑定参考，当前 upstream `tch` crate 为 `0.24.0`，与 PyTorch/libtorch 2.11 路线匹配，但默认构建暂不强依赖 `tch` crate。
- PyTorch C++ 路径短期优先使用 C++ ATen/libtorch bridge，避免直接 include 全量 `torch/torch.h`。
- `tch-rs` 的长期接入应优先使用独立/system-wide libtorch；`LIBTORCH_USE_PYTORCH=1` 只作为 fallback 或快速验证路径，避免把核心 Rust 路线重新耦合到 Python 环境。

## 下一步

- [ ] 将当前 correctness JSON 进一步整理成正式 report 文档，沉淀 M2 数学闭环结论。
- [ ] 扩展 correctness case，覆盖更大的 seq、更多 seed、float32 / mixed precision tolerance policy。
- [ ] 必要时增加 `max_rel_err` 并明确 tolerance policy。
- [ ] 将 Rust correctness model 继续拆分为 library + binary，便于后续 protocol / transport 复用。
- [ ] 在 cargo registry/network 可用后，增加 feature-gated `tch = 0.24.0` backend，并先实现 `tch_smoke`，再迁移 Ring Attention block update。
- [ ] 在 cargo registry/network 可用后，引入 optional `tch = 0.24.0` 并实现 `tch_smoke`。
- [ ] 为 `RingAttnMessage` 设计 serialization / deserialization。

## 重要模式与偏好

- 文档与 memory bank 使用中文。
- 不引入 HLPP high-boundary 语义到 HCP core。
- 优先保留最小、可见、可复现的实验闭环。
- 对 remote smoke 继承 phase3 的路径解析、解释器 pinning、clean build、report layout 纪律。
- 新核心代码优先 Rust + C++；Python 只作为辅助或历史对照。
- Git 纪律：每个任务节点实现并验证后应单独提交，避免会话结束时形成一个过大的混合 commit。结构化实验 report 可以提交作为项目进展资产；build 产物、临时日志、cache、大型二进制不应默认提交。
- sudo / 系统改动纪律：如果修复需要 `sudo`、root-owned path、`/opt`、系统 linker 或机器级配置，应停止并给用户最小命令让用户自己执行；不要为了绕过 sudo 擅自修改第三方二进制、install_name 或 vendor artifacts。

## 当前阻塞

- [2026-04-24] 当前沙箱环境不允许 Python worker 绑定本地端口，完整 Python smoke 会触发 `PermissionError: [Errno 1] Operation not permitted`。
- [2026-04-24] `ringattn_controller.py` 当前会将 `bytes` 放入 `json.dumps` payload，导致 `TypeError: Object of type bytes is not JSON serializable`。
- [2026-04-24] 普通 `cargo check` 会尝试访问 `rsproxy.cn` 并因 DNS 失败；当前使用 `cargo --offline` 可正常构建缓存依赖。
- [2026-04-24] PyTorch 2.11.0 在默认沙箱进程中 `mps_available=false`，原因是沙箱内 `MTLCopyAllDevices()` 返回 0；非沙箱进程可枚举 `Apple M1 Pro`，且 `torch.ones(..., device="mps")` 成功。
- [2026-04-24] 后续所有 Metal/MPS 相关验证必须在非沙箱/授权进程中运行；默认沙箱结果不能作为 MPS 不可用结论。
