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
- [2026-04-25] C++ ATen bridge 已增加 `HCP_TORCH_DEVICE=cuda|cuda:N` 解析；CUDA 成功时 report 使用 `status_code=3`。
- [2026-04-25] Rust build script 会在 CUDA 版 libtorch 库目录中发现 `torch_cuda` / `c10_cuda` 时自动追加链接；CPU/MPS libtorch 不受影响。
- [2026-04-25] 远端 GPU 机器首次运行若 Cargo cache 缺 `serde_json` 等依赖，应先 `cd rust && cargo fetch --locked`，或用 `CARGO_OFFLINE=0` 跑一次 smoke。
- [2026-04-25] 用户明确要求本机 libtorch smoke 直接使用非沙箱 MPS；CPU-only smoke 只算编译/链接 fallback，不作为本机硬件验证结论。
- [2026-04-25] Rust smoke 已改为强校验 torch bridge：`HCP_ENABLE_TORCH=1` 且请求设备未返回对应成功码时整体 smoke 失败；CLI summary 会打印 `torch_status` / `torch_device` / `torch_code`。
- [2026-04-25] 远端 CUDA smoke 返回 `torch_code=-2`，说明 C++ ATen bridge 抛异常；CLI 已补充失败时打印压缩 `torch_message`，完整异常保留在 JSON report。
- [2026-04-25] C++ ATen bridge 增加 CUDA backend preflight：请求 `cuda` / `cuda:N` 时 `at::hasCUDA()==false` 会返回 `torch_code=-5`，明确指向 CPU-only libtorch 或 `libtorch_cuda` / `c10_cuda` 未链接加载。
- [2026-04-25] 远端已确认 `/home/stark/libtorch/lib` 存在 `libtorch_cuda.so` / `libc10_cuda.so`，但 `ldd rust/target/debug/hcp-ringattn-rust` 未显示它们；`rust/build.rs` 已在 Linux CUDA libtorch 下用同一个 linker group 传入 `--push-state,--no-as-needed,-ltorch_cuda,-lc10_cuda,--pop-state`，避免 rustc / linker 参数重排导致 registration libraries 被丢弃。
- [2026-04-25] 远端 GPU smoke 已通过：`CARGO_OFFLINE=0 HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=cuda:0 bash scripts/run_rust_ringattn_smoke.sh` 输出 `torch_status=pass torch_device=cuda:0 torch_code=3`，`ldd` 已显示 `libtorch_cuda.so` / `libc10_cuda.so`。
- [2026-04-25] Rust 新增 protocol smoke：本地 P2P queue transport 按 Context Parallel ring order 转发 K/V block，并覆盖 softmax state / terminate 消息；report 新增 `protocol_smoke`。这里的 P2P 指 point-to-point message 语义，不绑定 IP/TCP。
- [2026-04-25] 当前 protocol smoke 默认 3 domains、10 个 source blocks、20 条 K/V block messages、1 条 softmax state、1 条 terminate，总计 `protocol_messages=22`。
- [2026-04-25] 远端 NVIDIA GPU host 为 `192.168.8.172`；不要直接编辑远端源码，代码变更必须本地 commit/push 后由远端 `git pull` 同步。
- [2026-04-25] Rust 新增双进程 / 双机器 remote P2P pair smoke：`tcp_remote_pair` 用长度前缀 JSON frame 发送 `RingAttnMessage`，server role 监听，client role 主动连接。
- [2026-04-25] 双机 remote P2P smoke 已通过：远端 GPU `192.168.8.172` 监听 `0.0.0.0:29172`，本机 `192.168.8.204` 连接；client report 为 `sent=2 received=1`，server report 为 `sent=1 received=2`，三类消息 `kv_block` / `softmax_state` / `terminate` 均验证通过。
- [2026-04-25] 远端非交互 SSH 默认 PATH 不包含 cargo；启动远端 Rust smoke 时需显式加 `PATH=/home/stark/.cargo/bin:$PATH`，不要修改远端 shell 配置文件作为临时 workaround。
- [2026-04-25] `reports/**/*.json` 已改为默认 git ignore，并从 git 索引移除历史 report JSON；实验结论应优先沉淀到 docs / memory-bank，除非用户明确要求提交 raw JSON。
- [2026-04-25] Rust 新增 `cp_ring_node_runtime` smoke：默认 3 个 domain 各起一个 Rust thread，每个节点同时承担 inbound receiver 和 outbound peer，持续转发 10 个 source blocks 产生的 20 条 K/V messages，并记录 30 次 compute updates。
- [2026-04-25] C++ ATen/libtorch bridge 新增 `torch_attention_bridge`：在请求设备上实际计算小尺寸 `softmax(QK^T / sqrt(d))V`，并与 CPU reference 比较；本机非沙箱 MPS 已通过 `torch_attention_status=pass torch_attention_code=2`，远端 CUDA 已通过 `torch_attention_status=pass torch_attention_code=3`。
- [2026-04-25] 远端非交互 SSH 默认也不加载 `LIBTORCH` / `LIBTORCH_INCLUDE` / `LIBTORCH_LIB` / `LD_LIBRARY_PATH`；跑 CUDA libtorch smoke 时需和 `PATH=/home/stark/.cargo/bin:$PATH` 一起显式传入。
- [2026-04-25] Rust 新增 `tcp_remote_cp_node` dual-role smoke 并已双机通过：Mac `node_index=0` 和 GPU `node_index=1` 均同时作为 listener + outbound peer；每个 node `messages_sent=4 messages_received=4 compute_updates=8`。
- [2026-04-25] Rust smoke 新增 `torch_block_update_bridge`：`cp_ring_node_runtime` 的 `compute_updates=30` 会传入 C++ ATen bridge，并在请求设备上执行同等次数的 `softmax(QK^T / sqrt(d))V` block compute；本机非沙箱 MPS 已通过 `torch_block_update_code=2`，远端 CUDA 已通过 `torch_block_update_code=3`。
- [2026-04-25] Rust smoke 新增 `torch_payload_block_bridge`：`RingAttnMessage.payload` 已变为 float32 K/V bytes，`cp_ring_node_runtime` 会捕获 30 个 compute payload block 并逐块交给 C++ ATen；本机非沙箱 MPS 已通过 `torch_payload_block_code=2 torch_payload_blocks=30/30`，远端 CUDA 已通过 `torch_payload_block_code=3 torch_payload_blocks=30/30`。
- [2026-04-25] `tcp_remote_cp_node` 已接入 payload-backed compute 并双机通过：Mac node `torch_payload_block_code=2 torch_payload_blocks=8/8`，GPU node `torch_payload_block_code=3 torch_payload_blocks=8/8`。本次还修复了 macOS accepted stream 继承 nonblocking 导致大 payload frame 读取 `WouldBlock` 的问题。
- [2026-04-25] `tcp_remote_cp_node` 已扩展到 3-node remote forwarding 并验证通过：Mac node0 -> GPU node1 -> Mac node2 -> Mac node0；每个 node `messages_sent=8 messages_received=8 compute_updates=12`，MPS nodes `torch_payload_block_code=2 torch_payload_blocks=12/12`，CUDA node `torch_payload_block_code=3 torch_payload_blocks=12/12`。
- [2026-04-25] 新增 `torch_payload_online_bridge`：C++ ATen 在请求设备上按 captured K/V payload block 流维护 running max / running sum / output，并与 full attention CPU reference 对比。本机 MPS / 远端 CUDA 主 smoke 均通过 `30/30`，3-node remote CP 每个 node 均通过 `12/12`。
- [2026-04-25] 新增 `torch_payload_chunk_bridge`：C++ ATen 将 online softmax state 扩展到小尺寸 Q chunk，输出 `[query, head, dim]` chunk tensor。本机 MPS / 远端 CUDA 主 smoke 均通过 `30/30`，3-node remote CP 每个 node 均通过 `12/12`。
- [2026-04-26] 新增 `torch_query_chunk_bridge`：Rust/domain-side 生成显式 float32 Q chunk payload，C++ ATen bridge 消费该 Q payload 与 captured K/V payload blocks，不再在该路径内部构造 Q。本机非沙箱 MPS 主 smoke 通过 `torch_query_chunk_code=2 30/30`，远端 CUDA 主 smoke 通过 `torch_query_chunk_code=3 30/30`。
- [2026-04-26] 尝试重跑 3-node remote CP query chunk smoke 时，node2 先启动后连接 node0 超时退出；随后 GPU host `192.168.8.172` SSH 返回 `No route to host` / `Host is down`。本机已确认无残留 remote CP/SSH 进程，待网络稳定后重跑 3-node。

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
- [ ] 抽出统一 transport trait，收敛 `local_p2p_queue`、`cp_ring_node_runtime`、`tcp_remote_pair`、`tcp_remote_cp_node` 的共用 send/recv/frame 语义，并保持当前 message schema / report 字段稳定。
- [ ] 将 Rust/domain-side deterministic Q payload 升级为真实 domain-local model activation / state lifecycle，并明确 state ownership。
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
- Git push 纪律：在 `main` 分支完成 commit 后，可以 push 到配置好的远端；任何 `git push --force`、`git push -f` 或 force-push 变体都禁止使用。
- sudo / 系统改动纪律：如果修复需要 `sudo`、root-owned path、`/opt`、系统 linker 或机器级配置，应停止并给用户最小命令让用户自己执行；不要为了绕过 sudo 擅自修改第三方二进制、install_name 或 vendor artifacts。
- 本机硬件 smoke 纪律：Mac 本机 libtorch smoke 应使用 `HCP_TORCH_DEVICE=mps` 并在非沙箱/授权进程运行；CPU smoke 不代表本机加速器路径有效。
- GPU hardware smoke 纪律：远端 `HCP_TORCH_DEVICE=cuda:0` 必须看到 `torch_status=pass`、`torch_code=3`，仅有 correctness `passed=3/3` 不足以证明 CUDA 路径有效。
- 远端 GPU smoke 排查经验：若 `torch_code=-2` 或 `torch_code=-5`，不要怀疑 `cuda:0` 设备名；优先看 `torch_message`，并检查 `LIBTORCH`、`LIBTORCH_LIB`、`LD_LIBRARY_PATH` / rpath、`ldd` 是否显示 `libtorch_cuda.so` / `libc10_cuda.so`。
- Protocol smoke 纪律：`protocol_status=pass` 需要同时覆盖 K/V block、softmax state、terminate；K/V block message 数应等于 source blocks * (domain_count - 1)。
- P2P 语义纪律：P2P 表示 point-to-point、非 collective；不要把 HCP protocol 本身等同于 IP/TCP。
- Remote GPU 纪律：`192.168.8.172` 只通过 git 同步代码；不要在远端直接编辑源码。
- Remote P2P 纪律：双机验证不能用 `127.0.0.1` 作为结论；server 应监听 `0.0.0.0` 或目标子网地址，client 应连接 `192.168.8.x` 子网内的 GPU host。
- Report 纪律：`reports/**/*.json` 是生成产物，默认不提交；如需长期记录实验进展，写入 docs 或 memory-bank。
- Remote CP node 启动纪律：Mac 侧可先启动 listener，再启动 GPU 节点；GPU 能连入 Mac，但必须确保 Mac listener 已实际运行。
- CP block update 纪律：`torch_query_chunk_status=pass` 证明 CP 消息 K/V payload 和 Rust/domain-side Q payload 已驱动设备侧 Q chunk online softmax output smoke；但在 Q 来自真实 domain-local model state 并接入完整 lifecycle 前，不应把它描述为完整 Ring Attention kernel。

## 当前阻塞

- [2026-04-24] 当前沙箱环境不允许 Python worker 绑定本地端口，完整 Python smoke 会触发 `PermissionError: [Errno 1] Operation not permitted`。
- [2026-04-24] `ringattn_controller.py` 当前会将 `bytes` 放入 `json.dumps` payload，导致 `TypeError: Object of type bytes is not JSON serializable`。
- [2026-04-24] 普通 `cargo check` 会尝试访问 `rsproxy.cn` 并因 DNS 失败；当前使用 `cargo --offline` 可正常构建缓存依赖。
- [2026-04-24] PyTorch 2.11.0 在默认沙箱进程中 `mps_available=false`，原因是沙箱内 `MTLCopyAllDevices()` 返回 0；非沙箱进程可枚举 `Apple M1 Pro`，且 `torch.ones(..., device="mps")` 成功。
- [2026-04-24] 后续所有 Metal/MPS 相关验证必须在非沙箱/授权进程中运行；默认沙箱结果不能作为 MPS 不可用结论。
- [2026-04-25] GPU 端 `CARGO_OFFLINE=1` 失败且提示 `no matching package named serde_json found` 时，优先判定为 Cargo registry cache miss，不是 CUDA/libtorch 问题。
- [2026-04-26] 3-node remote CP query chunk smoke 当前阻塞在 GPU host 网络可达性：`ssh stark@192.168.8.172` 返回 `No route to host` / `Host is down`。
