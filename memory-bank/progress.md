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
- [x] [2026-04-25] 已记录本机 libtorch smoke 纪律：Mac 本机默认使用非沙箱 MPS；CPU-only smoke 只作为 fallback。
- [x] [2026-04-25] Rust smoke summary 已增加 `torch_status` / `torch_device` / `torch_code`，并把 `HCP_ENABLE_TORCH=1` 下的 torch bridge 失败计入整体失败。
- [x] [2026-04-25] Rust smoke 在 torch bridge 失败时会打印压缩 `torch_message`，避免远端 CUDA 失败只看到 `torch_code=-2`。
- [x] [2026-04-25] C++ ATen bridge 已增加 `at::hasCUDA()` preflight；CUDA backend 不可用时返回 `torch_code=-5`，避免误判为设备名错误。
- [x] [2026-04-25] Rust build script 已在 Linux CUDA libtorch 下用单个 linker group 强制保留 `libtorch_cuda` / `c10_cuda`，防止 registration libraries 被链接器或 rustc 参数重排丢弃。
- [x] [2026-04-25] 远端 CUDA libtorch smoke 已通过：`torch_status=pass torch_device=cuda:0 torch_code=3`，且 `ldd` 显示 `libtorch_cuda.so` / `libc10_cuda.so`。
- [x] [2026-04-25] Rust protocol smoke 已建立：本地 P2P queue transport 可序列化、发送、接收、解码 K/V block、softmax state、terminate 消息；P2P 语义不绑定 IP/TCP。
- [x] [2026-04-25] `bash scripts/run_rust_ringattn_smoke.sh` 已输出 `protocol_status=pass protocol_messages=22`；非沙箱 MPS smoke 同样通过。
- [x] [2026-04-25] Rust remote P2P pair smoke 已建立：`--remote-p2p-role server|client` 使用 `tcp_remote_pair` transport 跨进程发送同一套 `RingAttnMessage`。
- [x] [2026-04-25] 双机 remote P2P smoke 已通过：GPU 节点 `192.168.8.172` 监听 `0.0.0.0:29172`，本机 client 连接后验证 `kv_block`、`softmax_state`、`terminate` 三类消息；结构化 report 已落在 `reports/rust-remote-p2p-20260425-123104/`。
- [x] [2026-04-25] `reports/**/*.json` 已加入 `.gitignore` 并从 git 索引移除，后续 smoke report 不再默认造成 dirty worktree 或远端 pull 前 stash。
- [x] [2026-04-25] Rust `cp_ring_node_runtime` smoke 已通过：3 个 domain thread 同时扮演 inbound receiver + outbound peer，10 个 source blocks 产生 20 条跨域 K/V messages，并记录 30 次 compute updates。
- [x] [2026-04-25] C++ ATen/libtorch `torch_attention_bridge` 已建立并在本机 MPS 与远端 CUDA 通过：实际计算 `softmax(QK^T / sqrt(d))V`，再与 CPU reference 比较；MPS 显示 `torch_attention_code=2`，CUDA 显示 `torch_attention_code=3`。
- [x] [2026-04-25] Rust `tcp_remote_cp_node` 双机 smoke 已通过：Mac/GPU 两个进程都同时作为 listener + outbound peer，每个 node 发送 4 个 source blocks、接收 4 个 peer blocks、记录 8 次 compute updates。
- [x] [2026-04-25] Rust `torch_block_update_bridge` 已接入 `cp_ring_node_runtime.compute_updates()`：默认 30 次 CP update 会驱动 30 次 C++ ATen attention block compute；本机 MPS 与远端 CUDA 均已通过。
- [x] [2026-04-25] Rust `torch_payload_block_bridge` 已消费 `RingAttnMessage.payload` 中的 float32 K/V bytes：默认 30 个 captured CP payload blocks 在本机 MPS 与远端 CUDA 均已通过 payload-backed ATen block compute。
- [x] [2026-04-25] 双机 `tcp_remote_cp_node` payload-backed compute 已通过：Mac MPS node 与 GPU CUDA node 均发送 4、接收 4、compute_updates=8，并各自完成 `torch_payload_blocks=8/8`。
- [x] [2026-04-25] 修复 macOS remote CP accepted stream 非阻塞读大 payload frame 的 `WouldBlock` 问题：`accept_with_retry` accept 成功后显式恢复 blocking mode。
- [x] [2026-04-25] 3-node remote CP forwarding smoke 已通过：Mac node0、GPU node1、Mac node2 三个独立进程形成 ring，每个 node `sent=8 received=8 compute_updates=12`，并完成 payload-backed ATen compute。
- [x] [2026-04-25] `torch_payload_online_bridge` 已通过：设备侧逐 block 维护 online softmax state，并与 full attention CPU reference 对比；本机 MPS、远端 CUDA、3-node remote CP 均已验证。
- [x] [2026-04-25] `torch_payload_chunk_bridge` 已通过：设备侧对小尺寸 Q chunk 逐 block 维护 online softmax state，输出 chunk tensor 并与 CPU reference 对比；本机 MPS、远端 CUDA、3-node remote CP 均已验证。
- [x] [2026-04-26] `torch_query_chunk_bridge` 已通过主 smoke：Rust/domain-side 显式 Q chunk payload 与 captured K/V payload blocks 一起进入 C++ ATen bridge；本机非沙箱 MPS 显示 `torch_query_chunk_code=2 torch_query_chunk_blocks=30/30`，远端 CUDA 显示 `torch_query_chunk_code=3 torch_query_chunk_blocks=30/30`。
- [x] [2026-04-26] `torch_query_chunk_bridge` 已通过 3-node remote CP smoke：Mac node0 / GPU node1 / Mac node2 均 `sent=8 received=8 compute_updates=12`，MPS nodes `torch_query_chunk_code=2 torch_query_chunk_blocks=12/12`，CUDA node `torch_query_chunk_code=3 torch_query_chunk_blocks=12/12`。
- [x] [2026-04-26] `DomainModelState` 已接入 CP payload path：每个 domain 持有本地 Q chunk 与 K/V storage，source K/V block 从 state 切片，target compute capture 携带本地 Q payload；本机 MPS、远端 CUDA 和 3-node remote CP 均已验证。
- [x] [2026-04-26] 已新增统一 3-node remote CP launcher：`scripts/run_rust_remote_cp_3node_smoke.sh` 会自动发现 Mac 子网地址、远端 GPU `git pull --ff-only`、预构建本机/远端 Rust crate，并统一启动 node0/node2 MPS 与 node1 CUDA。
- [x] [2026-04-26] `RUN_ID=rust-remote-cp-output-unified-20260426` 三节点 remote CP output digest smoke 已通过：node0/node2 MPS `torch_query_output_code=2 torch_query_output_blocks=12/12`，node1 CUDA `torch_query_output_code=3 torch_query_output_blocks=12/12`。
- [x] [2026-04-27] 已新增 `LayerActivationState` / output slot ownership：每个 domain-local layer state 明确持有 Q chunk、K cache、V cache、output slot，并把 output slot 元数据写入 CP report。
- [x] [2026-04-27] 临时 VPN remote CP smoke 已通过：`GPU_HOST=100.118.253.68 MAC_NODE_ADDR=100.121.35.138 RUN_ID=rust-remote-cp-modelstate-vpn-20260426` 下 node0/node2 MPS 与 node1 CUDA 均 `sent=8 received=8 compute_updates=12 torch_query_output_blocks=12/12`。
- [x] [2026-04-29] 已新增 projection-first Q/K/V 路径：`ModelLayerWeights` + domain-local hidden states 生成 Q chunk、K cache、V cache，不再直接公式生成 Q/K/V bytes。
- [x] [2026-04-29] projection 路径已通过本机和远端验证：`cargo test --offline` 4/4、`cargo clippy --offline -- -D warnings`、本机 MPS smoke `torch_query_output_blocks=30/30`、LAN 3-node remote CP `RUN_ID=rust-remote-cp-projection-lan-20260429` 三节点 `torch_query_output_blocks=12/12`。
- [x] [2026-04-30] Cargo registry 在线模式已恢复可用（rsproxy-sparse 正常）。
- [x] [2026-04-30] 已新增 optional `tch = "0.24.0"` 和 `tch-backend` feature gate。
- [x] [2026-04-30] 已新增 `rust/src/bin/tch_smoke.rs`：matmul + softmax + attention-like 三组 op，与 CPU reference 对比误差，输出 JSON report。
- [x] [2026-04-30] 已新增 `scripts/run_tch_ringattn_smoke.sh`，自动处理 `LIBTORCH` 与 `DYLD_LIBRARY_PATH`。
- [x] [2026-04-30] tch-rs CPU smoke 通过：`tch_status=pass tch_device=cpu tch_code=1 ops=3/3`。
- [x] [2026-04-30] tch-rs MPS smoke 通过：`tch_status=pass tch_device=mps tch_code=2 ops=3/3`。
- [x] [2026-04-30] `tch_backend.rs` attention block update 已实现：`tch::Tensor` 替代 C++ `RunTorchAttentionBlockUpdates`，CPU `tch_attention_code=1`、MPS `tch_attention_code=2` 均通过。
- [x] [2026-04-30] `main.rs` 已接入 `tch_attention_bridge` report，与 C++ `torch_attention_bridge` 并行存在，互不影响。
- [x] [2026-04-30] 远端 CUDA tch smoke 验证通过：`tch_status=pass tch_device=cuda:0 tch_code=3`。
- [x] [2026-04-30] Linux `--as-needed` 导致 `libtorch_cuda.so` 被丢弃的问题已通过 `build.rs` 的 `cargo:rustc-link-arg-bins` 修复。
- [x] [2026-04-30] `run_rust_ringattn_smoke.sh` 默认在有 `LIBTORCH` 时自动启用 `tch-backend` feature。
- [x] [2026-04-30] `docs/TCH_BACKEND_DESIGN.md` 已补充架构设计、数据流、构建链接和验证矩阵。
- [x] [2026-04-30] `tch_backend.rs` 已实现全部 6 个 C++ ATen 桥接对标函数（attention block updates、payload block、payload online、payload chunk、query chunk、query chunk output）。
- [x] [2026-04-30] `main.rs` 已接入全部 5 个 tch payload/query bridge report，与 C++ bridge 并行；`Report` / `RemoteCpNodeRunReport` / `run()` / cp-node 路径 / CLI summary / fail message 打印均已同步。
- [x] [2026-04-30] 本机 CPU tch 全桥接 smoke 通过：`status=pass`，全部 6 个 bridge `code=1`，payload/query 各 `30/30`，query output `groups=3`。
- [x] [2026-04-30] 本机 MPS + 远端 CUDA tch 全桥接 smoke 通过。
- [x] [2026-04-30] 3-node remote CP tch 全桥接 smoke 通过：node0/node2 MPS `code=2 12/12`，node1 CUDA `code=3 12/12`；C++ bridge 与 tch bridge 并行全部通过。
- [x] [2026-04-30] `scripts/run_rust_remote_cp_node.sh` 和 `run_rust_remote_cp_3node_smoke.sh` 已支持自动启用 `tch-backend` feature 并传递 `HCP_TCH_DEVICE`。
- [x] [2026-04-30] tch-backend 已接入实时 compute 路径：`compute_chunk_attention_step` 在 `run_cp_ring_node` / `run_remote_cp_node` / `receive_remote_cp_node_messages` 中被逐 block 调用；accumulator 通过 `Arc<Mutex<>>` 在 remote CP 的收发线程间共享。
- [x] [2026-04-30] 实时 compute 验证通过：CPU/MPS 单节点 checksum 与离线后验 smoke 完全一致；`main.rs` CLI 已输出 `tch_compute_output_checksum`。
- [x] [2026-04-30] RoPE 已接入 protocol，Q/K 逐 token 应用旋转位置编码。
- [x] [2026-04-30] LayerNorm 已接入 protocol，projection 前对 hidden states 逐 token 归一化。
- [x] [2026-04-30] o_proj + Residual Connection 已接入 protocol，attention output 经 o_proj 映射回 hidden_dim 后与原始 hidden states 相加。
- [x] [2026-04-30] 外部权重加载已接入 protocol：支持通过 `HCP_WEIGHTS_JSON` 环境变量从 JSON 文件加载 Q/K/V/O projection weights 与 LayerNorm gamma/beta；`DomainModelState::new_with_weights` 在有外部权重时替换默认合成权重；本地 CPU/MPS smoke 均验证通过，checksum 随权重变化。
- [x] [2026-04-30] VPN 三节点 remote CP tch-full 验证通过：`GPU_HOST=100.118.253.68 MAC_NODE_ADDR=100.121.35.138 RUN_ID=rust-remote-cp-tch-full-vpn-20260430 PORT_BASE=29335`，node0/node2 MPS `code=2 12/12`，node1 CUDA `code=3 12/12`；C++ bridge 与 tch bridge 全部通过。

## 进行中

- [ ] M2：Rust online softmax correctness report 与 tolerance policy 扩展。
- [ ] M3-tch：将 Ring Attention block update 迁移到 `tch-backend`，与 C++ ATen bridge 并行存在。
- [ ] M3：抽出统一 transport trait，减少 local queue / TCP pair / TCP CP node 的重复 frame 与 metrics 逻辑。
- [ ] M4：heterogeneous runtime stubs 与配置 / 环境纪律。
- [x] M5：将 deterministic projection weights 升级为真实权重加载 / layer config，并接入 RoPE、norm/residual 与完整 layer lifecycle。
- [ ] M6：memory / bandwidth scaling notes 与 context-length growth argument。

## 已知问题

- [2026-04-24] 完整 Python smoke 在当前沙箱下无法绑定本地端口。
- [2026-04-24] `ringattn_controller.py` 存在 `bytes` JSON 序列化问题。
- [2026-04-24] `ringattn_kernel_stub.py` 已有 correctness JSON report 入口，但还没有整理成正式 M2 report 文档。
- [2026-04-24] `tch` crate 未在本机 cargo cache 中；system-wide libtorch 已就绪，剩余阻塞是 cargo registry/network 拉取 `tch` / `torch-sys`。
- [2026-04-24] MPS 排查结论：沙箱进程隐藏 Metal device，非沙箱进程下 PyTorch 2.11.0 的 MPS 可用。
- [2026-04-25] GPU 远端默认 `CARGO_OFFLINE=1` 时可能因 cargo cache 缺 `serde_json` 等基础依赖失败；这不是 CUDA smoke 结果，需要先在线 fetch 或放开一次 `CARGO_OFFLINE=0`。
- [2026-04-25] 本机 CPU-only libtorch smoke 不能作为 hardware smoke 结论；需要以非沙箱 MPS report 为准。
- [2026-04-25] 旧版 CLI 只打印 `torch_compiled=true`，不能证明 CUDA/MPS 实际执行；需使用包含 `torch_status` / `torch_code` 的新版 smoke。
- [2026-04-25] 远端 CUDA smoke 历史问题已解决：根因是 Linux 链接阶段未保留 `libtorch_cuda` / `c10_cuda` registration libraries。
- [2026-04-30] projection weights 已从 deterministic 初始化升级为支持外部 JSON 权重加载（`HCP_WEIGHTS_JSON`）；RoPE、LayerNorm、o_proj、residual 均已接入 protocol；M5 目标已完成。
- [2026-04-26] 3-node remote CP query chunk smoke 的一次失败根因是 Mac 子网地址从 `192.168.8.204` 变化到 `192.168.8.239`；后续重跑已通过。后续 remote smoke 前应先用 `ifconfig | rg 'inet 192\\.168\\.8\\.'` 确认当前 Mac 地址。
- [2026-04-30] GPU host 当前 VPN 地址为 `100.118.253.68`，Mac 本机 VPN 地址为 `100.121.35.138`；LAN 地址 `192.168.8.172` / `192.168.8.239` 目前不可达。remote smoke 需使用当前可达地址。

## 里程碑

| 里程碑 | 状态 | 目标日期 |
|--------|------|----------|
| M0: 独立化完成 | 已完成 | [2026-04-24] |
| M1: 问题定义固定 | 已完成 | [2026-04-24] |
| M2: 数学闭环 | 进行中 | 待定 |
| M3: 协议闭环 | 进行中 | 待定 |
| M4: 异构 runtime 闭环 | 未开始 | 待定 |
| M5: 远端闭环 | 已完成 | [2026-04-30] |
| M6: 扩展性论证 | 未开始 | 待定 |
