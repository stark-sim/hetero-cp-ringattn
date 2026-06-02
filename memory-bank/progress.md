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
- [x] [2026-05-31] **white (RTX 4090) Rust tch-backend 端到端验证**：`cargo check --features tch-backend` ✅，`cargo test --lib --features tch-backend` 55/55 pass ✅。本地 2-node loopback smoke（CPU 模式，coordinator + 2 workers）完全成功：24 层 ring attention 全通，生成 `"The answer to life is not a destination,"`，exit=0，workers 优雅退出。
- [x] [2026-05-31] **跨节点 Mac MPS + white CUDA 异构 ring attention 验证**：Tailscale VPN 下 prefill 64 tokens + decode 多步通过 24 层，QUIC KV ring 交换正常（229KB/micro_block）。Tailscale 高延迟导致 decode 慢（每 layer recv 0–13s），但 correctness 无 regression，逻辑已验证。
- [x] [2026-06-02] **三平台异构分布式推理首次成功**（Mac MPS + white RTX 4090 CUDA + pearl RX 9060 XT HIP）：
  - 3-domain ring attention 通过 Tailscale VPN 完成 prefill + 3-step decode
  - Coordinator 生成 `"The quick brown"`，exit=0，workers 优雅退出
  - 24 层 × 2 rounds KV ring 交换全通（28672 bytes/micro_block）
  - 三平台容量：Mac 8192 MB / white 20805 MB / pearl uint64_max
  - **证明 HCP Ring Attention 协议完全不依赖同构假设**，MPS/CUDA/HIP 任意组合均可协同推理
  - 这是项目历史上首次三异构平台联合验证
- [x] [2026-06-02] **pearl capacity uint64_max 修复**（commit `1025838`）：根因是 `LD_PRELOAD` 导致 `rocm-smi` 子进程崩溃（exit=134），修复为 `.env_remove("LD_PRELOAD")`。验证：pearl capacity 从 `uint64_max` → `13992 MB`。
- [x] [2026-06-02] **2-domain Mac MPS + pearl HIP 跨节点验证**：64-token smoke pass，exit=0，`generated: jumps over the lazy dog. The quick brown fox`。capacity 正确：Mac 8192 MB / pearl 13992 MB。white CUDA 暂时下线。
- [x] [2026-06-02] **M6 扩展性论证文档完成**（commit `f3dfa31`）：`docs/SCALING_ARGUMENT.md` 完成，涵盖 memory wall、single-node ceiling、distributed scaling、network bandwidth、operating envelope。使用 Qwen2-0.5B 作为 concrete reference。
- [x] [2026-06-02] **2-domain HTTP API 跨节点 E2E 验证**（commit `613c443`）：Mac MPS + pearl HIP，`/health` workers_connected=2，`/metrics` 正常，`/v1/completions` non-streaming 生成 `1. The`，SSE streaming OK（data: events + [DONE]）。
- [x] [2026-06-02] **2-domain 并发 HTTP API 验证**（commit `a5da680`）：同时提交 2 个 `/v1/completions` 请求，req1=`1. The`，req2=`The lazy dog`，两者同时完成无错误。验证 coordinator 并发调度能力。
- [x] [2026-06-02] **2-domain MPS+HIP 规模矩阵验证**（commit `5a239cf` → `9ab4659`）：64→512→1024→2048→4096→8192 tokens 全部通过，exit=0，生成连贯文本。有效带宽 ~7–9 MB/s（Tailscale VPN），compute 稳定（pearl ~1.2ms/layer, Mac ~0.6ms/layer），recv 与 KV block 大小线性增长。8K 时 KV cache 自动拆分为 2×14MB micro blocks，compute 上升到 ~34ms/micro_block（O(n²) attention）。文档：`docs/2domain_mps_hip_scale_matrix.md`。
- [x] [2026-06-02] **Harness Subagent Review 完成 + Review Fixes 提交**（commits `e546dba`, `fc0eac3`）：
  - Harness subagent 对异构可行性证明进行全面 review，结论：👍 Thumbs Up with Reservations
  - P0 修复：多 GPU capacity 查询 bug（`Device::Cuda(idx)` 索引现在传递给 `nvidia-smi --id={idx}` 和 `rocm-smi -d {idx}`）
  - P1 修复：ISSUE-001 关闭（填写 root_cause/impact/resolution/prevention，移动到 resolved/）
  - P1 修复：DESIGN.md 添加历史文档 deprecation banner
  - P1 修复：smoke 脚本添加 cleanup trap（EXIT/INT/TERM）
  - Review 报告详见 subagent 输出日志
- [x] [2026-05-31] **三平台 torch 2.11.0 版本统一完成**：
  - white (RTX 4090): torch 2.11.0+cu130、vllm 0.22.0、CUDA 13.0 ✅
  - pearl (RX 9060 XT): torch 2.11.0+rocm7.2、ROCm 7.2、HIP 计算正常 ✅
  - Mac (M1 Pro): torch 2.11.0 (CPU, vllm-metal 绑定) ✅
- [x] [2026-04-24] 本地 C++ smoke 在 `SKIP_PYTHON_SMOKE=1` 下通过；Python correctness 已降为显式 opt-in。
- [x] [2026-04-24] 已新增 Rust crate，实现纯 Rust Ring Attention correctness model。
- [x] [2026-04-24] 已新增 Rust -> C ABI -> C++ runtime bridge，并成功调用 C++ `NoOpRingAttnRuntime`。
- [x] [2026-04-24] 已验证 `HCP_ENABLE_TORCH=1` 下 Rust -> C++ ATen/libtorch smoke 可编译执行，report 中 `torch_bridge.compiled=true`。
- [x] [2026-04-24] `cargo clippy -- -D warnings` 通过。
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
- [x] [2026-04-29] projection 路径已通过本机和远端验证：`cargo test` 4/4、`cargo clippy -- -D warnings`、本机 MPS smoke `torch_query_output_blocks=30/30`、LAN 3-node remote CP `RUN_ID=rust-remote-cp-projection-lan-20260429` 三节点 `torch_query_output_blocks=12/12`。
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
- [x] [2026-04-30] M2 correctness 扩展完成：7 个 cases（含大 seq 和边界条件）；`max_rel_err` 已添加到 correctness model 和全部 tch bridge；`--stress-test` 支持 5-seed 随机验证。
- [x] [2026-04-30] M3 protocol 优化完成：TCP frame I/O 统一为 `write_frame_to_stream`/`read_frame_from_stream`；`process_inbound_message` 提取消除 CP node runtime 重复逻辑；SSH ConnectTimeout=30 修复 VPN 远程 smoke 超时。
- [x] [2026-04-30] M3-tch Step 1-3 完成：`backend.rs` `ring_attention` 非因果路径已从 `compute_chunk_attention_step` 迁移到 `process_kv_block` 纯 tensor online softmax；因果/非因果输出统一为 `[batch, num_heads, seq_len, head_dim]`；`compute_runtime.rs` `TchComputeRuntime::compute_kv_block` 已内联 tensor 实现；清理死代码（`compute_chunk_attention_step`、`tensor_to_q_payload`、`tensor_to_kv_payload`）。
- [x] [2026-04-30] M4 异构 runtime 闭环完成：`ComputeRuntime` trait 提取，`TchComputeRuntime` 真实计算，`NoOpComputeRuntime` 仅作 fallback；计算路径与协议逻辑解耦；本地 MPS smoke 和 VPN 三节点 remote CP 验证通过。
- [x] [2026-04-30] M3 transport trait 重构完成：`MessageSender` + `MessageReceiver` trait 定义在 `protocol.rs` 中；`TcpStream`、`mpsc::Sender/Receiver<Vec<u8>>` 均实现对应 trait；`cp_ring_node_smoke`、`run_remote_cp_node`、`run_remote_p2p_server/client` 全部迁移；删除 `send_cp_node_frame`、`write_raw_message_frame`、`read_raw_message_frame` 等重复函数；18/18 测试通过，clippy 零警告，smoke 通过。
- [x] [2026-04-30] `tch-backend` 设为默认 feature：`Cargo.toml` `default = ["tch-backend"]`；修复由此暴露的全部 46 个 clippy 错误；`cargo test` 18/18 通过，`cargo clippy -- -D warnings` 零警告。
- [x] [2026-04-30] 明确 HCP 与 PyTorch 官方 Context Parallel 的边界：HCP 采用原始 Ring Attention 论文的 P2P 设计，支持异构/非均分；PyTorch 2.7+ CP 是同构 GPU 集群的 collective 优化。已记录于 `systemPatterns.md`。
- [x] [2026-05-01] Phase 1 Checkpoint 1-4: `safetensors`/`tokenizers`/`half` 依赖；`ModelConfig` 解析 HF `config.json`（Llama/Qwen2/Mistral 家族）；`ModelWeights` 从 `.safetensors` 加载并转换 F16/BF16→F32；`RmsNorm`、可配置 `RotaryEmbedding`、`SwiGLU` MLP；`GqaAttention`（RoPE + GQA + causal mask + KV cache）；`LocalAttentionBackend`（`AttentionBackend` trait）；`DecoderLayer`（Pre/Post-Norm + Residual）；`LlamaModel`（Embedding → N-layer → RMSNorm → LM Head）；`Generator`（prefill + decode 自回归 + temperature 采样）；inference CLI（`--infer-model-dir`/`--infer-prompt`/`--infer-max-tokens`）；合成 tiny 模型验证 pipeline 端到端跑通。
- [x] [2026-05-01] 修复 `protocol.rs` 中 `output_slot` 初始化大小 bug（`MODEL_HIDDEN_DIM` → `KV_NUM_HEADS * KV_HEAD_DIM`）；修复 `domain_model_state_projects_qkv_from_hidden_states` 测试未应用 RoPE 的问题。
- [x] [2026-05-01] 修复 inference pipeline 中 `Tensor::embedding` 参数顺序错误（tch-rs 为 `embedding(weight, indices, ...)`）；修复 MPS float64 限制（RmsNorm eps / RoPE inv_freq / attention scale 全部转为 f32 tensor）。
- [x] [2026-05-01] 修复 inference pipeline 中 causal mask NaN bug：`make_causal_mask`/`create_causal_mask` 原用 `mask.to_kind(Float) * NEG_INFINITY` 产生 NaN，改为 `zeros(...).masked_fill(&mask, NEG_INFINITY)`。
- [x] [2026-05-01] 修复 `test_chunk_step_vs_softmax_single_block` 中 `actual` tensor 形状构造错误（多余 `permute` 导致维度交换）。
- [x] [2026-05-01] 修复 `ring_attention` causal mask 未传递给 `compute_chunk_attention_step` 的问题：当 `attention_mask.is_some()` 时，直接使用已 mask 的 `scores` 做 online softmax 更新。
- [x] [2026-05-01] Phase 2 Checkpoint 5: `HcpRingAttentionBackend` 接入真实推理路径；`LlamaModel::from_weights` 根据 `num_domains` 选择 `LocalAttentionBackend`（默认）或 `HcpRingAttentionBackend`；新增 `--infer-num-domains` CLI 参数；Qwen2-0.5B 在 `num_domains=1/2/4` 下 greedy decode 输出完全一致。
- [x] [2026-05-01] 修复 GQA `repeat_kv` 关键 bug：Rust 原实现 `x.repeat([1, n_rep, 1, 1])` 与 Python transformers 的 `repeat_kv`（expand+reshape，每个 head 连续重复）语义不一致。修复后 Qwen2-0.5B greedy decode 与 Python transformers 输出完全一致；19/19 测试通过，clippy 零警告。
- [x] [2026-05-01] CUDA 推理验证通过：远端 `sd-1` GPU 节点使用 `cuda:1` 运行 Qwen2-0.5B greedy decode，输出与本地 CPU/MPS 完全一致；`HCP_TORCH_DEVICE=cuda:1` 环境变量被 `infer.rs` 正确解析。
- [x] [2026-05-01] M2：Rust online softmax correctness report 与 tolerance policy 扩展完成。新增 `ToleranceTier` 三级策略（Strict/Relaxed/EndToEnd），`--tolerance-tier` CLI 参数支持运行时切换；correctness JSON report 输出 `tolerance_tier` 和 `tolerance` 字段；`tch_backend.rs` 5 处硬编码 tolerance 统一为模块常量；`model.rs` 端到端断言使用命名常量；全部 18 单元测试通过，clippy 零警告。
- [x] [2026-05-01] M2 数学闭环正式文档已沉淀：`docs/CORRECTNESS_REPORT.md` 包含验证目标、方法论、7-case 测试矩阵、实测 metrics、分级 tolerance 推导依据（float32 epsilon、FMA、累加顺序、业界 PyTorch/NumPy/ICON-A 参考）和运行方式；`docs/RINGATTN_MODEL.md` 同步更新 case 列表和 tolerance 描述；全部 case 在 Strict tier 下通过，max_rel_err 与 threshold 保持至少 10 倍以上余量。
- [x] [2026-05-01] Phase B: 分布式 decode 路径打通。`seq_len <= 1` 回退已移除，decode 阶段走完整 `ring_attention` 路径；`seq_offset` 传入 `AttentionBackend::set_distributed`，`forward` 使用固定 `seq_offset` 代替 `position_ids.min()` 计算 `global_seq_start`；decode 阶段发送 KV 时排除新 append 的 token，避免 ring 中重复；`kv_chunks` 改用本地 KV 长度而非 Q 的 `seq_len`；新增 `LlamaModel::global_seq_len` 保证 decode position_ids 正确。新增 4 个单元测试验证 decode 路径数学正确性；23/23 测试通过，clippy 零警告。
- [x] [2026-05-01] 修复分布式 decode ~6 diff 根因：`LlamaModel::forward` 中 prefill 阶段错误地将 `global_seq_len` 设为本地 `seq_len`（domain0=8），导致 decode 时 `position_ids=8` 而非正确的全局位置 16。修复后 diff 从 6.7 降至 ~2e-6，与单节点参考一致。
- [x] [2026-05-01] Phase B++: A) `test_tcp_kv_transport_roundtrip` 验证 TCP 序列化无损（k_diff=0, v_diff=0）；B) `test_distributed_llama_model_multi_step_decode` 验证 4 步连续分布式 decode，每步 diff ~2e-6。修复多步 decode 根因：`history_len = k.size()[2] - 1` 在多步时会包含之前 decode append 的 token，引入 `prefill_kv_len` 字段确保只发送 prefill 分区。
- [x] [2026-05-01] Phase C: 分布式 Generator `DistributedGenerator`。单进程模拟多 domain CP 推理：prefill 分片到各 domain → 同步 global_seq_len → decode 循环广播 token → 采样。`test_distributed_generator_tokens_match_reference`：4 步贪婪 decode，domain0/domain1 token 完全一致，与单节点参考 logits diff ~1e-5。31/31 测试通过，clippy 零警告。
- [x] [2026-05-01] Phase D: `RingAttnMessage` serialization/deserialization 测试覆盖。新增 5 个测试：bincode roundtrip、payload 完整性（256 bytes）、schema version 字段、三种 message kind 全覆盖、TCP transport trait 端到端。30/30 测试通过，clippy 零警告。
- [x] [2026-05-01] Phase 3 Step 1-5: `KvTransport` trait 与 `KvBlock` 创建；`MockKvTransport` 支持 in-memory 测试；`HcpRingAttentionBackend` 集成 `KvTransport`（`send_local_kv` + `process_peer_block` + `global_seq_start`）；`LinkedMockKvTransport` 修复自环 bug（`peer_inbox`/`self_inbox` 分离）；测试代码修复 layer transport 覆盖 bug（每层独立 transport pair）；`test_distributed_llama_model_prefill` 端到端分布式 prefill 通过（2-layer、GQA、seq_len=16 拆成 2 domain），diff=2.79e-6；关键代码已补充详细中文注释。
- [x] [2026-04-30] **真实多进程分布式推理（Real Multi-Process Distributed Inference）完成**：`distributed_worker.rs`、`distributed_coordinator.rs`、`distributed_protocol.rs` 创建；Worker 加载模型权重做 KV ring 交换，Coordinator 加载 tokenizer+config 做 prompt 分片和 token 广播；Handshake（domain_id 排序）、`SyncGlobalSeqLen` 广播、`BidirectionalTcpKvTransport` 每层独立 TCP stream。本地 2-node CPU ✅、远端 CPU+CUDA:1 ✅、本地 MPS+CPU ✅、跨机器 MPS+CUDA:1 ✅、3-domain 本地 CPU×3 ✅、3-domain 跨机器 MPS+CUDA×2 ✅。性能基准见 `activeContext.md`。
- [x] [2026-04-30] **QUIC Transport 实现与验证**：新增 `quinn`/`rustls`/`rcgen`/`tokio` 依赖；`rust/src/quic_transport.rs` 实现 `QuicKvTransport`（单 connection + per-layer bidirectional stream）；修复 rustls `CryptoProvider` 未初始化、2-domain 对称连接死锁、quinn `open_bi` 不发送 STREAM 帧导致 `accept_bi` 挂起（1-byte dummy workaround）。本地 2-domain/3-domain/MPS+CPU 均通过，输出与 TCP baseline 一致。**跨机器 QUIC vs TCP 性能对比**（Mac MPS + 远端 CUDA:1，VPN ~150ms RTT，Qwen2-0.5B 11 prompt + 20 decode tokens）：TCP 107.3s，QUIC 76.4s，QUIC 快 **~29%**。
- [x] [2026-04-30] **Mask 优化**：分布式 prefill 阶段 `model.rs` 不再创建 `[seq_len, seq_len]` 密集 causal mask。`ring_attention` 已通过 `global_seq_start` + position 比较实现 causal，从不读取 mask 张量数据。改为：单节点时仍创建完整 mask；分布式（`num_domains > 1`）时传 `[1,1,1,1]` dummy zero tensor 作为 causal 标志。本地 2-domain/3-domain CPU smoke 验证通过，输出一致。
- [x] [2026-04-30] **动态不均等分片 Phase 1**：Coordinator CLI 新增 `--chunk-sizes`（逗号分隔，如 `7,4` 或 `5,3,3`），显式指定每个 domain 的 prompt chunk 长度。分片逻辑校验：长度必须等于 `num_domains`，总和必须等于 prompt token 数。测试验证：2-domain `7+4=11` ✅、3-domain `5+3+3=11` ✅，生成结果与参考一致。
- [x] [2026-04-30] **修复 1-token prefill 边界 bug**：原代码用 `seq_len > 1` 区分 prefill/decode，chunk=1 时被误判为 decode（如 `--chunk-sizes 10,1`）。`LlamaModel` 和 `HcpRingAttentionBackend` 均新增 `is_prefill_done` 标志，第一次 `forward` 无论 `seq_len` 都走 prefill 路径。验证矩阵：2-domain (6,5/7,4/8,3/9,2/10,1)、3-domain (4,4,3/5,3,3/6,3,2/7,2,2)、4-domain (3,3,3,2/4,3,2,2) 全部通过，输出与参考一致。31/31 单元测试通过，clippy 零警告。

## 进行中

- [x] [2026-05-02] **Phase 2: Capacity-Aware 不均等分片**：
  - 新建 `rust/src/capacity.rs`：跨平台 device capacity 查询（`nvidia-smi` for CUDA、`sysctl`/`/proc/meminfo` for CPU/MPS）+ largest-remainder 比例分配算法 + 最小 1-token 保证。
  - 11 个单元测试覆盖 equal/2:1/3:1/zero-capacity/min-token/realistic MPS+CUDA 场景。
  - Handshake 协议扩展为 16-byte fixed（domain_id + capacity_mb）。
  - Coordinator 三层分配优先级：`--chunk-sizes` > `--capacity-aware` > 均分。
  - `WorkerCommand::Prefill` 扩展为包含 `seq_offset`，worker 动态更新所有 layer backend 的 `seq_offset`，确保 capacity-aware 模式下 causal mask 使用正确的全局位置。
  - `HcpRingAttentionBackend::set_distributed` 仅在显式提供 transport 时才替换 `kv_transport`（`None` 表示保持现有）。
  - **验证**：42/42 单元测试通过，clippy 零警告；本地 2-node CPU smoke 三种模式（baseline even / `--capacity-aware` / `--chunk-sizes 7,4` override）输出均一致（`" in the universe."`）。
- [x] [2026-05-02] **修复 QUIC KV ring 大 block 死锁**：quinn 默认 `stream_receive_window` (~1.2MB) 不足以容纳 GQA repeat 后的 KV block（~15MB for 2K seq），导致所有 worker 在 `SendStream::write_all` 中阻塞等待 window update，但接收方也在 `write_all` 中发送自己的 block → 分布式死锁。修复：显式设置 `stream_receive_window=32MB`、`receive_window=128MB`；`write_frame` 添加 `flush`。2-domain CPU 4K ✅ 43s、3-domain CPU 4K ✅ 49s、短 prompt regression ✅。
- [x] [2026-05-11] **Rust lib.rs 重构 Commits 3-7/15**：提取 report types、reference algorithm、correctness tests、C++/tch bridges、remote networking。lib.rs 从 ~2500 行降至 555 行（含 384 行 run_cli，待 Commit 14 提取 distributed 内容后将进一步精简）。cargo check 通过，45 tests 通过。
- [x] [2026-05-11] **Step 1: N-domain ring 拓扑去硬编码**：`runtime.rs` 移除 `num_domains == 2` 硬编码分支，统一为并发 dial+accept；`mock.rs` 新增 `create_ring(n)` 支持任意 N-domain 环形连接；45/45 tests passed，commit `b0c040d`
- [x] [2026-05-11] **Step 2: Layer 内 Overlap — Split-Phase Transport + Pipeline**：
  - `KvTransport` trait 扩展为 split-phase API：`submit_send`（提交异步发送）、`poll_recv`（非阻塞轮询接收）、`flush_send`（等待发送完成）；旧同步方法（`send_kv_block`/`recv_kv_block`/`exchange_kv_block`）基于新 API 提供默认实现，保持向后兼容
  - `quic.rs` 重写为 async task + channel 架构：独立 send task（mpsc channel → QUIC write_all）和 recv task（QUIC read → 反序列化 → mpsc channel），主线程通过 `block_on` 与 channel 交互
  - `tcp.rs` / `mock.rs` 实现同步 split-phase：submit 缓冲到内部 buffer，flush 时统一写入；Mock 覆盖 `recv_kv_block` 避免 trait 默认忙等
  - `ring_attention` 从"先全部 exchange → 再统一 compute"串行结构，重构为 4-phase pipeline：Phase 0 启动 send → Phase 1 本地 KV compute（与 send 重叠）→ Phase 2 循环接收→处理→转发（compute 与下一轮 I/O 重叠）→ Phase 3 flush → Phase 4 提取输出
  - `QChunkState` 结构：每个 Q chunk 独立维护 (rm, rs, obh)，支持本地 KV 和 peer KV 分阶段处理，状态在阶段间持久化
  - 关键 bug 修复：串行 Mock 测试中先运行 domain 的 inbox 为空，`poll_recv` 返回 None 时如果简单忙等会死循环；修复策略为 `poll_recv` 返回 None 后改用 `recv_kv_block` 做确认性阻塞尝试（Mock 直接返回 None → break；QUIC 阻塞等待数据 → 正确）
  - 全部 45 cargo tests 通过（含 4 个端到端分布式 model tests），零 regression
- [x] [2026-05-12] **Step 3: Micro KV Block + A/B Overlap Quantification**：
  - `KvBlock` 新增 `micro_block_idx` / `total_micro_blocks` 字段，支持 KV block 细粒度切分
  - `HcpRingAttentionBackend` 新增 `disable_overlap`（串行对照模式）和 `micro_kv_block_size`（环境变量 `HCP_MICRO_KV_BLOCK_SIZE`，0=禁用）
  - `ring_attention` 双模式实现：Pipeline 模式（默认，4-phase overlap）vs Serial 模式（`HCP_DISABLE_OVERLAP=1`，先 exchange 再 compute）
  - 本地 2-domain CPU smoke：pipeline 与 serial 模式输出完全一致（`generated:  is not a`），correctness 无 regression
  - 45 cargo tests 通过，commit `7a2d33f` 已推送至 main
  - 新建 `scripts/run_cross_node_ab_test.sh`：自动化跨节点 A/B 对比测试脚本，支持 baseline/optimized 多配置批量运行和 TSV 报告输出
  - **跨节点异构 A/B 验证通过**（Mac MPS + white RTX 4090 CUDA，64-token prompt）：Serial vs Pipeline 输出完全一致（`jumps over the`），correctness 无 regression
  - **256-token 量化对比**（Tailscale VPN 非 LAN）：Serial 151s vs Pipeline 147s，差异 -4s (~2.6%)
  - **512-token 量化对比**（Mac MPS + white RTX 4090，Tailscale VPN ~107ms RTT）：Serial ~5min (300s) vs Pipeline ~3min (180s)，**Pipeline 快 ~40%**。收益因 KV block 增大到 ~900KB/layer，网络传输占比提高
  - **512-token 量化对比**（sd-1 RTX 4080 SUPER + white RTX 4090，Tailscale VPN ~78ms RTT）—— **关键新发现**：
    * Serial no-micro-block: **299s** | Serial micro-block=64: **330s** | Pipeline micro-block=64: **319s**
    * Pipeline no-micro-block: **connection lost**（917KB/layer 大传输导致 QUIC 不稳定）
    * **同 micro-block 条件下 Pipeline 仅快 3.3%**（319s vs 330s），远低于 Mac+white 的 40%
    * **根因**：sd-1+white 双 CUDA 计算快 + RTT 更好（78ms vs 107ms）→ compute_time >> network_time → overlap 能隐藏的网络时间很少
    * **Micro block 是稳定性必需品**：无 micro block 时大传输导致 connection lost；micro block 本身增加 ~10% 开销（299s→330s）
    * **公式验证**：Pipeline 收益 ≈ 1 - compute/(compute+network)。Mac MPS 计算慢 → compute≈network → 收益大；双 CUDA 计算快 → compute>>network → 收益趋近于 0
    * **Scaling insight**：当前 512-token 传输量太小（~22MB/round），不足以拉开差距。随着 seq_len 增加（4K→175MB/round, 8K→350MB/round）和 domain 增加（4-domain→3 rounds），network_time 线性增长而 compute_time 增长较慢 → ratio 逆转 → Pipeline 收益将显著增大。4K/8K+多 domain 才是 Pipeline 真正的战场
  - **Mac + white 弱网 A/B 测试**（2026-05-22）：
    * 2-domain ring: Mac MPS + white RTX 4090 CUDA, Tailscale VPN (~380ms RTT)
    * **成功**: 64/256/512 tokens 全部完成（Serial + Pipeline）
    * **Pipeline 收益递减**: 64-token +5% (60s→57s) → 256-token +2% (211s→207s) → 512-token **-2%** (383s→390s)
    * **512 tokens 是弱网可靠上限**: 1024/2048/4096 全部失败（但根因不同）
    * **[x] 1024+ shutdown hang 已修复**（commit `c4dcfc5`）: `write_frame_quic` 无 timeout，worker 断开时 `send.write_all` 无限期 hang。新增 `write_frame_quic_timeout` / `send_command_quic_timeout`，coordinator `shutdown_workers()` 使用 10s timeout + `finish()` streams + `endpoint.close()` + 2s cleanup sleep。
    * **4096 pipeline**: 2404s (~40min) 后 network failed
    * 报告: `reports/mac-white-weaknet-ab-20260522/README.md`
    * **公式验证**: `benefit ≈ 1 - compute/(compute+network)` — Mac MPS 计算慢，小序列时 compute≈network 有收益；512+ 时 network>>compute，Pipeline overhead 超过收益
    * **根因排查**: Pipeline Phase 2 实现缺陷（ISSUE-001）— 收集完全部 peer blocks 后才统一 process，receive 与 compute 完全不重叠。Serial 已天然双向并发，Pipeline 额外 overlap 被 micro block overhead 抵消
    * **已修复**（commit `cbefc49`）: Pipeline Phase 2 改为逐个 block 接收→立刻 process→转发。45/45 tests passed，零 regression
  - **4K 4-domain 异构测试**（Mac MPS + sd-1 + sd-2 + white）：
    * **Serial 模式**：✅ **成功完成，4988s（1h 23m）**。根因修复：`quic.rs` mpsc channel buffer 从 2 增大到 64，解决了 N-domain Serial 模式下 24 个 layer blocks 同时提交导致的分布式死锁。4 个 worker 全部完成 prefill（global_seq_len=4096），decode 生成 1 token（`over`），exit=0。报告：`reports/cross-node-4domain-4k-serial-20260522/`
    * **Pipeline 模式**：❌ prefill 阶段 connection lost（2166s，~36min）。d0/d1 PrefillDone 收到后 d2/d3 连接断开。根因：Tailscale VPN 大传输量（~528MB/worker）+ 长时间 → QUIC 连接不稳定。Pipeline 逻辑本身正确（d0 日志显示 3 rounds × 24 layers 正常推进）
    * **结论**：Serial 4K 4-domain 首次验证成功，证明 channel buffer 修复有效；Pipeline 4K 4-domain 仍受 VPN 稳定性限制，需要 LAN 环境才能对比
  - **4K 本地验证**：Serial/Pipeline 均正常（CPU 本地 ~30s），代码逻辑无 bug
  - **4K 跨节点失败**：网络不稳定导致连接丢失。根因：7.3MB/layer × 24 layers ≈ 175MB 总传输量，跨 VPN 慢网络下大 block 传输不稳定。需 micro block 切分改善
  - **QUIC recv_kv_block timeout 修复**：120s → 600s（`3759811`），覆盖大 block + 慢网络场景
  - **分析报告**：`reports/ab-analysis-20260513/README.md`
- [ ] M6：memory / bandwidth scaling notes 与 context-length growth argument。

## 已知问题

- [2026-05-03] 远程 4090 已恢复并通过验证（统一 QUIC 控制面 + 单进程多 domain worker，8K seq）。
- [2026-05-02] 单卡多 worker 虽共享权重，但 LM head + KV cache 仍按 domain 数倍增，不能突破单卡显存上限。生产默认 1 worker/GPU，开发环境可尝试 2 worker/GPU。
- [2026-04-24] 完整 Python smoke 在当前沙箱下无法绑定本地端口。
- [2026-04-24] `ringattn_controller.py` 存在 `bytes` JSON 序列化问题。
- [x] [2026-05-01] `ringattn_kernel_stub.py` 的 correctness JSON 已进一步整理为正式 M2 report 文档：`docs/CORRECTNESS_REPORT.md`。
- [2026-04-24] `tch` crate 未在本机 cargo cache 中；system-wide libtorch 已就绪，剩余阻塞是 cargo registry/network 拉取 `tch` / `torch-sys`。
- [2026-04-24] MPS 排查结论：沙箱进程隐藏 Metal device，非沙箱进程下 PyTorch 2.11.0 的 MPS 可用。
- [2026-04-25] GPU 远端默认 `CARGO_OFFLINE=1` 时可能因 cargo cache 缺 `serde_json` 等基础依赖失败；这不是 CUDA smoke 结果，需要先在线 fetch 或放开一次 `CARGO_OFFLINE=0`。
- [2026-04-25] 本机 CPU-only libtorch smoke 不能作为 hardware smoke 结论；需要以非沙箱 MPS report 为准。
- [2026-04-25] 旧版 CLI 只打印 `torch_compiled=true`，不能证明 CUDA/MPS 实际执行；需使用包含 `torch_status` / `torch_code` 的新版 smoke。
- [2026-04-25] 远端 CUDA smoke 历史问题已解决：根因是 Linux 链接阶段未保留 `libtorch_cuda` / `c10_cuda` registration libraries。
- [2026-04-30] projection weights 已从 deterministic 初始化升级为支持外部 JSON 权重加载（`HCP_WEIGHTS_JSON`）；RoPE、LayerNorm、o_proj、residual 均已接入 protocol；M5 目标已完成。
- [2026-04-26] 3-node remote CP query chunk smoke 的一次失败根因是 Mac 子网地址从 `192.168.8.204` 变化到 `192.168.8.239`；后续重跑已通过。后续 remote smoke 前应先用 `ifconfig | rg 'inet 192\.168\.8\.'` 确认当前 Mac 地址。
- [2026-04-30] GPU host 当前 VPN 地址为 `100.118.253.68`，Mac 本机 VPN 地址为 `100.121.35.138`；LAN 地址 `192.168.8.172` / `192.168.8.239` 目前不可达。remote smoke 需使用当前可达地址。
- [2026-05-09] **远程 GPU 已切换到 white**（100.64.0.2, user stark）。sd-1（100.64.0.93）因网络/代理不稳定弃用。
- [2026-05-09] **vllm-metal EngineCore 子进程残留**：父进程异常退出时 EngineCore 子进程可能未清理（macOS spawn 模式）。已添加 `pkill -9 -f EngineCore` 清理逻辑，待添加优雅退出 handler。
- [2026-05-09] **vllm-metal 初始化时间长**：首次 gloo init ~60s + Metal kernel warmup ~10-20s（M1 Pro）。预热后约 8-10s。QUIC peer accept 超时已从 30s 延长到 180s 覆盖。
- [2026-05-01] 网络环境切换：CUDA 节点通过 `user@sd-1`（IP `100.64.0.93`）访问，Mac 本机当前可达地址为 `100.64.0.95`；remote smoke 需使用 IP 地址（`GPU_HOST=100.64.0.93`），因为 Rust socket 解析不支持主机名。
- [2026-05-01] 3-node remote CP smoke 通过：`RUN_ID=rust-remote-cp-sd1-ip2-20260501 PORT_BASE=29430 GPU_HOST=100.64.0.93 GPU_USER=user MAC_NODE_ADDR=100.64.0.95`；node0/node2 MPS 全部 bridge `code=2 12/12`，node1 CUDA 全部 bridge `code=3 12/12`；tch compute checksum 分别为 71.35 / 238.88 / 406.41。
- [x] [2026-05-01] Transport trait 重构后 3-node regression 验证通过：`PORT_BASE=29250 GPU_HOST=100.64.0.93 GPU_USER=user MAC_NODE_ADDR=100.64.0.95`；全部节点 `sent=8 received=8 compute_updates=12`，C++ bridge 与 tch bridge 全部 `12/12` pass；checksum 与重构前完全一致（71.35 / 238.88 / 406.41），未引入 regression。
- [x] [2026-05-01] **已修复：分布式 decode 端到端 logits 与单节点参考的 ~6 diff**。根因是 `LlamaModel::forward` prefill 阶段 `global_seq_len` 被错误设为本地 `seq_len`（8）而非全局位置（16），导致 decode 时 RoPE 应用了错误位置。修复后 diff 降至 ~2e-6。
- [x] [2026-05-01] QUIC 大 block 死锁修复：`quinn` 默认 `stream_receive_window=~1.2MB`、`send_window=~10MB` 远小于 GQA repeat 后 KV block frame（16K seq ≈ 117MB），导致所有 worker 在 `write_all` 中阻塞等待 window update → 经典死锁。修复：显式设置 `stream_receive_window=128MB`、`receive_window=128MB`、`send_window=256MB`。`write_frame` 同时增加 `flush()`。
- [x] [2026-05-02] MPS 单节点大 seq 验证：2K ✅ 3.95s / 4K ✅ 4.82s / 8K ✅ 16.4s / 16K ❌ OOM（14GB attention scores 超过 MPS allocator 单 buffer 上限）。
- [x] [2026-05-02] CUDA 单节点大 seq 验证：4K ✅ 5.10s / 8K ✅ 6.39s。
- [x] [2026-05-02] CUDA 分布式大 seq 验证：2-domain 4K ✅ 1m11s / 2-domain 8K ✅ 2m19s / **2-domain 16K ✅ 4m43s**，输出均与单节点参考一致（`jumps over the lazy`）。
- [x] [2026-05-02] **单进程多 domain worker 实现**：`distributed_worker.rs` 支持 `--local-domain-ids 0,1`（同进程多 domain），权重只加载一次（`Arc<ModelWeights>` + `shallow_clone`），每个 domain 独立 LlamaModel + KV cache + coordinator stream + QUIC endpoint。`KvTransport` trait 添加 `Send` bound。移除了 `forward_lock`（会导致 ring attention 死锁），依赖 `no_grad` + 自然 CUDA stream 串行化。本地 2-domain/4-domain CPU smoke 均通过。
- [x] [2026-05-03] **统一 QUIC 控制面**：`distributed_protocol.rs` 新增 QUIC 版 frame I/O；`distributed_coordinator.rs` 从 TCP listener 改为 QUIC endpoint；`distributed_worker.rs` 从 TCP connect 改为 QUIC connect。本地 2-domain CPU smoke ✅（`generated: . The lazy dog is`），远程 4090 CUDA 8K smoke ✅（`generated:  lazy dog. The quick`）。全链路统一 QUIC，消除 TCP 高延迟下的 EAGAIN 问题。
- [x] [2026-05-04] **MPS 兼容性修复**：`backend.rs` `process_kv_block` 中 `arange_start` 改在 CPU 创建后 `to_device`；`masked_fill` 替换为 `add+mul`；`max_dim` 替换为 `amax`（避免 MPS 后端 argmax bug）。本地 MPS 单节点 smoke 通过。
- [x] [2026-05-05] **MPS NaN 回归修复**：`add+mul` workaround 在 CPU 上产生 NaN（`0.0 * NEG_INFINITY = NaN`），改用 `where_self` 替代。42/42 测试通过。
- [x] [2026-05-05] **LM head 长序列优化**：`LlamaModel::forward` 中当 `seq_len > 8192` 时只计算最后一个 token 的 logits。所有调用方（Generator、distributed_worker）prefill 阶段均只使用最后一个 logits 进行采样。消除了 ~20GB 的 LM head 峰值，使 32K+ 单节点推理在 24GB GPU 上可行。
- [x] [2026-05-05] **dense causal mask 跳过**：单节点长序列 prefill 不再创建 `[seq_len, seq_len]` 密集 causal mask（64K 时达 16GB）。`HcpRingAttentionBackend` 已通过全局位置比较实现 causal，mask 张量仅作为标志使用。`seq_len > 8192` 时传 `[1,1,1,1]` dummy zero tensor。
- [x] [2026-05-05] **32K 单节点验证通过**：RTX 4090 上 Qwen2-0.5B 32K prefill + 5 decode tokens，`generated: dog. The quick brown`，显存峰值 ~12.4GB，耗时 ~8-10min。
- [x] [2026-05-05] **2-domain 分布式 32K 验证通过**：同机 RTX 4090 `--local-domain-ids 0,1`，domain0 prefill 16K + domain1 prefill 16K，KV ring 交换正常，`generated: dog. The quick brown`，耗时 3m53s。
- [x] [2026-05-05] **64K 单节点验证通过**：RTX 4090 上 Qwen2-0.5B 64K prefill + 5 decode tokens，`generated: the lazy dog. The`，显存峰值 ~13GB，耗时 ~15-20min。
- [x] [2026-05-05] **64K 分布式 2-domain 验证通过**：同机 RTX 4090 `--local-domain-ids 0,1`，实际 70001 tokens prefill + 5 decode，`generated: The answer is: The`，`EXIT=0`。prefill 成功验证了 `exchange_kv_block` 并发修复 + QUIC 大窗口配置。
- [x] [2026-05-05] **Decode 性能优化**（commit `491a46c`）：decode 阶段 `kv_chunk_size` 从 `1` 提升到 `2048`，避免 70001 个 tiny chunks 的开销。64K 分布式 decode 从 ~10min 降至 ~1-2min，总时间从 ~12-13min 降至 ~4min。
- [x] [2026-05-05] **64K 分布式死锁修复**（commit `f1b1040`）：代码层 `exchange_kv_block()` 并发 send+recv + 配置层 QUIC 窗口增大（512MB stream / 1GB conn）。根因和修复方案已记录于 `activeContext.md`。
- [x] [2026-05-05] **MLP chunking 修复 cuBLAS 执行失败**：128K 单节点 prefill 在 MLP 层触发 `CUBLAS_STATUS_EXECUTION_FAILED`。`layers.rs` 中 `Mlp::forward` 对 `seq_len > 8192` 做 chunking，峰值中间内存从 ~3.6GB 降到 ~225MB。Commit `632393f`。
- [x] [2026-05-05] **Attention projection chunking 修复 cuBLAS 执行失败**：131071 prefill 在 q/k/v/o_proj matmul 触发 `CUBLAS_STATUS_EXECUTION_FAILED`（M=131071 超出 cuBLAS 限制）。`GqaAttention::forward` 对 projection 做 chunking（`seq_len > 8192`），与 MLP chunking 配合后 131071 prefill-only 成功通过。Commit `0fd39d9`。
- [x] [2026-05-05] **128K 边界问题诊断**：prompt=131072 tokens 时 prefill 成功，但 decode 阶段 `position_ids=131072` 超出 Qwen2-0.5B `max_position_embeddings=131072`（有效索引 `[0, 131071]`），导致 RoPE `index_select` CUDA assert。添加 `max_position_embeddings` guard 返回友好错误信息（commit `1ef6288`）。改用 131067 token prompt（max_tokens=5）验证端到端。
- [x] [2026-05-05] **131067 单节点端到端验证通过**：RTX 4090 上 Qwen2-0.5B 131067 prefill + 5 decode tokens 成功完成，`EXIT=0`，输出 `gregated dog חדר.worm`。验证了 projection+MLP chunking + `max_position_embeddings` guard 后，单节点可处理 ~131K context（模型理论上限）。显存峰值 ~10.6GB，耗时 ~30-40min。
- [x] [2026-05-04] **QUIC idle timeout 修复**：`quic_transport.rs` 添加 `max_idle_timeout(300s)`，防止 prefill 阶段（2-3min）连接因空闲而断开。
- [x] [2026-05-04] **跨节点异构 worker CP prefill 验证**：Mac MPS (domain 0) + 远程 RTX 4090 CUDA (domain 1) 通过 VPN 协同完成 64-token prefill，`worker 0 prefill done global_seq_len=32` + `worker 1 prefill done global_seq_len=64`。脚本 `scripts/run_cross_node_2domain_smoke.sh` 已创建。
- [x] [2026-05-05] **跨节点异构 decode 端到端验证**：Mac MPS + RTX 4090 CUDA 跨 VPN 完成 9-token prompt + 3 decode tokens，`generated: I am a`，exit code 0。验证了异构设备间 QUIC KV ring 交换正常。
- [x] [2026-05-05] **本地同机异构 8K 验证**：Mac MPS + CPU 同机 2-domain 完成 8801-token prefill + 5 decode，`generated: The quick brown fox jumps`。验证了代码逻辑正确性和 MPS GPU 使用。
- [x] [2026-05-05] **QUIC 高 RTT 超时增强**（commit `dab3ebf`）：accept 30s、keep_alive 1s、idle 3600s、exchange 120s 超时、MTU 1200。全部 timeout 已调至极限。
- [x] [2026-05-05] **跨节点异构验证矩阵**（网络恢复后 RTT ~380ms）：
  - 9 tokens: ~60s ✅ (`I am a`)
  - 111 tokens: ~8min ✅ (`The quick brown fox jumps`)
  - 221 tokens: ~15min ✅ (`How many times does the`)
  - **551 tokens: ~30min ✅** (`What is the sentiment of`)
- Mac MPS + RTX 4090 CUDA 跨节点 QUIC KV ring 交换在 551-token（~3.5MB KV block）下稳定工作。

## 新增：部署指南与可插拔架构

- [x] [2026-05-05] **Python Worker SDK Phase 1 控制面验证通过**（commit `dabd6fc`）：`test_worker_control_plane.py` 端到端跑通 prefill → decode → shutdown。修复 `WorkerResponse` dataclass 类方法/字段同名冲突（`error` → `from_error`），bytes JSON 序列化改用 hex 编码。新增 `NoOpKvTransport` stub，单节点跳过 peer 连接和 KV exchange。
- [x] [2026-05-05] **Python Worker SDK Phase 2: 2-domain Mock KV Ring 验证通过**（commit `1cacb35`）：`TransformersBackend` 接入 `past_key_values`（`DynamicCache` for transformers 4.57.6），实现 `get_kv_block` / `apply_peer_kv` / `recalculate_logits`。新增 `LinkedMockKvTransport` 内存 queue 单进程模拟 2-domain KV ring。`test_worker_2domain.py` coordinator + 2 worker threads 端到端通过，prefill 分片 → KV ring 交换 → decode → shutdown 全链路正常。**2-domain 输出与单节点参考完全一致**（` generated: ' in the universe. The'`）。修复 `_layer_kv_start` 按层跟踪、全局/本地 seq 索引转换、decode 阶段跳过 KV exchange 避免 cache 污染等关键 bug。
- [x] [2026-05-05] **Python Worker SDK Phase 1.5 vLLM Adapter MVP 验证通过**（commit `29bab9b`）：`VllmBackend` 接入 vLLM 0.6.4 `LLM` API，在远程 RTX 4090 上加载 Qwen2-0.5B，控制面 Prefill/Decode/Shutdown 全链路通过。输出与 transformers baseline 一致（` in the universe`）。vLLM 采用 --no-deps 安装 + 手动补核心依赖，避免 serving 重依赖树。
- [x] [2026-05-06] **Python Worker SDK Phase 2.5 vLLM 控制面流程验证**（commit `fd8649c` + `df22de7`）：新增 `test_worker_single.py` 自动化单 worker + coordinator 端到端验证。远端 RTX 4090 上单 vLLM worker 验证通过：Prefill/Decode/Shutdown 全链路正常，输出 ` in the universe. The` 与 transformers 参考完全一致。单卡双 vLLM 实例确认不可行（vLLM 显存管理假设独占 GPU），方案 B（多机/多卡每设备单实例）是正确路径。
- [x] [2026-05-06] **Python QUIC Transport 实现 + Python↔Rust 互通验证**（commit `16c14e7` + `99b9ceb`）：基于 `aioquic` 实现 `QuicKvTransport`，帧格式与 Rust `quinn` 完全兼容（4-byte BE length prefix + JSON metadata + raw f32 bytes + 1-byte dummy handshake）。自签名证书生成 + 开发环境跳过验证。`test_quic_kv.py` Python 内部闭环验证通过 ✅。`test_quic_python_rust.py` + Rust `quic-echo-server` 互通验证通过 ✅。
- [x] [2026-05-07] **Phase 3.1: Python bincode 编解码器 + QUIC 控制面接入 Rust coordinator**（commit `fc6a6de`）：
  - `python/hcp_worker_sdk/bincode.py`: 手写 bincode 1.3 兼容编解码器（enum tag u32 LE, usize/i64/u64 8-byte LE, Vec 8-byte len prefix）。6 个单元测试与 Rust `bincode::serialize` 输出逐 byte 匹配 ✅。
  - `python/hcp_worker_sdk/quic_control.py`: `QuicControlClient` 基于 `aioquic`，Handshake 16-byte LE + Command/Response [4-byte BE length][bincode payload]，与 Rust `distributed_protocol.rs` 完全互通。
  - `python/test_python_worker_rust_coord.py`: mock worker 端到端测试，完整控制面流程 Handshake → Prefill → SyncGlobalSeqLen → 3×Decode → Shutdown 通过 ✅。
  - `python/hcp_vllm_quic_worker.py`: vLLM backend QUIC worker，ready for 远程 GPU 部署。
- [x] [2026-05-07] **Phase 3.2: Python↔Python QUIC KV ring 交换验证**（commit `24d1fb8`）：
  - `python/hcp_worker_sdk/quic_server.py`: `QuicWorkerServer` 同时支持 QUIC 控制面（coordinator）和 QUIC 数据面（peer KV ring）
  - `test_quic_kv_ring.py`: 基础 Python↔Python KV block roundtrip ✅
  - `test_quic_kv_ring_concurrent.py`: 并发 send+recv 无死锁 ✅
  - `test_two_python_workers_quic.py`: 2 个 mock worker + Rust coordinator 全链路通过 ✅
  - 关键修复：aioquic client 创建 stream 后需写 1-byte dummy 触发 server `stream_handler`；`exchange_kv_block` 改用并发 send+recv
- [x] [2026-05-07] **Phase 3.3: 跨机器异构端到端验证通过**（commit `6b6265f`）：
  - `python/hcp_transformers_quic_worker.py`: Mac 端 TransformersBackend worker
  - `python/test_transformers_2domain_quic.py`: 本地同机 2-domain 分布式验证通过 ✅
  - `scripts/run_python_distributed_2node.sh`: 自动化跨机器启动脚本（Mac+GPU）
  - 远程 GPU 通过 `HTTP_PROXY=http://127.0.0.1:7890` 下载模型，vLLM 加载成功
  - **验证结果**：Mac CPU (worker 0, capacity=4096 MB) + RTX 4090 CUDA (worker 1, capacity=14467 MB)
  - Coordinator `generated: . I am`，vLLM decode ~155 it/s
  - 控制面 QUIC+bincode，数据面 QUIC KV ring，全部正常
- [x] [2026-05-08] **Mac 端 vLLM Metal backend 安装并单节点验证通过**（commit `b3e7c95`）：
  - vllm-metal 0.2.0 + vLLM 0.20.1+cpu 通过 install.sh 安装在 `~/.venv-vllm-metal`
  - `VllmBackend` 新增 `_vllm_generate()` 适配层，兼容 vLLM 0.6.x (`prompt_token_ids`) 和 0.20.x (`prompts`)
  - Mac worker 使用 vllm-metal 运行，模型加载在 **MPS (Metal GPU)** 上
  - 单节点测试：coordinator + vllm-metal worker，Prefill + 3×Decode + Shutdown 全部正常，输出 `generated: ! I'm`
- [x] [2026-05-09] **远程 GPU 从 sd-1 切换到 white**（100.64.0.2, user stark, RTX 4090）：sd-1 有网络/代理不稳定问题。white 环境已搭建完成。
- [x] [2026-05-09] **Python 包管理全面迁移到 uv**：本地 Mac `~/.venv-vllm-metal`，远程 white `~/venv-vllm`。不再使用 conda。
- [x] [2026-05-09] **white 远程环境搭建完成**：uv 0.11.7, Python 3.11.15, torch 2.5.1+cu124, vLLM 0.6.4, transformers 4.45.2, aioquic 1.3.0。model.safetensors 已复制到 white。
- [x] [2026-05-09] **vllm-metal 跨机器 E2E 超时根因定位**：worker 0 (Mac vllm-metal) 初始化 81s > worker 1 (white) peer accept 30s 超时，导致 worker 1 先退出。已修改 `quic_server.py` 超时（peer connect 10→30s, peer accept 30→180s）。
- [x] [2026-05-09] **vllm-metal EngineCore spawn 保护**：macOS 默认 spawn 模式，入口脚本必须有 `if __name__ == '__main__':` 保护，否则 EngineCore 子进程重新导入主模块导致递归崩溃。
- [x] [2026-05-09] **跨机器异构 E2E 验证通过（vllm-metal + white RTX 4090）**：`scripts/run_python_distributed_2node.sh` 成功运行，Mac vllm-metal (MPS, 预热后 8.39s 初始化) + white RTX 4090 (vLLM 0.6.4 CUDA) 完整端到端通过。Coordinator 输出 `generated: . I am`。QUIC 控制面 + 数据面双链路稳定，异构后端协同正常。
- [x] [2026-05-09] **大规模跨机器验证矩阵完成**（一个节点一个 worker，Mac vllm-metal + white RTX 4090）：
  - T0 回归（2 tokens + 3 decode）：`. I am` ✅ ~40s
  - T1 规模（111 tokens + 5 decode）：`quick brown fox jumps over` ✅ ~2min
  - T2 极限（551 tokens + 5 decode）：`100 dog.` ✅ ~40s
  - 性能亮点：white RTX 4090 551-token prefill 达 968 tok/s，decode 105-109 it/s；vllm-metal warm-up 后 276-token prefill 仅 1.10s
  - 与 Rust 基线对比：Python Worker SDK + vLLM 侧 551 tokens 仅需 ~40s，远低于 Rust 基线 ~30min（vLLM optimized kernels 优势明显）
- [x] [2026-05-09] **EngineCore 优雅退出实现并验证**：
  - `VllmBackend.shutdown()` 跨版本 cleanup（vLLM 0.6.x / 0.20.x）
  - `QuicWorkerServer` 支持 `shutdown_event` 信号中断
  - `hcp_vllm_quic_worker.py` SIGTERM/SIGINT handler + finally cleanup
  - cleanup 脚本改为先 graceful、后 fallback
  - E2E 验证无 EngineCore 残留 ✅
- [x] [2026-05-09] **更长序列不均等分片验证（25% Mac / 75% CUDA）**：
  - T3: 1024 tokens + 5 decode, chunk-sizes 256,768 → `jumps over the lazy dog` ✅
  - T4: 2048 tokens + 5 decode, chunk-sizes 512,1536 → `dog jumps over the lazy` ✅
  - Mac MPS 512-token prefill 1.69s (303 tok/s)，white RTX 4090 1536-token prefill ~0.32s (4788 tok/s)
  - 验证了 coordinator `--chunk-sizes` 不均等分片在跨机器异构场景下稳定工作
- [x] [2026-05-05] **Rust Worker SDK 实现完成**：`worker_sdk/backend.rs` (`WorkerBackend` trait)、`worker_sdk/runtime.rs` (`WorkerRuntime<B>` 协议循环)、`worker_sdk/tch_backend.rs` (`TchWorkerBackend` 默认 tch-rs 后端)、`distributed_worker.rs` 重构为薄壳（解析参数 → 创建后端 → 运行 runtime）。`cargo test` 42/42 通过，SDK 相关 clippy 警告已清理。
  - 解耦目的：协议层与模型计算层完全分离，外部框架（vLLM/TensorRT-LLM/MLX）只需实现 `WorkerBackend` trait 即可接入 HCP 分布式网络。
  - 单元测试验证：分布式 prefill（`test_distributed_llama_model_prefill` diff=2.79e-6 ✅）、4-step decode（`test_distributed_llama_model_decode` diff~2e-6 ✅）、generator token 一致性（`test_distributed_generator_tokens_match_reference` logits diff~1e-5 ✅）。
  - **真实权重性能回归测试**（Qwen2-0.5B）：单节点 CPU 2.39s（vs 历史 2.87s ✅）、单节点 MPS 2.19s（vs 历史 2.06s ✅）、2-domain CPU 分布式 9.59s（decode 20 tokens）✅。全部配置输出一致，性能无回归。
  - **跨节点异构验证**（Mac MPS + RTX 4090 CUDA）：11-token prompt + 5 decode，`generated: The quick brown fox jumps`，exit=0，~40s。`WorkerRuntime` + `TchWorkerBackend` 在真实异构网络环境中端到端通过。
- [x] [2026-05-09] **Rust Static Batching 实现与 correctness 验证**：
  - `BatchGenerator`：支持 batch > 1 的并行 prefill + decode，等长 prompts 约束（避免 padding mask 引入 correctness 风险）
  - `generate_batch` / `generate_batch_from_ids`：从 prompts 或原始 token IDs 启动 batch generation
  - 早期停止：单 request EOS 后继续喂 0 token，KV cache 形状保持一致，不影响其他 request 的 attention 计算
  - `test_batch_forward_correctness`：验证 `LlamaModel::forward` batch=2 时，每个 sample 的 logits 与独立 batch=1 一致（diff ~1e-6），4-step decode token 完全一致
  - `test_batch_generator_correctness`：验证 `BatchGenerator` batch=2 的输出与两个独立单 request 生成结果完全一致
  - 全部 24 个 model tests 通过，无 regression
- [x] [2026-05-09] **Rust 分布式推理服务化**：
  - Protocol 添加 `request_id`：`WorkerCommand`/`WorkerResponse` 所有 variant 携带 request ID，支持多请求生命周期隔离
  - Worker 新请求自动隔离：`TchWorkerBackend::prefill` 在每次 Prefill 时自动重建 KV cache，避免旧请求污染新请求
  - Worker 优雅退出：`WorkerRuntime::run()` 检测到 connection lost / stream closed 时正常返回 Ok，不再 panic
  - Coordinator 多请求串行处理：新增 `--prompts-file` 参数（每行一个 prompt），循环处理每个请求，错误只影响当前请求
  - 本地 2-domain CPU smoke 验证：2 个短 prompt 串行处理（`The answer to life` → ` is not a`，`Once upon a time` → `, there was`），Worker 优雅退出，无 panic，全部 45 tests 通过 ✅
- [x] [2026-05-11] **跨节点异构多请求 E2E 验证通过**（Mac MPS + white RTX 4090 CUDA）：
  - 根因：启动脚本 `DYLD_LIBRARY_PATH` export 在 coordinator 启动之后，dyld 找不到 `libtorch_cpu.dylib` → SIGABRT（Abort trap 6）
  - 修复：环境变量 export 提到所有 libtorch-linked binary 启动之前；远程 worker 补 `LD_LIBRARY_PATH`
  - 验证：2 个 prompt 串行处理，exit=0，Worker 0/1 均优雅退出，零 panic ✅
- [x] [2026-05-11] **工程化代码重构**（14 commits）：
  - 将纯 binary crate 重构为 **lib + bin** 结构，`main.rs` 从 2,694 行精简到 3 行
  - 按"操作对象"分组拆分大文件，创建 20+ 个新模块：
    - `protocol.rs` (2,671) → `protocol/message.rs` + `transport.rs` + `framing.rs` + `node.rs`
    - `model/backend.rs` (1,282) → `model/attention/backend.rs` + `attention/ring.rs`
    - `model/generate.rs` (678) → `model/sampling.rs` + `generator.rs` + `distributed_generator.rs`
    - `model/layers.rs` (584) → `layers/norm.rs` + `rotary.rs` + `mlp.rs` + `attention.rs`
    - `model/kv_transport.rs` (367) → `model/transport/block.rs` + `trait.rs` + `tcp.rs` + `mock.rs`
    - 新建 `cli.rs`, `error.rs`, `report.rs`, `remote.rs`, `smoke/`, `distributed/`
  - 全部 45 cargo tests 通过，零 regression
  - 已推送至 main 分支
- [x] [2026-04-30] **手动部署指南** (`docs/DEPLOYMENT_GUIDE.md`)：从零开始的手动部署文档，覆盖：
  - 单节点本地部署（Mac MPS / GPU CUDA 独立验证）
  - 双节点异构部署（Mac MPS + remote RTX 4090 CUDA）完整步骤
  - 统一启动脚本使用说明
  - 高级配置（手动不均等分片、capacity-aware、单进程多 domain、高延迟 VPN 调参）
  - 故障排查手册（coordinator 卡 waiting、worker connect 失败、CUDA 不可用、MPS NaN、长序列 OOM）
  - 验证清单
- [x] [2026-04-30] **可插拔域内后端架构** (`docs/PLUGIN_ARCHITECTURE.md`)：
  - 核心思想：HCP 定义跨域 P2P 协议，域内实现是黑盒
  - 接口契约：控制面（WorkerCommand/WorkerResponse）、数据面（KvTransport）、模型面（HcpWorkerBackend）
  - 适配器分层设计：框架原生层 → HCP 适配层 → HCP 协议层
  - vLLM 适配器设计决策（PagedAttention KV 格式转换、后处理模式、online softmax 实现位置）
  - 最小 MVP 实现路径（Phase 0-4）
  - Python Worker SDK 接口定义参考
  - TensorRT-LLM / MLX 适配要点
- [x] [2026-04-30] **vLLM Worker 适配器详细设计** (`docs/VLLM_INTEGRATION.md`)：
  - vLLM 选型理由（PagedAttention、Continuous Batching、CUDA kernel）
  - 系统架构图（Coordinator → vLLM + HCP Adapter）
  - 三大挑战与解决方案：PagedAttention KV cache 格式转换、attention 替换策略（非侵入式 Wrapper）、模型并行 all-gather
  - 最小可运行代码骨架（`hcp_vllm_worker.py`）
  - 部署命令示例（同构 vLLM 集群、异构混合 Mac+GPU）
  - 性能预期与风险缓解
- [x] [2026-04-30] **Python Worker SDK** (`python/hcp_worker_sdk/`)：
  - `types.py`: `KvBlock`, `WorkerCommand`, `WorkerResponse`, `WorkerHandshake`（bincode/JSON 序列化）
  - `backend.py`: `HcpWorkerBackend` 抽象接口（load_model/prefill/decode/get_kv_block/apply_peer_kv/capacity_mb）
  - `transport.py`: `KvTransport` + `TcpKvTransport`（length-prefixed JSON + raw f32 bytes）
  - `server.py`: `HcpWorkerServer` 通用事件循环（handshake → command loop → response）

## 里程碑

| 里程碑 | 状态 | 目标日期 |
|--------|------|----------|
| M0: 独立化完成 | 已完成 | [2026-04-24] |
| M1: 问题定义固定 | 已完成 | [2026-04-24] |
| M2: 数学闭环 | 已完成 | [2026-04-30] |
| M3: 协议闭环 | 已完成 | [2026-04-30] |
| M4: 异构 runtime 闭环 | 已完成 | [2026-04-30] |
| M5: 远端闭环 | 已完成 | [2026-04-30] |
| M5+: 单进程多 domain worker | 已完成 | [2026-05-02] 本地 CPU 验证通过 |
| M5++: 统一 QUIC 控制面 | 已完成 | [2026-05-03] 本地 CPU + 远程 4090 CUDA 验证通过 |
| M5+++: 跨节点异构 worker CP | **已完成** | [2026-05-05] Mac MPS + RTX 4090 CUDA prefill ✅、decode 端到端 ✅（9-token prompt + 3 decode tokens） |
| M6: 扩展性论证 | **已完成** | [2026-05-05] 32K/64K/131067 单节点 ✅、32K/64K 分布式 ✅ |
| M7: Python Worker SDK + vLLM 异构 E2E | **控制面已完成** | [2026-05-09] Mac vllm-metal + white RTX 4090 vLLM 跨机器控制面（Prefill/Decode/Shutdown）通过，但 KV 数据面仍为 stub |
| M8: Transformers 真实 KV + online softmax correctness | **已完成** | [2026-05-09] `test_worker_2domain.py` (mock) ✅、`test_transformers_2domain_quic.py` (QUIC) ✅。`recalculate_logits()` + `DynamicCache` 兼容层 + 仅最后一个 domain 重算 |
| M9: 冻结 Python 层，聚焦 Rust + C++ + libtorch | **已决策** | [2026-05-09] Python 层进入维护模式不再扩展，Rust 层是唯一主干 |
| M10: Rust Static Batching | **已完成** | [2026-05-09] `BatchGenerator` 支持 batch > 1 的 prefill/decode，correctness 验证通过（batch=2 vs 两个独立 batch=1 完全一致），24/24 tests 无 regression |
| M10.1: Rust 分布式推理服务化 | **已完成** | [2026-05-09] Protocol 添加 request_id、Worker 新请求自动隔离 KV cache、Coordinator 支持 `--prompts-file` 多请求串行处理、Worker 优雅退出。本地 2-domain CPU smoke 验证 2 个 prompt 串行通过，无 panic |
| M10.2: Rust HTTP API 服务化 | **已完成** | [2026-05-22] axum OpenAI-compatible `/v1/completions` + `/health` + `/metrics`。Coordinator 双模式：batch vs HTTP API（默认）。Request queue + oneshot。45/45 tests passed。Commit `e3eafe9`。本地 E2E (`test_http_api_local.sh`) 21s ✅ `jumps over the lazy dog`。跨节点异构 E2E (`test_http_api_cross_node.sh`) 59s ✅ Mac MPS + white RTX 4090 CUDA，Tailscale VPN，`generated: 1` |
| M10.3: Rust Request-Level Parallelism | **已完成** | [2026-05-22] Coordinator 并发请求处理：`Arc<Mutex>` 保护 `worker_streams`，`rt.spawn_blocking()` 并发处理，`max_concurrent=4` 信号量，`ActiveRequestGuard` RAII。`/metrics` 新增 `queued_requests` + `active_requests`。本地并发 E2E (`test_http_api_concurrent_local.sh`) 通过：2 个请求同时提交均正确返回，metrics 计数器准确。45/45 tests passed。Commit `5ffa83f` |
| M10.3.1: Worker 多请求状态隔离修复 | **已完成** | [2026-05-23] Harness Reviewer 独立验证发现并发请求下 worker panic。Bug A: `HcpRingAttentionBackend::is_prefill_done` / `prefill_kv_len` 未在请求间重置 → `set_distributed()` 中重置。Bug B: prompt tokens < num_domains 时空 chunk → `process_single_request` 中检查 size==0 并返回错误。Reviewer 验证 5/5 通过（编译、单元测试、并发 E2E、顺序请求、panic 检查），confidence=high。Commit `eb71401` |
| M10.4: Rust 性能优化与生产化 | **待启动** | 量化（暂不实施，correctness 优先）、RDMA transport |
| M11: vLLM Block-Aware Ring | **远景** | [2026-05-09] 核心洞察：ring 在 vLLM block 层面运作。详见 `docs/BLOCK_RING_FUSION.md` |
| M13 Phase 1-2: Continuous Batching Scheduler + Per-Request KV Cache | **已完成** | [2026-05-23] Coordinator `BatchScheduler` 迭代调度 + `DecodeBatch` 协议 + Worker `RequestContext` per-request KV cache 隔离。本地 E2E：单请求 regression ✅ (`jumps over the`)，2 请求 batch ✅ (`jumps over the` + `, there was`)。48/48 tests passed。Commit `ea111c9` |
| M12: PagedAttention Block Table | **已完成** | [2026-05-24] `KvCache` trait 抽象 + `ContiguousKvCache` refactor + `BlockTableKvCache` prototype。`AttentionBackend::forward` 接受 `&mut dyn KvCache`。可行性研究：vLLM 单节点 backend = HIGH，vLLM KV ring = LOW。推荐混合架构。53/53 tests passed。Commits: `cfe4cde`, `fc5c15d`, `c98e45f` |
| M12.1: BlockTable 接入真实推理路径 | **已完成** | [2026-05-09] `KvCacheImpl` enum 替代硬编码类型，`create_kv_caches()` 支持 `HCP_KV_CACHE_BLOCK_TABLE=1` 运行时切换。集成测试 `test_block_table_through_model_forward` 验证 prefill + decode 全路径 diff < 1e-6。Commit `3efbdf0` |
| M12.2: BlockTable 分布式 E2E 验证 | **已完成** | [2026-05-09] `test_block_table_e2e_local.sh` 运行 2-domain HTTP API E2E（coordinator + 2 workers），BlockTableKvCache (block_size=4) 与 ContiguousKvCache 输出完全一致：`jumps over the lazy dog`。Commit `c8ec740` |
| M12.3: Request memory leak 修复 | **已完成** | [2026-05-09] `WorkerCommand::ReleaseRequest` 协议扩展 + `WorkerBackend::release_request()` trait 方法。TchWorkerBackend 释放 KV cache，VllmWorkerBackend/TransformersBackend/VllmBackend 清理 `_request_states`。Coordinator batch mode + HTTP mode 均在请求完成后发送。54/54 tests pass。Commit `ee6bd0e` |
| Docs: DEPLOYMENT_GUIDE.md vLLM backend 部署文档 | **已完成** | [2026-05-09] 新增 §6.5，覆盖 Mac vllm-metal + GPU vLLM 0.6.4 CUDA 环境准备、启动命令、HTTP API 验证、已知问题。Commit `a9cbefc` |
| Fix: flaky test_batch_forward_correctness | **已完成** | [2026-05-09] BATCH_TOL 1e-5→1e-4 + token agreement assertion。根因：CPU BLAS 非确定性。5/5 连续通过，55/55 tests pass。Commit `f84f441` |
| HTTP API SSE Streaming | **已完成** | [2026-05-09] `/v1/completions` 支持 `stream: true`，OpenAI-compatible SSE 格式。Dual-channel InferenceJob (oneshot+mpsc)。Coordinator decode loop 每 iteration emit StreamChunk。E2E 验证通过。55/55 tests pass。Commit `89efef1` |
| M13 Step 2/5: VllmWorkerBackend 原型 | **已完成** | [2026-05-24] Rust `VllmWorkerBackend` (JSON-over-stdio pipe) + Python `hcp_worker_process.py` (mock/transformers/vllm 三后端)。`--backend-type` CLI 参数支持 tch/vllm 切换。Harness Review: Guard=APPROVE, Examiner=CONDITIONAL → 2 blockers 修复后通过（backend_type shadowing + TransformersBackend KV cache reuse）。本地 E2E: mock ✅、transformers ✅、vllm-metal ✅、cross-backend (tch vs transformers) ✅。53/53 tests passed。Commits: `cc9f5c0`→`abed260` |
| M13 Step 3/5: vLLM CUDA E2E (white) | **待验证** | [2026-05-31] white 恢复，代码同步到 `dbf3871`，cargo check ✅，cargo test 55/55 ✅，torch 2.11.0+cu130 ✅，vLLM 0.22.0 ✅，模型 Qwen2-0.5B ✅。Rust tch-backend 本地 2-node loopback smoke 通过（`generated: is not a destination,`）。`--backend-type vllm` 验证待进行。 |
| pearl (AMD RX 9060 XT) Rust + libtorch GPU 路径 | **已跑通** | [2026-06-02] libtorch 2.11.0+rocm7.2 + tch-rs 0.24.0 + HIP patch。关键突破：`LD_PRELOAD=libtorch_hip.so` 强制加载 HIP kernel 注册库（ROCm 构建中 `libtorch_cpu.so` 不自动加载 `libtorch_hip.so`）。单节点推理 ✅、本地 2-node loopback smoke ✅、`cargo test` 55/55 ✅、`tch_smoke` 3/3 ✅。新增 `scripts/patch_torch_sys_hip.sh`。 |
| **三平台异构分布式推理** | **已完成** | [2026-06-02] Mac MPS + white RTX 4090 CUDA + pearl RX 9060 XT HIP 三平台 3-domain ring attention 首次联合成功。Coordinator 生成 `"The quick brown"`。证明 HCP 协议完全不依赖同构假设。 |
| M13 Phase 3-5: Full Continuous Batching with PagedAttention | **待启动** | 在 PagedAttention 基础上实现 kernel-level batch decode + dynamic join/leave |
