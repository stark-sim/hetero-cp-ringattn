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
- [2026-05-05] **跨节点长序列限制**：Tailscale VPN 当前带宽 0.47Mbps/RTT 1.2s，111-token 跨节点 prefill 预计需 10+ 分钟，8K 预计需数小时。长序列跨节点异构验证待网络恢复后重试。

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
| M6: 扩展性论证 | **进行中** | [2026-05-05] 32K 单节点 ✅、32K 分布式 ✅、64K 单节点 ✅、64K 分布式 ✅、128K 待验证 |
