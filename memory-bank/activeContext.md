# 当前上下文

## 当前焦点

[2026-05-31] **Infra 环境同步与三平台状态评估**（已完成规范化）：
- **white (RTX 4090, CUDA) 恢复 + Python 3.12 升级完成**：Tailscale 修复后 SSH 连通，代码从 `b9a0bd3` fast-forward 到 `d9a1eb2`（+43 commits）。cargo check 13.98s ✅，cargo test 55/55 ✅。**`.venv` 规范化完成**：repo 内 `uv venv --python 3.12`，通过清华镜像安装 torch 2.11.0+cu130 + vllm 0.22.0 + transformers 5.9.0 + aioquic 1.3.0 等全部依赖，CUDA RTX 4090 识别正常（`torch.cuda.is_available()=True`, `CUDA 13.0`, `device_count=1`）。standalone libtorch (`~/libtorch`, 2.11.0) 保持用于 Rust 编译（环境变量已持久化到 `~/.bashrc`）。
- **pearl (RX 9060 XT, ROCm/HIP) 首次接入 + 规范化**：AMD Radeon RX 9060 XT (gfx1200)，ROCm 7.2，torch 2.12.0+rocm7.2。代码同步到 `d9a1eb2` ✅。`.venv` 确认已是 uv 管理 ✅。Rust CPU 测试 55/55 pass ✅。Python HIP 计算正常 ✅。**模型已下载**：hf-mirror.com 绕过防火墙，Qwen2-0.5B 完整下载（942MB）✅。
- **关键发现：pip torch 不能用于 Rust tch-rs 编译！** white 上测试将 `LIBTORCH` 指向 pip torch 路径后，`torch-sys` build script 编译失败：`torch::_assert_tensor_metadata` 参数数量不匹配（pip torch C++ API 与 standalone libtorch 不同）。这说明 pearl 的 GPU panic 也是同一根因——**tch-rs 0.24.0 的 C++ 绑定与 pip torch 2.12.0 的 C++ ABI 不兼容**。
- **关键发现：tch-rs 0.25.0 支持 libtorch 2.12.0**。GitHub 主分支 `build.rs` 显示 `TORCH_VERSION = "2.12.0"`。升级路径：tch-rs 0.24.0 → 0.25.0，同时 white standalone libtorch 2.11.0 → 2.12.0，即可实现两平台 libtorch 版本完全对齐。
- **pearl 网络问题**：IPv4 HTTPS 对特定域名阻断（huggingface.co、google.com 超时），但 github.com 和 hf-mirror.com 正常。已通过 `HF_ENDPOINT=https://hf-mirror.com` 绕过。
- **本地 push 17 commits 到 origin/main**：`abed260` → `d9a1eb2`。
- **项目 unconditional tch 依赖发现**：`cargo check --no-default-features` 编译失败（25 errors），`infer.rs` 等模块在 `#[cfg(feature = "tch-backend")]` 保护外直接引用 `tch`。`tch-backend` 名义上是 optional，实际上已变为 required。

[2026-05-09] **HTTP API SSE Streaming 支持完成**（commit `89efef1`）：
- `/v1/completions` 新增 `stream: true` 支持，返回 Server-Sent Events (SSE)
- `InferenceJob` 双通道设计：`tx` (oneshot) 非 streaming + `stream_tx` (mpsc) streaming
- `ActiveRequest` 携带 `stream_tx` 通过 scheduler
- Coordinator decode 循环每 iteration 为每个 streaming 请求发送 `StreamChunk`
- SSE 格式：每 token 一个 `data: {json}` event，`[DONE]` 标记结束
- 非 streaming 回归：`jumps over the lazy dog` ✅
- Streaming E2E：` over`→` the`→` lazy`→` dog`→`finish_reason:length`→`[DONE]` ✅
- 55/55 tests pass
- **Phase 2 TODO**: 增量解码优化（当前单 token decode 对 subword token 可能产生不干净的 delta）

[2026-05-09] **Flaky test 修复完成**（commit `f84f441`）：
- `test_batch_forward_correctness` 因 CPU BLAS 非确定性频繁失败（batched vs single matmul 累加顺序不同）
- BATCH_TOL 从 1e-5 放宽到 1e-4，添加 single-step decode token 一致性断言（argmax 必须匹配）
- 5/5 连续运行通过，全套件 55/55 tests pass

[2026-05-09] **DEPLOYMENT_GUIDE.md vLLM backend 部署文档更新完成**（commit `a9cbefc`）：
- 新增 §6.5 "vLLM Backend 部署（单节点高吞吐）"
- 覆盖 Mac vllm-metal 和 GPU vLLM 0.6.4 CUDA 的环境准备、启动命令、关键环境变量表
- HTTP API 验证步骤
- 已知问题文档化：初始化慢、EngineCore 残留、token drift
- 明确标注 vLLM backend 的当前限制：单节点 only，不参与 KV ring

[2026-05-09] **Request memory leak 修复完成**（commit `ee6bd0e`）：
- 根因：coordinator 检测请求完成（EOS/max_tokens）后仅从 scheduler 移除，从未通知 workers 释放 per-request 状态
- `WorkerCommand::ReleaseRequest { request_id }` 新增到控制协议
- `WorkerBackend::release_request()` trait 方法，默认 no-op（向后兼容）
- `TchWorkerBackend`: `request_contexts.remove(request_id)` 释放 KV cache tensors
- `VllmWorkerBackend`: 发送 release_request JSON 命令到 Python 子进程
- Python `TransformersBackend` / `VllmBackend`: `del _request_states[request_id]`
- Coordinator batch mode + HTTP mode 均在请求完成后发送 ReleaseRequest 给所有 workers
- 验证：54/54 tests pass，HTTP API E2E `jumps over the lazy dog` ✅，worker 日志确认 `[TchWorkerBackend] released request 1`

[2026-05-09] **M12 BlockTable 运行时切换 + 集成测试 + E2E 验证完成**（commits `3efbdf0`, `c8ec740`）：
- `KvCacheImpl` enum（Contiguous | BlockTable）替代硬编码 `ContiguousKvCache`，避免 `Box<dyn>` 生命周期问题
- `create_kv_caches()` 运行时切换：环境变量 `HCP_KV_CACHE_BLOCK_TABLE=1` 启用 BlockTable，`HCP_KV_CACHE_BLOCK_SIZE=N` 调整 block 大小（默认 16）
- `test_block_table_through_model_forward`：BlockTableKvCache 通过完整 `LlamaModel::forward` prefill + 3-step decode 路径验证，block_size=4 跨越 block 边界，diff < 1e-6
- `test_block_table_e2e_local.sh`：真实 2-domain 分布式 HTTP API E2E 验证（coordinator + worker0 + worker1），BlockTable 与 Contiguous 输出完全一致：`jumps over the lazy dog`
- 所有 53 非 flaky tests 通过，零 regression
- **Trade-off**: BlockTable.update() 仍调用 Tensor::cat()，无即时内存/性能收益；此为未来 custom kernel 消费 `k_blocks()`/`v_blocks()` 的结构基础

[2026-05-24] **M13 Step 2: VllmWorkerBackend 原型完成 + Review 修复**（commits `cc9f5c0` → `abed260`）：
- `VllmWorkerBackend`：通过子进程 + JSON-over-stdio pipe 与 Python vLLM worker 通信，实现 `WorkerBackend` trait
  - Handshake 获取 num_layers / capacity_mb
  - Prefill/Decode/DecodeBatch 全命令覆盖
  - Graceful shutdown via Drop，stdout 隔离（非 JSON 行自动跳过）
- `python/hcp_worker_process.py`：多后端 worker 进程（mock / transformers / vllm），统一 JSON 协议
  - TransformersBackend：MPS/CPU 自动检测，prefill→decode KV cache 复用（Reviewer blocker #2 修复）
  - VllmBackend：vLLM 0.6.x / 0.20.x 双版本兼容，one-hot logits 重建
- `worker.rs` 新增 `--backend-type` CLI 参数：tch（默认）或 vllm。vllm 模式下跳过 tch 模型权重加载
  - backend_type shadowing 修复（Reviewer blocker #1）：移除冗余变量和错误 casing 的 assert
- `WorkerBackend` trait 从 `worker_sdk/mod.rs` 重新导出，支持 `Box<dyn WorkerBackend>` 在 worker entry 中使用
- Harness Review：Guard=APPROVE，Examiner=CONDITIONAL → 2 blockers 修复后通过
- 53/53 cargo tests passed（1 flaky `test_batch_forward_correctness` 在并行运行时偶发 CPU BLAS 非确定性，单独运行通过）
- 本地 E2E 全部通过：mock ✅、transformers ✅、vllm-metal ✅、cross-backend (tch vs transformers) ✅

[2026-05-24] **M12 PagedAttention Block Table + vLLM Feasibility 完成**：
- `KvCache` trait 抽象：`ContiguousKvCache`（现有行为）+ `BlockTableKvCache`（block 化存储）
- `AttentionBackend::forward` 签名改为 `Option<&mut dyn KvCache>`，零行为变更
- `BlockTableKvCache`：可配置 `block_size`（默认 16），逻辑 block 边界，供未来 custom kernel 消费
- 可行性研究结论：vLLM 作为单节点 backend = HIGH feasibility；vLLM 参与 HCP KV ring = LOW feasibility（需深度修改 vLLM 内部）
- 推荐架构：混合路径（`TchWorkerBackend` 处理分布式 ring attention + `VllmWorkerBackend` 处理单节点高吞吐）
- 53/53 cargo tests passed。Commits: `cfe4cde` (trait), `fc5c15d` (feasibility), `c98e45f` (BlockTable)

[2026-05-23] **M13 Phase 1-3: Continuous Batching Scheduler + Per-Request KV Cache + Batch Decode 完成**：
- Coordinator `BatchScheduler`：pending/active/completed 请求池，固定 `max_batch_size` 调度
- `DecodeBatch` / `DecodeBatchDone` 协议扩展并已在 worker runtime 实现
- Worker `RequestContext`：per-request KV cache 隔离
- HTTP mode iteration-based 调度循环
- `test_decode_batch_isolation`：验证 batch decode logits 与独立 decode 一致，4 步无交叉污染
- Scheduler edge-case tests：batch 动态大小变化 + 状态修改
- **本地 Batch E2E 验证**：2 请求先后到达，输出正确，metrics 准确
- 53/53 cargo tests passed。Commits: `ea111c9` (Phase 1-2), `4f3d3dc` (Phase 3 tests)

[2026-05-22] **M10.3 Request-Level Parallelism 完成**：
- Coordinator 并发请求处理：`worker_streams` 用 `Arc<std::sync::Mutex>` 保护，`rt.spawn_blocking()` 并发处理 HTTP 请求
- `max_concurrent=4` 信号量限制并发数，防止 worker 过载
- `ActiveRequestGuard` RAII 自动管理 `active_counter`
- `/metrics` 新增 `queued_requests` 和 `active_requests` 字段
- **本地并发 E2E 验证**（`scripts/test_http_api_concurrent_local.sh`）：同时提交 2 个请求
  - Request 1: `jumps over the lazy dog` ✅
  - Request 2: `there was a man` ✅
  - `/metrics` → `{"total_requests":2,"completed_requests":2,"queued_requests":0,"active_requests":0}` ✅
  - 无 panic，无 command 交错，correctness 无 regression
- 45/45 cargo tests passed。Commit `5ffa83f`

[2026-05-22] **M10.2 HTTP API 服务化完成 + 本地 E2E 验证通过**：
- `POST /v1/completions` — 标准 completions API（prompt, max_tokens, temperature, top_p）
- `GET /health` — workers_connected, num_domains
- `GET /metrics` — total/completed/failed request counters
- 双模式：Batch mode（`--prompts-file` 处理完退出）vs HTTP API mode（默认无 prompt 时启动服务）
- `process_single_request()` 从 `run()` 提取，batch 和 HTTP 复用同一核心逻辑
- HTTP server 独立线程 + tokio runtime，通过 `mpsc::unbounded_channel` + `oneshot` 与 coordinator 主循环通信
- 45/45 tests passed，零 regression。Commit `e3eafe9`
- **本地 E2E 验证**（`scripts/test_http_api_local.sh`）：coordinator HTTP mode + 2 local workers
  - `/health` → `{"status":"ok","workers_connected":2,"num_domains":2}` ✅
  - `/metrics` → counters all zero ✅
  - `/v1/completions` → `{"text":" jumps over the lazy dog","finish_reason":"length"}` ✅
  - 21s 完成，commit `4a08293`
- **跨节点异构 E2E 验证**（`scripts/test_http_api_cross_node.sh`）：Mac MPS (coordinator+worker0) + white RTX 4090 CUDA (worker1)，Tailscale VPN
  - `/health` → `{"status":"ok","workers_connected":2,"num_domains":2}` ✅
  - `/v1/completions` (512-token prompt, max_tokens=1) → `{"text":"1","finish_reason":"length"}` ✅
  - 59s 完成，commit `a8507d3`

[2026-05-22] **Coordinator shutdown hang 修复**（commit `c4dcfc5`）：
- Root cause: `write_frame_quic` 无 timeout，worker 断开时 `send.write_all` 无限期 hang
- Fix: `write_frame_quic_timeout` / `read_frame_quic_timeout` / `send_command_quic_timeout` / `recv_command_quic_timeout`
- Coordinator `shutdown_workers()` 使用 10s timeout 发送 Shutdown，finish() streams，close endpoint，sleep 2s
- 45/45 tests passed

[2026-05-22] **Mac + white 弱网 A/B 测试完成 + Pipeline Phase 2 修复**：
- **测试**: 64/256/512 tokens 全部完成（Serial + Pipeline），512 是弱网可靠上限
- **Pipeline 收益递减**: 64-token +5% → 256-token +2% → 512-token **-2%**
- **根因排查**（ISSUE-001）: Pipeline Phase 2 收集完全部 blocks 后才 process，receive 阻塞 compute，overlap 未实现
- **已修复**（`cbefc49`）: Phase 2 改为逐个 block 接收→立刻 process→转发，实现 true streaming compute
- 45/45 tests passed，零 regression
- 报告: `reports/mac-white-weaknet-ab-20260522/README.md`

[2026-05-22] **4-domain 4K Serial 异构测试首次成功** — 4988s（1h 23m）：
- **Serial 模式**：✅ **成功完成**。`quic.rs` mpsc channel buffer 2→64 修复了 N-domain Serial 死锁。4 个 worker 全部完成 prefill（4096 tokens），decode 生成 1 token（`over`）。exit=0。报告：`reports/cross-node-4domain-4k-serial-20260522/`
- **Pipeline 模式**：❌ 仍待验证。上次尝试 2166s 后 connection lost（Tailscale VPN 大传输不稳定）。需要 LAN 环境才能做 Serial vs Pipeline 4K 4-domain A/B 对比。

Python 层冻结。Rust 层为主干。

---

## 近期变化

- [2026-05-09] **Rust 分布式推理服务化**：
  - Protocol 添加 `request_id`：`WorkerCommand` / `WorkerResponse` 所有 variant 携带 request ID，支持多请求生命周期隔离
  - Worker 新请求自动隔离：`TchWorkerBackend::prefill` 在每次 Prefill 时自动重建 KV cache（`create_kv_caches()`），避免旧请求污染新请求的 attention 计算
  - Worker 优雅退出：`WorkerRuntime::run()` 检测到 "connection lost" / "stream closed" 等连接关闭信号时打印日志并正常返回 Ok，不再 panic
  - Coordinator 多请求串行处理：新增 `--prompts-file` 参数（每行一个 prompt），循环处理每个请求，全部完成后统一 Shutdown workers
  - Coordinator 错误处理改进：单个请求的失败（logits size mismatch、sample_token error）只影响当前请求，继续处理下一个请求
  - 本地 2-domain CPU smoke 验证：2 个短 prompt 串行处理，Request 1 → ` is not a`，Request 2 → `, there was`，Worker 优雅退出，无 panic ✅
  - **跨节点异构验证**（Mac MPS + white RTX 4090 CUDA）：2 个 prompt 串行处理，Worker 0/1 均优雅退出，exit=0，零 panic ✅
  - 全部 45 个 tests 通过，无 regression
- [2026-05-09] **Rust Static Batching 实现与验证**：
  - `BatchGenerator`：等长 prompts 约束 + 0-token EOS 填充 + greedy/temperature/top-p 采样
  - `test_batch_forward_correctness`：batch=2 vs batch=1，logits diff ~1e-6，token 完全一致 ✅
  - `test_batch_generator_correctness`：`BatchGenerator` batch=2 与两个独立 `Generator` 输出完全一致 ✅
  - 全部 24 model tests 通过，无 regression
- [2026-05-09] **Correctness-First 开发纪律确立**：在 correctness 流程完全走完之前，禁止实施任何可能损害服务质量的优化（量化、近似 attention、非 deterministic kernel、投机解码等）。提出优化前必须完成四问 trade-off 分析。详见 `systemPatterns.md` "Correctness-First 开发纪律"章节。
- [2026-05-09] **远程 GPU 从 sd-1 切换到 white**（100.64.0.2, user stark, RTX 4090）。sd-1 有网络/代理不稳定问题。
- [2026-05-09] **Python 包管理全面迁移到 uv**：本地 Mac 用 `~/.venv-vllm-metal`，远程 white 用 `~/venv-vllm`。不再使用 conda。
- [2026-05-09] **vllm-metal 0.2.0 安装完成**：官方 `install.sh` 安装到 `~/.venv-vllm-metal`，`vllm==0.20.1+cpu` 从源码编译 + `vllm-metal==0.2.0` wheel。使用 MLX + Metal GPU backend（`PyTorch device set to: mps`）。
- [2026-05-09] **`VllmBackend` API 兼容性修复**：新增 `_vllm_generate()` 适配层，vLLM 0.6.x 用 `prompt_token_ids`，vLLM 0.20.x (vllm-metal) 用 `prompts=[token_ids]`。
- [2026-05-09] **Mac 单节点 vllm-metal E2E 验证通过**：coordinator + vllm-metal worker，Prefill + 3×Decode + Shutdown，输出 `generated: ! I'm`。
- [2026-05-09] **远程 white 环境搭建完成**：uv 0.11.7, Python 3.11.15, torch 2.5.1+cu124, vLLM 0.6.4, transformers 4.45.2, aioquic 1.3.0。model.safetensors 已复制到 white。
- [2026-05-09] **脚本更新**：`scripts/run_python_distributed_2node.sh` 默认 `GPU_ADDR=100.64.0.2`, `GPU_USER=stark`，使用 uv venv 而非 conda。
- [2026-05-09] **QUIC 超时修复**：`quic_server.py` peer connect 10→30s，peer accept 30→180s，覆盖 vllm-metal 长初始化时间。

---

## 近期变化

- [2026-05-09] **远程 GPU 从 sd-1 切换到 white**（100.64.0.2, user stark, RTX 4090）。sd-1 有网络/代理不稳定问题。
- [2026-05-09] **Python 包管理全面迁移到 uv**：本地 Mac 用 `~/.venv-vllm-metal`，远程 white 用 `~/venv-vllm`。不再使用 conda。
- [2026-05-09] **vllm-metal 0.2.0 安装完成**：官方 `install.sh` 安装到 `~/.venv-vllm-metal`，`vllm==0.20.1+cpu` 从源码编译 + `vllm-metal==0.2.0` wheel。使用 MLX + Metal GPU backend（`PyTorch device set to: mps`）。
- [2026-05-09] **`VllmBackend` API 兼容性修复**：新增 `_vllm_generate()` 适配层，vLLM 0.6.x 用 `prompt_token_ids`，vLLM 0.20.x (vllm-metal) 用 `prompts=[token_ids]`。
- [2026-05-09] **Mac 单节点 vllm-metal E2E 验证通过**：coordinator + vllm-metal worker，Prefill + 3×Decode + Shutdown，输出 `generated: ! I'm`。
- [2026-05-09] **远程 white 环境搭建完成**：uv 0.11.7, Python 3.11.15, torch 2.5.1+cu124, vLLM 0.6.4, transformers 4.45.2, aioquic 1.3.0。model.safetensors 已复制到 white。
- [2026-05-09] **脚本更新**：`scripts/run_python_distributed_2node.sh` 默认 `GPU_ADDR=100.64.0.2`, `GPU_USER=stark`，使用 uv venv 而非 conda。
- [2026-05-09] **QUIC 超时修复**：`quic_server.py` peer connect 10→30s，peer accept 30→180s，覆盖 vllm-metal 长初始化时间。

## 活跃决策

- [2026-05-09] **Correctness-First 纪律**：在 correctness 流程完全走完之前，禁止实施任何可能损害服务质量的优化。提出优化前必须完成四问 trade-off 分析（为什么默认存在、牺牲了什么、被牺牲的东西的作用、对本项目的影响）。详见 `systemPatterns.md`。
- [2026-05-09] vllm-metal EngineCore 使用 multiprocessing.spawn（macOS 默认），入口脚本必须有 `if __name__ == '__main__':` 保护，否则子进程重新导入主模块导致递归崩溃。
- [2026-05-09] vllm-metal 首次 Metal kernel warmup 在 M1 Pro 上约 60-90 秒（gloo init 60s + kernel compile 10-20s）。预热后后续初始化约 8-10 秒。
- [2026-05-09] uv 替代 conda 作为 Python 包管理工具，本地和远程均统一使用。
- [2026-05-09] 跨机器异构测试必须考虑最慢 worker 的初始化时间，peer accept 超时需覆盖该时间。

## 下一步

- [x] [2026-05-11] **Rust lib.rs 重构 Commits 3-7**：提取 report types (`src/report.rs`)、reference algorithm (`src/smoke/reference_algo.rs`)、correctness tests (`src/smoke/correctness.rs`)、C++/tch bridge wrappers (`src/smoke/bridges/cxx.rs`, `src/smoke/bridges/tch.rs`)、remote networking (`src/remote.rs`)。lib.rs 从 ~2500 行降至 555 行。`cargo check --features tch-backend` 通过，`cargo test --features tch-backend` 45/45 通过。
- [x] [2026-05-11] **Step 1: N-domain ring 拓扑去硬编码**：`runtime.rs` 移除 `num_domains == 2` 硬编码分支，统一为并发 dial+accept；`mock.rs` 新增 `create_ring(n)`；45/45 tests passed，已提交 `b0c040d`
- [x] [2026-05-11] **Step 2: Layer 内 Overlap — Split-Phase Transport + Pipeline**：
  - `KvTransport` trait 扩展 split-phase API：`submit_send` / `poll_recv` / `flush_send`，旧方法提供默认阻塞实现（向后兼容）
  - QUIC transport 重写为内部 async task + channel 架构：send task / recv task 独立运行，主线程通过 mpsc channel 交互，channel 中只传 `Vec<u8>`（避免 Tensor 跨线程移动）
  - TCP/Mock transport 同步 split-phase 实现：submit 缓冲到内部 buffer，recv 覆盖默认实现避免忙等
  - `ring_attention` 重构为 4-phase pipeline：Phase 0 submit_send(first_block) → Phase 1 本地 KV compute（与 send 重叠）→ Phase 2 循环 poll_recv→process→submit_send 转发（compute 与下一轮 network I/O 重叠）→ Phase 3 flush_send → Phase 4 提取输出
  - 关键修复：Mock 测试中先运行 domain 的 inbox 为空，`poll_recv` 返回 None 后改用 `recv_kv_block` 做确认性阻塞尝试，区分"数据暂未到"和"stream 已关闭/peer 不会发送"，避免死循环
  - 全部 45 cargo tests 通过（含 `test_distributed_llama_model_prefill/decode/multi_step_decode`），零 regression
- [x] [2026-05-12] **Step 3: Micro KV Block + A/B Overlap Quantification**：
  - `KvBlock` 新增 `micro_block_idx` / `total_micro_blocks` 字段，支持 KV block 的细粒度切分
  - `HcpRingAttentionBackend` 新增 `disable_overlap`（串行对照模式）和 `micro_kv_block_size`（环境变量 `HCP_MICRO_KV_BLOCK_SIZE` 配置，默认 0=禁用）
  - `ring_attention` 重构为支持 micro block 的双模式：
    * Pipeline 模式（默认）：Phase0 submit_send → Phase1 本地 compute → Phase2 循环 recv→process→forward → Phase3 flush
    * 串行模式（`HCP_DISABLE_OVERLAP=1`）：先全部 exchange 再统一 compute，用于 A/B baseline 对比
  - 本地 2-domain CPU smoke 验证：pipeline 与 serial 模式输出完全一致（`generated:  is not a`），correctness 无 regression
  - 45 cargo tests 通过，commit `7a2d33f` 已推送至 main
  - 新建 `scripts/run_cross_node_ab_test.sh`：自动化跨节点 A/B 对比测试脚本，支持 baseline/optimized 多配置批量运行
  - **跨节点异构 A/B 验证通过**（Mac MPS + white RTX 4090 CUDA，64-token prompt，3 decode tokens）：
    * Baseline Serial (`HCP_DISABLE_OVERLAP=1`)：`generated:  jumps over the` ✅
    * Pipeline Default（overlap on）：`generated:  jumps over the` ✅
    * 两种模式输出完全一致，correctness 无 regression；micro block 传输日志正常（`received micro_block 1/1, 229376 bytes`）
  - **256-token A/B 量化对比**（Tailscale VPN，非 LAN，带宽受限）：
    * Serial: **151s** | Pipeline: **147s** | 差异: **-4s (~2.6%)**
    * 输出一致（`the`），correctness 无 regression
  - **512-token A/B 量化对比**（Mac MPS + white RTX 4090，Tailscale VPN ~107ms RTT）：
    * Serial: **~5min (300s)** | Pipeline: **~3min (180s)** | **Pipeline 快 ~40%**
    * 输出一致（`brown`），correctness 无 regression
  - **512-token A/B 量化对比**（sd-1 RTX 4080 SUPER + white RTX 4090，Tailscale VPN ~78ms RTT）—— **关键新发现**：
    * Serial no-micro-block: **299s** | Serial micro-block=64: **330s** | Pipeline micro-block=64: **319s**
    * Pipeline no-micro-block: **connection lost**（大传输导致 QUIC 不稳定）
    * **同 micro-block 下 Pipeline 仅快 3.3%**，远低于 Mac+white 的 40%
    * **根因**：双 CUDA 计算快 + RTT 更好（78ms vs 107ms）→ compute >> network → overlap 收益趋近于 0
    * **Micro block 是稳定性必需品**：无 micro block → connection lost；micro block 增加 ~10% 开销
    * **公式验证**：Pipeline 收益 ≈ 1 - compute/(compute+network)。计算越慢/网络越差 → 收益越大
  - **4K 本地验证**：Serial 和 Pipeline 均正常（CPU 本地 ~30s），代码逻辑无 bug
  - **4K 跨节点失败**：网络不稳定导致连接断开。根因：7.3MB/layer × 24 layers ≈ 175MB 总传输量，跨 VPN 慢网络下大 block 传输触发连接丢失。需要 micro block 切分或更稳定网络才能进行 4K+ 跨节点对比
  - **QUIC recv_kv_block timeout 修复**：120s → 600s（commit `3759811`）。4K 跨节点 KV block 传输超过 120s 导致 timeout panic，600s 覆盖大 block + 慢网络场景。512-token 验证通过
  - **核心公式化结论**：Pipeline 收益 ≈ 1 - (compute_time / (compute_time + network_time))。本地/小 scale 收益 ≈ 0%；异构慢计算+慢网络 ≈ 40%；同构快计算+较好网络 ≈ 0-5%；micro block 是稳定性必需品但增加 5-15% 开销
  - **分析报告**：`reports/ab-analysis-20260513/README.md` 完整记录测试矩阵、量化数据、根因分析、下一步建议
- [x] [2026-05-09] **验证跨机器 E2E通过**：`scripts/run_python_distributed_2node.sh` 成功运行，Mac vllm-metal (MPS, 8.39s 初始化) + white RTX 4090 (CUDA) 完整端到端通过，生成 `. I am`。QUIC 超时修复（peer accept 180s）生效。
- [x] [2026-05-09] **大规模跨机器验证矩阵完成**（一个节点一个 worker）：
  - T0 回归（2 tokens + 3 decode）：`. I am` ✅ ~40s
  - T1 规模（111 tokens + 5 decode）：`quick brown fox jumps over` ✅ ~2min
  - T2 极限（551 tokens + 5 decode）：`100 dog.` ✅ ~40s
  - 关键发现：vllm-metal warm-up 后 551-token prefill 仅 1.10s（276 tok），white RTX 4090 达 968 tok/s prefill + 105-109 it/s decode。Python Worker SDK 侧跨机器性能远超 Rust 基线（Rust 551 tokens ~30min）。
- [x] [2026-05-09] **EngineCore 子进程优雅退出完成**：
  - `VllmBackend.shutdown()` 添加跨版本兼容 cleanup（stop_remote_worker_execution_loop、del llm、gc.collect、CUDA empty_cache、psutil 终止 EngineCore 子进程）
  - `QuicWorkerServer.run()` 支持 `shutdown_event` 参数，command loop 可响应外部信号
  - `hcp_vllm_quic_worker.py` 注册 SIGTERM/SIGINT handler，finally 块调用 `server.cleanup()` + `backend.shutdown()`
  - `run_python_distributed_2node.sh` cleanup 改为先 SIGTERM、sleep 2s、仅残留时 fallback 到 `pkill -9`
  - E2E 验证无 EngineCore 残留 ✅
- [x] [2026-05-09] **更长序列验证完成（25% Mac / 75% CUDA 分片）**：
  - T3: 1024 tokens + 5 decode, chunk-sizes 256,768 → `jumps over the lazy dog` ✅
  - T4: 2048 tokens + 5 decode, chunk-sizes 512,1536 → `dog jumps over the lazy` ✅
  - Mac MPS 512-token prefill 1.69s (303 tok/s)，white RTX 4090 1536-token prefill ~0.32s (4788 tok/s)
- [x] **Phase 3.4: Transformers 路径真实 KV + online softmax correctness 验证**（已完成）：
  - `test_worker_2domain.py` (mock transport) ✅、`test_transformers_2domain_quic.py` (QUIC) ✅
  - 关键修复：`recalculate_logits()` + `DynamicCache` 兼容层 + 仅最后一个 domain 重算
  - **架构决策：冻结 Python 层投入**。Python 层的存在理由只有 vLLM 适配。transformers correctness 已由 Rust 层覆盖，继续维护两套 SDK 不划算。后续以 Rust + C++ + libtorch 为主干。
- [x] **Phase 4.1: Rust 层 Static Batching**（ correctness 优先）：
  - `BatchGenerator` 实现：支持 batch > 1 的 prefill + decode，所有 prompts 必须等长（避免 padding mask 复杂度）
  - `generate_batch_from_ids`：核心 batch generation API，支持 greedy/temperature/top-p 采样
  - 早期停止：单个 request 遇到 EOS 后继续喂 0 token 保持 KV cache 形状一致，不影响其他 request
  - correctness 验证：`test_batch_forward_correctness`（batch=2 vs batch=1，prefill + 4-step decode，logits diff ~1e-6，token 完全一致）✅
  - correctness 验证：`test_batch_generator_correctness`（`BatchGenerator` batch=2 vs 两个独立 `Generator`，token 序列完全一致）✅
  - 无 regression：全部 24 个 model tests 通过 ✅
- [x] [2026-05-22] **Phase 4.2: Rust 层 HTTP API 服务化**：
  - Coordinator 添加 axum HTTP server，OpenAI-compatible `/v1/completions` API
  - `GET /health` + `GET /metrics` endpoints
  - Request queue (`tokio::sync::mpsc`) + oneshot result channel
  - 双模式：batch mode（`--prompts-file`）vs HTTP API mode（默认）
  - Commit `e3eafe9`，45/45 tests passed ✅
- [x] [2026-05-23] **M13 Phase 1-2: Continuous Batching Scheduler + Per-Request KV Cache**：
  - `BatchScheduler`：pending/active/completed 请求池，固定 `max_batch_size` 迭代调度
  - `DecodeBatch` / `DecodeBatchDone` 协议扩展
  - Worker `RequestContext`：per-request KV cache 隔离（`HashMap<u64, RequestContext>`）
  - HTTP mode 改为 iteration-based 调度循环
  - 48/48 tests passed，batch E2E 通过（2 请求先后到达输出正确）。Commit `ea111c9` ✅
- [ ] **M13 Step 3/5: vLLM Backend 集成测试（white RTX 4090 CUDA）**：
  - 在 white RTX 4090 上运行 `cargo run --bin distributed_worker -- --backend-type vllm`
  - 验证单节点 vLLM backend 与 Rust coordinator 控制面互通
  - 使用 mock backend 做本地协议回归测试（无需 vLLM 安装）✅ 已完成
  - **Blocked**: white RTX 4090 当前不可达（Tailscale down），等待恢复后验证
- [ ] **M13 Step 4/5: vLLM Logits 精确提取**：
  - 当前 `VllmBackend` 使用 one-hot logits（sampled token=1e9，其余=-1e9），因 vLLM 公共 API 不暴露完整 logits
  - vllm-metal token drift 已知：vLLM 内部采样器即使 temperature=0.0 也使用独立随机状态，sampled token 可能与 true argmax 不同
  - 方案 A：vLLM `model_executor.execute_model(output_logits=True)` 直接提取
  - 方案 B：vLLM `logits_processors` 或自定义 sampler 捕获
  - 与 tch backend 做逐 token logits diff 对比
- [ ] **M13 Step 5/5: vLLM Backend 端到端验证**：
  - 跨节点异构：`--backend-type vllm` on white + `--backend-type tch` on Mac
  - 单节点吞吐：vLLM 单 worker vs tch backend 单 worker 性能对比
  - Correctness 验证：相同 prompt 输出 token 一致，logits diff < 1e-4
- [ ] **M12: PagedAttention Block Table**（待启动）：
  - 替换 `KvCache` 为 block 化分配，支持 ragged batching
  - 不等长请求的 true continuous batching 基础设施
- [ ] **M13 Phase 3-5: Full Continuous Batching with PagedAttention**（待启动）：
  - Worker runtime 填充 `DecodeBatch` stub + coordinator 构造 batch token 列表
  - Dynamic join/leave（新请求在 decode iteration 间插入，已完成的请求离开 batch）
  - Kernel-level batch decode（依赖 PagedAttention block table）
- [ ] **Phase 4.3: Rust 层性能优化与生产化**（长期）：
  - 量化支持（FP8/INT8 KV cache）— **暂不实施**，correctness 流程尚未完全走完
  - 更高效的 transport（RDMA / GPUDirect）
- [ ] **Phase 5: vLLM Block-Aware Ring**（远景）：
  - 让 ring 在 vLLM PagedAttention block 层面运作
  - 详见 `docs/BLOCK_RING_FUSION.md`

[2026-05-23] **M10.3 Worker 侧 Bug 修复**（commit `eb71401`）：
- **Bug A: HcpRingAttentionBackend 状态未重置** — Reviewer 独立验证发现并发请求下 worker panic
  - Root cause: `TchWorkerBackend::prefill()` 重置了 `LlamaModel::is_prefill_done` 但未重置 `HcpRingAttentionBackend::is_prefill_done` / `prefill_kv_len`
  - Fix: `HcpRingAttentionBackend::set_distributed()` 中重置 `is_prefill_done=false` + `prefill_kv_len=0`
  - Reviewer 验证: 2 并发 + 1 顺序请求全部通过，无 panic ✅
- **Bug B: 空 chunk 分配** — prompt tokens < num_domains 时 domain 得到 0-token chunk
  - Root cause: `process_single_request` chunk 分配在 seq_len=1, num_domains=2 时给 domain1 分配 0
  - Fix: chunk_sizes 计算后检查 size==0，返回友好错误
  - Reviewer 验证: 1-token prompt 正确返回错误（不再 panic）✅
- **Harness Reviewer 独立验证通过**: APPROVE (guard=APPROVE, examiner=APPROVE, validator=APPROVE)
  - 5/5 测试通过，confidence=high
