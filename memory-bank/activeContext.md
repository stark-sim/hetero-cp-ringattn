# 当前上下文

## 当前焦点

[2026-05-21] **4-domain 4K 异构测试完成** — 发现了 Serial 死锁 bug 和 VPN 稳定性上限：
- **Serial 模式**：❌ decode 死锁。根因：`exchange_kv_block` 默认实现只支持 2-domain（发 1 收 1），4-domain 下发 1 收 3 → `recv_kv_block` 永远等待。需修复为逐 round 转发
- **Pipeline 模式**：❌ prefill connection lost（2166s）。d2/d3 在 ~36min 大传输后断开。Tailscale VPN 不适合 4K+ 长时测试（~528MB/worker）
- **512-token 仍是 VPN 可靠上限**：已验证 Serial/Pipeline 均可行
- **Pipeline 逻辑本身正确**：d0 日志证实 3 rounds × 24 layers KV 交换正常推进

上一阶段 **sd-1+white 512-token A/B**（Pipeline 收益 3.3% vs Mac+white 40%，公式验证 compute/network 比例决定收益）已完成。Python 层冻结。

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
- [ ] **Phase 4.2: Rust 层 HTTP API 服务化**：
  - Coordinator 添加 HTTP server（axum），提供 OpenAI-compatible `/v1/completions` API
  - Request queue + 异步处理，支持并发请求接入
  - Health check `/health` 和 metrics `/metrics` endpoint
- [ ] **Phase 4.3: Rust 层性能优化与生产化**（长期）：
  - 量化支持（FP8/INT8 KV cache）— **暂不实施**，correctness 流程尚未完全走完
  - 连续 batching / 动态 request 调度
  - 更高效的 transport（RDMA / GPUDirect）
- [ ] **Phase 5: vLLM Block-Aware Ring**（远景）：
  - 让 ring 在 vLLM PagedAttention block 层面运作
  - 详见 `docs/BLOCK_RING_FUSION.md`
