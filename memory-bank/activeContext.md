# 当前上下文

## 当前焦点

[2026-05-09] **Rust 分布式推理服务化已完成**。Coordinator 从单请求 smoke 工具升级为支持多请求串行处理的完整推理服务。commit 待填。

上一阶段 **跨机器异构 E2E（Mac vllm-metal + white RTX 4090）** 已在 Python 层完成验证并冻结。Python 层不再扩展。

---

## 近期变化

- [2026-05-09] **Rust 分布式推理服务化**：
  - Protocol 添加 `request_id`：`WorkerCommand` / `WorkerResponse` 所有 variant 携带 request ID，支持多请求生命周期隔离
  - Worker 新请求自动隔离：`TchWorkerBackend::prefill` 在每次 Prefill 时自动重建 KV cache（`create_kv_caches()`），避免旧请求污染新请求的 attention 计算
  - Worker 优雅退出：`WorkerRuntime::run()` 检测到 "connection lost" / "stream closed" 等连接关闭信号时打印日志并正常返回 Ok，不再 panic
  - Coordinator 多请求串行处理：新增 `--prompts-file` 参数（每行一个 prompt），循环处理每个请求，全部完成后统一 Shutdown workers
  - Coordinator 错误处理改进：单个请求的失败（logits size mismatch、sample_token error）只影响当前请求，继续处理下一个请求
  - 本地 2-domain CPU smoke 验证：2 个短 prompt 串行处理，Request 1 → ` is not a`，Request 2 → `, there was`，Worker 优雅退出，无 panic ✅
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
