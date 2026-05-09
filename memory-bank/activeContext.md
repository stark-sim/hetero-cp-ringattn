# 当前上下文

## 当前焦点

[2026-05-09] **跨机器异构 E2E：Mac vllm-metal (MPS) + white RTX 4090 (vLLM 0.6.4 CUDA)**。

**根因定位**：跨机器 E2E 测试超时的核心原因是 **vllm-metal 初始化时间（~81秒）> Python QUIC peer accept 超时（30秒）**。

时间线分析：
- worker 1 (white CUDA): 09:40:57 开始加载 → 09:41:00 完成 → listen for peer → **09:41:30 超时退出**
- worker 0 (Mac vllm-metal): 09:41:02 开始加载 → gloo init 60s → Metal kernel warmup → **09:42:23 才完成**
- worker 0 完成时 worker 1 已退出 93 秒，peer connect 失败

**已采取修复**：
- `python/hcp_worker_sdk/quic_server.py`: peer connect timeout 10s → 30s，peer accept timeout 30s → 180s
- 预热脚本 `/tmp/warmup_vllm.py` 必须用 `if __name__ == '__main__':` 保护（vllm-metal EngineCore 使用 multiprocessing.spawn，macOS 默认 spawn 模式）
- vllm-metal 预热成功：EngineCore 初始化 8.5s（含 Metal kernel warmup），prefill 5.7s，seq_len=2 ✅

**待验证**：修改超时后重跑跨机器 E2E。

**EngineCore 子进程残留问题**：父进程异常退出时 vllm-metal EngineCore 子进程可能未清理，Mac 本地曾出现 3 个残留 EngineCore 进程。已添加清理逻辑（`pkill -9 -f EngineCore`）。

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
- [ ] **EngineCore 子进程优雅退出**：在 `hcp_vllm_quic_worker.py` 中添加 signal/atexit handler，确保父进程退出时正确关闭 LLM EngineCore（当前 cleanup 用 `pkill -9` 粗暴终止）
- [ ] **Phase 3.4: vLLM 真实 KV 提取**（长期）：vLLM 0.6.4/0.20.x `LLM` API 不暴露 KV cache，需探索 `LLMEngine` 底层 API 或 vllm-metal 的 MLX KV cache 访问
