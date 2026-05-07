# 当前上下文

## 当前焦点

[2026-05-05] **Python Worker SDK Phase 2: 2-domain Mock KV Ring 验证通过**（commit `1cacb35`）。
- `TransformersBackend` 接入 `past_key_values`（`DynamicCache` for transformers 4.57.6），prefill/decode 均使用 KV cache 增量计算。
- `get_kv_block` / `apply_peer_kv` 实现按层 seq_start 跟踪，支持 prepend/append 正确拼接 peer KV。
- 新增 `LinkedMockKvTransport`：内存 queue 单进程模拟 2-domain KV ring 交换。
- `test_worker_2domain.py` 单进程起 coordinator + 2 worker threads，prefill 分片 → KV ring 交换 → decode 循环 → shutdown 全链路通过。
- **2-domain 输出与单节点参考完全一致**：` generated: ' in the universe. The'`（tokens: [304, 279, 15494, 13, 576]）。
- 修复 `seq_start` 全局/本地索引混淆 bug（`get_kv_block` 需按层 `_layer_kv_start` 转换）。
- 移除 decode 阶段 KV exchange（所有 worker 已同步），避免重复拼接污染 KV cache。

[2026-05-05] **Python Worker SDK Phase 1 控制面验证通过**（commit `dabd6fc`）。
- 修复 `WorkerResponse` dataclass 字段 `error` 与类方法 `error` 同名冲突（Python dataclass 将类方法作为字段默认值），重命名为 `from_error`。
- 修复 bytes JSON 序列化：改用 hex 编码替代 `list()`，避免类型歧义。
- 新增 `NoOpKvTransport` stub，单节点模式跳过 peer 连接和 KV exchange。
- `test_worker_control_plane.py` --coordinator + --worker 端到端通过：prefill → 3× decode → shutdown，生成 `' in the universe'`。

[2026-05-05] **Rust Worker SDK 重构完成并验证通过**（commit `dfbb517` + `02230d0`）。`distributed_worker.rs` 解耦为协议运行时 (`WorkerRuntime`) + 可插拔后端 (`WorkerBackend` trait) + 默认 tch-rs 后端 (`TchWorkerBackend`）。`cargo test` 42/42 通过，`cargo check` 通过。SDK 相关 clippy 警告已清理。解耦目的已记录于 `systemPatterns.md`。

**跨节点异构验证已通过**（commit `719a168` 后，white/RTX 4090）：
- Mac MPS (domain 0) + white RTX 4090 CUDA (domain 1) 跨新 tailnet 完成端到端验证
- 11-token: `generated: The quick brown fox jumps`，exit=0，~40s
- **111-token: `generated: 10000`（5× decode token 16/15），exit=0，prefill 分片 [0,56)/[56,111)**
- Worker 0/1 prefill → KV ring 交换 → decode → shutdown 全部正常
- 551-token 测试正在后台运行（预计 ~30min）

**性能回归测试已完成**（真实 Qwen2-0.5B 权重，commit `02230d0` 后）：

| 配置 | Prompt | Decode | 时间 | 历史基准 | 结论 |
|------|--------|--------|------|----------|------|
| 单节点 CPU | 11 tok | 20 | 2.393s | 2.87s | ✅ 更快 |
| 单节点 MPS | 11 tok | 20 | 2.186s | 2.06s | ✅ 同一数量级 |
| 单节点 MPS (num_domains=2) | 11 tok | 20 | 3.241s | — | ✅ 正常 |
| 2-domain CPU 分布式 | 11 tok | 20 | 9.594s | 3-domain 13.20s | ✅ 合理 |
| 2-domain CPU 分布式 | 30 tok | 5 | 7.566s | — | ✅ 正常 |

- **功能正确性**：单节点 CPU/MPS/2-domain 分布式输出完全一致（` in the universe.` / ` The quick brown fox jumps`）
- **性能无回归**：单节点 CPU 比历史更快（2.39s vs 2.87s），MPS 与历史同一数量级（2.19s vs 2.06s），分布式 2-domain 9.6s 与 3-domain 历史 13.2s 合理
- **重构前后对比**：解耦没有引入额外网络往返、tensor 拷贝或 trait dispatch 开销

[2026-05-05] **Decode 性能优化已完成并验证通过**（commit `491a46c`）。decode 阶段 `kv_chunk_size` 从 `1`（70001 个 tiny chunks）提升到 `2048`，大幅缩减循环次数。

[2026-05-05] **QUIC 高 RTT 超时增强**（commit `dab3ebf`）。针对 RTT 1.2s 的 Tailscale VPN 跨节点路径，增加全部 timeout：
- `accept` 超时 3s → 30s（QUIC 握手需 2-3 RTT）
- `keep_alive_interval` 1s（NAT 表项刷新）
- `max_idle_timeout` 300s → 3600s（防止长 prefill 期间 idle 断开）
- `exchange_kv_block` 添加 120s 超时（fail-fast 替代无限 hang）
- 禁用 MTU discovery，`initial_mtu=1200`（Tailscale WireGuard MTU=1280）

**根因结论**：跨节点 111-token 测试仍因 Tailscale VPN 在 sustained 400KB+ UDP 流量下约 2-3 分钟后完全断开而失败。这是网络基础设施限制，代码 timeout 已调至极限。9-token 短序列跨节点验证 ✅ 通过。
- 优化前：64K 分布式 decode 5 tokens 约 10min
- 优化后：64K 分布式 decode 5 tokens 约 1-2min
- 64K 分布式验证（70001 tokens，2-domain 同机 RTX 4090）：prefill+5 decode 成功，`EXIT=0`，输出 `The answer is: The`，总时间约 4min

[2026-05-05] **64K 分布式死锁修复与验证通过**（commit `f1b1040`）。根因：QUIC `stream_receive_window=128MB` < 64K 双 domain KV block 大小（224MB），双方同时 `write_all` 填满 send buffer 后互相等待，无人 recv → 死锁。修复分两层：
1. **代码逻辑层**：`KvTransport` trait 新增 `exchange_kv_block()`，`QuicKvTransport` 用 `tokio::join!` 并发执行 send+recv，打破串行 send→recv 的死锁模式；提取 `send_kv_block_to_stream` / `recv_kv_block_from_stream` 为纯 async 函数，允许独立借用 `send`/`recv` 字段。
2. **传输配置层**：`stream_receive_window=512MB`、`receive_window=1GB`、`send_window=1GB`，覆盖 128K 四 domain 场景。

---

[2026-05-05] **HCP kernel 长序列验证里程碑**：32K 单节点 ✅、32K 分布式 2-domain ✅、64K 单节点 ✅、64K 分布式 2-domain ✅、128K 单节点 ⏳。decode 端到端 ✅（同机 2-domain CUDA+CUDA + 跨节点 MPS+CUDA）。

**跨节点异构验证状态**：
- ✅ 本地同机 8K MPS+CPU 分布式 prefill+decode 成功（`generated: The quick brown fox jumps`）
- ✅ 跨节点 Mac MPS + RTX 4090 CUDA 9-token decode 成功（`generated: I am a`），验证异构设备间 QUIC KV ring 交换正常
- ✅ 跨节点异构验证矩阵（网络恢复后 RTT ~380ms）：
  - 9 tokens: ~60s ✅
  - 111 tokens: ~8min ✅ (`The quick brown fox jumps`)
  - 221 tokens: ~15min ✅ (`How many times does the`)
  - **551 tokens: ~30min ✅** (`What is the sentiment of`)
- Mac MPS + RTX 4090 CUDA 跨节点 QUIC KV ring 交换在 551-token（~3.5MB KV block）下稳定工作

**关键修复与优化**：
1. **`quic_transport.rs` + `backend.rs` 64K 分布式死锁修复**（commit `f1b1040`）：`stream_receive_window=128MB` < 64K KV block（224MB），双方同时 `write_all` 填满 send buffer 后互相等待 → 死锁。代码层：`KvTransport` trait 新增 `exchange_kv_block()`，`QuicKvTransport` 用 `tokio::join!` 并发 send+recv；配置层：`stream_receive_window=512MB`、`receive_window=1GB`、`send_window=1GB`。
2. `backend.rs` MPS NaN 回归修复（commit `7900a8e`）：`add+mul` workaround 产生 NaN（`0.0 * NEG_INFINITY = NaN`），改用 `where_self` 替代。42/42 测试通过。
3. `model.rs` LM head 长序列优化（commit `85d24cc`）：`seq_len > 8192` 时只计算最后一个 token 的 logits，消除 ~20GB 峰值，使 32K+ 在 24GB GPU 可行。
4. `model.rs` dense causal mask 跳过（commit `f8734a7`）：单节点长序列 prefill 传 `[1,1,1,1]` dummy mask 替代 `[seq_len, seq_len]` 密集 mask（64K 时达 16GB）。
5. `layers.rs` MLP chunking（commit `632393f`）：128K 单节点 prefill 在 MLP 层触发 `CUBLAS_STATUS_EXECUTION_FAILED`。MLP 是逐 token 独立的，安全 chunk（`seq_len > 8192`），峰值中间内存从 ~3.6GB 降到 ~225MB。

**验证结果**：
| 配置 | Seq | 结果 | 耗时 | 输出 |
|------|-----|------|------|------|
| 单节点 CUDA | 16K | ✅ | 2m8s | `jumps over the lazy dog` |
| 单节点 CUDA | 24K | ✅ | ~6min | `the lazy dog. The` |
| 单节点 CUDA | 32K | ✅ | ~8-10min | `dog. The quick brown` |
| 2-domain CUDA | 32K | ✅ | 3m53s | `dog. The quick brown` |
| 单节点 CUDA | 64K | ✅ | ~15-20min | `the lazy dog. The` |
| 2-domain 分布式 | 64K | ✅ | ~4min | `The answer is: The`（decode 优化后） |
| 同机 2-domain decode | 9 | ✅ | ~5s | `. The lazy dog is` |
| 跨节点 MPS+CUDA decode | 9 | ✅ | ~60s | `I am a` |
| 本地同机 MPS+CPU | 8K | ✅ | ~5min | `The quick brown fox jumps` |

**跨节点异构 decode 状态**：
- [x] [2026-05-05] Mac MPS (worker 0 + coordinator) + RTX 4090 CUDA (worker 1) 跨 VPN 完成 9-token prompt + 3 decode tokens 完整端到端。Coordinator `generated: I am a`，exit code 0。Worker 0/1 prefill → KV ring → decode → shutdown 全部正常。
- **架构纪律**：异构验证中每个平台都必须有 worker。coordinator 只负责 tokenizer 分片和 token 广播，不做模型计算。Mac 端同时跑 coordinator + worker 0 (MPS)，GPU 端跑 worker 1 (CUDA)，双方均执行 forward 和 KV ring 交换。
- HCP 跨节点异构 distributed CP 端到端验证通过（短序列）。长序列受限于 VPN 带宽。

**长序列验证矩阵**：
| 配置 | Seq | 结果 | 耗时 | 输出 |
|------|-----|------|------|------|
| 单节点 CUDA | 16K | ✅ | 2m8s | `jumps over the lazy dog` |
| 单节点 CUDA | 24K | ✅ | ~6min | `the lazy dog. The` |
| 单节点 CUDA | 32K | ✅ | ~8-10min | `dog. The quick brown` |
| 单节点 CUDA | 64K | ✅ | ~15-20min | `the lazy dog. The` |
| 单节点 CUDA | 128K | ❌ | — | `position_ids=131072` 超出 `max_position_embeddings=131072`（RoPE cache 索引越界） |
| 单节点 CUDA | 131071 prefill | ✅ | ~30-40min | prefill-only 成功（projection+MLP chunking 解决 cuBLAS） |
| 单节点 CUDA | 131067 | ✅ | ~30-40min | `gregated dog חדר.worm`（prefill+5 decode 端到端通过） |
| 2-domain CUDA | 32K | ✅ | 3m53s | `dog. The quick brown` |
| 2-domain 分布式 | 64K | ✅ | ~12-13min | `The answer is: The` |
| 同机 2-domain decode | 9 | ✅ | ~5s | `. The lazy dog is` |
| 跨节点 MPS+CUDA decode | 9 | ✅ | ~30-40s | `. The lazy` |

---

[2026-05-03] **统一 QUIC 控制面已完成并验证通过**（commit `7d62b46`）。Worker-coordinator 控制面从 TCP 迁移到 QUIC，与 KV ring 共用同一链路，享受 256MB window + 流控优化。

**本地验证**：2-domain CPU smoke ✅，`generated: . The lazy dog is`
**远程验证**：4090 CUDA 8K smoke ✅，`generated:  lazy dog. The quick`

**核心实现**（commit `7d62b46`）：
1. `distributed_protocol.rs`：新增 QUIC 版 frame I/O（`write_frame_quic`/`read_frame_quic`/`send_command_quic`/`recv_command_quic`/`write_handshake_quic`/`read_handshake_quic`）
2. `distributed_coordinator.rs`：Coordinator 从 TCP listener 改为 QUIC endpoint，`accept` incoming connections
3. `distributed_worker.rs`：Worker 从 TCP connect 改为 QUIC connect 到 coordinator
4. `quic_transport.rs`：导出 `read_exact` 供 protocol 复用

**前后对比**：
| 链路 | 之前 | 现在 |
|------|------|------|
| Worker ↔ Coordinator 控制面 | TCP（高延迟下 EAGAIN） | QUIC（256MB window） |
| Worker ↔ Worker KV ring | QUIC | QUIC |
| 全链路协议 | 混合（TCP+QUIC） | **统一 QUIC** |

---

[2026-05-02] **单进程多 domain worker（Multi-Domain Worker）已实现并本地验证通过**。下一步：远程 GPU 恢复后验证 CUDA，然后尝试 32K/64K。

**核心实现**（commit `8c1b4d0`）：
1. `distributed_worker.rs` 新增 `--local-domain-ids 0,1` 参数，单个进程内加载权重一次，通过 `Arc<ModelWeights>` + `shallow_clone` 共享给多个 domain thread。
2. 每个 domain thread 独立持有：
   - 自己的 `LlamaModel` 实例（权重 shallow_clone 共享底层 tensor）
   - 自己的 KV cache（不共享）
   - 自己的 coordinator TCP stream
   - 自己的 QUIC endpoint（per-domain）
3. `KvTransport` trait 添加 `Send` bound，支持跨线程使用。
4. **移除了 `forward_lock`**（原用于串行化 GPU 访问），原因是 ring attention 中 domain0 在 `recv_kv_block` 阻塞等待 domain1，但 domain1 被 lock 挡在外面 → 死锁。替代方案：`no_grad_guard()` + 自然 CUDA stream 串行化（推理时不保存计算图，MPS/CUDA 天然按 stream 顺序执行）。

**验证**：
- 本地 2-domain CPU smoke ✅ 通过，输出与参考一致（` in the universe.`）。
- 本地 4-domain CPU smoke ✅ 通过。

**显存模型分析**（单卡多 worker 的限制）：
- 权重共享：1 份 ~4.6GB（Qwen2-0.5B F32）
- 每 domain KV cache：`2 × num_layers × num_kv_heads × seq_len × head_dim × 4B`
- 每 domain LM head output：`batch × seq_len × vocab_size × 4B`（最大显存 spike）
  - 8K seq → 4.64GB ✅
  - 16K seq → 9.27GB（16GB 卡紧张）
  - 32K seq → 18.55GB ❌（超过单卡上限）
- **结论**：32K 必须 4-domain+（每 domain ≤8K），64K 必须 8-domain+。单卡双 worker 只能节省一份权重，但 LM head 和 KV cache 仍按 domain 数倍增，不能突破单卡显存上限。

**默认策略**：
- **生产默认 1 worker / 1 GPU**（最稳定、显存最可控）
- **开发环境可尝试 2 worker / 1 GPU**（权重共享省 ~4.6GB，适合小 seq 验证）
- **大 seq 场景必须多卡**：4-domain 32K 用 2 workers × 2 cards，8-domain 64K 用 4 workers × 2 cards 或 8 workers × 4 cards

---

[2026-05-02] **Phase 3: 大 Seq 工程验证 — 2K/4K/8K/16K 全矩阵通过**。下一步：32K/64K/128K 验证。

**问题**：3-domain CPU 4K 和 2-domain CPU 4K 分布式测试均出现 workers 卡死、coordinator 超时现象。Worker 进程 CPU 0%，coordinator `recv PrefillDone` 返回 `UnexpectedEof`。

**根因诊断**：
- Quinn 默认 `stream_receive_window = ~1.2MB`（由 `STREAM_RWND = MAX_STREAM_BANDWIDTH / 1000 * EXPECTED_RTT` 计算得出）。
- GQA repeat 后每个 KV block 的 frame 大小 = `2 * num_heads * seq_len * head_dim * 4 bytes` + metadata。
  - 2-domain 2048 tokens: ~14.7 MB
  - 3-domain 1365 tokens: ~9.8 MB
- 当 `write_all` 发送的数据超过接收方 stream window 时，quinn 阻塞等待 window update。但接收方也在 `write_all` 中发送自己的 KV block，没有调用 `recv_kv_block` 读取数据 → 经典分布式死锁。

**修复**：
1. `rust/src/quic_transport.rs`: `create_endpoint` 中显式增大 quinn transport window：
   - `stream_receive_window(32MB)` — 单 stream 可接收 32MB，覆盖最大 KV block
   - `receive_window(128MB)` — connection-level 总接收窗口
2. `rust/src/distributed_protocol.rs`: `write_frame` 添加 `stream.flush()`，确保控制帧及时到达对端。
3. `scripts/run_distributed_3node_smoke.sh`: 改为 `--release` build（与 2-node 脚本一致），避免 debug build 性能问题。

**验证**：
- 2-domain CPU 4K: ✅ 43s 完成，输出 `the lazy dog.`
- 3-domain CPU 4K (capacity-aware): ✅ 49s 完成，输出 `the lazy dog.`
- 2-domain/3-domain 短 prompt smoke: ✅ 回归通过

**后续发现与修复**：
- 16K 分布式（每 domain 8K）再次卡住。诊断发现：quinn 默认 `send_window = ~10MB`，而 8K seq 的 KV block 约 58MB。发送方写满 10MB 后阻塞等待 ACK，接收方也在 write_all 中发送自己的 block → **死锁**。
- 修复：`send_window=256MB`（与 `receive_window` 匹配）。

**完整大 seq 验证矩阵**：

| 配置 | Seq | 结果 | 耗时 | 输出 |
|------|-----|------|------|------|
| MPS 单节点 | 2K | ✅ | 3.95s | `dog. The quick` |
| MPS 单节点 | 4K | ✅ | 4.82s | `the lazy dog.` |
| MPS 单节点 | 8K | ✅ | 16.4s | `brown fox jumps over` |
| MPS 单节点 | 16K | ❌ OOM | — | `Invalid buffer size: 14.00 GiB` |
| MPS+CPU 分布式 | 8K | ✅ | 2m40s | `brown fox jumps over` |
| CUDA 单节点 | 4K | ✅ | 5.10s | `the lazy dog.` |
| CUDA 单节点 | 8K | ✅ | 6.39s | `brown fox jumps over` |
| CUDA 分布式 2-domain | 4K | ✅ | 1m11s | `the lazy dog.` |
| CUDA 分布式 2-domain | 8K | ✅ | 2m19s | `brown fox jumps over` |
| CUDA 分布式 2-domain | 16K | ✅ | 4m43s | `jumps over the lazy` |

关键结论：
1. MPS 单节点 16K OOM（attention scores 14GB 超过 MPS allocator 上限），分布式是扩展长 seq 的必需路径。
2. CUDA 单节点 8K 只需 6.4s，分布式 16K 4m43s（主要耗时在 24 层 × 2 轮 ring KV 交换）。
3. 所有分布式测试输出与单节点参考一致，验证正确性。

---

[2026-05-02] **Phase 2: Capacity-Aware 不均等分片已完成并验证**。

在 Phase 1 手动 `--chunk-sizes` 基础上，实现了自动 capacity 感知分配：
1. **跨平台 capacity 查询**：新建 `rust/src/capacity.rs`。
   - CUDA：`nvidia-smi --query-gpu=memory.free` 子进程解析。
   - MPS (macOS)：`sysctl hw.memsize` 总 RAM / 2（unified memory 保守启发式）。
   - CPU：`/proc/meminfo MemAvailable` (Linux) / `sysctl` (macOS) / 4（CPU 计算慢，故意 under-weight）。
2. **Largest-remainder 分配算法**：`allocate_by_capacity(L, c[])` 保证 `sum == L`、`min >= 1`、单调性。
3. **Handshake 协议扩展**：16-byte fixed（domain_id u64 LE + capacity_mb u64 LE）。
4. **Coordinator 三层优先级**：`--chunk-sizes`（精确手动）> `--capacity-aware`（自动比例）> 均分（默认 fallback）。
5. **动态 seq_offset 同步**：`WorkerCommand::Prefill` 扩展为包含 `seq_offset`，worker 收到后同时更新 `LlamaModel.seq_offset` 和所有 layer backend 的 `seq_offset`，确保 capacity-aware 模式下 causal mask 使用正确的全局位置。
6. **验证**：42/42 单元测试通过，clippy 零警告。本地 2-node CPU smoke 三种模式（baseline even / `--capacity-aware` / `--chunk-sizes 7,4` override）输出均一致（`" in the universe."`）。

---

[2026-04-30] **QUIC Transport、Mask 优化、动态不均等分片 Phase 1、1-token prefill bug 修复已完成并验证**。

在原有 TCP KV ring transport 基础上，本阶段完成了四项关键工程改进：

### 1. QUIC Transport
- 新增 `rust/src/quic_transport.rs`：`QuicKvTransport` 基于 `quinn` 0.11，使用单 QUIC connection + 每层一个 bidirectional stream，替代原来每层一对独立 TCP stream。
- 自签名证书生成（`rcgen`）+ 客户端跳过证书验证（`rustls::client::danger::SkipServerVerification`）。
- `tokio` async runtime 用于 quinn 事件循环；`KvTransport` trait 方法内部用 `block_on` 桥接同步调用。
- **修复 rustls 0.23 `CryptoProvider` 未安装默认 provider** 的运行时 panic。
- **修复 2-domain 对称连接死锁**：quinn 在 loopback 上同时 dial/accept 同一地址时可能合并 connection，`domain_id==0` 负责 dial、`domain_id==1` 只 accept，共享同一个 connection handle。
- **修复 quinn `open_bi()` 不立即发送 STREAM 帧** 导致 `accept_bi()` 永远挂起：stream 建立时 sender 先写入 1-byte dummy，receiver 首次 `recv_kv_block` 跳过该 byte。
- 本地验证通过：QUIC 2-domain CPU ✅、QUIC 3-domain CPU ✅、QUIC 2-domain MPS+CPU ✅；生成结果与 TCP baseline 和 Python transformers 参考完全一致。
- **跨机器性能对比**（Mac MPS + 远端 GPU CUDA:1，Tailscale VPN ~150ms RTT，Qwen2-0.5B 11 prompt + 20 decode tokens greedy）：
  - TCP KV ring: **107.3s**
  - QUIC KV ring: **76.4s**
  - **QUIC 比 TCP 快 ~29%**。高延迟网络下 QUIC 的 connection 复用、内置 TLS 和流控优势显现。

### 2. Mask 优化
- 分布式 prefill 阶段 `model.rs` 不再创建 `[seq_len, seq_len]` 密集 causal mask（O(seq²) 内存爆炸根因）。
- `HcpRingAttentionBackend::ring_attention` 已通过 `global_seq_start` + position 比较实现 causal，从不读取 mask 张量数据。
- 改为：单节点时仍创建完整 mask；分布式（`num_domains > 1`）时传 `[1,1,1,1]` dummy zero tensor 作为 causal 标志。
- 本地 2-domain/3-domain CPU smoke 验证通过，输出一致。

### 3. 动态不均等分片 Phase 1
- Coordinator CLI 新增 `--chunk-sizes`（逗号分隔，如 `7,4` 或 `5,3,3`），显式指定每个 domain 的 prompt chunk 长度。
- 分片逻辑校验：长度必须等于 `num_domains`，总和必须等于 prompt token 数。
- 测试验证：2-domain `7+4=11` ✅、3-domain `5+3+3=11` ✅，生成结果与参考一致。

### 4. 1-token prefill 边界 bug 修复
- 原代码用 `seq_len > 1` 区分 prefill/decode，chunk=1 时被误判为 decode（如 `--chunk-sizes 10,1`）。
- `LlamaModel` 和 `HcpRingAttentionBackend` 均新增 `is_prefill_done` 标志，第一次 `forward` 无论 `seq_len` 都走 prefill 路径。
- 验证矩阵：2-domain (6,5/7,4/8,3/9,2/10,1)、3-domain (4,4,3/5,3,3/6,3,2/7,2,2)、4-domain (3,3,3,2/4,3,2,2) 全部通过，输出与参考一致。31/31 单元测试通过，clippy 零警告。

---

[2026-04-30] **真实多进程分布式推理（Real Multi-Process Distributed Inference）已完成并验证**。

核心实现：
1. 新增 `distributed_worker.rs`：加载 LlamaModel，连接 peer（每层独立 TCP）做 KV ring 交换，连接 coordinator 执行 Prefill/Decode/Shutdown 命令循环。
2. 新增 `distributed_coordinator.rs`：加载 tokenizer+config（不加载权重），accept worker 连接（handshake 读取 domain_id 排序），prompt 分片 → 广播 `SyncGlobalSeqLen` → decode 循环广播 token → 采样 → 检查 EOS。
3. 新增 `distributed_protocol.rs`：`WorkerCommand`（Prefill/Decode/SyncGlobalSeqLen/Shutdown）与 `WorkerResponse`（PrefillDone/DecodeDone/Error）bincode 序列化，4-byte BE length prefix frame I/O。
4. `main.rs`：新增 `--distributed-role worker|coordinator` CLI 分发；解析到 role 后停止解析其余参数，避免冲突。
5. `BidirectionalTcpKvTransport`：封装每层一对 outbound/inbound `TcpStream`，实现 `KvTransport` trait。

关键 bug 修复：
- **Handshake**：worker 连接 coordinator 后发送 domain_id，coordinator 排序后再分配 chunk。否则 accept 顺序与 domain 顺序错位，导致 prefill chunk 发错 worker，生成结果完全错误。
- **SyncGlobalSeqLen**：prefill 后 coordinator 广播 max global_seq_len。否则 worker0 的 global_seq_len 仅为本地 chunk 长度（6），decode 时 RoPE 位置错误，输出与参考不一致。
- **main.rs CLI passthrough**：原 `parse_cli_args` 不认识 worker/coordinator 私有参数（`--domain-id` 等）并报 "unknown argument"。改为遇到 `--distributed-role` 后 `break`，让子模块自行解析 `std::env::args()`。

验证结果：
- **本地 2-node CPU smoke**：生成 " in the universe."，与 Python transformers 参考 **完全一致**。
- **远端 2-node 多进程（CPU + CUDA:1）**：生成结果一致。
- **本地非沙箱 MPS + CPU 混合**：生成结果一致。
- **跨机器 MPS + CUDA:1**：生成结果一致。VPN 内网无防火墙，直接 0.0.0.0 bind 即可互通。
- **3-domain 本地 CPU×3**：true ring forwarding 验证通过，生成 " in the universe."，与参考一致。
- **3-domain 跨机器（MPS + CUDA:1 + CUDA:2）**：本地 Mac worker0(MPS) + 远端 worker1(CUDA:1) + 远端 worker2(CUDA:2) 形成真正 ring，生成 " in the universe."，与参考 **完全一致**。验证了 3-domain 异构跨机器分布式推理端到端可行。

性能基准（Qwen2-0.5B，prompt 11 tokens，generate 20 tokens，greedy）：
| 配置 | 总耗时 | 说明 |
|---|---|---|
| 单节点 MPS | 2.06s | Mac M1 Pro 本地基线 |
| 单节点 CPU | 2.87s | Mac CPU 本地基线 |
| 3-domain 本地 CPU×3 | 13.20s | 同机 3 进程，模型权重重复加载 + loopback TCP KV 交换 |
| 3-domain 跨机器 (MPS+CUDA:1+CUDA:2) | 125.46s | VPN 延迟 ~75ms RTT，decode 每步需网络往返 |
| 跨机器 TCP (MPS+CUDA:1) | 107.3s | 2-domain，prompt 11 + decode 20 |
| 跨机器 QUIC (MPS+CUDA:1) | 76.4s | 2-domain，prompt 11 + decode 20，比 TCP 快 29% |

当前无阻塞。

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
- [2026-04-25] 新增 `torch_payload_chunk_bridge`：C++ ATen 将 online softmax state 扩展到小尺寸 Q chunk，输出 `[query, head_dim]` chunk tensor。本机 MPS / 远端 CUDA 主 smoke 均通过 `30/30`，3-node remote CP 每个 node 均通过 `12/12`。
- [2026-04-26] 新增 `torch_query_chunk_bridge`：Rust/domain-side 生成显式 float32 Q chunk payload，C++ ATen bridge 消费该 Q payload 与 captured K/V payload blocks，不再在该路径内部构造 Q。本机非沙箱 MPS 主 smoke 通过 `torch_query_chunk_code=2 30/30`，远端 CUDA 主 smoke 通过 `torch_query_chunk_code=3 30/30`。
- [2026-04-26] 尝试重跑 3-node remote CP query chunk smoke 时，node2 先启动后连接 node0 超时退出；随后 GPU host `192.168.8.172` SSH 返回 `No route to host` / `Host is down`。本机已确认无残留 remote CP/SSH 进程，待网络稳定后重跑 3-node。
- [2026-04-26] 3-node remote CP query chunk smoke 已重跑通过：失败根因是 Mac 的 `192.168.8.x` 地址从 `192.168.8.204` 变化到 `192.168.8.239`；使用当前地址后 node0/node2 MPS 均通过 `torch_query_chunk_code=2 12/12`，GPU node1 CUDA 通过 `torch_query_chunk_code=3 12/12`。
- [2026-04-26] Rust protocol 新增 `DomainModelState`：每个 domain 持有本地 Q chunk 与 K/V storage；K/V 消息从 source state 切片生成，compute capture 携带 target state 的 Q payload。`torch_query_chunk_bridge_report` 已按 `compute_domain` 分组，避免不同 domain blocks 混用同一个 Q。
- [2026-04-26] `DomainModelState` 路径已验证：`cargo test` 通过 2 个 state 单元测试；本机非沙箱 MPS 主 smoke、远端 CUDA 主 smoke、`RUN_ID=rust-remote-cp-modelstate-20260426` 三节点 remote CP smoke 均通过。
- [2026-04-26] 新增 `scripts/run_rust_remote_cp_3node_smoke.sh` 统一 launcher：自动发现当前 Mac `192.168.8.x` 地址、远端 GPU `git pull --ff-only`、本机/远端 cargo preflight build，并统一启动 node0/node2 MPS 与 node1 CUDA。
- [2026-04-26] `RUN_ID=rust-remote-cp-output-unified-20260426 PORT_BASE=29285 bash scripts/run_rust_remote_cp_3node_smoke.sh` 已通过；三节点均 `sent=8 received=8 compute_updates=12`，MPS nodes `torch_query_output_code=2 12/12`，CUDA node `torch_query_output_code=3 12/12`。
- [2026-04-27] `DomainModelState` 已拆出 `LayerActivationState`，明确每个 domain-local layer 拥有 Q chunk、K cache、V cache 和 output slot；`torch_query_output_bridge` 会校验 output value 数匹配本地 output slot。
- [2026-04-27] 临时 VPN 路由已验证：CUDA 节点 `100.118.253.68`，Mac 节点 `100.121.35.138`。`RUN_ID=rust-remote-cp-modelstate-vpn-20260426 PORT_BASE=29295 GPU_HOST=100.118.253.68 MAC_NODE_ADDR=100.121.35.138 bash scripts/run_rust_remote_cp_3node_smoke.sh` 三节点通过，node0/node2 MPS `torch_query_output_code=2 12/12`，node1 CUDA `torch_query_output_code=3 12/12`。
- [2026-04-29] `DomainModelState` 已从直接 Q/K/V fixture 生成推进到 projection 数据流：domain-local hidden states 通过 `ModelLayerWeights` 的 Q/K/V projection 生成 Q chunk、K cache、V cache；当前 projection weights deterministic 初始化，用于可复现 smoke，后续可替换真实权重加载。
- [2026-04-29] `RUN_ID=rust-remote-cp-projection-lan-20260429 PORT_BASE=29315 GPU_HOST=192.168.8.172 MAC_NODE_ADDR=192.168.8.239 bash scripts/run_rust_remote_cp_3node_smoke.sh` 已通过；远端 CUDA node 从 `ccffedc` fast-forward 到 `46c9e18`，三节点均 `sent=8 received=8 compute_updates=12 torch_query_output_blocks=12/12`。
- [2026-04-30] `cargo check` 在线模式已确认可用（rsproxy-sparse 可达），之前 memory-bank 记录的 "rsproxy.cn DNS 失败" 为临时问题。
- [2026-04-30] Rust `Cargo.toml` 已新增 optional `tch = "0.24.0"` 和 `tch-backend` feature gate；`torch-sys` build script 只需要 `LIBTORCH=/Users/stark_sim/libtorch` 环境变量，**不要**同时设置 `LIBTORCH_INCLUDE` 或 `LIBTORCH_LIB`，否则 `torch-sys` 会将其当作 `LIBTORCH` 重复追加 `/include` 导致 `torch/torch.h` 找不到。
- [2026-04-30] 新增 `rust/src/bin/tch_smoke.rs`：在 `tch-backend` feature 下运行 matmul / softmax / attention-like 三组 op，与 CPU reference 对比 `max_abs_err` / `mean_abs_err`，输出结构化 JSON report。
- [2026-04-30] 新增 `scripts/run_tch_ringattn_smoke.sh`：自动设置 `LIBTORCH`、`DYLD_LIBRARY_PATH`、`LD_LIBRARY_PATH`，支持 `HCP_TCH_DEVICE=cpu|mps|cuda` 和 `RUN_ID`。
- [2026-04-30] tch-rs CPU smoke 通过：`tch_status=pass tch_device=cpu tch_code=1 ops=3/3`。
- [2026-04-30] tch-rs MPS smoke 通过：`tch_status=pass tch_device=mps tch_code=2 ops=3/3`（当前进程可直接访问 Metal device）。
- [2026-04-30] 新增 `rust/src/tch_backend.rs`：用 `tch::Tensor` 实现 attention block update smoke，对标 C++ `RunTorchAttentionBlockUpdates`；CPU/MPS 均通过 `tch_attention_status=pass`。
- [2026-04-30] 远端 CUDA tch 验证通过：基础 `tch_smoke` `tch_code=3`，主 binary `tch_attention_bridge` `tch_code=3`；根因是 Linux `--as-needed` 丢弃 `libtorch_cuda.so`，通过 `build.rs` 的 `cargo:rustc-link-arg-bins` 修复。
- [2026-04-30] `run_rust_ringattn_smoke.sh` 已默认在有 `LIBTORCH` 时自动启用 `--features tch-backend`，同时设置 `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH`；无 `LIBTORCH` 时 `tch_attention=disabled` 不影响现有流程。
- [2026-04-30] 新增 `docs/TCH_BACKEND_DESIGN.md`，记录架构定位、数据流、构建链接注意事项、验证矩阵和与 C++ bridge 的对应关系。
- [2026-04-30] `main.rs` 已新增 `tch_attention_bridge` report 字段，CLI summary 同步输出 `tch_attention_status` / `tch_attention_code`；当 `tch-backend` feature 未启用时状态为 `disabled`，不影响 C++ ATen 路径。
- [2026-04-30] `tch_backend.rs` 已扩展全部 6 个桥接函数：`run_attention_block_updates`、`run_payload_block_smoke`、`run_payload_online_smoke`、`run_payload_chunk_smoke`、`run_query_chunk_smoke`、`run_query_chunk_output_smoke`（通过 `run_query_chunk_smoke` 返回 checksum/max_err/output_values）。
- [2026-04-30] `main.rs` 已完整接入全部 5 个 tch payload/query 桥接 report：`tch_payload_block_bridge`、`tch_payload_online_bridge`、`tch_payload_chunk_bridge`、`tch_query_chunk_bridge`、`tch_query_output_bridge`。`Report` 和 `RemoteCpNodeRunReport` 均已包含对应字段；`run()` 和 cp-node 路径均已调用并纳入 `status` pass/fail 逻辑；CLI summary 已输出全部 tch 字段；fail message 打印块已补充。
- [2026-04-30] 本机 CPU tch 全桥接 smoke 通过：`status=pass`，全部 6 个 tch bridge `status=pass code=1`，payload/query 各 `30/30`，query output `groups=3`。
- [2026-04-30] 本机 MPS tch 全桥接 smoke 通过：全部 6 个 tch bridge `code=2`。
- [2026-04-30] 远端 CUDA tch 全桥接 smoke 通过：全部 6 个 tch bridge `code=3`。
- [2026-04-30] 3-node remote CP tch 全桥接 smoke 通过：`RUN_ID=rust-remote-cp-tch-full-20260430 PORT_BASE=29325`，node0/node2 MPS `code=2 12/12`，node1 CUDA `code=3 12/12`；C++ bridge 和 tch bridge 并行全部通过。
- [2026-04-30] `scripts/run_rust_remote_cp_node.sh` 已支持自动启用 `tch-backend` feature 并显式指定 `--bin hcp-ringattn-rust`；`scripts/run_rust_remote_cp_3node_smoke.sh` 已传递 `HCP_TCH_DEVICE` 到所有节点并在 preflight build 中启用 tch-backend。
- [2026-04-30] tch-backend 已接入实时 compute 路径：`protocol.rs` 的 `run_cp_ring_node` 和 `run_remote_cp_node` 不再只是 NoOp 计数，而是在每个 K/V block 到达时立即调用 `compute_chunk_attention_step` 更新 online softmax accumulator；ring 结束时 finalize 到 `output_slot`。
- [2026-04-30] 实时 compute 验证：CPU `tch_compute_output_checksum=1357.6399246966466` 与离线后验 smoke 的 3 个 group checksum 总和完全一致；MPS `checksum=1357.639936434105`，差异 ~1.2e-8。
- [2026-04-30] RoPE 已接入 protocol：`apply_rope` 在 Q chunk 和 K cache 构建时对每个 token 按 head 应用旋转位置编码。
- [2026-04-30] LayerNorm 已接入 protocol：`LayerNormWeights` + `layer_norm` 在 projection 之前对 hidden states 做逐 token 归一化。
- [2026-04-30] o_proj + Residual Connection 已接入 protocol：`ModelLayerWeights.o_proj` 将 attention output `[num_heads, head_dim]` 映射回 `[hidden_dim]`；`finalize_tch_compute_output` 执行 `residual_input + o_proj_out` 写入 `output_slot`。
- [2026-04-30] 接入 RoPE/LayerNorm/o_proj/Residual 后 CPU/MPS smoke 均通过，checksum 完全一致（`1093.5917292535305`）。
- [2026-04-30] `scripts/run_rust_remote_cp_node.sh` 已支持自动启用 `tch-backend` feature 并显式指定 `--bin hcp-ringattn-rust`；`scripts/run_rust_remote_cp_3node_smoke.sh` 已传递 `HCP_TCH_DEVICE` 到所有节点并在 preflight build 中启用 tch-backend。
- [2026-04-30] 外部权重加载已接入 protocol：`ModelWeightsJson` 支持从 JSON 文件解析 `layers` 数组，每个 layer 包含 `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gamma`/`beta`；`DomainModelState::new_with_weights` 可在有外部权重时替换默认合成权重。
- [2026-04-30] 权重加载路径验证：本地默认 smoke `tch_compute_output_checksum=1093.59...`；加载 `config/test_weights.json` 后 checksum 变为 `2810.30...`，确认外部权重确实参与计算；非 tch-backend 编译和运行均不受影响。
- [2026-04-30] VPN 三节点 remote CP tch-full 验证通过：`GPU_HOST=100.118.253.68 MAC_NODE_ADDR=100.121.35.138 RUN_ID=rust-remote-cp-tch-full-vpn-20260430 PORT_BASE=29335`，node0/node2 MPS `code=2 12/12`，node1 CUDA `code=3 12/12`；C++ bridge 与 tch bridge 全部通过，实时 compute checksum 分别为 71.35 / 238.88 / 406.41。
- [2026-04-30] M2 correctness 扩展完成：cases 从 3 个扩展到 7 个（含大 seq 512+256+256、单 block、unit block）；新增 `max_rel_err` 到 correctness model 和 tch-backend；新增 `--stress-test` CLI flag（5 seeds，自动跳过 seq>256）；本地 MPS smoke 和 VPN 三节点 remote CP 均验证通过。
- [2026-04-30] M3 protocol 优化完成：统一 TCP frame I/O（`write_frame_to_stream`/`read_frame_from_stream`）；提取 `process_inbound_message` 去重 `run_cp_ring_node` 与 `receive_remote_cp_node_messages` 的消息处理逻辑；SSH ConnectTimeout=30 修复 VPN 远程 smoke 连接超时。
- [2026-04-30] M4 异构 runtime 闭环完成：提取 `ComputeRuntime` trait，`TchComputeRuntime` 执行真实 tensor 计算，`NoOpComputeRuntime` 仅作为无 tch-backend feature 时的编译兼容 fallback；计算路径从 protocol 逻辑中完全解耦；本地 MPS smoke 和 VPN 三节点 remote CP 均验证通过，checksum 与重构前完全一致。
- [2026-05-01] 修复 inference pipeline 中 causal mask 的 NaN bug：`make_causal_mask`（backend.rs 测试用）和 `create_causal_mask`（model.rs prefill 用）均使用 `mask.to_kind(Float) * NEG_INFINITY`，其中 False→0.0 与 NEG_INFINITY 相乘产生 NaN；改为 `zeros(...).masked_fill(&mask, NEG_INFINITY)` 后修复。Qwen2-0.5B 和 tiny model 均从全 "!" / "<unk>" 恢复为有意义的生成输出。
- [2026-05-01] 修复 `test_chunk_step_vs_softmax_single_block`：`compute_chunk_attention_step` 输出布局为 `[num_heads, query_len, head_dim]`，但测试代码额外加了 `.permute(&[1, 0, 2])` 导致 `actual` 变为 `[1, query_len, num_heads, head_dim]`，与 expected `[1, num_heads, query_len, head_dim]` 不匹配；去掉 permute 后 diff=2.9e-8。
- [2026-05-01] 修复 `ring_attention` causal mask 路径：`compute_chunk_attention_step` 内部从 q/k/v payload 重新计算 scores，不感知 causal mask。当 `attention_mask.is_some()` 时，改为在 `ring_attention` 中直接使用已应用 causal mask 的 `scores` tensor 做 online softmax 更新（与 `compute_chunk_attention_step` 数学等价，但尊重 mask）。`test_ring_attention_matches_local_causal` diff=9.4e-7。
- [2026-05-01] Qwen2-0.5B 权重重新下载完成（942MB），safetensors 验证通过，无全零层。
- [2026-05-01] 修复 MPS 上 `HcpRingAttentionBackend` 设备不匹配：`ring_attention` 方法中 `output_acc` 从 CPU buffer 通过 `Tensor::from_slice` 重建，导致输出 tensor 留在 CPU；最终与 MPS 上的 `o_proj` matmul 时报 "mat1 is on CPU"。修复：在 `ring_attention` 返回前加 `.to(q.device())`。修复后 MPS `num_domains=2` 输出与 `num_domains=1` 一致（`, theT. \`），16/16 单元测试仍通过。
- [2026-05-01] 消除 inference 路径中所有 CPU fallback：
  - `ring_attention` causal prefill 路径改为全程 device tensor online softmax（`rm`/`rs`/`obh` 直接在 MPS 上创建和更新，彻底移除每个 KV block 的 D2H/H2D round-trip）。无 mask 的 protocol smoke 路径保留 `compute_chunk_attention_step` 兼容。
  - `GqaAttention::forward`、`HcpRingAttentionBackend::local_attention_scores`、`RmsNorm::forward` 中的 `Tensor::from(scale).to_kind(Float)` 和 `Tensor::from(eps).to_kind(Float)` 均改为直接标量乘法/加法，避免创建 CPU 标量 tensor。
  - 验证 `Tensor::embedding` Int64 on MPS 和 RoPE `index_select` Int64 on MPS 均无需 fallback（之前的 "Placeholder storage" 报错未复现，可能是特定版本或特定条件下的 transient issue）。
  - MPS `num_domains=1/2/4` 输出一致；16/16 单元测试通过；MPS 全量 smoke（tch_attention + 5 个 payload/query bridge）均通过。
- [2026-05-01] Phase 1 Checkpoint 1-4: `safetensors`/`tokenizers`/`half` 依赖；`ModelConfig` 解析 HF `config.json`（Llama/Qwen2/Mistral 家族）；`ModelWeights` 从 `.safetensors` 加载并转换 F16/BF16→F32；`RmsNorm`、可配置 `RotaryEmbedding`、`SwiGLU` MLP；`GqaAttention`（RoPE + GQA + causal mask + KV cache）；`LocalAttentionBackend`（`AttentionBackend` trait）；`DecoderLayer`（Pre/Post-Norm + Residual）；`LlamaModel`（Embedding → N-layer → RMSNorm → LM Head）；`Generator`（prefill + decode 自回归 + temperature 采样）；inference CLI（`--infer-model-dir`/`--infer-prompt`/`--infer-max-tokens`）；合成 tiny 模型验证 pipeline 端到端跑通。
- [x] [2026-05-01] Phase 2 Checkpoint 5 完成：`HcpRingAttentionBackend` 已接入真实推理路径；`LlamaModel` 支持 `num_domains` 切换（`--infer-num-domains` CLI 参数）；Qwen2-0.5B 验证 `num_domains=1/2/4` 输出一致。
- [x] [2026-05-01] 外部权重加载：支持通过 `HCP_WEIGHTS_JSON` 环境变量从 JSON 文件加载 Q/K/V/O projection weights 和 LayerNorm gamma/beta；已验证本地 CPU/MPS smoke 均正常，checksum 随权重变化而变化。
- [x] [2026-05-01] VPN 三节点 remote CP 验证：MPS + CUDA 异构 domain 通过 C++ bridge 与 tch bridge 全量 smoke；M5 远端闭环已完成。
- [x] [2026-05-01] Phase B: 分布式 decode 路径打通。`seq_len <= 1` 回退已移除，decode 阶段走完整 `ring_attention` 路径；`seq_offset` 传入 `AttentionBackend::set_distributed`，`forward` 使用固定 `seq_offset` 代替 `position_ids.min()` 计算 `global_seq_start`；decode 阶段发送 KV 时排除新 append 的 token，避免 ring 中重复；`kv_chunks` 改用本地 KV 长度而非 Q 的 `seq_len`；新增 `LlamaModel::global_seq_len` 保证 decode position_ids 正确。新增 4 个单元测试验证 decode 路径数学正确性；23/23 测试通过，clippy 零警告。
- [x] [2026-05-01] 修复分布式 decode ~6 diff 根因：`LlamaModel::forward` 中 prefill 阶段错误地将 `global_seq_len` 设为本地 `seq_len`（domain0=8），导致 decode 时 `position_ids=8` 而非正确的全局位置 16。修复后 diff 从 6.7 降至 ~2e-6，与单节点参考一致。
- [x] [2026-05-01] Phase B++: A) `test_tcp_kv_transport_roundtrip` 验证 TCP 序列化无损（k_diff=0, v_diff=0）；B) `test_distributed_llama_model_multi_step_decode` 验证 4 步连续分布式 decode，每步 diff ~2e-6。修复多步 decode 根因：`history_len = k.size()[2] - 1` 在多步时会包含之前 decode append 的 token，引入 `prefill_kv_len` 字段确保只发送 prefill 分区。
- [x] [2026-05-01] Phase C: 分布式 Generator `DistributedGenerator`。单进程模拟多 domain CP 推理：prefill 分片到各 domain → 同步 global_seq_len → decode 循环广播 token → 采样。`test_distributed_generator_tokens_match_reference`：4 步贪婪 decode，domain0/domain1 token 完全一致，与单节点参考 logits diff ~1e-5。31/31 测试通过，clippy 零警告。
- [x] [2026-05-01] Phase D: `RingAttnMessage` serialization/deserialization 测试覆盖。新增 5 个测试：bincode roundtrip、payload 完整性（256 bytes）、schema version 字段、三种 message kind 全覆盖、TCP transport trait 端到端。30/30 测试通过，clippy 零警告。
- [x] [2026-05-01] Phase 3 Step 1-5: `KvTransport` trait 与 `KvBlock` 创建；`MockKvTransport` 支持 in-memory 测试；`HcpRingAttentionBackend` 集成 `KvTransport`（`send_local_kv` + `process_peer_block` + `global_seq_start`）；`LinkedMockKvTransport` 修复自环 bug（`peer_inbox`/`self_inbox` 分离）；测试代码修复 layer transport 覆盖 bug（每层独立 transport pair）；`test_distributed_llama_model_prefill` 端到端分布式 prefill 通过（2-layer、GQA、seq_len=16 拆成 2 domain），diff=2.79e-6；关键代码已补充详细中文注释。
- [x] [2026-04-30] **真实多进程分布式推理完成**：`distributed_worker.rs`、`distributed_coordinator.rs`、`distributed_protocol.rs` 创建；Worker 加载模型权重做 KV ring 交换，Coordinator 加载 tokenizer+config 做 prompt 分片和 token 广播；Handshake（domain_id 排序）、`SyncGlobalSeqLen` 广播、`BidirectionalTcpKvTransport` 每层独立 TCP stream。本地 2-node CPU ✅、远端 CPU+CUDA:1 ✅、本地 MPS+CPU ✅、跨机器 MPS+CUDA:1 ✅、3-domain 本地 CPU×3 ✅、3-domain 跨机器 MPS+CUDA×2 ✅。
- [x] [2026-04-30] **QUIC Transport 完成**：`quic_transport.rs` 实现 `QuicKvTransport`，基于 `quinn` 单 connection + per-layer bidirectional stream；修复 rustls `CryptoProvider` 未初始化、2-domain 对称连接死锁、quinn `open_bi` 不发送 STREAM 帧导致 `accept_bi` 挂起（1-byte dummy workaround）。本地 2-domain/3-domain/MPS+CPU 均通过，输出与 TCP baseline 一致。**跨机器 QUIC vs TCP 性能对比**（Mac MPS + 远端 CUDA:1，VPN ~150ms RTT）：TCP 107.3s，QUIC 76.4s，QUIC 快 **~29%**。
- [x] [2026-04-30] **Mask 优化完成**：分布式 prefill 不再分配 `[seq_len, seq_len]` 密集 causal mask；`ring_attention` 已用 `global_seq_start` + position 比较实现 causal。单节点时创建完整 mask；分布式时传 `[1,1,1,1]` dummy zero tensor 作 causal 标志。本地 2-domain/3-domain CPU smoke 验证通过。
- [x] [2026-04-30] **动态不均等分片 Phase 1 完成**：Coordinator CLI 新增 `--chunk-sizes`（逗号分隔，如 `7,4` 或 `5,3,3`），显式指定每个 domain 的 prompt chunk 长度。分片逻辑校验：长度必须等于 `num_domains`，总和必须等于 prompt token 数。2-domain `7+4=11` ✅、3-domain `5+3+3=11` ✅，生成结果与参考一致。
- [x] [2026-04-30] **1-token prefill 边界 bug 修复**：`LlamaModel` 和 `HcpRingAttentionBackend` 均新增 `is_prefill_done` 标志，第一次 `forward` 无论 `seq_len` 都走 prefill 路径。验证矩阵：2-domain (6,5/7,4/8,3/9,2/10,1)、3-domain (4,4,3/5,3,3/6,3,2/7,2,2)、4-domain (3,3,3,2/4,3,2,2) 全部通过。31/31 单元测试通过，clippy 零警告。

## 活跃决策

- HCP 与 HLPP 保持边界清晰：本仓只做 intra-layer / low-boundary Ring Attention。
- 跨异构 domain 坚持 P2P，不把 collective 作为主通信假设。
- 当前优先级是 correctness -> protocol -> transport -> remote heterogeneous deployment -> scaling argument。
- 每个实验阶段都应产生结构化 report，而不是只保留日志或口头结论。
- Rust + C++ 是后续核心工程路径；`tch-rs` 作为 PyTorch Rust 绑定参考，当前 upstream `tch` crate 为 `0.24.0`，与 PyTorch/libtorch 2.11 路线匹配，但默认构建暂不强依赖 `tch` crate。
- PyTorch C++ 路径短期优先使用 C++ ATen/libtorch bridge，避免直接 include 全量 `torch/torch.h`。
- `tch-rs` 的长期接入应优先使用独立/system-wide libtorch；`LIBTORCH_USE_PYTORCH=1` 只作为 fallback 或快速验证路径，避免把核心 Rust 路线重新耦合到 Python 环境。
- `tch-backend` feature 已接入：只需 `LIBTORCH=/Users/stark_sim/libtorch`，不要同时设 `LIBTORCH_INCLUDE`/`LIBTORCH_LIB`；macOS 运行时需要 `DYLD_LIBRARY_PATH` 包含 `${LIBTORCH}/lib`。
- 远端 GPU host `~/.profile` 已收敛环境变量配置（`LIBTORCH`、`LD_LIBRARY_PATH`、`PATH`），本地 `scripts/run_rust_remote_cp_3node_smoke.sh` 已移除 `remote_env_exports()` 显式传入，改为完全依赖远端 `bash -l` 加载 `.profile`。
- [2026-05-01] 当前网络环境已切换：CUDA 节点通过 `user@sd-1`（SSH 别名，HostName `sd-1` → IP `100.64.0.93`）访问，Mac 本机当前可达地址为 `100.64.0.95`。
- [2026-05-01] 3-node remote CP smoke 已通过：`GPU_HOST=100.64.0.93 GPU_USER=user MAC_NODE_ADDR=100.64.0.95`，node0/node2 MPS `code=2 12/12`，node1 CUDA `code=3 12/12`。注意：`GPU_HOST` 必须用 IP 地址，Rust `SocketAddr` 解析不支持主机名（`sd-1:29410` 会报 `invalid socket address syntax`）。
- [2026-05-01] **Transport trait 重构后 3-node regression 验证通过**：`PORT_BASE=29250 GPU_HOST=100.64.0.93 GPU_USER=user MAC_NODE_ADDR=100.64.0.95`；node0/node2 MPS `code=2`，node1 CUDA `code=3`；全部 C++ bridge 与 tch bridge `12/12` pass；checksum 分别为 71.35 / 238.88 / 406.41，与重构前完全一致。transport trait 统一未引入 regression。
- [2026-05-01] **M4 异构 runtime stub 核心完成**：`TchComputeRuntime` 接入设备配置（`select_tch_device_from_env()` 从 `HCP_TCH_DEVICE`/`HCP_TORCH_DEVICE` 解析 cpu/mps/cuda/cuda:N）；移除硬编码 `tch::Device::Cpu`；本地 CPU/MPS smoke checksum 一致（1093.59）；remote 3-node smoke MPS+CUDA 异构通过，checksum 与重构前一致。
- [2026-04-30] 明确 HCP 与 PyTorch 官方 Context Parallel 的边界：HCP 采用原始 Ring Attention 论文的 P2P 设计，支持异构/非均分；PyTorch 2.7+ CP 是同构 GPU 集群的 collective 优化。已记录于 `systemPatterns.md`。

## 下一步

- [x] decode 性能优化：decode 阶段 `kv_chunk_size = 2048`，64K 分布式 decode 从 ~10min 降至 ~1-2min
- [x] 远程 GPU 验证：64K 分布式 2-domain CUDA ✅（decode 优化后 ~4min）
- [x] 跨节点长序列异构验证：551 tokens ✅（~30min）。更长序列受限于 Tailscale VPN 带宽
- [ ] 多 worker launcher 脚本：扩展 `run_rust_remote_cp_3node_smoke.sh` 支持 `--local-domain-ids` 参数传递
- [ ] M6：memory / bandwidth scaling notes 与 context-length growth argument。
- [x] 动态不均等分片 Phase 2：worker handshake 上报 `free_vram_mb`，coordinator 按 capacity 比例自动分配 chunk sizes ✅ 已完成
- [ ] KV 传输压缩：当前 KV 以原始 float32 传输，大 seq 场景（32K+/64K+/128K）带宽压力巨大。探索量化（INT8/FP8）或差分编码。
- [x] 大 seq 工程验证：32K ✅ / 64K ✅ / 128K 边界 131067 ✅（单节点）
- [ ] 将 Rust correctness model 继续拆分为 library + binary，便于后续 protocol / transport 复用。
- [x] 远端多进程分布式验证：在 `sd-1` GPU 节点上启动 worker，Mac 本机启动 coordinator，验证跨机器 QUIC ring + 异构设备 ✅
- [ ] 多 worker launcher 脚本：扩展 `run_distributed_2node_smoke.sh` 支持远程 SSH 启动 worker、统一日志收集。
- [x] 分级 tolerance policy 已落地：`ToleranceTier` 三级（Strict/Relaxed/EndToEnd），`--tolerance-tier` CLI 参数支持切换，correctness report 包含 tier 信息。
- [x] **Phase 1: Python Worker SDK 控制面验证**（单节点，commit `dabd6fc`）
  - ✅ `TransformersBackend` 实现 `HcpWorkerBackend` trait
  - ✅ `HcpWorkerServer` 事件循环接收 Prefill/Decode/Shutdown，返回 logits
  - ✅ 简化 Coordinator 端到端生成验证通过
- [x] **Phase 1.5: Python Worker SDK + vLLM Adapter MVP**（单节点 vLLM，commit `29bab9b`）
  - ✅ `VllmBackend` 接入 vLLM 0.6.4 `LLM` API，在远程 RTX 4090 上加载 Qwen2-0.5B
  - ✅ 控制面 Prefill/Decode/Shutdown 全链路通过，输出与 transformers baseline 一致：` in the universe`
  - ✅ `NoOpKvTransport` 单节点跳过 KV exchange
- [x] **Phase 2: Python Worker SDK 2-domain Mock KV Ring 验证**（commit `1cacb35`）
  - ✅ `LinkedMockKvTransport` 单进程模拟 2-domain KV 交换
  - ✅ `TransformersBackend` `get_kv_block` / `apply_peer_kv` 正确性验证
  - ✅ 2-domain 输出与单节点参考完全一致
- [x] **Phase 2.5: vLLM Adapter 控制面流程验证**（commit `fd8649c` + `df22de7`）
  - `VllmBackend` 接入 vLLM 0.6.4 `LLM` API，控制面 Prefill/Decode/Shutdown 全链路通过
  - 新增 `test_worker_single.py`：自动化单 worker + coordinator 端到端验证（subprocess 启动，避免 vLLM 与 multiprocessing 嵌套冲突）
  - 远端 RTX 4090 验证通过：`generated: ' in the universe. The'`，与 transformers 参考完全一致 ✅
  - **单卡 2-domain vLLM 不可行**：vLLM `gpu_memory_utilization` 基于总显存比例，首个实例加载后导致第二个实例 `kv_cache_size` 为负。方案 B（多机/多卡，每设备单实例）是正确路径
  - 真实 KV ring 需 Phase 3 接入 vLLM `LLMEngine` 底层 API

- [x] **Python QUIC Transport 实现 + Python↔Rust 互通验证**（commit `16c14e7` + `99b9ceb`）
  - 基于 `aioquic` 实现 `QuicKvTransport`，帧格式与 Rust `quinn` 完全兼容
  - 关键兼容点：1-byte dummy handshake（Rust `open_bi()` 后立即写 `\x00`，接收方首次 `read` 跳过）
  - 自签名证书生成 + 开发环境跳过验证
  - `test_quic_kv.py` Python 内部闭环验证通过 ✅
  - `test_quic_python_rust.py` + `quic-echo-server` Rust server 互通验证通过 ✅
    - Python client 发送 KV block → Rust server 接收 → k/v + 1.0 → 回显 → Python 验证 checksum 一致

- [x] **Phase 3.1: Python bincode 编解码器 + QUIC 控制面接入 Rust coordinator**（commit `fc6a6de`）
  - `python/hcp_worker_sdk/bincode.py`: 手写 bincode 1.3 兼容编解码器，覆盖 `WorkerCommand`/`WorkerResponse`/`WorkerHandshake`
    - enum tag: u32 LE (4 bytes); usize/i64/u64: 8 bytes LE; Vec: 8-byte len prefix + raw bytes
    - 6 个单元测试全部通过，与 Rust `bincode::serialize` 输出逐 byte 匹配 ✅
  - `python/hcp_worker_sdk/quic_control.py`: `QuicControlClient` 基于 `aioquic`
    - Handshake: 16 bytes LE (domain_id u64 + capacity_mb u64)，无 length prefix
    - Command/Response: [4-byte BE length][bincode payload]
    - 与 Rust `distributed_protocol.rs` 的 `read_handshake_quic` / `send_command_quic` / `recv_response_quic` 完全互通
  - `python/test_python_worker_rust_coord.py`: mock worker 端到端测试
    - 启动 Rust coordinator（`--distributed-role coordinator`）+ Python mock worker
    - 完整控制面流程验证：Handshake → Prefill → SyncGlobalSeqLen → 3×Decode → Shutdown ✅
  - `python/hcp_vllm_quic_worker.py`: vLLM backend QUIC worker（ready for remote GPU deployment）
    - 命令行参数：`--model-dir`, `--coordinator-host`, `--coordinator-port`, `--domain-id`
    - 复用 `VllmBackend` prefill/decode，通过 `QuicControlClient` 收发 bincode 命令

- [x] **Phase 3.2: Python↔Python QUIC KV ring 交换验证**（commit `24d1fb8`）
  - `python/hcp_worker_sdk/quic_server.py`: `QuicWorkerServer` 同时支持 QUIC 控制面（coordinator）和 QUIC 数据面（peer KV ring）
  - `test_quic_kv_ring.py`: 基础 Python↔Python KV block roundtrip ✅
  - `test_quic_kv_ring_concurrent.py`: 并发 send+recv（双方同时 exchange_kv_block），无死锁 ✅
  - `test_two_python_workers_quic.py`: 2 个 Python mock worker + Rust coordinator，完整 Handshake→Prefill→KV exchange→SyncGlobalSeqLen→Decode→Shutdown ✅
  - 关键修复：aioquic server 只在收到 STREAM 数据帧时才调用 `stream_handler`，client 创建 stream 后需立即写 1-byte dummy 触发 server 回调
  - `QuicKvTransport.exchange_kv_block` 改用并发 send+recv（`asyncio.gather`），避免经典 ring 死锁

- [x] **Phase 3.3: 跨机器异构端到端验证通过**（commit `6b6265f`）
  - `python/hcp_transformers_quic_worker.py`: Mac 端 TransformersBackend worker，完整 QuicWorkerServer 接入
  - `python/test_transformers_2domain_quic.py`: 本地同机 2-domain TransformersBackend 分布式验证通过 ✅（`generated: . I am`）
  - `scripts/run_python_distributed_2node.sh`: 自动化跨机器启动脚本
    - Mac: coordinator (0.0.0.0:26001) + worker 0 (TransformersBackend, domain 0)
    - Remote GPU: worker 1 (VllmBackend, domain 1) via SSH
    - 远程通过 `HTTP_PROXY=http://127.0.0.1:7890` 下载 HuggingFace 模型
  - **验证结果**：Mac CPU (worker 0, capacity=4096 MB) + RTX 4090 CUDA (worker 1, capacity=14467 MB)
  - Coordinator `generated: . I am`，vLLM decode ~155 it/s
  - 控制面 QUIC+bincode，数据面 QUIC KV ring，全部正常

- [ ] **Phase 3.4: vLLM 真实 KV 提取**
  - vLLM 0.6.4 `LLM` API 不暴露 KV cache，需接入 `LLMEngine` 底层 API
  - PagedAttention KV 格式转换需避免性能损失

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
- Remote GPU 纪律：`sd-1`（SSH 主机名，用户 `user`）只通过 git 同步代码；不要在远端直接编辑源码。
- Remote P2P 纪律：双机验证不能用 `127.0.0.1` 作为结论；server 应监听 `0.0.0.0` 或目标子网地址，client 应连接 `100.64.0.x` 子网内的 GPU host。
- Report 纪律：`reports/**/*.json` 是生成产物，默认不提交；如需长期记录实验进展，写入 docs 或 memory-bank。
- Remote CP node 启动纪律：正式 3-node remote CP smoke 优先使用 `scripts/run_rust_remote_cp_3node_smoke.sh`，让 Mac 地址发现、GPU git 同步、preflight build、节点启动和日志路径统一收敛；手工启动只用于排查。
- Remote CP 地址纪律：Mac 的可达地址可能在 `192.168.8.x` 子网或 `100.x` VPN/overlay 网络之间切换；统一 launcher 使用 `MAC_NODE_ADDR` 覆盖本机可达地址，`GPU_HOST` 覆盖 CUDA host。
- CP block update 纪律：`torch_query_chunk_status=pass` 证明 CP 消息 K/V payload 和 Rust/domain-side Q payload 已驱动设备侧 Q chunk online softmax output smoke；当前 Q/K/V 已来自 hidden states + projection weights，但在权重加载、RoPE、norm/residual 和完整 layer lifecycle 接入前，不应把它描述为完整 Transformer layer。

## 当前阻塞

- [2026-05-07] **Phase 3.1 + 3.2 + 3.3 全部完成**。Python bincode + QUIC 控制面 + Python↔Python QUIC KV ring 交换 + 跨机器异构端到端全部验证通过。
- [2026-05-07] **跨机器异构端到端通过**（commit `6b6265f`）：Mac (TransformersBackend/CPU) + Remote RTX 4090 (VllmBackend/CUDA)，Tailscale VPN，控制面 QUIC+bincode，数据面 QUIC KV ring。Coordinator `generated: . I am`。vLLM decode ~155 it/s。
- [2026-05-02] **单卡多 worker 显存上限不变**：权重共享只能省一份权重 (~4.6GB)，但 LM head output 和 KV cache 仍按 domain 数倍增。2 domain × 8K 同卡 ≈ 2×(4.64GB LM head + KV) + 4.6GB 权重 ≈ 14GB+，仍可能 OOM。大 seq 必须多卡分布。
- [2026-04-24] 当前沙箱环境不允许 Python worker 绑定本地端口，完整 Python smoke 会触发 `PermissionError: [Errno 1] Operation not permitted`。
- [2026-04-24] `ringattn_controller.py` 当前会将 `bytes` 放入 `json.dumps` payload，导致 `TypeError: Object of type bytes is not JSON serializable`。
- [2026-04-30] 环境默认在线（rsproxy-sparse 可达），`cargo test` / `cargo clippy` / `cargo build` 均在线执行。
- [2026-04-24] PyTorch 2.11.0 在默认沙箱进程中 `mps_available=false`，原因是沙箱内 `MTLCopyAllDevices()` 返回 0；非沙箱进程可枚举 `Apple M1 Pro`，且 `torch.ones(..., device="mps")` 成功。
- [2026-04-24] 后续所有 Metal/MPS 相关验证必须在非沙箱/授权进程中运行；默认沙箱结果不能作为 MPS 不可用结论。
- [2026-04-25] GPU 端 `CARGO_OFFLINE=1` 失败且提示 `no matching package named serde_json found` 时，优先判定为 Cargo registry cache miss，不是 CUDA/libtorch 问题。
- [2026-05-01] Qwen2-0.5B 推理输出已与 Python transformers 完全一致（greedy decode `" I am a 20 year old girl from"`）；temperature/top-p 采样已接入（`--infer-top-p` CLI），可作为后续优化方向。
- [2026-05-01] **CUDA 推理验证通过**：远端 GPU 节点 (`sd-1`) 使用 `cuda:1` 运行 greedy decode，输出与本地 CPU/MPS 完全一致（`" I am a 20 year old girl from"` / `" The capital of France is Paris."`）；`HCP_TORCH_DEVICE=cuda:1` 环境变量被正确解析。GPU 0 存在显存残留（15.3GB/16GB 被占，无活跃进程），推理需使用 `cuda:1/2/3`。
- [2026-04-30] **修复 GQA `repeat_kv` 关键 bug**：Rust 原实现使用 `x.repeat([1, n_rep, 1, 1])`，将整组 KV head 循环重复（如 [A,B,A,B,A,B]），而 Python transformers 的 `repeat_kv` 是每个 head 连续重复 n_rep 次（如 [A,A,A,B,B,B]）。这导致 query head 与 key/value head 的对应关系完全错乱，是推理 logits 与 Python 完全不同的根因。修复后改用 `unsqueeze(2).expand(...).reshape(...)`，与 Python 语义一致。修复后 Qwen2-0.5B greedy decode 输出与 Python transformers 完全一致（`" I am a 20 year old girl from"`），CPU 和 MPS 均验证通过；19/19 单元测试通过，clippy 零警告。
