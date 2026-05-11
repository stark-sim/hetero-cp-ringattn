# HCP Ring Attention 手动体验指南

> 目标：从零开始，亲手启动每一个进程，观察每一步日志，理解分布式推理的设计内涵。

---

## 前置准备

### 1. 环境变量（每次新开 terminal 都要设）

```bash
# Mac 本地
export LIBTORCH=/Users/stark_sim/libtorch
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH

# 验证 libtorch 可用
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH ./target/release/hcp-ringattn-rust --distributed-role coordinator --listen-addr 127.0.0.1:9999
# 应该看到 "[coordinator] starting..."，而不是 dyld 错误
# Ctrl+C 退出
```

### 2. 确认模型路径

```bash
ls /Users/stark_sim/models/qwen2-0.5b/config.json
ls /Users/stark_sim/models/qwen2-0.5b/tokenizer.json
ls /Users/stark_sim/models/qwen2-0.5b/model.safetensors
```

### 3. 准备测试 prompts

```bash
cat > /tmp/test_prompts.txt << 'EOF'
The answer to life
Once upon a time
EOF
```

---

## Stage 0：代码结构速查（理解再动手）

重构后的代码按"操作对象"分组。启动进程之前，先花 5 分钟了解每个模块管什么：

| 你想理解什么 | 文件 | 核心职责 |
|-------------|------|----------|
| **Coordinator 怎么调度 prompt 分片** | `src/distributed/coordinator.rs` | 接受 worker 连接 → tokenize → 三 tier 分片 → 广播 Prefill/Decode → 采样 token |
| **Worker 怎么启动、怎么连 peer** | `src/distributed/worker.rs` | 解析参数 → 加载权重（一次）→ 每 domain 一个线程 → 运行 WorkerRuntime |
| **Worker 事件循环长什么样** | `src/worker_sdk/runtime.rs` | QUIC 网络初始化 → per-layer stream → command loop → graceful exit |
| **Ring Attention 算法本身** | `src/model/attention/ring.rs` | online softmax、KV block 交换、causal mask、prefill/decode 路径 |
| **KV 怎么跨省界发出去** | `src/distributed/transport/quic.rs` | QUIC endpoint、1GB window、并发 send+recv（防死锁）、dummy handshake |
| **单节点参考输出怎么算** | `src/infer.rs` | LlamaModel 加载 → greedy/temperature 采样 → 自回归生成 |
| **Smoke 测试框架** | `src/smoke/` | 参考 attention 算法、correctness case、C++/tch bridge 验证 |

---

## Stage 1：单节点推理（理解 baseline）

> **设计内涵**：先确认单节点能跑通，掌握参考输出。这是后续分布式对比的基准。单节点必须覆盖 **CPU / MPS / CUDA** 三种设备，确保模型本身无 device-specific bug。

### 1a. CPU 单节点（最基础的正确性验证）

```bash
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH \
  HCP_TCH_DEVICE=cpu \
  ./target/release/hcp-ringattn-rust \
    --infer-model-dir /Users/stark_sim/models/qwen2-0.5b \
    --infer-prompt "The answer to life" \
    --infer-max-tokens 3 \
    --infer-temperature 0.0 \
    --infer-num-domains 1
```

**预期输出**：
```
[infer] device: Cpu
[infer] loading weights from /Users/stark_sim/models/qwen2-0.5b
[infer] building model (24 layers, 14 heads)
[infer] generating (max_tokens=3, temperature=0)...
 is not a
```

**观察重点**：
- `device: Cpu` — 确认 fallback 路径正常
- 记录输出：`The answer to life` → ` is not a`（这是后续所有 distributed 验证的**黄金参考**）

### 1b. MPS 单节点（Mac GPU 验证）

```bash
DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH \
  HCP_TCH_DEVICE=mps \
  ./target/release/hcp-ringattn-rust \
    --infer-model-dir /Users/stark_sim/models/qwen2-0.5b \
    --infer-prompt "The answer to life" \
    --infer-max-tokens 3 \
    --infer-temperature 0.0 \
    --infer-num-domains 1
```

**预期输出**：与 CPU 完全一致 ` is not a`

**观察重点**：
- `device: Mps` — 确认 Metal GPU 被选中
- 输出必须与 CPU 参考一致，否则说明 MPS backend 有数值问题

### 1c. CUDA 单节点（远程 GPU 验证）

```bash
ssh stark@100.64.0.2

cd ~/hetero-cp-ringattn/rust
export PATH=/home/stark/.cargo/bin:$PATH
export LIBTORCH=/home/stark/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export HCP_TCH_DEVICE=cuda

./target/release/hcp-ringattn-rust \
  --infer-model-dir ~/hetero-cp-ringattn/models/Qwen2-0.5B \
  --infer-prompt "The answer to life" \
  --infer-max-tokens 3 \
  --infer-temperature 0.0 \
  --infer-num-domains 1
```

**预期输出**：与 CPU/MPS 完全一致 ` is not a`

**观察重点**：
- `device: Cuda(0)` — 确认 RTX 4090 被选中
- 输出必须与 CPU 参考一致，否则说明 CUDA backend 有数值问题

### 1d. 单节点 device 一致性验证

| Device | 预期输出 | 验证标准 |
|--------|----------|----------|
| CPU | ` is not a` | 基准参考 |
| MPS | ` is not a` | 与 CPU 逐 token 一致 |
| CUDA | ` is not a` | 与 CPU 逐 token 一致 |

如果三个 device 输出不一致，说明模型实现存在 device-specific bug（例如 MPS `masked_fill` 行为差异、CUDA dtype 问题），必须先修复再进入分布式验证。

---

## Stage 2：本地双节点（理解分布式分片）

> **设计内涵**：同一台机器上启动 3 个进程（1 coordinator + 2 workers），观察 prompt 如何被切成两块、各自计算、通过本地网络交换 KV。

### 打开 3 个 terminal

#### Terminal A：Coordinator

```bash
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
export DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:$DYLD_LIBRARY_PATH

./target/release/hcp-ringattn-rust --distributed-role coordinator \
  --model-dir /Users/stark_sim/models/qwen2-0.5b \
  --prompts-file /tmp/test_prompts.txt \
  --max-tokens 3 \
  --temperature 0.0 \
  --num-domains 2 \
  --listen-addr 0.0.0.0:9500
```

**观察重点**：
- `worker 0 connected (accept order 0), capacity=8192 MB`
- `worker 1 connected (accept order 1), capacity=...`
- `sent Prefill chunk [0, 2) to worker 0` — 前 2 个 token 给 worker 0
- `sent Prefill chunk [2, 4) to worker 1` — 后 2 个 token 给 worker 1
- `generated:  is not a` — 与单节点输出一致

#### Terminal B：Worker 0（domain 0）

等 Coordinator 启动后再执行：

```bash
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
export DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:$DYLD_LIBRARY_PATH
export HCP_TCH_DEVICE=mps

./target/release/hcp-ringattn-rust --distributed-role worker \
  --domain-id 0 \
  --model-dir /Users/stark_sim/models/qwen2-0.5b \
  --listen-addr 0.0.0.0:9510 \
  --next-peer-addr 127.0.0.1:9511 \
  --coordinator-addr 127.0.0.1:9500 \
  --num-domains 2
```

**观察重点**：
- `QUIC connection to next peer established` — 连上了 worker 1
- `QUIC connection to coordinator established`
- `Prefill { request_id: 1, chunk: [12522, 5193], seq_offset: 0 }` — 收到 coordinator 发来的 chunk
- `[ring_attention] round 0 layer 2: sent KV block 14336 bytes, received 14336 bytes` — **KV 交换的直接证据**
- `Decode { request_id: 1, token: 374 }` — 收到 coordinator 广播的 decode token
- `coordinator connection closed, exiting gracefully` — 优雅退出

#### Terminal C：Worker 1（domain 1）

等 Worker 0 启动后再执行：

```bash
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
export DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:$DYLD_LIBRARY_PATH
export HCP_TCH_DEVICE=mps

./target/release/hcp-ringattn-rust --distributed-role worker \
  --domain-id 1 \
  --model-dir /Users/stark_sim/models/qwen2-0.5b \
  --listen-addr 0.0.0.0:9511 \
  --next-peer-addr 127.0.0.1:9510 \
  --coordinator-addr 127.0.0.1:9500 \
  --num-domains 2
```

**观察重点**：
- `QUIC connection from prev peer established` — worker 0 主动连过来了
- `Prefill { request_id: 1, chunk: [311, 2272], seq_offset: 2 }` — seq_offset=2，说明是后半段
- 同样有 `sent KV block / received KV block` 日志

### 分片验证

对比两个 worker 的 Prefill chunk：
- Worker 0: `[12522, 5193]`（offset=0）
- Worker 1: `[311, 2272]`（offset=2）

这就是 prompt `The answer to life` 被 tokenizer 切成的 4 个 token，前两个给 worker 0，后两个给 worker 1。

### 结果验证

Coordinator 输出的 `generated:  is not a` 应与 Stage 1 单节点完全一致。

---

## Stage 3：跨节点异构（真实分布式）

> **设计内涵**：Mac MPS 与远程 RTX 4090 CUDA 协同工作，通过 Tailscale VPN 交换 KV。这是 HCP 的核心价值场景——异构算力池化。

### 拓扑

```
Coordinator (Mac, 100.64.0.95:9500)
    ├─ Worker 0 (Mac MPS, 100.64.0.95:9510) ──QUIC KV ring──┐
    └─ Worker 1 (white CUDA, 100.64.0.2:9511) ───────────────┘
```

### 前提

- Mac 和 white 都能访问 GitHub（`git pull` 同步代码）
- 两边都已编译好 release binary
- white 上已设好环境：`export LIBTORCH=/home/stark/libtorch`

### 打开 3 个 terminal

#### Terminal A：Coordinator（Mac）

```bash
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
export DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:$DYLD_LIBRARY_PATH

./target/release/hcp-ringattn-rust --distributed-role coordinator \
  --model-dir /Users/stark_sim/models/qwen2-0.5b \
  --prompts-file /tmp/test_prompts.txt \
  --max-tokens 3 \
  --temperature 0.0 \
  --num-domains 2 \
  --listen-addr 0.0.0.0:9500
```

#### Terminal B：Worker 0（Mac MPS）

等 Coordinator 启动后：

```bash
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
export DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:$DYLD_LIBRARY_PATH
export HCP_TCH_DEVICE=mps

./target/release/hcp-ringattn-rust --distributed-role worker \
  --domain-id 0 \
  --model-dir /Users/stark_sim/models/qwen2-0.5b \
  --listen-addr 0.0.0.0:9510 \
  --next-peer-addr 100.64.0.2:9511 \
  --coordinator-addr 100.64.0.95:9500 \
  --num-domains 2
```

**注意**：`--next-peer-addr` 指向 white，`--coordinator-addr` 指向本机 Mac 的 Tailscale IP。

#### Terminal C：Worker 1（white CUDA，SSH 过去）

等 Worker 0 启动后：

```bash
ssh stark@100.64.0.2

cd ~/hetero-cp-ringattn/rust
export PATH=/home/stark/.cargo/bin:$PATH
export LIBTORCH=/home/stark/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export HCP_TCH_DEVICE=cuda

./target/release/hcp-ringattn-rust --distributed-role worker \
  --domain-id 1 \
  --model-dir ~/hetero-cp-ringattn/models/Qwen2-0.5B \
  --listen-addr 0.0.0.0:9511 \
  --next-peer-addr 100.64.0.95:9510 \
  --coordinator-addr 100.64.0.95:9500 \
  --num-domains 2
```

### 跨节点特有的观察重点

1. **Peer 连接方向**：
   - Worker 0 日志：`QUIC connection to next peer established`（主动连 white）
   - Worker 1 日志：`QUIC connection from prev peer established`（被动接受 Mac 连接）

2. **Capacity 差异**：
   - Worker 0: `capacity=8192 MB`（Mac MPS 可用内存）
   - Worker 1: `capacity=20788 MB`（RTX 4090 显存）

3. **KV 交换跨省界**：
   - 两层都有 `sent KV block 14336 bytes, received 14336 bytes`
   - 这 14KB 数据真的通过 Tailscale VPN 从 Mac 发到 white、再从 white 发回 Mac

4. **输出一致性**：
   - Coordinator 输出应与 Stage 1/2 完全一致：` is not a` / `, there was`

---

## Stage 4：代码阅读路线（按业务逻辑）

手动跑通之后，按以下顺序读代码，理解每个模块的业务逻辑：

### 4a. Coordinator 调度流程

```
src/distributed/coordinator.rs
  → run() [入口]
    → 创建 QUIC endpoint，监听 worker 连接
    → accept_workers() [按 domain_id 排序]
    → tokenize prompt
    → allocate_chunks() [三 tier：chunk-sizes > capacity-aware > 均分]
    → 循环每个 request：
      → 发送 Prefill 到每个 worker
      → 收集 PrefillDone，max(global_seq_len)
      → 广播 SyncGlobalSeqLen
      → 采样第一个 token
      → decode 循环：广播 Decode → 收集 logits → 采样 → 广播下一个 token
    → 全部完成后 Shutdown
```

### 4b. Worker 生命周期

```
src/distributed/worker.rs
  → run() [入口]
    → select_device() [env > MPS > CUDA > CPU]
    → 加载 ModelConfig + ModelWeights（一次，shallow_clone 共享）
    → 每 domain 一个线程：
      → TchWorkerBackend::load() / from_model()
      → WorkerRuntime::new() [网络初始化]
      → runtime.run() [事件循环]
    → ResetBarrier::wait() [多 domain 同步]

src/worker_sdk/runtime.rs
  → new() [网络初始化]
    → create_endpoint()
    → dial next peer + accept prev peer
    → open per-layer bidirectional streams
    → write 1-byte dummy handshake
    → connect to coordinator
    → backend.setup_kv_transports()
    → send WorkerHandshake
  → run() [事件循环]
    → loop:
      → recv_command_quic()
      → dispatch: Prefill → backend.prefill() → PrefillDone
      → dispatch: Decode → backend.decode() → DecodeDone
      → dispatch: Shutdown → break
    → graceful exit on connection lost
```

### 4c. Ring Attention 核心算法

```
src/model/attention/ring.rs
  → HcpRingAttentionBackend::forward()
    → Q/K/V projection + RoPE
    → 如果是分布式（num_domains > 1）：
      → ring_attention()
        → 构建本地 KvBlock
        → for round in 0..num_domains-1:
          → transport.exchange_kv_block() [并发 send+recv]
          → 收到 peer block 后 online_update_block()
            → online softmax: max, sum_exp, output
        → 最终输出 = output / sum_exp
    → 如果是单节点：local_attention_scores()
    → O-projection + residual
```

### 4d. QUIC 传输层

```
src/distributed/transport/quic.rs
  → create_endpoint()
    → 自签名证书 + SkipServerVerification
    → 1GB send/receive window（防 KV block 死锁）
    → 1s keep-alive（NAT/防火墙存活）
    → 1200 MTU（Tailscale/WireGuard 兼容）
  → QuicKvTransport::exchange_kv_block()
    → tokio::join!(send_kv_block, recv_kv_block) [并发，防死锁]
    → 120s timeout
    → 帧格式：4-byte BE length + JSON metadata + raw f32 bytes
```

---

## 设计内涵速查表

| 你看到什么 | 它意味着什么 |
|-----------|------------|
| `sent Prefill chunk [0, 2)` | Coordinator 把 prompt 切成多块，每块发给一个 worker |
| `seq_offset: 0` vs `seq_offset: 2` | Worker 0 处理前 2 个 token，Worker 1 处理后 2 个 token |
| `SyncGlobalSeqLen { len: 4 }` | 所有 worker 必须知道全局长度，否则 causal mask 会错 |
| `sent KV block 14336 bytes` | Worker 把自己的 KV cache 发送给 peer |
| `received KV block 14336 bytes` | Worker 收到 peer 的 KV cache，用于计算 attention |
| `Decode { request_id: 1, token: 374 }` | Coordinator 采样的 token 广播给所有 worker |
| `coordinator connection closed, exiting gracefully` | 没有 panic，没有资源泄漏 |

---

## 故障排查

| 现象 | 原因 | 修复 |
|------|------|------|
| `dyld: Library not loaded: libtorch_cpu.dylib` | `DYLD_LIBRARY_PATH` 没设 | 每次启动进程前 export |
| `worker connect coordinator timed out` | Coordinator 还没启动或已崩溃 | 确认 Coordinator 日志正常 |
| `QUIC connection to next peer: timed out` | 对端 worker 还没启动 | 按顺序启动：Coordinator → Worker 1 → Worker 0 |
| 输出与单节点不一致 | KV 没交换成功 | 检查 `[ring_attention]` 日志是否存在 |
| Worker 1 启动报 `cargo: command not found` | white 上 PATH 没加载 cargo | 加 `export PATH=/home/stark/.cargo/bin:$PATH` |
| MPS 输出与 CPU 不一致 | MPS backend 数值差异 | 检查 `ring_attention` 中 `masked_fill`/`where_self` 的 MPS workaround |
| CUDA 输出与 CPU 不一致 | CUDA dtype/accumulation 差异 | 检查 f32 vs f16、cuBLAS 执行失败、position_ids 越界 |

---

## 进阶：不均等分片

如果想观察 capacity-aware 分片（Mac 少分点、CUDA 多分点）：

```bash
# Coordinator 加 --chunk-sizes
./target/release/hcp-ringattn-rust --distributed-role coordinator \
  ... \
  --chunk-sizes 1,3   # worker 0 处理 1 个 token，worker 1 处理 3 个 token
```

观察两个 worker 的 Prefill chunk 长度差异。
