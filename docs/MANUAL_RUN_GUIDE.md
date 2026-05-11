# HCP Ring Attention 手动体验指南

> 目标：从零开始，亲手启动每一个进程，观察每一步日志，理解分布式推理的设计内涵。

---

## 前置准备

### 1. 环境变量（每次新开 terminal 都要设）

```bash
# Mac 本地
export LIBTORCH=/Users/stark_sim/libtorch
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH
export HCP_TCH_DEVICE=mps   # 或 cpu，如果用 cpu 就把 mps 改成 cpu

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

## Stage 1：单节点推理（理解 baseline）

> **设计内涵**：先确认单节点能跑通，掌握参考输出。这是后续分布式对比的基准。

### 命令

```bash
cd ~/VSCodeProjects/hetero-cp-ringattn/rust
DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH \
  HCP_TCH_DEVICE=mps \
  ./target/release/hcp-ringattn-rust \
    --infer-model-dir /Users/stark_sim/models/qwen2-0.5b \
    --infer-prompt "The answer to life" \
    --infer-max-tokens 3 \
    --infer-temperature 0.0 \
    --infer-num-domains 1
```

### 预期输出

```
[infer] device: Mps
[infer] loading weights from /Users/stark_sim/models/qwen2-0.5b
[infer] building model (24 layers, 14 heads)
[infer] generating (max_tokens=3, temperature=0)...
 is not a
```

### 观察重点

1. `device: Mps` — 确认确实跑在 GPU 上，不是 fallback 到 CPU
2. `24 layers, 14 heads` — Qwen2-0.5B 的模型结构
3. `--infer-num-domains 1` — 单节点不分片，全部计算在一台机器上完成
4. 记录输出：`The answer to life` → ` is not a`

### 再跑第二个 prompt

```bash
DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH \
  HCP_TCH_DEVICE=mps \
  ./target/release/hcp-ringattn-rust \
    --infer-model-dir /Users/stark_sim/models/qwen2-0.5b \
    --infer-prompt "Once upon a time" \
    --infer-max-tokens 3 \
    --infer-temperature 0.0 \
    --infer-num-domains 1
```

预期输出：`, there was`

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
