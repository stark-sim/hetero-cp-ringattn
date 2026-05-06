# HCP Ring Attention 手动部署指南

> 目标：让你从零开始，亲手把 HCP 推理系统跑起来，体验跨节点异构推理的完整工程链路。

---

## 1. 架构概览

HCP 推理系统由两类进程组成：

| 进程 | 职责 | 是否加载模型权重 | 典型部署位置 |
|------|------|-----------------|-------------|
| **Coordinator** | Tokenizer 分词、Prompt 分片、Token 广播、采样 | ❌ 只加载 tokenizer + config | 任意机器（建议 Mac） |
| **Worker** | 模型 Forward、KV Ring 交换 | ✅ 加载完整 safetensors 权重 | 每个 domain 一台机器 |

**关键原则**：Coordinator 不做任何模型计算。每个参与 heterogeneous 协作的平台必须至少运行一个 Worker。

**网络拓扑**（以 2-domain 为例）：

```
┌─────────────────┐                    ┌─────────────────┐
│   Mac (MPS)     │ ←── QUIC KV ring ─→│  RTX 4090 (CUDA)│
│  Coordinator    │                    │    Worker 1     │
│  + Worker 0     │ ←── QUIC control ─→│                 │
└─────────────────┘                    └─────────────────┘
        ↑                                      ↑
   domain_id=0                           domain_id=1
   listen: 0.0.0.0:29450                 listen: 0.0.0.0:29451
   next_peer: 192.168.x.x:29451          next_peer: 192.168.x.x:29450
```

---

## 2. 前置条件

### 2.1 所有节点通用

- **Rust** 1.75+ (`rustc --version`)
- **CMake** 3.16+ (`cmake --version`)
- 模型目录（HuggingFace 格式）：`config.json` + `model.safetensors` + `tokenizer.json`
  - 推荐：Qwen2-0.5B（小、快、已验证）
  - 下载：`huggingface-cli download Qwen/Qwen2-0.5B --local-dir ./models/Qwen2-0.5B`
- 仓库代码：`git clone <repo-url> && cd hetero-cp-ringattn`

### 2.2 Mac 节点（Coordinator + Worker 0）

- **macOS** 12.3+（MPS 需要）
- **libtorch** 2.11.0 for macOS（CPU+MPS）
  ```bash
  # 假设 libtorch 解压到 /Users/<you>/libtorch
  export LIBTORCH=/Users/<you>/libtorch
  export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"
  ```

### 2.3 GPU 节点（Worker 1）

- **Linux** with NVIDIA GPU + CUDA 12.x
- **libtorch** 2.11.0 for Linux with CUDA
  ```bash
  export LIBTORCH=/home/<you>/libtorch
  export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
  ```

### 2.4 网络

- 节点间互相可达（建议 Tailscale VPN 或同一局域网）
- 确认连通性：
  ```bash
  # Mac 上
  ping <GPU_NODE_IP>
  # GPU 上
  ping <MAC_NODE_IP>
  ```

---

## 3. 单节点本地部署（验证最小闭环）

在任意一台机器上先验证单节点推理能力，排除模型权重、libtorch、tokenizer 问题。

```bash
cd hetero-cp-ringattn/rust

# 单节点 MPS（Mac）
HCP_TCH_DEVICE=mps \
LIBTORCH=/Users/<you>/libtorch \
DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH" \
cargo run --features tch-backend --bin hcp-ringattn-rust -- \
  --infer-model-dir /path/to/models/Qwen2-0.5B \
  --infer-prompt "The quick brown fox jumps over the lazy dog" \
  --infer-max-tokens 20 \
  --infer-num-domains 1

# 单节点 CUDA（GPU）
HCP_TCH_DEVICE=cuda:0 \
LIBTORCH=/home/<you>/libtorch \
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" \
cargo run --features tch-backend --bin hcp-ringattn-rust -- \
  --infer-model-dir /path/to/models/Qwen2-0.5B \
  --infer-prompt "The quick brown fox jumps over the lazy dog" \
  --infer-max-tokens 20 \
  --infer-num-domains 1
```

预期输出：
```
[infer] device: Mps   (或 Cuda(0))
[infer] loading config from ...
[infer] generating (max_tokens=20, temperature=0.7)...
[infer] generated: <一段连贯的英文>
```

---

## 4. 双节点异构部署（Mac MPS + GPU CUDA）

### 4.1 步骤总览

1. **Mac 启动 Coordinator**（监听 worker 连接）
2. **Mac 启动 Worker 0**（MPS 设备，连接 coordinator）
3. **GPU 启动 Worker 1**（CUDA 设备，连接 coordinator）
4. Coordinator 收到全部 worker 后，自动开始 prefill → decode → 输出

### 4.2 地址规划

| 角色 | 地址 | 说明 |
|------|------|------|
| Coordinator listen | `0.0.0.0:29450` | 接受所有 worker 连接 |
| Worker 0 listen | `0.0.0.0:29451` | Mac 接受 peer KV |
| Worker 0 next_peer | `<GPU_IP>:29452` | Worker 0 发送 KV 到 GPU |
| Worker 1 listen | `0.0.0.0:29452` | GPU 接受 peer KV |
| Worker 1 next_peer | `<MAC_IP>:29451` | Worker 1 发送 KV 到 Mac |
| Worker 0 coordinator | `127.0.0.1:29450` | Worker 0 连本机 coordinator |
| Worker 1 coordinator | `<MAC_IP>:29450` | Worker 1 连 Mac coordinator |

### 4.3 Mac 端：启动 Coordinator

```bash
cd hetero-cp-ringattn/rust

HCP_TCH_DEVICE=mps \
LIBTORCH=/Users/<you>/libtorch \
DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH" \
cargo run --features tch-backend --bin hcp-ringattn-rust -- \
  --distributed-role coordinator \
  --model-dir /path/to/models/Qwen2-0.5B \
  --prompt "The quick brown fox jumps over the lazy dog" \
  --max-tokens 50 \
  --num-domains 2 \
  --listen-addr 0.0.0.0:29450 \
  --worker-addrs "" \
  --temperature 0.7
```

**注意**：`--worker-addrs ""` 表示 coordinator 只 listen、不主动 dial。Worker 会主动连上来。

你会看到：
```
[coordinator] starting, num_domains=2, workers=[], listen=0.0.0.0:29450
[coordinator] prompt tokens: 9
[coordinator] QUIC endpoint listening on 0.0.0.0:29450
[coordinator] waiting for workers...
```

### 4.4 Mac 端：启动 Worker 0（MPS）

另开终端：

```bash
cd hetero-cp-ringattn/rust

HCP_TCH_DEVICE=mps \
LIBTORCH=/Users/<you>/libtorch \
DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH" \
cargo run --features tch-backend --bin hcp-ringattn-rust -- \
  --distributed-role worker \
  --domain-id 0 \
  --model-dir /path/to/models/Qwen2-0.5B \
  --listen-addr 0.0.0.0:29451 \
  --next-peer-addr <GPU_IP>:29452 \
  --coordinator-addr 127.0.0.1:29450 \
  --num-domains 2
```

预期输出：
```
[worker 0] starting, listen=0.0.0.0:29451, next_peer=<GPU_IP>:29452, coordinator=127.0.0.1:29450
[worker 0] device: Mps
[worker 0] loaded model weights once for 1 domain(s)
[worker 0] QUIC endpoint bound to 0.0.0.0:29451
[worker 0] QUIC connection to next peer established
[worker 0] distributed domain setup complete
[worker 0] handshake sent to coordinator
[worker 0] waiting for command...
```

### 4.5 GPU 端：启动 Worker 1（CUDA）

SSH 到 GPU 节点：

```bash
cd hetero-cp-ringattn/rust

PATH=/home/<you>/.cargo/bin:$PATH \
HCP_TCH_DEVICE=cuda:0 \
LIBTORCH=/home/<you>/libtorch \
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" \
cargo run --features tch-backend --bin hcp-ringattn-rust -- \
  --distributed-role worker \
  --domain-id 1 \
  --model-dir /path/to/models/Qwen2-0.5B \
  --listen-addr 0.0.0.0:29452 \
  --next-peer-addr <MAC_IP>:29451 \
  --coordinator-addr <MAC_IP>:29450 \
  --num-domains 2
```

预期输出：
```
[worker 1] starting, listen=0.0.0.0:29452, next_peer=<MAC_IP>:29451, coordinator=<MAC_IP>:29450
[worker 1] device: Cuda(0)
[worker 1] loaded model weights once for 1 domain(s)
[worker 1] QUIC connection from prev peer established
[worker 1] distributed domain setup complete
[worker 1] handshake sent to coordinator
[worker 1] waiting for command...
```

### 4.6 观察 Coordinator 输出

当两个 worker 都连上后，coordinator 自动推进：

```
[coordinator] worker 0 connected, capacity=16384 MB
[coordinator] worker 1 connected, capacity=24576 MB
[coordinator] sent Prefill chunk [0, 5) to worker 0
[coordinator] sent Prefill chunk [5, 9) to worker 1
[coordinator] worker 0 prefill done, global_seq_len=9
[coordinator] worker 1 prefill done, global_seq_len=9
[coordinator] max_global_seq_len = 9
[coordinator] generated: <模型生成的文本>
```

---

## 5. 使用统一启动脚本（推荐）

手动三窗口启动容易出错（时序、地址、环境变量）。使用统一脚本：

```bash
# Mac 端（会自动 SSH 到 GPU 节点同步代码并启动）
GPU_HOST=192.168.8.172 \
GPU_USER=stark \
MAC_NODE_ADDR=192.168.8.xxx \
RUN_ID=my-first-hetero-smoke \
PORT_BASE=29450 \
bash scripts/run_rust_remote_cp_3node_smoke.sh
```

脚本会自动：
1. 发现 Mac 的局域网地址
2. SSH 到 GPU 节点执行 `git pull --ff-only` + cargo preflight build
3. 在 GPU 节点启动 worker 1
4. 在 Mac 启动 coordinator + worker 0
5. 收集所有日志到 `reports/my-first-hetero-smoke/`

---

## 6. 高级配置

### 6.1 手动不均等分片

默认 coordinator 将 prompt 均分给所有 domain。若显存不均，可手动指定：

```bash
# Coordinator 参数追加：
--chunk-sizes 400,151  # domain0 处理 400 tokens，domain1 处理 151 tokens
```

约束：`len(chunk-sizes) == num-domains` 且 `sum(chunk-sizes) == prompt_tokens`

### 6.2 Capacity-Aware 自动分片

让 worker 上报可用显存，coordinator 按比例自动分配：

```bash
# Coordinator 参数追加：
--capacity-aware
```

### 6.3 单进程多 Domain（开发测试）

在单张卡上模拟多 domain（显存倍增，仅用于开发）：

```bash
cargo run --features tch-backend --bin hcp-ringattn-rust -- \
  --distributed-role worker \
  --local-domain-ids 0,1 \
  --listen-addrs 0.0.0.0:29451,0.0.0.0:29452 \
  --next-peer-addrs 127.0.0.1:29452,127.0.0.1:29451 \
  --coordinator-addr 127.0.0.1:29450 \
  --num-domains 2 \
  --model-dir /path/to/model
```

权重只加载一次（shallow_clone），各 domain 独立 KV cache。

### 6.4 高延迟网络（Tailscale VPN）调参

跨 VPN 时，QUIC 默认超时可能不够：

```rust
// rust/src/quic_transport.rs 中已配置（无需手动修改）：
// keep_alive_interval: 1s
// max_idle_timeout: 3600s
// stream_receive_window: 512MB
// exchange_kv_block timeout: 120s
```

如遇隧道崩溃，减小 prompt 长度或等待网络恢复后重试。

---

## 7. 故障排查

### 7.1 Coordinator 卡在 "waiting for workers"

- Worker 的 `--coordinator-addr` 是否正确？
- 防火墙是否放行了 coordinator 端口？
- 节点间是否可以互相 ping 通？

### 7.2 Worker 报 "connect to next peer failed"

- Worker 的 `--next-peer-addr` 是否指向了对端的 listen 地址？
- 在 2-domain 场景下：domain 0 先 dial、domain 1 先 accept。确保 domain 0 的 next_peer 是 domain 1 的 listen 地址。

### 7.3 "torch_code=-5"（CUDA 不可用）

- libtorch 是否是 CUDA 版本？检查 `$LIBTORCH/lib/libtorch_cuda.so` 是否存在
- `LD_LIBRARY_PATH` 是否包含 libtorch 的 lib 目录？

### 7.4 MPS 上 tensor 结果错误 / NaN

- 确保在非沙箱环境运行（普通 Terminal，不是某些 IDE 的内置终端）
- `HCP_TCH_DEVICE=mps` 是否设置正确？
- 查看日志中 `device: Mps` 确认设备选择

### 7.5 长序列 OOM

- 减少 `--max-tokens`
- 增加 domain 数量 `--num-domains 4`
- 使用 `--capacity-aware` 让 coordinator 自动分配

---

## 8. 验证清单

完成部署后，对照以下清单确认每个环节正常：

- [ ] 单节点本地推理输出连贯文本
- [ ] Coordinator 成功接受所有 worker 连接
- [ ] 每个 Worker 日志显示正确的设备（Mps / Cuda(0)）
- [ ] KV Ring 交换无死锁（无 "exchange_kv_block timeout"）
- [ ] Prefill 阶段所有 worker 返回 `global_seq_len` 一致
- [ ] Decode 阶段生成 token 直到 EOS 或 max_tokens
- [ ] Coordinator 最终输出完整生成文本
- [ ] 各节点 GPU/CPU 利用率符合预期（分布式下 GPU 利用率会间歇性出现峰值）

---

## 9. 下一步

- 阅读 [`PLUGIN_ARCHITECTURE.md`](PLUGIN_ARCHITECTURE.md) 了解如何将同构 domain 内的 worker 替换为 vLLM 实现
- 阅读 [`docs/DESIGN.md`](DESIGN.md) 理解 Ring Attention 的 online softmax 数学原理
- 阅读 [`docs/VALIDATION_PLAN.md`](VALIDATION_PLAN.md) 了解 correctness 验证策略
