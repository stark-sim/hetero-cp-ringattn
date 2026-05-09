# HCP 分布式推理完整流程 — 从 Prompt 到生成结果

> 本文档基于代码分析和实际运行日志，完整追踪一个 prompt 从输入到推理结果输出的生命周期。覆盖 Coordinator (Rust)、Worker (Python/vLLM)、QUIC 控制面/数据面三个层面。

---

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Coordinator (Rust)                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐  │
│  │Tokenizer │→ │ 分片逻辑  │→ │QUIC Command  │→ │ 采样逻辑  │→ │ Decode  │  │
│  │(tokenize)│  │(chunking)│  │(Prefill/...) │  │(argmax)   │  │ loop    │  │
│  └──────────┘  └──────────┘  └──────────────┘  └──────────┘  └──────────┘  │
│                         ↑                           ↑                       │
│                    WorkerResponse              WorkerResponse               │
└─────────────────────────┼───────────────────────────┼───────────────────────┘
                          │ QUIC 控制面                │ QUIC 控制面
                          │ (bincode over QUIC stream) │
                          ↓                           ↓
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│      Worker 0 (Mac vllm-metal)   │    │      Worker 1 (white vLLM CUDA) │
│  ┌──────────┐  ┌──────────────┐ │    │  ┌──────────┐  ┌──────────────┐ │
│  │ vLLM     │  │ KV Ring      │ │    │  │ vLLM     │  │ KV Ring      │ │
│  │ Prefill  │→ │ Exchange     │ │←──→│  │ Prefill  │→ │ Exchange     │ │
│  │ (MPS)    │  │ (QUIC stream)│ │    │  │ (CUDA)   │  │ (QUIC stream)│ │
│  └──────────┘  └──────────────┘ │    │  └──────────┘  └──────────────┘ │
│  ┌──────────┐                   │    │  ┌──────────┐                   │
│  │ vLLM     │                   │    │  │ vLLM     │                   │
│  │ Decode   │                   │    │  │ Decode   │                   │
│  │ (MPS)    │                   │    │  │ (CUDA)   │                   │
│  └──────────┘                   │    │  └──────────┘                   │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

---

## 2. 阶段一：Coordinator 初始化与 Prompt 处理

### 2.1 输入

```
prompt_text: &str          // 用户输入的原始文本，如 "Hello world"
model_dir: &str            // 模型目录，如 "models/Qwen2-0.5B"
num_domains: usize         // domain 数量，如 2
chunk_sizes: Option<Vec<usize>>  // 可选手动分片，如 [256, 768]
```

### 2.2 处理过程

**Step 1: 加载 tokenizer 和 config**

```rust
// rust/src/distributed_coordinator.rs:96-99
let config_path = Path::new(&args.model_dir).join("config.json");
let config = ModelConfig::from_file(&config_path)?;
let tokenizer_path = Path::new(&args.model_dir).join("tokenizer.json");
let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)?;
```

**Step 2: Tokenize prompt**

```rust
// distributed_coordinator.rs:110-112
let encoding = tokenizer.encode(prompt_text.as_str(), true)?;
let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
// 例: "Hello world" → [9906, 1917] (2 tokens)
```

**Step 3: 创建 QUIC endpoint，等待 workers 连接**

```rust
// distributed_coordinator.rs:118-121
let endpoint = create_endpoint(listen_addr)?;
// 监听 0.0.0.0:26001，等待 worker QUIC 连接
```

**Step 4: 接收 worker handshake**

每个 worker 连接后发送 16-byte handshake：

```
[8-byte LE domain_id (u64)] [8-byte LE capacity_mb (u64)]
```

Coordinator 收集并排序：

```rust
// distributed_coordinator.rs:144-156
// worker 0: domain_id=0, capacity_mb=4096  (Mac MPS)
// worker 1: domain_id=1, capacity_mb=14451 (white RTX 4090)
```

### 2.3 输出

```
prompt_ids: Vec<i64>       // token ID 序列，如 [9906, 1917]
worker_streams: Vec<(SendStream, RecvStream)>  // 与每个 worker 的 QUIC 双向流
chunk_boundaries: Vec<usize>  // 分片边界，如 [0, 1, 2]（均分）或 [0, 256, 1024]（不均分）
```

---

## 3. 阶段二：Prefill 分片与分发

### 3.1 输入

```
prompt_ids: Vec<i64>       // 完整 token 序列
chunk_sizes: Option<Vec<usize>>  // 手动指定或 capacity-aware 自动计算
```

### 3.2 处理过程

**Step 1: 计算分片大小**

三层优先级：

```rust
// distributed_coordinator.rs:169-197
let chunk_sizes = if let Some(ref sizes) = args.chunk_sizes {
    // 1. 手动指定，如 --chunk-sizes 256,768
    sizes.clone()
} else if args.capacity_aware {
    // 2. 按 worker capacity 比例分配
    allocate_by_capacity(seq_len, &worker_capacities)
} else {
    // 3. 默认均分
    seq_len.div_ceil(num_domains).max(1)
};
```

**实际验证案例（2048 tokens，25%/75% 分片）：**

```
--chunk-sizes 512,1536
chunk_boundaries = [0, 512, 2048]
worker 0 (Mac)  处理 chunk [0, 512)   = 512 tokens
worker 1 (CUDA) 处理 chunk [512, 2048) = 1536 tokens
```

**Step 2: 构造并发送 Prefill 命令**

对每个 worker 发送 `WorkerCommand::Prefill`：

```rust
// distributed_coordinator.rs:205-215
let cmd = WorkerCommand::Prefill {
    chunk: chunk.to_vec(),      // 该 worker 的 token 子序列
    seq_offset: start as i64,   // 全局起始位置（用于 RoPE position_ids）
};
send_command_quic(send, &cmd, rt.handle())?;
```

**bincode 帧格式（控制面）：**

```
[4-byte BE length][bincode payload]

Prefill 命令的 bincode 结构：
  u32 tag = 0 (CMD_PREFILL)
  Vec<i64> chunk:
    u64 len
    i64[0], i64[1], ...
  i64 seq_offset
```

### 3.3 输出

```
对每个 worker:
  → WorkerCommand::Prefill { chunk, seq_offset }
```

---

## 4. 阶段三：Worker Prefill 执行

### 4.1 Worker 0 (Mac vllm-metal) 的处理流程

**输入：**

```
cmd["chunk"]: List[int]     // 512 个 token IDs
cmd["seq_offset"]: int      // 0
```

**处理过程：**

**Step 1: 调用 vLLM prefill**

```python
# python/hcp_vllm_worker.py:60-76 (VllmBackend.prefill)
self._history = list(chunk)  # 保存 token 历史
outputs = _vllm_generate(
    self.llm,
    self._history,
    SamplingParams(max_tokens=1, temperature=0),
)
completion = outputs[0].outputs[0]
token_id = completion.token_ids[0]  # vLLM 生成的第一个 token
```

vllm-metal 内部执行：
- MLX Metal backend 执行 model forward
- PagedAttention (Metal kernels)
- 返回 logits 和生成的 token

**Step 2: 构造 one-hot logits**

```python
# hcp_vllm_worker.py:73-76
logits = torch.full((self.vocab_size,), -1e9, dtype=torch.float32)
logits[token_id] = 0.0
# 例: vocab_size=151936, token_id=304 → logits[304]=0.0, 其余=-1e9
```

> **注意**：当前 vLLM backend 不暴露完整 logits，只返回 one-hot。这是 Phase 1.5 MVP 的务实方案。

**Step 3: KV Ring 交换**

```python
# python/hcp_worker_sdk/quic_server.py:167-185
for layer_idx in range(self.backend.num_layers):  # 24 layers
    seq_start = self.seq_offset          # 0
    seq_end = self.global_seq_len        # 512 (prefill 后)
    
    local_block = self.backend.get_kv_block(layer_idx, seq_start, seq_end)
    # 当前为 stub: 返回空 tensor (Phase 1.5)
    
    for _round in range(self.num_domains - 1):  # 1 round for 2-domain
        peer_block = await self.kv_transport._exchange_kv_block(local_block)
        self.backend.apply_peer_kv(layer_idx, peer_block)
        local_block = peer_block  # 转发到下一个 peer
```

**KV 数据面帧格式：**

```
[4-byte BE meta_len] [JSON metadata] [k_bytes] [v_bytes]

metadata 示例：
{
  "layer_idx": 0,
  "global_seq_start": 0,
  "global_seq_end": 512,
  "k_shape": [1, 2, 512, 64],      // [batch, kv_heads, seq, head_dim]
  "v_shape": [1, 2, 512, 64],
  "k_bytes": 262144,               // 1*2*512*64*4 bytes (float32)
  "v_bytes": 262144
}
```

**并发 send+recv 避免死锁：**

```python
# quic_transport.py:174-185
send_task = asyncio.create_task(self._send_kv_block(block))
recv_task = asyncio.create_task(self._recv_kv_block())
await asyncio.wait([send_task, recv_task], return_when=asyncio.ALL_COMPLETED)
```

**Step 4: 返回 PrefillDone**

```python
# quic_server.py:171-176
return {
    "kind": "PrefillDone",
    "last_logits_bytes": logits_bytes,  // f32 bytes, len=151936*4=607744
    "global_seq_len": self.global_seq_len,  // 512
}
```

bincode 编码：

```
u32 tag = 0 (RESP_PREFILL_DONE)
Vec<u8> last_logits_bytes:
  u64 len = 607744
  bytes[0..607744]
u64 global_seq_len = 512
```

### 4.2 Worker 1 (white vLLM CUDA) 的处理流程

与 Worker 0 类似，但：
- chunk = [512, 2048) 的 1536 tokens
- seq_offset = 512
- vLLM 0.6.4 + XFormers backend (CUDA)
- Prefill ~0.32s @ 4788 tok/s

**输出：**

```
WorkerResponse::PrefillDone {
    last_logits_bytes: Vec<u8>,  // 607744 bytes
    global_seq_len: 2048,
}
```

---

## 5. 阶段四：Coordinator 收集与同步

### 5.1 输入

```
来自每个 worker 的 PrefillDone 响应
```

### 5.2 处理过程

**Step 1: 收集 prefill 响应**

```rust
// distributed_coordinator.rs:218-233
let mut max_global_seq_len = 0usize;
let mut last_logits_bytes: Vec<u8> = Vec::new();

for (domain_id, (_send, recv)) in worker_streams.iter_mut().enumerate() {
    let resp = recv_response_quic(recv, rt.handle())?;
    match resp {
        WorkerResponse::PrefillDone { last_logits_bytes: bytes, global_seq_len } => {
            max_global_seq_len = max_global_seq_len.max(global_seq_len);
            if domain_id == args.num_domains - 1 {
                last_logits_bytes = bytes;  // 用最后一个 domain 的 logits
            }
        }
    }
}
```

**Step 2: 同步 global_seq_len**

```rust
// distributed_coordinator.rs:236-240
for (send, _recv) in worker_streams.iter_mut() {
    let cmd = WorkerCommand::SyncGlobalSeqLen(max_global_seq_len);
    send_command_quic(send, &cmd, rt.handle())?;
}
// 所有 worker 的 global_seq_len 设为 2048
```

**Step 3: 采样第一个 token**

```rust
// distributed_coordinator.rs:242-252
let vocab_size = config.vocab_size as usize;  // 151936
let logits_vec: Vec<f32> = last_logits_bytes.chunks_exact(4)
    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    .collect();
let logits_tensor = Tensor::from_slice(&logits_vec);
let next_token = sample_token(&logits_tensor, temperature=0.0, top_p=1.0)? as i64;
// temperature=0.0 → greedy argmax
```

### 5.3 输出

```
max_global_seq_len: 2048
next_token: i64        // 第一个生成的 token ID
```

---

## 6. 阶段五：Decode 循环

### 6.1 每轮 Decode 的流程

**Coordinator 侧：**

```rust
// distributed_coordinator.rs:258-293
for step in 0..args.max_tokens {
    let token = next_token as u32;
    generated_ids.push(token);
    
    // 1. 广播 token 给所有 workers
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::Decode(next_token);
        send_command_quic(send, &cmd, rt.handle())?;
    }
    
    // 2. 接收 worker 0 的 decode 响应
    let resp = recv_response_quic(&mut worker_streams[0].1, rt.handle())?;
    let logits_bytes = match resp {
        WorkerResponse::DecodeDone { logits_bytes } => logits_bytes,
    };
    
    // 3. 同步接收其他 worker 的响应（保持流同步）
    for (_send, recv) in worker_streams.iter_mut().skip(1) {
        let _ = recv_response_quic(recv, rt.handle())?;
    }
    
    // 4. 采样下一个 token
    let decode_logits: Vec<f32> = logits_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let decode_tensor = Tensor::from_slice(&decode_logits);
    next_token = sample_token(&decode_tensor, args.temperature, args.top_p)? as i64;
}
```

**Worker 侧（每个 worker）：**

```python
# quic_server.py:178-189 (_handle_decode)
token = cmd["token"]
logits = self.backend.decode(token)

# VllmBackend.decode:
# 1. self._history.append(token)
# 2. vLLM generate(self._history, max_tokens=1)
# 3. 返回 one-hot logits

logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()
return {
    "kind": "DecodeDone",
    "logits_bytes": logits_bytes,
}
```

### 6.2 关键观察

- **当前 vLLM backend 每次 decode 都重新 prefill**：`self._history` 包含全部 token，vLLM 不支持增量输入。这是 Phase 1.5 MVP 的性能代价。
- **KV exchange 在 decode 阶段被跳过**：`quic_server.py:184` `# await self._exchange_kv_ring(prefill=False)`。所有 worker 在 prefill 后已同步，decode 不需要再交换 KV。

---

## 7. 阶段六：Shutdown 与结果输出

### 7.1 Coordinator 发送 Shutdown

```rust
// distributed_coordinator.rs:296-298
for (send, _recv) in worker_streams.iter_mut() {
    let _ = send_command_quic(send, &WorkerCommand::Shutdown, rt.handle());
}
```

### 7.2 Worker 响应 Shutdown

```python
# quic_server.py:90-94
elif kind == "Shutdown":
    print(f"[worker {self.domain_id}] shutting down")
    break

# finally 块:
# await self.control_client.close()
# backend.shutdown()  // 释放 vLLM 资源
```

### 7.3 Coordinator 解码生成结果

```rust
// distributed_coordinator.rs:301-302
let text = tokenizer.decode(&generated_ids, true)?;
println!("[coordinator] generated: {}", text);
```

---

## 8. 协议帧格式速查表

### 8.1 控制面（Coordinator ↔ Worker）

| 层级 | 格式 | 说明 |
|------|------|------|
| 传输 | `[4-byte BE length][payload]` | length-prefixed frame |
| Handshake | `[8-byte LE domain_id][8-byte LE capacity_mb]` | 16 bytes，无 length prefix |
| Prefill 命令 | `tag=0u32 + Vec<i64> chunk + i64 seq_offset` | bincode LE |
| Decode 命令 | `tag=1u32 + i64 token` | bincode LE |
| SyncGlobalSeqLen | `tag=2u32 + u64 global_seq_len` | bincode LE |
| Shutdown | `tag=3u32` | bincode LE |
| PrefillDone | `tag=0u32 + Vec<u8> logits + u64 global_seq_len` | bincode LE |
| DecodeDone | `tag=1u32 + Vec<u8> logits` | bincode LE |

### 8.2 数据面（Worker ↔ Worker KV Ring）

| 层级 | 格式 | 说明 |
|------|------|------|
| Dummy | `0x00` | Rust quinn workaround，首次发送前写 1 byte |
| 传输 | `[4-byte BE meta_len][JSON][k_bytes][v_bytes]` | JSON 包含 shape 和 bytes 长度 |
| KV tensor | float32 numpy bytes | `torch.Tensor → cpu().float32().numpy().tobytes()` |

---

## 9. 关键数据流验证点

### 9.1 Tokenize → Chunk → Prefill

| 步骤 | 输入 | 输出 | 验证方式 |
|------|------|------|---------|
| Tokenize | "Hello world" | `[9906, 1917]` (2 tokens) | coordinator log |
| 均分分片 | 2 tokens, 2 domains | chunk0=[9906], chunk1=[1917] | coordinator log |
| 不均分分片 | 2048 tokens, 25/75 | chunk0=[0,512), chunk1=[512,2048) | coordinator log |
| Prefill | chunk + seq_offset | logits (one-hot) + global_seq_len | worker log |

### 9.2 KV Ring Exchange

| 步骤 | 输入 | 输出 | 备注 |
|------|------|------|------|
| get_kv_block | layer_idx, seq_start, seq_end | KvBlock(k, v) | **当前为 stub**，返回空 tensor |
| _send_kv_block | KvBlock | QUIC stream bytes | 包含 JSON metadata + raw f32 bytes |
| _recv_kv_block | QUIC stream bytes | KvBlock(k, v) | 反序列化为 torch.Tensor |
| apply_peer_kv | layer_idx, peer_block | None | **当前为 no-op** |

### 9.3 Decode Loop

| 步骤 | 输入 | 输出 | 验证方式 |
|------|------|------|---------|
| Broadcast token | `WorkerCommand::Decode(token)` | — | coordinator log |
| Worker decode | token → vLLM generate | one-hot logits | worker log |
| Coordinator sample | logits f32 bytes | next_token | coordinator log |
| 生成文本 | generated_ids | decoded string | coordinator log |

---

## 10. 当前限制与下一步

| 限制 | 说明 | 影响 |
|------|------|------|
| vLLM 不暴露完整 logits | `LLM.generate()` 只返回 token，backend 构造 one-hot | 无法做真实采样，greedy decode 可用但 temperature/top-p 无效 |
| KV block 为 stub | `get_kv_block` / `apply_peer_kv` 返回空/no-op | **KV ring 交换存在但无实际数据**，这是最大 gap |
| 每次 decode 重新 prefill | vLLM `LLM` API 不支持增量输入 | 性能极差，O(seq²) 每步 |
| 真实多步 decode 未验证 | 当前只验证了 3-5 步 decode | 更长 decode 可能暴露一致性 bug |

### 下一步（Phase 3.4）

**真实 KV 提取**是打通完整 pipeline 的关键：

1. **vLLM 0.6.4 CUDA**: 探索 `LLMEngine.model_executor.driver_worker.model_runner.model.model.layers[i].self_attn.kv_cache` 或 `CacheEngine` 底层 API
2. **vllm-metal 0.20.x**: 探索 MLX KV cache 的访问方式（`vllm_metal` 内部可能用 `mx.array` 管理 KV）
3. **KV 格式转换**: PagedAttention 的 block table → 连续 tensor → `KvBlock` 序列化
4. **Online softmax 合并**: 收到 peer KV 后，在 Rust 侧或 Python 侧用 ring attention 算法合并增量贡献

---

## 附录：实际运行日志片段

### 2048 tokens, 25%/75% 分片运行日志

```
[coordinator] prompt tokens: 2048
[coordinator] sent Prefill chunk [0, 512) to worker 0
[coordinator] sent Prefill chunk [512, 2048) to worker 1
[coordinator] worker 0 prefill done, global_seq_len=512
[coordinator] worker 1 prefill done, global_seq_len=2048
[coordinator] max_global_seq_len = 2048
[coordinator] generated:  dog jumps over the lazy

[worker 0] Prefill: 1.69s @ 302 tok/s (512 tokens, MPS)
[worker 1] Prefill: ~0.32s @ 4788 tok/s (1536 tokens, CUDA)
[worker 1] Decode: 16.6-18.9 it/s
```
