# HCP 可插拔域内后端架构

> 目标：在同构计算域内，通过接口实现的形式，把默认的 Rust/tch-rs worker 替换为 vLLM、TensorRT-LLM、MLX 等社区成熟实现，最大化复用社区轮子。

---

## 1. 核心思想

HCP 的边界是 **跨域低层协议**（P2P KV ring + online softmax），**域内实现是黑盒**。

这意味着：
- **Coordinator 不关心** domain 内部用 Rust、Python 还是 C++ 跑模型
- **Coordinator 只关心**：Worker 能否正确响应 `WorkerCommand`，能否在 KV ring 中收发 block
- **Domain 内部**：可以用 vLLM 的 `LLMEngine`、TensorRT-LLM 的 `GptSession`、MLX 的 `nn.Module`——只要外面包一层 HCP 协议适配器

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Coordinator (Rust)                           │
│  - Tokenizer + Config                                                │
│  - Prompt chunking                                                   │
│  - WorkerCommand {Prefill, Decode, SyncGlobalSeqLen, Shutdown}      │
│  - Logits sampling                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │ QUIC control stream
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HCP Worker Interface (协议边界 — 所有实现必须满足)                    │
│  ─────────────────────────────────────────────────────────────────  │
│  输入: WorkerCommand::Prefill { chunk, seq_offset }                 │
│       WorkerCommand::Decode(token)                                  │
│  输出: WorkerResponse::PrefillDone { last_logits_bytes }            │
│       WorkerResponse::DecodeDone { logits_bytes }                   │
│  侧信道: KvTransport::exchange_kv_block() — QUIC stream            │
└─────────────────────────────────────────────────────────────────────┘
                              │ 接口实现层
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
    │ HCP-Rust Worker │ │ vLLM Adapter │ │ MLX Adapter  │
    │ (tch-rs default)│ │ (Python)     │ │ (Swift/C++)  │
    └─────────────────┘ └──────────────┘ └──────────────┘
```

---

## 2. 为什么需要这个架构

### 2.1 社区轮子的价值

| 后端 | 优势 | HCP 默认实现的差距 |
|------|------|-------------------|
| **vLLM** | PagedAttention、Continuous Batching、CUDA kernel 优化、前缀缓存、多 LoRA | 我们的 Rust 实现是 correctness-first，kernel 未优化 |
| **TensorRT-LLM** | NVIDIA 官方 kernel 融合、FP8/INT8 KV Cache、Inflight Batching | 不支持 TRT engine、无量化 |
| **MLX** | Apple Silicon 原生优化、统一内存、内存高效 attention | 我们的 MPS 路径 workaround 多，未做 Metal kernel 优化 |
| **SGLang** | RadixAttention、Structured Decoding、FastAPI 服务化 | 无服务化能力 |

### 2.2 HCP 的独特定位

这些框架**各自都是同构集群的王者**，但：
- vLLM 假设所有 GPU 是同构的 NCCL process group
- MLX 只在 Apple Silicon 上跑
- TensorRT-LLM 只跑在 NVIDIA 上

**HCP 的独特价值是**：让异构设备（MPS + CUDA + 未来 NPU）在同一个 attention layer 内协作。域内用最强社区框架，域间用 HCP P2P 协议——这是 vLLM 们天然不会做的事。

---

## 3. 接口契约

### 3.1 控制面接口（Worker ↔ Coordinator）

所有 Worker 实现必须支持以下协议：

```rust
// distributed_protocol.rs
pub enum WorkerCommand {
    Prefill { chunk: Vec<i64>, seq_offset: i64 },
    Decode(i64),
    SyncGlobalSeqLen(usize),
    Shutdown,
}

pub enum WorkerResponse {
    PrefillDone { last_logits_bytes: Vec<u8>, global_seq_len: usize },
    DecodeDone { logits_bytes: Vec<u8> },
    Error(String),
}
```

**序列化格式**：bincode（length-prefixed frame over QUIC stream）

**Worker 生命周期**：
1. 启动后通过 QUIC 连接 coordinator
2. 发送 `WorkerHandshake { domain_id, capacity_mb }`
3. 等待 `WorkerCommand::Prefill` → 执行 prefill → 返回 `PrefillDone`
4. 等待 `WorkerCommand::SyncGlobalSeqLen` → 更新全局 seq len
5. 循环等待 `WorkerCommand::Decode` → 执行 decode → 返回 `DecodeDone`
6. 收到 `Shutdown` → 优雅退出

### 3.2 数据面接口（Worker ↔ Worker，KV Ring）

```rust
pub trait KvTransport: Send {
    fn exchange_kv_block(&mut self, block: &KvBlock) -> Result<Option<KvBlock>, String>;
}

pub struct KvBlock {
    pub layer_idx: usize,
    pub global_seq_start: usize,
    pub global_seq_end: usize,
    pub k: Tensor,  // [batch, num_heads, seq_len, head_dim]
    pub v: Tensor,  // [batch, num_heads, seq_len, head_dim]
}
```

**关键要求**：
- `exchange_kv_block` 是**原子操作**：同时 send 本地 KV block 到 next peer，并从 prev peer recv KV block
- 必须支持并发 send+recv，防止大 KV block 死锁
- KV tensor 以原始 float32 bytes 传输（可直接用 `torch.Tensor` 的 `numpy().tobytes()` / `frombuffer()`）

### 3.3 模型状态接口（Worker 内部）

每个 Worker 内部需要维护：
- **模型权重**：从 `--model-dir` 加载（safetensors 或框架原生格式）
- **KV Cache**：每个 layer 一个，支持 append 操作
- **seq_offset**：本 domain 在全局序列中的起始位置
- **global_seq_len**：当前全局序列总长度（由 coordinator 同步）

---

## 4. 适配器分层设计

### 4.1 分层模型

```
Layer 3: 框架原生层
  ├─ vLLM: LLMEngine, Worker, CacheEngine, GPUModelRunner
  ├─ TensorRT-LLM: GptSession, KVCacheManager
  └─ MLX: nn.Module, KV cache array

Layer 2: HCP 适配层（必须自己实现）
  ├─ CommandHandler: 将 WorkerCommand 映射到框架 API
  ├─ KvTransportBridge: 将框架 KV cache 序列化为 KvBlock / 反序列化注入
  ├─ OnlineSoftmaxAggregator: 在框架 attention 输出上叠加 peer KV 贡献
  └─ HandshakeReporter: 上报 capacity_mb、device 信息

Layer 1: HCP 协议层（可复用）
  ├─ QUIC endpoint 管理（quinn wrapper）
  ├─ Frame I/O（length-prefixed bincode）
  ├─ WorkerCommand / WorkerResponse 序列化
  └─ KvTransport trait 实现（QUIC/TCP）
```

### 4.2 适配器核心职责

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **CommandHandler** | 接收 coordinator 指令，调用框架执行 forward | `WorkerCommand` | `WorkerResponse` |
| **KvRingBridge** | 在框架 attention 前后插入 KV ring 交换 | 本地 K/V tensors | peer K/V tensors（通过 transport） |
| **SoftmaxMerger** | 将本地 attention 结果与 peer block 结果用 online softmax 合并 | 本地 output, peer outputs | 全局等价 output |
| **CacheSynchronizer** | 管理本 domain 的 KV cache，支持 prefill 写入和 decode append | token ids | 更新后的 KV cache |

---

## 5. vLLM 适配器设计（示例）

### 5.1 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                    vLLM Worker Adapter (Python)                   │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ HCP Protocol │  │ vLLM Engine  │  │ HCP KV Ring Bridge   │   │
│  │ Server       │  │ (LLMEngine)  │  │ (Online Softmax)     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│         │ 1. Prefill/     │ 2. schedule/execute  │ 3. exchange   │
│         │    Decode cmd   │    forward           │    KV blocks  │
│         │                 │                      │               │
│         └─────────────────┴──────────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
                              │
                    QUIC (control + KV ring)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            Coordinator            Peer Worker
```

### 5.2 关键设计决策

#### 决策 1：vLLM 的 KV Cache 格式转换

vLLM 内部使用 **PagedAttention** 的 block table 格式管理 KV cache，不是连续的 `[layers, 2, seq, heads, dim]`。HCP 需要：

```python
# vLLM → HCP: 从 block table 提取连续 K/V tensor
def extract_kv_from_block_table(
    cache_engine: CacheEngine,
    block_table: List[int],
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 vLLM 的 paged cache 提取 [layers, heads, seq, dim] 格式 K/V"""
    # 按 block_table 顺序读取 GPU 上的 KV cache block
    # 返回 K, V: [num_layers, num_heads, seq_len, head_dim]
    pass

# HCP → vLLM: 将收到的 peer KV 注入到本地 block table
def inject_kv_to_block_table(
    cache_engine: CacheEngine,
    peer_k: torch.Tensor,  # [layers, heads, peer_seq, dim]
    peer_v: torch.Tensor,
    block_table: List[int],
    seq_offset: int,
) -> None:
    """将 peer KV 写入 paged cache 的指定位置"""
    pass
```

**Trade-off**：每次 layer forward 前后做一次 D2D / D2H 格式转换，有 overhead。但对于异构协作场景，network transfer 本身就是瓶颈，格式转换 overhead 相对较小。

#### 决策 2：Attention 替换策略

**方案 A：Hook vLLM Attention 层（侵入式）**
- 在 `vllm.attention.layer.DecoderSelfAttention.forward` 中插入 KV ring 交换
- 优点：最自然，online softmax 直接在 attention 内部完成
- 缺点：需要修改 vLLM 源码，维护成本高

**方案 B：Wrapper 层（非侵入式，推荐）**
- 不修改 vLLM，而是在 vLLM 完成一层 forward 后：
  1. 提取该层的 KV cache
  2. 通过 HCP transport 交换 peer KV
  3. 用 HCP 的 `process_kv_block` 计算 peer KV 对当前层输出的修正
  4. 将修正后的输出写回
- 优点：不修改 vLLM，版本升级无负担
- 缺点：每层多做一次 forward（本地 + peer），计算量翻倍

**推荐方案 B**，因为：
- vLLM 升级频繁，侵入式 hook 维护成本高
- 方案 B 的"每层两次 forward" 在分布式场景下被网络等待掩盖
- 可以先跑通 correctness，再优化性能

#### 决策 3：Online Softmax 的实现位置

```python
class HcpVllmAdapter:
    def __init__(self, vllm_engine: LLMEngine, domain_id: int, num_domains: int):
        self.engine = vllm_engine
        self.domain_id = domain_id
        self.num_domains = num_domains
        self.kv_transport: KvTransport = ...  # QUIC/TCP
        
    def forward_with_ring_attention(self, input_tokens, seq_offset):
        # Step 1: vLLM 执行本地 forward（包含本地 KV 的 attention）
        local_output, local_kv = self.engine.execute_model(input_tokens)
        
        # Step 2: KV Ring 交换
        peer_kv_blocks = []
        for round in range(self.num_domains - 1):
            peer_block = self.kv_transport.exchange_kv_block(local_kv)
            if peer_block:
                peer_kv_blocks.append(peer_block)
                local_kv = peer_block  # 转发给下一个 peer
        
        # Step 3: Online Softmax 合并（HCP 核心算法）
        # 用 Rust 端相同的 process_kv_block 逻辑
        merged_output = self.online_softmax_merge(
            local_output, peer_kv_blocks, seq_offset
        )
        return merged_output
```

### 5.3 最小 MVP 实现路径

1. **Phase 1**：单节点 vLLM 跑通 HCP 控制协议
   - 实现 `HcpVllmWorker` 类，支持 `Prefill` / `Decode` / `Shutdown`
   - 暂时不接入 KV ring，先验证控制面通信
   - Coordinator 连接 Python Worker，跑通 end-to-end 生成

2. **Phase 2**：接入 Mock KV Transport
   - 用 `LinkedMockKvTransport` 在单进程内模拟 2-domain
   - 验证 vLLM 输出 + HCP online softmax 合并后数值正确

3. **Phase 3**：接入 QUIC Transport
   - 用 Python `aioquic` 或 `quinn` Python binding 实现 KvTransport
   - 双节点：Mac 跑 Rust Worker 0，GPU 跑 Python vLLM Worker 1

4. **Phase 4**：优化（可选）
   - 批量 KV 传输（减少 round trip）
   - vLLM PagedAttention block table 直连 HCP transport（零拷贝）

---

## 6. 接口定义参考（Python 版 Worker SDK）

未来可以提供一个官方 Python SDK，降低适配器开发门槛：

```python
# hcp_worker_sdk.py — 概念性接口
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class KvBlock:
    layer_idx: int
    global_seq_start: int
    global_seq_end: int
    k: torch.Tensor  # [batch, num_heads, seq, head_dim]
    v: torch.Tensor

class KvTransport(ABC):
    @abstractmethod
    def exchange_kv_block(self, block: KvBlock) -> KvBlock | None:
        """Send local block to next peer, receive block from prev peer."""
        pass

class HcpWorkerBackend(ABC):
    """框架适配器必须实现的接口。"""
    
    @abstractmethod
    def load_model(self, model_dir: str, device: str) -> None:
        """加载模型权重。"""
        pass
    
    @abstractmethod
    def prefill(self, chunk: list[int], seq_offset: int) -> tuple[torch.Tensor, int]:
        """
        执行 prefill forward。
        返回: (last_token_logits [vocab_size], global_seq_len)
        """
        pass
    
    @abstractmethod
    def decode(self, token: int) -> torch.Tensor:
        """
        执行单 token decode forward。
        返回: logits [vocab_size]
        """
        pass
    
    @abstractmethod
    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        """从框架 KV cache 提取指定范围的 K/V block。"""
        pass
    
    @abstractmethod
    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        """将 peer KV block 合并到当前层的 online softmax 状态中。"""
        pass
    
    @property
    @abstractmethod
    def capacity_mb(self) -> int:
        """上报可用显存/内存（用于 coordinator 分片）。"""
        pass

class HcpWorkerServer:
    """通用 Worker 协议服务器，框架开发者只需实现 HcpWorkerBackend。"""
    def __init__(self, backend: HcpWorkerBackend, transport: KvTransport):
        self.backend = backend
        self.transport = transport
    
    def run(self, coordinator_addr: str, listen_addr: str) -> None:
        """启动事件循环，处理 WorkerCommand/WorkerResponse。"""
        pass
```

---

## 7. TensorRT-LLM / MLX 适配要点

### 7.1 TensorRT-LLM

- TRT-LLM 的 KV cache 由 `GptSession` 内部管理，对外接口是 `generation_logits` + `present_key_values`
- 需要从 `present_key_values` 提取每层 K/V，序列化后走 HCP transport
- TRT-LLM 的 `In-flight Batching` 与 HCP 的单序列 decode 循环不冲突（可以关闭 batching）

### 7.2 MLX

- MLX 的 KV cache 是普通的 `mx.array`，可以在 CPU/GPU 间自由移动（unified memory）
- 可以在 Mac 上直接用 MLX 替代 tch-rs，避免 MPS workaround
- 但 MLX 目前不支持跨设备 P2P（这正是 HCP 的价值所在）

---

## 8. 路线图

| 阶段 | 目标 | 优先级 |
|------|------|--------|
| Phase 0 | Rust 默认 Worker 稳定（已完成） | ✅ |
| Phase 1 | Python SDK + vLLM Adapter MVP（单节点） | 🔥 高 |
| Phase 2 | vLLM Adapter 跨节点 smoke（Rust Worker ↔ vLLM Worker） | 🔥 高 |
| Phase 3 | TensorRT-LLM Adapter（GPU 域内替换） | 中 |
| Phase 4 | MLX Adapter（Mac 域内替换） | 中 |
| Phase 5 | 多框架混合：Mac(MLX) + GPU(vLLM) | 低（远景） |

---

## 9. 总结

HCP 不是要重新发明 vLLM/TensorRT-LLM/MLX，而是：

> **在异构设备的边界上定义最小协议，让每个域内可以自由选择最强的社区框架。**

这就像 Linux 的 VFS：ext4、btrfs、XFS 各自是文件系统，VFS 定义了统一的 open/read/write 接口。HCP 就是分布式长上下文 attention 的 "VFS"——定义跨域 P2P KV 交换和 online softmax 的接口，域内爱用啥用啥。
