# vLLM Worker 适配器详细设计

> 本文档描述如何将 vLLM 的 `LLMEngine` 包装为 HCP 分布式 Worker，让 vLLM 负责同构 GPU 域内的所有优化（PagedAttention、Continuous Batching、CUDA kernel），HCP 负责跨域 KV Ring 交换。

---

## 1. 为什么先选 vLLM

vLLM 是当前开源社区最成熟的 LLM 推理服务框架：

- **PagedAttention**：将 KV cache 分页管理，消除内存碎片
- **Continuous Batching**：动态批处理，提高 GPU 利用率
- **CUDA Kernel 优化**：FlashAttention、FP8、Speculative Decoding
- **生态丰富**：OpenAI API 兼容、多 LoRA、Prefix Caching

HCP 默认的 Rust/tch-rs Worker 是 correctness-first 实现，kernel 层面远未达到生产级。在同构 NVIDIA GPU 集群内部，没有理由不用 vLLM。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Coordinator                                 │
│                           (Rust / Python)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │ Tokenizer   │  │ Chunker     │  │ Sampler (temperature, top_p)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ QUIC control streams
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
    ┌──────────────────────────────┐    ┌──────────────────────────────┐
    │   Domain 0: Mac MPS/CPU      │    │   Domain 1: GPU CUDA         │
    │   HCP Rust Worker (default)  │    │   vLLM + HCP Adapter         │
    │                              │    │                              │
    │  ┌────────────────────────┐  │    │  ┌────────────────────────┐  │
    │  │ LlamaModel (tch-rs)    │  │    │  │ LLMEngine (vLLM)       │  │
    │  │ - LocalAttention       │  │    │  │ - PagedAttention       │  │
    │  │ - HcpRingAttention     │  │    │  │ - FlashAttention       │  │
    │  └────────────────────────┘  │    │  │ - CUDA Kernels         │  │
    │                              │    │  └────────────────────────┘  │
    │  ┌────────────────────────┐  │    │  ┌────────────────────────┐  │
    │  │ KvTransport (QUIC)     │◄─┼────┼─►│ KvTransport (QUIC)     │  │
    │  │ exchange_kv_block()    │  │    │  │ exchange_kv_block()    │  │
    │  └────────────────────────┘  │    │  └────────────────────────┘  │
    └──────────────────────────────┘    └──────────────────────────────┘
```

---

## 3. vLLM 侧的关键挑战

### 3.1 挑战 1：PagedAttention KV Cache 格式

vLLM 的 KV cache 不是连续的 `[layers, 2, seq, heads, dim]`，而是：

```python
# vLLM KV cache 结构（概念）
class CacheConfig:
    block_size: int          # 每个 block 容纳的 token 数（默认 16）
    num_gpu_blocks: int      # GPU 上分配的 block 数
    num_cpu_blocks: int      # CPU 上分配的 block 数（offload）

# 每个 sequence 的 KV cache 由 block_table 索引
block_table: List[int] = [3, 7, 12, 45, ...]  # 指向 KV cache block 的序号

# 实际 KV cache 存储（GPU 上）
# kv_cache[layer_idx] shape: [2, num_gpu_blocks, block_size, num_heads, head_dim]
# 其中 kv_cache[layer_idx][0] 是 K，kv_cache[layer_idx][1] 是 V
```

**HCP 需要的是**：每层一个连续的 `[2, seq_len, num_heads, head_dim]` tensor。

**解决方案**：

```python
def gather_kv_from_block_table(
    kv_cache: List[torch.Tensor],  # [layers] each [2, num_blocks, block_size, heads, dim]
    block_table: List[int],
    num_layers: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 vLLM 的 paged KV cache 提取连续 K/V。
    返回: K [num_layers, seq_len, num_heads, head_dim]
          V [num_layers, seq_len, num_heads, head_dim]
    """
    k_list, v_list = [], []
    for layer_idx in range(num_layers):
        layer_cache = kv_cache[layer_idx]  # [2, num_blocks, block_size, heads, dim]
        # 按 block_table 顺序 gather
        blocks = layer_cache[:, block_table, :, :, :]  # [2, num_blocks_in_table, block_size, heads, dim]
        # flatten block dimension → [2, seq_len, heads, dim]
        kv = blocks.view(2, -1, num_heads, head_dim)
        # 截取实际 seq_len（最后一个 block 可能没满）
        kv = kv[:, :seq_len, :, :]
        k_list.append(kv[0])
        v_list.append(kv[1])
    
    K = torch.stack(k_list, dim=0)  # [num_layers, seq_len, num_heads, head_dim]
    V = torch.stack(v_list, dim=0)
    return K, V
```

### 3.2 挑战 2：Attention 替换

vLLM 的 `LLMEngine` 是高层 API，对外暴露的是 `schedule` + `execute_model`。我们无法直接 hook 内部的 attention forward。

**推荐方案：后处理模式（Post-process）**

不替换 vLLM attention，而是：
1. 让 vLLM 用本地 KV cache 正常跑完一层
2. 提取该层的 output hidden states
3. 通过 HCP transport 获取 peer KV
4. 用 HCP online softmax 计算 peer KV 对当前层输出的增量贡献
5. 合并到 output

```python
class HcpVllmAttentionMerger:
    """
    在 vLLM 每层 forward 后，用 HCP online softmax 合并 peer KV 贡献。
    """
    def __init__(self, num_heads: int, head_dim: int, num_layers: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.scale = 1.0 / (head_dim ** 0.5)
    
    def merge_peer_kv_into_layer(
        self,
        layer_idx: int,
        q: torch.Tensor,                    # [batch, num_heads, q_len, head_dim]
        local_output: torch.Tensor,         # [batch, seq_len, hidden_size]
        peer_kv_blocks: List[Tuple[torch.Tensor, torch.Tensor]],  # [(K, V), ...]
        seq_offset: int,
    ) -> torch.Tensor:
        """
        对当前层的 Q，逐个处理 peer KV block，用 online softmax 合并。
        返回修正后的 output。
        """
        # 初始化 online softmax state
        batch, num_heads, q_len, head_dim = q.shape
        rm = torch.full((batch, num_heads, q_len), float('-inf'), device=q.device)
        rs = torch.zeros((batch, num_heads, q_len), device=q.device)
        obh = torch.zeros((batch, num_heads, q_len, head_dim), device=q.device)
        
        # 先处理本地 KV（已包含在 local_output 中，需要反推 local attention 的 softmax 状态）
        # 实际上 vLLM 已经算完了，我们只需要额外处理 peer KV
        
        for peer_k, peer_v in peer_kv_blocks:
            # peer_k, peer_v: [batch, num_heads, peer_seq, head_dim]
            scores = torch.matmul(q, peer_k.transpose(-2, -1)) * self.scale
            
            # causal mask: peer tokens 在 Q 左边才能被看到
            q_pos = torch.arange(seq_offset, seq_offset + q_len, device=q.device)
            k_pos = torch.arange(peer_k.shape[2], device=q.device)  # 需要知道 peer 的全局位置
            # ... (与 HCP Rust 端相同的 online softmax 逻辑)
            
            local_max = scores.amax(dim=-1, keepdim=False)
            weights = torch.exp(scores - local_max.unsqueeze(-1))
            local_sum = weights.sum(dim=-1)
            local_pv = torch.matmul(weights, peer_v)
            
            new_max = torch.maximum(rm, local_max)
            exp_prev = torch.exp(rm - new_max)
            exp_local = torch.exp(local_max - new_max)
            new_sum = exp_prev * rs + exp_local * local_sum
            
            obh = (exp_prev.unsqueeze(-1) * rs.unsqueeze(-1) * obh +
                   exp_local.unsqueeze(-1) * local_pv) / new_sum.unsqueeze(-1)
            
            rm = new_max
            rs = new_sum
        
        # obh 是 peer KV 贡献的 attention output
        # 需要加到 local_output 上（通过 O-projection）
        return self._merge_output(local_output, obh, layer_idx)
```

### 3.3 挑战 3：vLLM 的模型并行

如果 domain 内部有多张 GPU（TP/PP），vLLM 已经处理了这些。HCP 只关心 domain 对外的 KV 交换：

- 在 TP=2 时，vLLM 内部的 attention head 被分到两张卡上
- HCP adapter 需要在所有卡上收集完整的 K/V（all-gather within domain）
- 然后以 domain 为单位对外发送

**解决方案**：在 vLLM Worker 进程内，先执行 domain 内部的 all-gather（NCCL），再执行 domain 之间的 HCP P2P。

---

## 4. 最小可运行代码骨架

```python
# hcp_vllm_worker.py —— 最小可运行骨架
#!/usr/bin/env python3
"""vLLM-based HCP Worker."""

import asyncio
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

# vLLM imports
# from vllm import LLM, SamplingParams
# from vllm.worker.worker import Worker
# from vllm.core.scheduler import Scheduler

# HCP protocol imports (假设已实现)
from hcp_worker_sdk import HcpWorkerBackend, KvBlock, KvTransport


class VllmHcpBackend(HcpWorkerBackend):
    """vLLM 实现的 HCP Worker 后端。"""
    
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        # self.engine = LLM(model=model_dir, dtype="float16", tensor_parallel_size=1)
        # self.worker = self.engine.llm_engine.model_executor.driver_worker
        # self.cache_engine = self.worker.cache_engine
        self.num_layers = 24  # 从 config 读取
        self.num_heads = 14
        self.head_dim = 64
        self.kv_cache = []  # 简化：直接用 list of tensors
        self.seq_len = 0
        
    def load_model(self, model_dir: str, device: str) -> None:
        """加载模型。"""
        pass  # vLLM 在 __init__ 中已加载
    
    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        """执行 prefill，返回 last token logits 和 global_seq_len。"""
        # 简化版：直接跑 vLLM 的 forward
        # 实际实现需要 hook vLLM 的 execute_model
        
        input_tensor = torch.tensor([chunk], dtype=torch.long, device=self.device)
        # output = self.engine.generate(...)  # vLLM API
        
        # 提取 last token logits
        # logits = output[-1]  # [vocab_size]
        logits = torch.randn(151936)  # placeholder
        self.seq_len = len(chunk)
        return logits, self.seq_len
    
    def decode(self, token: int) -> torch.Tensor:
        """执行单 token decode。"""
        # input_tensor = torch.tensor([[token]], dtype=torch.long, device=self.device)
        # output = self.engine.generate(...)
        logits = torch.randn(151936)  # placeholder
        self.seq_len += 1
        return logits
    
    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        """从 vLLM KV cache 提取 block。"""
        # k = self.kv_cache[layer_idx][0][:, seq_start:seq_end, :, :]
        # v = self.kv_cache[layer_idx][1][:, seq_start:seq_end, :, :]
        k = torch.randn(1, 14, seq_end - seq_start, 64)
        v = torch.randn(1, 14, seq_end - seq_start, 64)
        return KvBlock(layer_idx, seq_start, seq_end, k, v)
    
    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        """将 peer KV 合并到当前层（online softmax）。"""
        # 这里需要实现与 HCP Rust 端等价的 online softmax
        pass
    
    @property
    def capacity_mb(self) -> int:
        """上报可用显存。"""
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[1] // (1024 * 1024)
        return 16384  # fallback


class HcpVllmWorker:
    """完整的 vLLM HCP Worker 进程。"""
    
    def __init__(
        self,
        model_dir: str,
        domain_id: int,
        num_domains: int,
        device: str = "cuda",
    ):
        self.backend = VllmHcpBackend(model_dir, device)
        self.domain_id = domain_id
        self.num_domains = num_domains
        # self.transport = QuicKvTransport(...)  # 待实现
        
    async def run(self, coordinator_addr: str, listen_addr: str) -> None:
        """主事件循环。"""
        # 1. 连接 coordinator
        # 2. 发送 handshake
        # 3. 建立 peer 连接（QUIC/TCP）
        # 4. 循环处理 WorkerCommand
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--coordinator-addr", required=True)
    parser.add_argument("--listen-addr", default="0.0.0.0:29452")
    parser.add_argument("--next-peer-addr", required=True)
    args = parser.parse_args()
    
    worker = HcpVllmWorker(
        model_dir=args.model_dir,
        domain_id=args.domain_id,
        num_domains=args.num_domains,
    )
    asyncio.run(worker.run(args.coordinator_addr, args.listen_addr))
```

---

## 5. 部署命令示例

### 5.1 同构 vLLM 集群（2x A100）

```bash
# Node 0 (A100-0)
python hcp_vllm_worker.py \
  --model-dir /models/Llama-3-8B \
  --domain-id 0 \
  --num-domains 2 \
  --coordinator-addr 10.0.0.1:29450 \
  --listen-addr 0.0.0.0:29451 \
  --next-peer-addr 10.0.0.2:29452

# Node 1 (A100-1)
python hcp_vllm_worker.py \
  --model-dir /models/Llama-3-8B \
  --domain-id 1 \
  --num-domains 2 \
  --coordinator-addr 10.0.0.1:29450 \
  --listen-addr 0.0.0.0:29452 \
  --next-peer-addr 10.0.0.1:29451
```

### 5.2 异构混合（Mac MLX + GPU vLLM）

```bash
# Mac 端：Rust Worker 0 (MPS/MLX)
HCP_TCH_DEVICE=mps cargo run --bin hcp-ringattn-rust -- --distributed-role worker ...

# GPU 端：vLLM Worker 1 (CUDA)
python hcp_vllm_worker.py --model-dir /models/Qwen2-0.5B --domain-id 1 ...
```

---

## 6. 性能预期

| 指标 | HCP Rust Worker | vLLM Worker | 备注 |
|------|----------------|-------------|------|
| Prefill (8K) | ~30s (MPS) | ~2s (A100) | vLLM FlashAttention 优势明显 |
| Decode per token | ~200ms | ~10ms | vLLM PagedAttention + kernel 优化 |
| KV cache 内存效率 | 连续分配，有碎片 | 分页管理，无碎片 | vLLM 可支持更长序列 |
| 跨域传输 overhead | 同量级 | 同量级 | 取决于网络，非框架差异 |

**结论**：在同构 NVIDIA GPU 域内，vLLM Worker 应该比 HCP Rust Worker 快 10-30 倍。跨域传输 overhead 固定，所以整体加速比取决于计算/通信比例。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| vLLM 版本升级导致 API 变化 | 适配器需要跟着改 | 用最小子集 API，定期升级测试 |
| PagedAttention KV 提取 overhead | 每层 forward 后额外 gather | 异步 prefetch，与计算 overlap |
| vLLM 与 HCP 的调度冲突 | vLLM 有自己的 scheduler | 在 adapter 中禁用 vLLM scheduler，改为 coordinator 驱动 |
| 多卡 TP 时 KV 不完整 | domain 对外发送的 KV 被 shard | domain 内部先 all-gather，再对外 P2P |

---

## 8. 下一步行动

1. **实现 Python HCP Worker SDK** (`python/hcp_worker_sdk/`)
   - `WorkerCommand` / `WorkerResponse` 的 bincode 序列化
   - `KvTransport` 的 Python 实现（先用 TCP，再升级 QUIC）
   - `HcpWorkerServer` 通用事件循环

2. **实现 vLLM 最小适配器**
   - 用 vLLM 的 `LLM` 类跑通单节点 prefill/decode
   - 接入 Mock KV Transport 验证 2-domain correctness
   - 接入 TCP/QUIC Transport 跑跨节点 smoke

3. **Correctness 验证**
   - Rust Worker ↔ vLLM Worker 的数值对齐（tolerance tier: Relaxed）
   - Long context（32K+）的端到端生成一致性
