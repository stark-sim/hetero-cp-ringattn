# Block 与 Ring 的融合架构 — 从二选一到协同

> 核心洞察：**vLLM 的 PagedAttention block 不是障碍，而是 Ring Attention 的天然粒度单位**。抛弃 block 能力 = 抛弃 vLLM 的全部价值，二者必须协同而非二选一。

---

## 1. 错误路径：提取连续 tensor

之前 `docs/VLLM_INTEGRATION.md` §3.1 假设的方向是：

```
vLLM block table → gather → 连续 tensor [layers, seq, heads, dim]
                                        ↓ 传输
                                   连续 tensor
                                        ↓ scatter
                               vLLM block table
```

**为什么错了**：
1. 来回转换毫无意义 — PagedAttention 的设计就是为了避免连续分配
2. 转换 overhead（gather/scatter）+ 传输 overhead 叠加
3. 失去了 vLLM 最核心的内存管理能力（block reuse、preemption、swap）
4. 如果最终结果还是连续 tensor，那用 transformers 的 `past_key_values` 更干净

---

## 2. 正确路径：Ring 在 Block 层面运作

### 2.1 核心思想

让 **Ring 的 KV block 直接对齐 vLLM 的 PagedAttention block**。不需要提取连续 tensor，不需要 gather/scatter。

```
vLLM block_size = 16 tokens (可配置)

Worker 0 负责 chunk [0, 512):
  Prefill 产生 blocks: [B0, B1, ..., B31]  (每个 Bi = 16 tokens 的 KV)
  
Worker 1 负责 chunk [512, 1536):
  Prefill 产生 blocks: [B32, B33, ..., B95]

Ring Exchange:
  Round 1: Worker 0 发送 [B0..B31] → Worker 1
           Worker 1 发送 [B32..B95] → Worker 0
           
Post-Exchange Worker 0 的 block table:
  [0, 1, ..., 31, 32, 33, ..., 95]
   ↑本地           ↑peer

vLLM Attention Kernel:
  仍然通过 block_table 索引，不关心 block 来源
  online softmax 在 kernel 内部处理跨 block 的 max/sum 更新
```

### 2.2 两种具体模式

#### 模式 A：Inter-block Ring（Block 之间做 Ring）

Ring 交换的**单位是 vLLM block**（默认 16 tokens）。

```
Worker 0 持有逻辑 blocks [0..31]（seq [0, 512)）
Worker 1 持有逻辑 blocks [32..95]（seq [512, 1536)）

Ring 遍历：
  Step 1: Worker 0 把 blocks [0..31] 发给 Worker 1
          Worker 0 从 Worker 1 收到 blocks [32..95]
          
  Step 2: Worker 0 现在有完整的 blocks [0..95]
          计算 attention(Q[0:512), K[0:1536), V[0:1536))
          
  Step 3: Worker 1 同理，计算 attention(Q[512:1536), K[0:1536), V[0:1536))
```

**优点**：
- 与 vLLM 的 block table 完全对齐
- 无需格式转换，传输后直接可用
- vLLM 的内存管理（block reuse、preemption）仍然有效

**缺点**：
- 需要修改 vLLM 的 `CacheEngine`，支持"远程 block"概念
- Block index 需要全局协调（Coordinator 分配）

#### 模式 B：Intra-block Ring（Block 内部做 Ring）

如果 block_size = 16 太大，可以在一个 block 内部进一步分片。

```
Block B0 内部（16 tokens 的 KV）:
  Worker 0 计算 tokens [0..8) 的 KV
  Worker 1 计算 tokens [8..16) 的 KV
  
  通过 ring 交换后，Block B0 完整
```

**适用场景**：超长序列、超多 worker，需要更细粒度并行。

**注意**：这本质上是把 vLLM 的 block_size 设得更小（如 8 或 4），或把 ring block 设得比 vLLM block 更小。实现复杂度更高，通常不需要。

---

## 3. 为什么 vLLM 的 Block 管理能力不能丢

| vLLM Block 能力 | 如果丢了，损失什么 |
|----------------|-----------------|
| **动态分配** | 无法按需分配 KV cache，必须预分配最大长度 |
| **Block reuse** | 相同 prefix 的 KV 无法共享（如 system prompt） |
| **Preemption** | GPU 内存不足时无法 swap blocks 到 CPU，直接 OOM |
| **Continuous Batching** | 无法动态合并不同 request 的 batch |
| **Prefix Caching** | 无法缓存常用 prefix 的 KV |

如果为了 ring attention 放弃这些，vLLM 相比 transformers 的优势就归零了。**那不如直接用 transformers。**

---

## 4. 与 Transformers 路径的对比

| 维度 | Transformers + CP Ring-Attn | vLLM + Block-Aware Ring |
|------|---------------------------|------------------------|
| **KV 格式** | `past_key_values` tuple，连续 tensor | PagedAttention block table |
| **KV 提取** | 零摩擦：`past_key_values[layer][:, :, start:end, :]` | 无需提取，直接传 blocks |
| **内存管理** | 连续分配，有碎片 | Block 级动态管理，无碎片 |
| **Batching** | 无（单 sequence） | 支持（block table 天然支持多 seq） |
| **kernel 优化** | 无（纯 PyTorch） | FlashAttention、cutlass 等 |
| **跨节点传输** | 传输连续 tensor slices | 传输 blocks（粒度更小，可流水线） |
| **实现复杂度** | **低**（已验证） | **高**（需改 vLLM CacheEngine） |
| **正确性验证** | ✅ 已验证 | 待验证 |
| **性能上限** | 低（无 kernel 优化） | 高（保留 vLLM 全部优化） |

**结论**：
- **短期**：Transformers 路径是验证 correctness 的最快路径
- **长期**：vLLM 路径必须走 Block-Aware Ring，否则不如不用 vLLM

---

## 5. Block-Aware Ring 的技术挑战

### 5.1 Block Index 全局协调

```python
# Coordinator 需要给每个 worker 分配全局唯一的 block index 范围
def allocate_block_indices(seq_len, block_size, num_domains, capacities):
    """
    返回每个 worker 的 block index 范围。
    
    例: seq_len=2048, block_size=16, 2 domains
        Worker 0: chunk [0, 512)  → blocks [0..31]
        Worker 1: chunk [512, 2048) → blocks [32..127]
    """
    pass
```

### 5.2 CacheEngine 扩展

```python
# vLLM CacheEngine 需要支持 "远程 block"
class DistributedCacheEngine:
    """扩展 vLLM CacheEngine，支持从 peer worker 接收 blocks。"""
    
    def __init__(self, local_block_range, ...):
        self.local_blocks = {}  # 本地计算的 blocks
        self.remote_blocks = {}  # 从 peer 接收的 blocks
        self.block_table = []  # 混合了 local + remote 的索引
    
    def receive_remote_block(self, block_idx, k_bytes, v_bytes):
        """从 transport 接收 peer block，写入 GPU memory。"""
        # 分配 GPU block slot
        # 写入 K/V bytes
        # 更新 block_table
        pass
    
    def get_kv_cache(self, layer_idx):
        """返回该层的 KV cache（包含 remote blocks）。"""
        # vLLM attention kernel 仍然正常索引
        pass
```

### 5.3 Online Softmax 在 Block 粒度

Ring Attention 的 online softmax 不需要修改：
- 每个 block 是一个 KV chunk
- `process_kv_block` 处理每个 block 时，block 内部是连续的
- Block 之间通过 online softmax 的 max/sum 状态传递

```
Worker 0 计算 Q[0:512) vs K[0:512)  →  local_output, running_max, running_sum
收到 peer block B32 (K[512:528), V[512:528)):
  → 计算 Q[0:512) vs K[512:528)  →  peer_scores
  → 更新 running_max, running_sum
  → 更新 output
收到 peer block B33 (K[528:544), V[528:544)):
  → 同上
...
```

### 5.4 GQA 优化边界

vLLM 内部 KV cache 按 `num_kv_heads`（如 2）存储，attention 时 broadcast 到 `num_heads`（如 14）。

**关键决策**：在 GQA broadcast **之前** 还是 **之后** 做 ring 交换？

| 方案 | 传输体积 | 后续处理 |
|------|---------|---------|
| **GQA 前交换**（推荐） | 48MB → **~7MB** (7× 减少) | 收到后 broadcast 到 14 heads，再做 online softmax |
| **GQA 后交换** | 48MB | 直接 online softmax |

原始 Ring Attention 论文也在 GQA 之前分片。HCP Rust 端的 `repeat_kv` 是在收到 KV block 后做的，所以协议层应该传 `num_kv_heads` 格式。

---

## 6. 修正 VLLM_INTEGRATION.md 的错误方向

之前文档 §3.1 的 `gather_kv_from_block_table` 和 §3.2 的 `HcpVllmAttentionMerger`（后处理模式）都是**次优路径**：

- `gather_kv_from_block_table`：破坏了 PagedAttention 的设计
- `HcpVllmAttentionMerger`：每层两次 forward，计算量翻倍

**正确方向**：
1. **不改 vLLM attention 层** — 让 vLLM 正常通过 block table 索引
2. **改 CacheEngine** — 支持远程 block 接收和索引
3. **改 block allocation** — Coordinator 全局协调 block index
4. **online softmax 在 kernel 内部** — 或作为 kernel wrapper

---

## 7. 路线图修正

| 阶段 | 之前方向 | 修正后方向 |
|------|---------|-----------|
| **Phase 3.4 短期** | 从 vLLM `LLM` API 提取连续 KV tensor | **弃用**。回到 transformers backend 验证真实 KV + online softmax correctness |
| **Phase 3.4 中期** | 深入 `CacheEngine` 底层 API | transformers 路径 correctness 验证通过后，再评估 vLLM Block-Aware Ring |
| **Phase 4 长期** | vLLM 后处理模式（每层两次 forward） | **Block-Aware Ring**：改 CacheEngine + 全局 block index 协调 |
| **Phase 5 远景** | 多框架混合 | 保留。Mac 端用 transformers/MLX，GPU 端用 vLLM Block-Aware Ring |

---

## 8. 一句话总结

> **Block 和 Ring 不是二选一，而是同一枚硬币的两面。**
> 
> vLLM 的 PagedAttention block 是内存管理的最优解，Ring Attention 的 KV 交换是跨节点扩展的最优解。
> 让 ring 交换的粒度 = block 大小，二者协同，才是正确的融合方式。
