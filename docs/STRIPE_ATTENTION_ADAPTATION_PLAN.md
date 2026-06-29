# HCP Striped Attention 适配计划

> 目标：将 Striped Attention（Brandon et al., arXiv:2311.09431）引入 HCP，以缓解因果 attention 在异构 Ring Attention 中的负载不均。

---

## 1. 为什么优先 Striped Attention

- **P2P-only 友好**：与原始 Ring Attention 一样，只依赖 Q 固定 + KV 沿 ring P2P 传递，不需要 all-to-all / all-gather。
- **异构负载均衡**：把每个 domain 的 token 均匀散布在原始序列中，使每轮计算的上三角 mask 比例大致相同，避免小/慢 domain 在 Phase 2 成为瓶颈。
- **实现改动小**：本质是一次输入 permutation + mask 调整；online softmax 和传输逻辑不变。
- **训练收益对 HCP 意义有限**：论文面向训练，HCP 当前聚焦推理，且消费级异构互联不适合训练；因此只取推理价值。

---

## 2. Striped Attention 核心机制回顾

- **Permutation**：device `i` 持有原始序列中下标满足 `index % N == i` 的 token。等价于把序列重排为 `[device0_tokens, device1_tokens, ..., device(N-1)_tokens]`，每个 device 在重排后的序列中占一段连续 chunk。
- **Mask 调整**：因果 mask 仍基于**原始序列顺序**。在每次 block-wise attention 中，比较 query/key 的原始位置来决定 mask，而不是用本地 block 的三角 mask。
- **Workload**：每 device 每轮计算量从 Ring 的 `c²` 上限降到约 `c²/2`（理论极限 2× speedup）。

---

## 3. HCP 适配设计

### 3.1 数据流改动

```text
原始 prompt tokens:  [t0, t1, t2, t3, t4, t5, ...]
原始 position ids:   [0,  1,  2,  3,  4,  5,  ...]

permute schedule (N=2, uniform stripe):
  device 0: [t0, t2, t4, ...]   -> chunk_0 in permuted space
  device 1: [t1, t3, t5, ...]   -> chunk_1 in permuted space

permuted tokens:     [t0, t2, t4, ..., t1, t3, t5, ...]
permuted positions:  [0,  2,  4,  ..., 1,  3,  5,  ...]

domain 0 Q/K/V: first chunk_0 entries of permuted sequence
domain 1 Q/K/V: next  chunk_1 entries of permuted sequence
```

### 3.2 不均等 chunk 的推广

HCP 需要 capacity-aware 不均等分片（如 3:1）。直接把 `% N` 会强制均分。解决方案：

- 使用**细粒度 scheduling unit**（如 1 个 token 或一个固定小 block）。
- 按容量比例循环分配 scheduling unit：例如 3:1 比例下，模式为 `[0,0,0,1,0,0,0,1,...]`。
- 这样每个 device 的 token 数可以任意比例，同时仍近似均匀散布在原始序列中。
- 当 scheduling unit 远小于序列长度时，负载均衡性质与均匀 stripe 几乎一致。

### 3.3 需要修改的模块

| 模块 | 改动 |
|------|------|
| `distributed_coordinator.rs` | 在分配 chunk 后生成 `permutation` / `inverse_permutation` 和 `permuted_position_ids`，随 `Prefill` 命令下发。 |
| `model/attention.rs` / `backend.rs` | 用**原始位置 id** 计算 causal mask，而不是本地 block 的下三角/上三角 mask。 |
| `distributed_worker.rs` / `infer.rs` | 在 embedding 层之前对 input token ids 和 position ids 应用 permutation；在输出层之后做 inverse permutation。 |
| `kv_transport.rs` / `quic_transport.rs` | 不变；仍按 permuted chunk 顺序传递 K/V block。 |
| `correctness.rs` | 新增 striped correctness case：比较 HCP-striped 与 reference attention（原始顺序）。 |

### 3.4 Mask 实现细节

HCP 目前使用 local block index 判断 causal。改为：

```rust
// q_pos / k_pos: [block_size] 原始全局位置
let mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1); // true -> mask out
```

这天然支持任意 permutation，包括 striped 和非均等 scheduling。

### 3.5 Decode 阶段的特殊处理

- 新 token 总是追加到原始序列末尾，其原始位置为 `global_seq_len`。
- 根据 scheduling 规则决定该 token 归属哪个 domain，并只在该 domain 的 KV cache 中追加。
- 采样时只需要所有 domain 中对应**最后原始位置**的 logits；coordinator 负责 inverse-permute 后取最终 token。

---

## 4. 实现步骤（建议顺序）

1. **Step 0 — 在 correctness model 中验证 permutation 正确性**
   - 用纯 Python / Rust correctness test 生成 permuted Q/K/V，按位置比较 mask，验证输出与 reference attention 一致。
   - 先做单设备模拟，确认 online softmax + striped mask 数值正确。

2. **Step 1 — coordinator permutation 生成**
   - 给定 `chunk_sizes` 和 `global_seq_len`，生成 `permuted_position_ids` 和 `inverse_permutation`。
   - 支持 uniform stripe 和比例 stripe 两种 schedule。

3. **Step 2 — worker 输入/输出 permutation**
   - 在 prefill 前对 token ids / position ids 应用 permutation。
   - 在 attention output 之后、LM head 之前做 inverse permutation。

4. **Step 3 — attention backend 改位置 mask**
   - 将 `backend.rs` 中的 causal mask 从 block-index-based 改为 `orig_position`-based。
   - 注意与 RoPE 的交互：RoPE 必须在 original position 上计算，而不是 permuted index。

5. **Step 4 — 分布式 smoke**
   - 2-domain uneven stripe（如 3:1）与 vanilla Ring Attention 对比 correctness。
   - 测量每 domain 每轮实际计算时间，验证负载均衡改善。

6. **Step 5 — 性能评估**
   - 在不同 seq len 和 chunk 比例下跑 benchmark，量化 striped 对 pearl 类慢节点的收益。
   - 若收益显著，再考虑与 Ring Flash Attention kernel 集成。

---

## 5. 风险与注意事项

| 风险 | 说明 | 缓解 |
|------|------|------|
| RoPE/位置编码出错 | permutation 后容易把 permuted index 当位置 | 显式传递并检查 `permuted_position_ids` |
| Decode 采样位置错乱 | 需要 inverse permutation 后再取最后 token | coordinator 层统一做 inverse permute |
| 非均等 stripe 负载不均衡 | scheduling unit 太大时分布不再均匀 | 使用 token-level 或很小 block-level scheduling |
| FlashAttention kernel 不支持任意位置 mask | HCP 当前用 tch-rs 逐元素 mask；kernel 集成需额外工作 | 先 correctness 验证，kernel 优化后置 |
| 与现有 `--chunk-sizes` 语义冲突 | 当前 chunk 是连续序列段；stripe 后 chunk 是 permuted 段 | 在 coordinator 中把 `chunk_sizes` 解释为 permuted-space 长度 |

---

## 6. 成功标准

- `cargo test` 新增 striped correctness case 通过，float32 diff ≤ 1e-5。
- 2-domain uneven stripe smoke（white+pearl 或本地 loopback）exit=0，token 输出一致。
- 理论/模拟证明 pearl 类慢节点在 striped 下的 per-round workload 方差显著低于 vanilla ring。

---

## 7. 参考

- Brandon et al., *Striped Attention: Faster Ring Attention for Causal Transformers*, arXiv:2311.09431.
- Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*, arXiv:2310.01889.
- zhuzilin/ring-flash-attention (GitHub)：提供 ring / zigzag / stripe Flash Attention 实现参考。
