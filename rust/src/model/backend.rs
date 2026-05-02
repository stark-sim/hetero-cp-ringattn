use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};
#[cfg(feature = "tch-backend")]
use crate::model::kv_transport::{KvBlock, KvTransport};

/// Trait for attention computation backends.
#[cfg(feature = "tch-backend")]
pub trait AttentionBackend {
    /// Forward pass: compute attention output for the given hidden states.
    ///
    /// `hidden_states`: `[batch, seq_len, hidden_size]`
    /// `position_ids`: `[batch, seq_len]` (Int64)
    /// `kv_cache`: Optional KV cache for autoregressive decoding
    /// `attention_mask`: Optional causal mask for prefill (shape `[1, 1, seq_len, seq_len]`)
    ///
    /// Returns: `[batch, seq_len, hidden_size]`
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError>;

    /// Optional: configure distributed transport, domain id, and sequence offset.
    /// Only `HcpRingAttentionBackend` implements this; others are no-ops.
    #[cfg(feature = "tch-backend")]
    #[allow(dead_code)]
    fn set_distributed(&mut self, _domain_id: usize, _seq_offset: usize, _transport: Option<Box<dyn KvTransport>>) {
        // Local backend 不需要分布式配置，noop
    }
}

/// Local (non-distributed) attention backend using standard GQA.
#[cfg(feature = "tch-backend")]
pub struct LocalAttentionBackend {
    pub attention: super::layers::GqaAttention,
}

#[cfg(feature = "tch-backend")]
impl AttentionBackend for LocalAttentionBackend {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        self.attention.forward(hidden_states, position_ids, kv_cache, attention_mask)
    }
}

/// Ring-attention backend that splits sequence into chunks and computes
/// attention via online softmax over K/V blocks.
///
/// This is a single-process simulation of multi-domain ring attention.
/// In Phase 3 it will be extended to true multi-process / multi-node.
#[cfg(feature = "tch-backend")]
pub struct HcpRingAttentionBackend {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    rope: super::layers::RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    num_domains: usize,
    layer_idx: usize,
    #[cfg(feature = "tch-backend")]
    kv_transport: Option<Box<dyn KvTransport>>,
    #[allow(dead_code)]
    local_domain_id: usize,
    seq_offset: usize,
    /// Length of KV cache produced during prefill. In decode phase we only send
    /// the prefill partition (not decode-appended tokens) to peers.
    prefill_kv_len: usize,
    /// Whether the first forward (prefill) has completed.
    is_prefill_done: bool,
}

#[cfg(feature = "tch-backend")]
impl HcpRingAttentionBackend {
    pub fn from_weights(
        weights: &super::ModelWeights,
        layer: usize,
        config: &super::ModelConfig,
        rope: &super::layers::RotaryEmbedding,
        num_domains: usize,
    ) -> Result<Self, ModelError> {
        let q_bias = weights.get(&super::WeightNames::q_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        let k_bias = weights.get(&super::WeightNames::k_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        let v_bias = weights.get(&super::WeightNames::v_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        Ok(Self {
            q_proj: weights.get(&super::WeightNames::q_proj_weight(layer))?.shallow_clone(),
            k_proj: weights.get(&super::WeightNames::k_proj_weight(layer))?.shallow_clone(),
            v_proj: weights.get(&super::WeightNames::v_proj_weight(layer))?.shallow_clone(),
            o_proj: weights.get(&super::WeightNames::o_proj_weight(layer))?.shallow_clone(),
            q_bias,
            k_bias,
            v_bias,
            rope: rope.clone(),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads(),
            head_dim: config.head_dim(),
            scale: 1.0 / (config.head_dim() as f64).sqrt(),
            num_domains: num_domains.max(1),
            layer_idx: layer,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        })
    }

    #[cfg(feature = "tch-backend")]
    #[allow(dead_code)]
    pub fn set_transport(&mut self, transport: Box<dyn KvTransport>) {
        self.kv_transport = Some(transport);
    }

    #[allow(dead_code)]
    pub fn set_local_domain_id(&mut self, id: usize) {
        self.local_domain_id = id;
    }

    #[allow(dead_code)]
    pub fn set_seq_offset(&mut self, offset: usize) {
        self.seq_offset = offset;
    }

    /// 【处理单个 KV block】用一组 Q 去和一组 K/V 做 attention，然后更新 online softmax 状态。
    /// 
    /// 所有位置参数都必须是【全局位置】，不能是本地索引。
    /// 例如 domain1 的本地索引 0 对应全局位置 8。
    #[allow(clippy::too_many_arguments)]
    fn process_kv_block(
        &self,
        q_chunk: &Tensor,            // 当前 Q chunk，shape: [batch, num_heads, q_chunk_len, head_dim]
        q_global_start: usize,       // 这个 Q chunk 在全局序列中的起始位置
        q_global_end: usize,         // 这个 Q chunk 在全局序列中的结束位置
        k_chunk: &Tensor,            // 当前 K block，shape: [batch, num_heads, kv_chunk_len, head_dim]
        v_chunk: &Tensor,            // 当前 V block，shape: [batch, num_heads, kv_chunk_len, head_dim]
        kv_global_start: usize,      // 这个 K/V block 在全局序列中的起始位置
        kv_global_end: usize,        // 这个 K/V block 在全局序列中的结束位置
        rm: &mut Tensor,             // 【running max】当前见过的最大 score（可变的引用）
        rs: &mut Tensor,             // 【running sum】当前 softmax 分母的累加和
        obh: &mut Tensor,            // 【output buffer】当前加权累加的输出
        apply_causal_mask: bool,     // 【是否应用因果掩码】true=因果路径, false=非因果路径
    ) {
        let kv_chunk_len = (kv_global_end - kv_global_start) as i64;

        // ====== Early Return 优化（仅因果路径）======
        // 因果 Attention 的规则：当前 token 只能看到自己和过去的 token，不能看到未来的 token。
        // 如果 kv_global_start >= q_global_end，说明这个 KV block 的所有位置都在 Q chunk 的右边，
        // 对当前 Q 来说全部是"未来"，应该被 mask 掉。直接 return 可以省掉一次矩阵乘法。
        // 
        // 非因果路径（如 protocol smoke）没有"未来"概念，所有 KV 都应该被处理，不能跳过。
        if apply_causal_mask && (kv_chunk_len <= 0 || kv_global_start >= q_global_end) {
            return;
        }
        // 非因果路径下，空 block 仍然跳过。
        if kv_chunk_len <= 0 {
            return;
        }

        // ====== 第一步：计算 Attention Scores ======
        // Attention 的核心公式：score = Q @ K^T / sqrt(head_dim)
        // 
        // q_chunk shape: [batch, num_heads, q_chunk_len, head_dim]
        // k_chunk shape: [batch, num_heads, kv_chunk_len, head_dim]
        // k_chunk.transpose(2, 3) → [batch, num_heads, head_dim, kv_chunk_len]
        // matmul 结果 shape: [batch, num_heads, q_chunk_len, kv_chunk_len]
        // 最后乘 self.scale（即 1/sqrt(head_dim)）做缩放，防止数值过大导致 softmax 梯度消失。
        let mut scores = q_chunk.matmul(&k_chunk.transpose(2, 3)) * self.scale;

        // 用全局位置构造因果掩码，确保当前 token 看不到未来的 token。
        // 
        // 原理：对于每个 query 位置 i 和 key 位置 j，如果 i >= j 则允许 attention，否则 mask 为 -inf。
        // 
        // q_pos: 当前 Q chunk 的全局位置，shape [1, 1, q_chunk_len, 1]
        // k_pos: 当前 K/V block 的全局位置，shape [1, 1, 1, kv_chunk_len]
        // causal = q_pos.ge_tensor(&k_pos): element-wise 比较，返回 bool tensor。
        // masked_fill(&causal.logical_not(), NEG_INFINITY):
        //   把 causal=False 的位置填入负无穷，softmax 后权重变为 0。
        if apply_causal_mask {
            let q_pos = Tensor::arange_start(
                q_global_start as i64,
                q_global_end as i64,
                (Kind::Int64, q_chunk.device()),
            )
            .unsqueeze(1)
            .unsqueeze(0)
            .unsqueeze(0);
            let k_pos = Tensor::arange_start(
                kv_global_start as i64,
                kv_global_end as i64,
                (Kind::Int64, q_chunk.device()),
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0);
            let causal = q_pos.ge_tensor(&k_pos);
            scores = scores.masked_fill(&causal.logical_not(), f64::NEG_INFINITY);
        }

        // ====== 第三步：Online Softmax 更新 ======
        // Ring Attention 的核心技巧：不需要等所有 KV block 都处理完再算 softmax，
        // 而是每处理一个 KV block 就更新一次结果，始终保持正确的 softmax 输出。
        //
        // 标准 softmax 公式：
        //   softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
        //
        // 但如果直接对很多个 block 分别做 softmax 再平均，结果是错的。
        // Online softmax 的关键：维护一个"全局最大值"和"全局累加和"，
        // 每来一个新 block，就根据新的最大值重新调整之前的累加和。
        //
        // 数学推导：
        // 假设之前处理了一些 KV，得到：
        //   prev_max = max(prev_scores)
        //   prev_sum = sum(exp(prev_scores - prev_max))
        //   prev_out = sum(exp(prev_scores - prev_max) * V_prev) / prev_sum
        //
        // 现在来了一个新 block，得到：
        //   local_max = max(local_scores)
        //   local_sum = sum(exp(local_scores - local_max))
        //   local_pv = sum(exp(local_scores - local_max) * V_local)
        //
        // 合并后的全局最大值：new_max = max(prev_max, local_max)
        //
        // 为了合并，需要把之前的值"对齐"到新的最大值：
        //   exp_prev = exp(prev_max - new_max)   （之前的值需要缩小多少）
        //   exp_local = exp(local_max - new_max) （新值需要缩小多少）
        //
        // 新的累加和：new_sum = exp_prev * prev_sum + exp_local * local_sum
        //
        // 新的输出：
        //   new_out = (exp_prev * prev_sum * prev_out + exp_local * local_pv) / new_sum
        //           = (exp_prev.unsqueeze(3) * rs.unsqueeze(3) * obh + exp_local.unsqueeze(3) * local_pv) / new_sum.unsqueeze(3)

        // local_max: 当前 KV block 中每个 query 的最大 score。
        // max_dim(3, false): 在第 3 维（kv_chunk_len 维）上取最大值，不保留维度。
        // shape: [batch, num_heads, q_chunk_len]
        let (local_max, _) = scores.max_dim(3, false);

        // weights: 当前 block 的 score 减去最大值后取指数。
        // 减最大值是为了数值稳定性（防止 exp 爆炸）。
        // shape: [batch, num_heads, q_chunk_len, kv_chunk_len]
        let weights = (&scores - local_max.unsqueeze(3)).exp();

        // local_sum: 当前 block 的权重之和（softmax 的分母的一部分）。
        // sum_dim_intlist(&[3i64][..], false, Kind::Float): 在第 3 维求和。
        // shape: [batch, num_heads, q_chunk_len]
        let local_sum = weights.sum_dim_intlist(&[3i64][..], false, Kind::Float);

        // local_pv: 当前 block 的加权 Value 之和（softmax 的分子的一部分）。
        // weights [batch, num_heads, q_chunk_len, kv_chunk_len] @ V [batch, num_heads, kv_chunk_len, head_dim]
        // = [batch, num_heads, q_chunk_len, head_dim]
        let local_pv = weights.matmul(v_chunk);

        // new_max: 全局最大值 = max(之前见过的最大值, 当前 block 的最大值)
        let new_max = rm.max_other(&local_max);

        // exp_prev: 把之前的状态缩放到新的最大值下。
        // &*rm 中的 * 是解引用（因为 rm 是 &mut Tensor，需要解引用才能做减法）。
        let exp_prev = (&*rm - &new_max).exp();

        // exp_local: 把当前 block 的状态缩放到新的最大值下。
        let exp_local = (&local_max - &new_max).exp();

        // new_sum: 合并后的 softmax 分母。
        // &*rs 同理，解引用 rs（&mut Tensor）得到 Tensor。
        let new_sum = &exp_prev * &*rs + &exp_local * &local_sum;

        // 更新输出 obh。
        // 公式：new_out = (exp_prev * prev_sum * prev_out + exp_local * local_pv) / new_sum
        // unsqueeze(3) 把 [batch, num_heads, q_chunk_len] 变成 [batch, num_heads, q_chunk_len, 1]，
        // 以便和 [batch, num_heads, q_chunk_len, head_dim] 做 element-wise 乘法（广播）。
        *obh = (&exp_prev.unsqueeze(3) * &rs.unsqueeze(3) * &*obh
            + &exp_local.unsqueeze(3) * &local_pv)
            / &new_sum.unsqueeze(3);

        // 更新 running max 和 running sum，供下一个 KV block 使用。
        *rm = new_max;
        *rs = new_sum;
    }

    /// 【Ring Attention 核心算法】
    /// 
    /// 传统 Attention 需要一次性加载整个 K/V 矩阵到显存，当序列很长时（如 1M tokens）会爆显存。
    /// Ring Attention 的思想：把 Q 切成多个小 chunk，把 K/V 也切成多个 block，
    /// 一次只处理一个 Q chunk + 一个 K/V block，用 online softmax 逐步合并结果。
    /// 
    /// 在分布式场景下，每个 domain 只持有本地序列的 K/V，peer domain 的 K/V 需要通过网络传输获取。
    fn ring_attention(
        &mut self,
        q: &Tensor,                  // Query，shape: [batch, num_heads, seq_len, head_dim]
        k: &Tensor,                  // Key（GQA repeat 后），shape: [batch, num_heads, seq_len, head_dim]
        v: &Tensor,                  // Value（GQA repeat 后），shape: [batch, num_heads, seq_len, head_dim]
        attention_mask: Option<&Tensor>, // 因果掩码，Some 表示 prefill 阶段，None 表示非因果测试
        global_seq_start: usize,     // 本地序列在全局序列中的起始位置（domain0=0, domain1=8）
    ) -> Tensor {
        let batch = q.size()[0];
        let num_heads = q.size()[1];
        let seq_len = q.size()[2];
        let head_dim = q.size()[3];

        // 如果只有一个 domain（非分布式），直接用本地 attention。
        // 注意：decode 阶段（seq_len == 1）在分布式场景下也必须走 ring 路径，
        // 因为每个 domain 只持有部分 KV cache，需要交换 peer KV 才能计算完整 attention。
        if self.num_domains == 1 {
            return self.local_attention_scores(q, k, v, attention_mask);
        }

        // ====== 第一步 + 第二步：Ring KV 交换 ======
        // 在真正的 N-domain ring 中，每个 domain 只与 next/prev 两个邻居直接通信。
        // 每个 domain 把自己的 KV 发给 next，从 prev 接收 KV；收到的 KV 在下一轮再转发给 next。
        // 经过 num_domains - 1 轮后，每个 domain 都收到了所有其他 domain 的 KV。
        //
        // 例如 3-domain (0→1→2→0):
        //   Round 0: 0 发 K0→1，收 K2；1 发 K1→2，收 K0；2 发 K2→0，收 K1
        //   Round 1: 0 发 K2→1，收 K1；1 发 K0→2，收 K2；2 发 K1→0，收 K0
        //   最终每个 domain 持有 K0/K1/K2 全部三块。
        //
        // 【decode 阶段特殊处理】seq_len == 1 时，只发送 prefill 阶段产生的 KV 分区
        //（记录在 self.prefill_kv_len 中），decode 新 token 的 KV 完全由每个节点本地持有。
        let mut peer_blocks: Vec<super::kv_transport::KvBlock> = Vec::new();

        if let Some(ref mut transport) = self.kv_transport {
            let (k_to_send, v_to_send, send_seq_end) = if seq_len == 1 {
                let history_len = self.prefill_kv_len as i64;
                (
                    k.narrow(2, 0, history_len),
                    v.narrow(2, 0, history_len),
                    global_seq_start + self.prefill_kv_len,
                )
            } else {
                (k.shallow_clone(), v.shallow_clone(), global_seq_start + k.size()[2] as usize)
            };

            let mut current_block = KvBlock {
                layer_idx: self.layer_idx,
                global_seq_start,
                global_seq_end: send_seq_end,
                k: k_to_send,
                v: v_to_send,
            };

            for round in 0..self.num_domains.saturating_sub(1) {
                if let Err(e) = transport.send_kv_block(&current_block) {
                    eprintln!("[ring_attention] round {round} send_kv_block failed: {e}");
                    break;
                }

                match transport.recv_kv_block() {
                    Ok(Some(peer_block)) => {
                        peer_blocks.push(peer_block.clone());
                        current_block = peer_block;
                    }
                    Ok(None) => {
                        eprintln!("[ring_attention] round {round} recv_kv_block returned None");
                        break;
                    }
                    Err(e) => {
                        eprintln!("[ring_attention] round {round} recv_kv_block failed: {e}");
                        break;
                    }
                }
            }
        }

        // ====== 第三步：确定 chunk 大小 ======
        let q_chunk_size = (seq_len as usize).div_ceil(self.num_domains).max(1);
        let kv_chunk_size = q_chunk_size;

        // 存储每个 Q chunk 的输出，最后 cat 拼接。
        let mut outputs = Vec::new();

        // ====== 第四步：逐个处理 Q chunk ======
        // 外层循环遍历每个 Q chunk（比如 [0,4) 和 [4,8)）。
        for q_start in (0..seq_len as usize).step_by(q_chunk_size) {
            let q_end = (q_start + q_chunk_size).min(seq_len as usize);
            let q_chunk_len = (q_end - q_start) as i64;

            // narrow(2, start, len) 在第 2 维（seq_len 维）上截取一段。
            // q_chunk shape: [batch, num_heads, q_chunk_len, head_dim]
            let q_chunk = q.narrow(2, q_start as i64, q_chunk_len);

            // 把本地 K/V 也切成多个 block，方便逐个处理。
            // 注意：KV 的长度可能与 Q 不同（decode 阶段 Q len=1，但 KV cache 很长）。
            // 因此用 k.size()[2]（本地 KV 的 seq_len 维）而不是 q 的 seq_len。
            let local_kv_len = k.size()[2] as usize;
            let kv_chunks: Vec<(usize, usize)> = (0..local_kv_len)
                .step_by(kv_chunk_size)
                .map(|start| (start, (start + kv_chunk_size).min(local_kv_len)))
                .collect();

            // 区分两种路径：
            // - attention_mask.is_some()：因果 prefill，用纯 tensor 在线 softmax（在 GPU 上执行）。
            // - attention_mask.is_none()：非因果路径（协议 smoke 测试），用 CPU buffer。
            if attention_mask.is_some() {
                // ====== Online Softmax 状态初始化 ======
                // Ring Attention 的核心是 online softmax：不需要一次性看到所有 K/V，
                // 而是逐块处理，每处理一块就更新当前的最佳估计。
                // 
                // 三个状态变量：
                // - rm (running max): 当前见过的最大 score 值，shape [batch, num_heads, q_chunk_len]
                //   初始化为负无穷，表示还没有处理任何 KV block。
                // - rs (running sum): 当前 softmax 分母的累加和，shape 同上。
                //   初始化为 0。
                // - obh (output buffer head): 当前加权累加的输出，shape [batch, num_heads, q_chunk_len, head_dim]
                //   初始化为 0。
                let mut rm = Tensor::full(
                    [batch, num_heads, q_chunk_len],
                    f64::NEG_INFINITY,
                    (Kind::Float, q.device()),
                );
                let mut rs = Tensor::zeros([batch, num_heads, q_chunk_len], (Kind::Float, q.device()));
                let mut obh = Tensor::zeros(
                    [batch, num_heads, q_chunk_len, head_dim],
                    (Kind::Float, q.device()),
                );

                // 计算当前 Q chunk 的全局位置范围。
                // 例如 domain1 的 q_start=0 → q_global_start=8+0=8，q_global_end=8+4=12。
                let q_global_start = global_seq_start + q_start;
                let q_global_end = global_seq_start + q_end;

                // 4a. 先处理【本地 KV blocks】
                // 本地 KV 是当前 domain 自己计算的 K/V，不需要网络传输。
                for (kv_start, kv_end) in &kv_chunks {
                    let kv_chunk_len = (*kv_end - *kv_start) as i64;

                    // 从本地 K/V tensor 中截取当前 block
                    let k_chunk = k.narrow(2, *kv_start as i64, kv_chunk_len);
                    let v_chunk = v.narrow(2, *kv_start as i64, kv_chunk_len);

                    // 调用 process_kv_block 更新 online softmax 状态。
                    // 注意：kv_start/kv_end 是本地索引，需要加上 global_seq_start 才能得到全局位置。
                    self.process_kv_block(
                        &q_chunk, q_global_start, q_global_end,
                        &k_chunk, &v_chunk,
                        global_seq_start + kv_start, global_seq_start + kv_end,
                        &mut rm, &mut rs, &mut obh,
                        true,  // 因果路径
                    );
                }

                // 4b. 再处理【peer KV blocks】
                // peer KV 是从其他 domain 通过网络接收到的 K/V。
                // 这些 block 已经被预取到 peer_blocks 向量中，所有 Q chunk 共享。
                for peer_block in &peer_blocks {
                    self.process_kv_block(
                        &q_chunk, q_global_start, q_global_end,
                        &peer_block.k, &peer_block.v,
                        peer_block.global_seq_start, peer_block.global_seq_end,
                        &mut rm, &mut rs, &mut obh,
                        true,  // 因果路径
                    );
                }

                // 这个 Q chunk 的处理完成，obh 就是该 chunk 的 attention 输出。
                outputs.push(obh); // shape: [batch, num_heads, q_chunk_len, head_dim]
            } else {
                // ====== 非因果路径（protocol smoke / 全可见 attention）======
                // 非因果路径意味着每个 Q 都能看到所有 K/V（没有 causal mask）。
                // 之前这里调用 C++ compute_chunk_attention_step（CPU buffer 方式），
                // 现在改为和因果路径相同的纯 tch tensor online softmax，全程在设备上执行。
                let mut rm = Tensor::full(
                    [batch, num_heads, q_chunk_len],
                    f64::NEG_INFINITY,
                    (Kind::Float, q.device()),
                );
                let mut rs = Tensor::zeros([batch, num_heads, q_chunk_len], (Kind::Float, q.device()));
                let mut obh = Tensor::zeros(
                    [batch, num_heads, q_chunk_len, head_dim],
                    (Kind::Float, q.device()),
                );

                // 非因果路径下，所有本地 KV 和 peer KV 都是可见的。
                // global_seq_start 对 non-causal 没有实际意义（因为不用 causal mask），
                // 但为了接口统一，传入 0。
                for (kv_start, kv_end) in &kv_chunks {
                    let kv_chunk_len = (*kv_end - *kv_start) as i64;
                    let k_chunk = k.narrow(2, *kv_start as i64, kv_chunk_len);
                    let v_chunk = v.narrow(2, *kv_start as i64, kv_chunk_len);
                    self.process_kv_block(
                        &q_chunk, 0, seq_len as usize,
                        &k_chunk, &v_chunk,
                        0, seq_len as usize,
                        &mut rm, &mut rs, &mut obh,
                        false,  // 非因果路径，不应用 causal mask
                    );
                }

                for peer_block in &peer_blocks {
                    self.process_kv_block(
                        &q_chunk, 0, seq_len as usize,
                        &peer_block.k, &peer_block.v,
                        0, seq_len as usize,
                        &mut rm, &mut rs, &mut obh,
                        false,  // 非因果路径
                    );
                }

                outputs.push(obh);
            }
        }

        // ====== 第五步：拼接所有 Q chunk 的输出 ======
        // 无论因果还是非因果路径，每个 output 的 shape 都是 [batch, num_heads, q_chunk_len, head_dim]。
        // 在第 2 维（seq_len 维）上拼接，恢复完整的 seq_len。
        Tensor::cat(&outputs, 2)
    }

    /// Standard local attention for short sequences or single-token decode.
    fn local_attention_scores(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let scores = q.matmul(&k.transpose(2, 3)) * self.scale;

        let scores = if let Some(mask) = attention_mask {
            scores + mask
        } else {
            scores
        };

        let attn_weights = scores.softmax(-1, Kind::Float);
        attn_weights.matmul(v)
    }
}

#[cfg(feature = "tch-backend")]
impl AttentionBackend for HcpRingAttentionBackend {
    fn set_distributed(&mut self, domain_id: usize, seq_offset: usize, transport: Option<Box<dyn KvTransport>>) {
        self.local_domain_id = domain_id;
        self.seq_offset = seq_offset;
        self.kv_transport = transport;
    }
    fn forward(
        &mut self,
        hidden_states: &Tensor,      // 【输入】上一层的输出，shape: [batch, seq_len, hidden_size]
        position_ids: &Tensor,       // 【位置编码】每个 token 在完整序列中的绝对位置，shape: [batch, seq_len]
        kv_cache: Option<&mut crate::model::cache::KvCache>, // 【KV 缓存】用于自回归生成时复用之前的 K/V
        attention_mask: Option<&Tensor>, // 【注意力掩码】因果掩码，防止当前 token 看到未来的 token
    ) -> Result<Tensor, ModelError> {
        // 获取输入的各个维度大小
        // batch: 批量大小（一次处理多少条句子，测试里通常是 1）
        // seq_len: 当前输入的 token 数量（比如 8）
        // hidden_size: 隐藏层维度（比如 32）
        let batch = hidden_states.size()[0];
        let seq_len = hidden_states.size()[1];
        let hidden_size = hidden_states.size()[2];

        // ====== 第一步：线性投影（Linear Projection）======
        // Attention 的核心思想：用三个不同的权重矩阵把输入 hidden_states 映射成 Q、K、V。
        // 
        // Q（Query）：当前 token 的"查询向量"，用来问"我和哪些过去的 token 相关？"
        // K（Key）：每个 token 的"关键词向量"，用来回答查询
        // V（Value）：每个 token 的"价值向量"，最终加权平均的就是 V
        //
        // 数学上：q = hidden_states @ W_q^T
        // matmul 是矩阵乘法，transpose(0, 1) 把权重矩阵转置（因为 PyTorch/HF 格式是 [out, in]）
        let mut q = hidden_states.matmul(&self.q_proj.transpose(0, 1));
        if let Some(ref bias) = self.q_bias { q += bias; }  // 如果有偏置，加上去
        let mut k = hidden_states.matmul(&self.k_proj.transpose(0, 1));
        if let Some(ref bias) = self.k_bias { k += bias; }
        let mut v = hidden_states.matmul(&self.v_proj.transpose(0, 1));
        if let Some(ref bias) = self.v_bias { v += bias; }

        // ====== 第二步：reshape 成多头格式 ======
        // 原始 shape: [batch, seq_len, num_heads * head_dim]
        // 先 view 成: [batch, seq_len, num_heads, head_dim]
        // 再 transpose(1, 2) 交换第 1、2 维，变成: [batch, num_heads, seq_len, head_dim]
        // 
        // 这样每个 head 独立处理一部分维度，可以并行计算。
        let q = q.view([batch, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let k = k.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let v = v.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // ====== 第三步：RoPE（旋转位置编码）======
        // RoPE 给 Q 和 K 注入位置信息，让模型知道 token 的先后顺序。
        // 它根据 position_ids（全局位置）对 Q/K 的每一对维度做旋转变换。
        // 旋转角度取决于位置：位置越远，旋转角度越大。
        let (q, k) = self.rope.apply(&q, &k, Some(position_ids));

        // ====== 第四步：更新 KV Cache（可选）======
        // 在自回归生成时，每次只输入一个新 token，但需要看到之前所有 token 的 K/V。
        // KV Cache 把之前的 K/V 存起来，新 token 的 K/V append 到后面，避免重复计算。
        // 
        // 在 prefill 阶段（第一次处理完整 prompt），kv_cache 为 None，直接返回当前 K/V。
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k.shallow_clone(), v.shallow_clone())
        };

        // 记录 prefill 阶段的 KV 长度。decode 阶段发送 peer KV 时，
        // 只发送 prefill 分区，不包含 decode 阶段 append 的新 token。
        if !self.is_prefill_done {
            self.prefill_kv_len = k.size()[2] as usize;
            self.is_prefill_done = true;
        }

        // ====== 第五步：GQA 头重复 ======
        // GQA（Group Query Attention）是一种节省显存的技术：
        // - 标准 MHA：num_kv_heads == num_heads（每个 query head 对应一个 key/value head）
        // - GQA：num_kv_heads < num_heads（多个 query head 共享同一个 key/value head）
        //
        // 例如 num_heads=4, num_kv_heads=1：
        // - 4 个 query head，但只有一个 key head 和一个 value head
        // - 为了让矩阵乘法维度匹配，需要把 K/V 在 head 维度上重复 4 次
        // - repeat(&[1, 4, 1, 1]) 表示：batch 不重复，head 重复 4 次，seq_len 不重复，head_dim 不重复
        let num_rep = self.num_heads / self.num_kv_heads;
        let k = if num_rep > 1 {
            let shape = k.size();
            k.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            k
        };
        let v = if num_rep > 1 {
            let shape = v.size();
            v.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            v
        };

        // ====== 第六步：计算全局序列起始位置 ======
        // 在分布式场景下，domain1 处理的是完整序列的后半段（比如 [8, 16)）。
        // 但 causal mask 必须使用"全局位置"才能正确判断哪些 K 对当前 Q 可见。
        // 
        // 例如 domain1 的 position_ids = [8, 9, ..., 15]，min = 8。
        // global_seq_start = 8 意味着本地索引 0 对应全局位置 8。
        // 如果没有传输层（单进程本地模式），global_seq_start = 0。
        // 使用固定的 seq_offset 作为全局序列起始位置。
        // 在 prefill 阶段，seq_offset = domain_id * chunk_size（如 domain0=0, domain1=8）。
        // 在 decode 阶段，seq_offset 保持不变（domain0 始终负责全局位置 0..8 的 KV）。
        // 不能用 position_ids.min()，因为 decode 阶段 position_ids 是当前 token 的全局位置，
        // 与该 domain 负责的历史 KV 的起始位置不同。
        let global_seq_start = self.seq_offset;

        // ====== 第七步：Ring Attention（核心分布式注意力计算）======
        // 把 Q 切成多个 chunk，逐个处理本地 KV 和 peer KV，用 online softmax 合并结果。
        let attn_output = self.ring_attention(&q, &k, &v, attention_mask, global_seq_start);

        // ====== 第八步：输出投影（O-projection）======
        // attn_output 的 shape: [batch, num_heads, seq_len, head_dim]
        // transpose(1, 2) → [batch, seq_len, num_heads, head_dim]
        // view → [batch, seq_len, hidden_size]（把多头的结果拼接起来）
        // 最后再乘一个 o_proj 权重矩阵，映射回 hidden_size 维度
        let attn_output = attn_output.transpose(1, 2).contiguous().view([batch, seq_len, hidden_size]);
        Ok(attn_output.matmul(&self.o_proj.transpose(0, 1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "tch-backend")]
    fn create_test_attention(device: tch::Device) -> super::super::layers::GqaAttention {
        let hidden_size = 64i64;
        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;

        super::super::layers::GqaAttention {
            q_proj: Tensor::randn([(num_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            k_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            v_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            o_proj: Tensor::randn([hidden_size, (num_heads * head_dim) as i64], (Kind::Float, device)),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            num_heads,
            num_kv_heads,
            head_dim,
            rope: super::super::layers::RotaryEmbedding::new(head_dim, 128, 10000.0, device),
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Create a causal mask for local attention testing.
    #[cfg(feature = "tch-backend")]
    fn make_causal_mask(seq_len: i64, device: tch::Device) -> Tensor {
        let mask = Tensor::ones([seq_len, seq_len], (Kind::Float, device))
            .triu(1)
            .to_kind(Kind::Bool);
        Tensor::zeros([seq_len, seq_len], (Kind::Float, device))
            .masked_fill(&mask, f64::NEG_INFINITY)
            .unsqueeze(0)
            .unsqueeze(0)
    }

    /// 【验证 process_kv_block 非因果模式与标准 softmax 等价】
    ///
    /// 这个测试验证：当 apply_causal_mask=false 时，process_kv_block 的 online softmax
    /// 输出与直接 softmax(QK^T/sqrt(d))V 完全一致。
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_chunk_step_vs_softmax_single_block() {
        let device = tch::Device::Cpu;
        let query_len = 4i64;
        let block_len = 4i64;
        let num_heads = 2i64;
        let head_dim = 8i64;

        tch::manual_seed(42);
        let q = Tensor::randn([1, num_heads, query_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, block_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, block_len, head_dim], (Kind::Float, device));

        // 标准 softmax attention（参考值）
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // 用 process_kv_block（非因果模式）计算
        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 1,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };

        let mut rm = Tensor::full(
            [1i64, num_heads, query_len],
            f64::NEG_INFINITY,
            (Kind::Float, device),
        );
        let mut rs = Tensor::zeros([1i64, num_heads, query_len], (Kind::Float, device));
        let mut obh = Tensor::zeros(
            [1i64, num_heads, query_len, head_dim],
            (Kind::Float, device),
        );

        // 非因果模式：所有 Q 都能看到所有 K/V
        backend.process_kv_block(
            &q, 0, query_len as usize,
            &k, &v,
            0, block_len as usize,
            &mut rm, &mut rs, &mut obh,
            false,
        );

        // obh 形状: [1, num_heads, query_len, head_dim]
        let actual = obh;

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Single block diff = {}", diff_val);
        assert!(diff_val < 1e-4, "Single block chunk step differs from softmax: {}", diff_val);
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_local_attention_backend_shape() {
        let device = tch::Device::Cpu;
        let attn = create_test_attention(device);
        let hidden_size = attn.q_proj.size()[1];

        let mut backend = LocalAttentionBackend { attention: attn };
        let batch = 1i64;
        let seq_len = 5i64;
        let hidden = Tensor::randn([batch, seq_len, hidden_size], (Kind::Float, device));
        let pos_ids = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);

        let out = backend.forward(&hidden, &pos_ids, None, None).unwrap();
        assert_eq!(out.size(), vec![batch, seq_len, hidden_size]);
    }

    /// Build a ring-attention backend and verify it matches local full attention
    /// (non-causal, all positions attend to all positions).
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_matches_local_full() {
        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let seq_len = 16i64;
        let num_domains = 4usize;

        tch::manual_seed(123);
        let q = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // Create a minimal backend (only needs q/k/v/o_proj for local_attention_scores)
        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let mut backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };

        let actual = backend.ring_attention(&q, &k, &v, None, 0);

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Ring vs local full diff = {}", diff_val);
        assert!(diff_val < 1e-5, "Ring attention differs from local full: {}", diff_val);
    }

    /// Verify ring attention with causal mask matches local causal attention.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_matches_local_causal() {
        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let seq_len = 16i64;
        let num_domains = 4usize;

        tch::manual_seed(456);
        let q = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;

        let mask = make_causal_mask(seq_len, device);
        let scores_masked = scores + mask.shallow_clone();
        let attn = scores_masked.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let mut backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };

        let actual = backend.ring_attention(&q, &k, &v, Some(&mask), 0);

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Ring vs local causal diff = {}", diff_val);
        assert!(diff_val < 1e-4, "Ring attention differs from local causal: {}", diff_val);
    }

    /// Verify ring attention with seq_len=1 (decode) and long KV cache matches local attention.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_decode_matches_local() {
        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let q_len = 1i64;
        let kv_len = 17i64; // simulate KV cache with 17 tokens
        let num_domains = 2usize;

        tch::manual_seed(999);
        let q = Tensor::randn([1, num_heads, q_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let mut backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };

        // Without transport, ring_attention processes only local KV.
        // Since local KV = full KV (17 tokens), output should match local_attention_scores.
        let actual = backend.ring_attention(&q, &k, &v, None, 0);

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Decode ring vs local diff = {}", diff_val);
        assert!(diff_val < 1e-5, "Decode ring attention differs from local: {}", diff_val);
    }

    /// Verify distributed decode (seq_len=1) with peer KV matches local attention.
    /// Simulates real decode KV cache where new token is appended after history.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_decode_with_peer_kv() {
        use super::super::kv_transport::MockKvTransport;

        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let q_len = 1i64;
        let kv_len = 17i64; // tokens 0..16

        tch::manual_seed(777);
        let q_all = Tensor::randn([1, num_heads, q_len, head_dim], (Kind::Float, device));
        let k_all = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));
        let v_all = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));

        // Reference: local attention on full KV [0..16]
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores_ref = q_all.matmul(&k_all.transpose(2, 3)) * scale;
        let attn_ref = scores_ref.softmax(-1, Kind::Float);
        let expected = attn_ref.matmul(&v_all);

        // Simulate 2-domain distributed decode:
        // domain0 holds history [0..8) + new token 16
        // domain1 holds history [8..16) + new token 16
        let half = (kv_len - 1) / 2; // 8

        // domain0 local KV = [0..7] + [16]
        let k0_local = Tensor::cat(&[k_all.narrow(2, 0, half), k_all.narrow(2, kv_len - 1, 1)], 2);
        let v0_local = Tensor::cat(&[v_all.narrow(2, 0, half), v_all.narrow(2, kv_len - 1, 1)], 2);
        // domain1 local KV = [8..15] + [16]
        let k1_local = Tensor::cat(&[k_all.narrow(2, half, half), k_all.narrow(2, kv_len - 1, 1)], 2);
        let v1_local = Tensor::cat(&[v_all.narrow(2, half, half), v_all.narrow(2, kv_len - 1, 1)], 2);

        // History for sending (exclude new token)
        let k0_hist = k_all.narrow(2, 0, half);
        let v0_hist = v_all.narrow(2, 0, half);
        let k1_hist = k_all.narrow(2, half, half);
        let v1_hist = v_all.narrow(2, half, half);

        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);

        // Worker 0: receives peer KV [8..15] from worker 1
        let mut transport0 = MockKvTransport::new();
        transport0.push(super::super::kv_transport::KvBlock {
            layer_idx: 0,
            global_seq_start: half as usize,
            global_seq_end: kv_len as usize - 1,
            k: k1_hist.shallow_clone(),
            v: v1_hist.shallow_clone(),
        });

        let mut backend0 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope: rope.clone(),
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport0)),
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };
        let out0 = backend0.ring_attention(&q_all, &k0_local, &v0_local, None, 0);

        // Worker 1: receives peer KV [0..7] from worker 0
        let mut transport1 = MockKvTransport::new();
        transport1.push(super::super::kv_transport::KvBlock {
            layer_idx: 0,
            global_seq_start: 0,
            global_seq_end: half as usize,
            k: k0_hist.shallow_clone(),
            v: v0_hist.shallow_clone(),
        });

        let mut backend1 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport1)),
            local_domain_id: 1,
            seq_offset: half as usize,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };
        let out1 = backend1.ring_attention(&q_all, &k1_local, &v1_local, None, half as usize);

        let diff0 = (&expected - &out0).abs().mean(Kind::Float).double_value(&[]);
        let diff1 = (&expected - &out1).abs().mean(Kind::Float).double_value(&[]);
        let diff01 = (&out0 - &out1).abs().mean(Kind::Float).double_value(&[]);
        println!("Decode with peer KV diff ref-vs-domain0 = {}", diff0);
        println!("Decode with peer KV diff ref-vs-domain1 = {}", diff1);
        println!("Decode with peer KV diff domain0-vs-domain1 = {}", diff01);

        assert!(diff0 < 1e-4, "Decode distributed (domain0) differs from reference: {}", diff0);
        assert!(diff1 < 1e-4, "Decode distributed (domain1) differs from reference: {}", diff1);
        assert!(diff01 < 1e-4, "Decode distributed domain0 differs from domain1: {}", diff01);
    }

    /// Verify HcpRingAttentionBackend::forward matches GqaAttention::forward
    /// when num_domains=1 (no distributed transport) on decode (seq_len=1).
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_backend_matches_gqa_on_decode() {
        let device = tch::Device::Cpu;
        let hidden_size = 32i64;
        let num_heads = 4usize;
        let num_kv_heads = 1usize;
        let head_dim = 8usize;
        let seq_len = 1i64;
        let cache_len = 17i64; // simulate prefill of 16 tokens + 1 decode token

        tch::manual_seed(555);

        // Create GqaAttention
        let mut gqa = super::super::layers::GqaAttention {
            q_proj: Tensor::randn([(num_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            k_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            v_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            o_proj: Tensor::randn([hidden_size, (num_heads * head_dim) as i64], (Kind::Float, device)),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            num_heads,
            num_kv_heads,
            head_dim,
            rope: super::super::layers::RotaryEmbedding::new(head_dim, 128, 10000.0, device),
            scale: 1.0 / (head_dim as f64).sqrt(),
        };

        // Create HcpRingAttentionBackend with same weights
        let rope = gqa.rope.clone();
        let mut ring_backend = HcpRingAttentionBackend {
            q_proj: gqa.q_proj.shallow_clone(),
            k_proj: gqa.k_proj.shallow_clone(),
            v_proj: gqa.v_proj.shallow_clone(),
            o_proj: gqa.o_proj.shallow_clone(),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: gqa.scale,
            num_domains: 1,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };

        // hidden states for decode step
        let hidden_states = Tensor::randn([1, seq_len, hidden_size], (Kind::Float, device));
        let position_ids = Tensor::from_slice(&[16i64]).unsqueeze(0); // position 16

        // Pre-populate KV caches with synthetic history (16 tokens)
        let history_k = Tensor::randn([1, num_kv_heads as i64, cache_len - 1, head_dim as i64], (Kind::Float, device));
        let history_v = Tensor::randn([1, num_kv_heads as i64, cache_len - 1, head_dim as i64], (Kind::Float, device));

        let mut gqa_cache = super::super::cache::KvCache::new();
        let _ = gqa_cache.update(&history_k, &history_v).unwrap();

        let mut ring_cache = super::super::cache::KvCache::new();
        let _ = ring_cache.update(&history_k, &history_v).unwrap();

        // Run both forwards
        let gqa_out = gqa.forward(&hidden_states, &position_ids, Some(&mut gqa_cache), None).unwrap();
        let ring_out = ring_backend.forward(&hidden_states, &position_ids, Some(&mut ring_cache), None).unwrap();

        let diff = (&gqa_out - &ring_out).abs().mean(Kind::Float).double_value(&[]);
        println!("Ring backend vs GqaAttention decode diff = {}", diff);
        assert!(diff < 1e-4, "Ring backend differs from GqaAttention on decode: {}", diff);
    }

    /// Verify distributed ring attention with MockKvTransport matches single-process causal attention.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_with_mock_transport() {
        use super::super::kv_transport::MockKvTransport;

        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let seq_len = 16i64;
        let half = seq_len / 2;

        tch::manual_seed(789);
        let q = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));

        // Expected: single-process causal attention
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let mask = make_causal_mask(seq_len, device);
        let scores_masked = scores + mask.shallow_clone();
        let attn = scores_masked.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // Split into two workers
        let q0 = q.narrow(2, 0, half);
        let k0 = k.narrow(2, 0, half);
        let v0 = v.narrow(2, 0, half);
        let q1 = q.narrow(2, half, half);
        let k1 = k.narrow(2, half, half);
        let v1 = v.narrow(2, half, half);

        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);

        // Worker 0: receives peer KV from worker 1
        let mut transport0 = MockKvTransport::new();
        transport0.push(super::super::kv_transport::KvBlock {
            layer_idx: 0,
            global_seq_start: half as usize,
            global_seq_end: seq_len as usize,
            k: k1.shallow_clone(),
            v: v1.shallow_clone(),
        });

        let mut backend0 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope: rope.clone(),
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport0)),
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };
        let out0 = backend0.ring_attention(&q0, &k0, &v0, Some(&mask), 0);

        // Worker 1: receives peer KV from worker 0
        let mut transport1 = MockKvTransport::new();
        transport1.push(super::super::kv_transport::KvBlock {
            layer_idx: 0,
            global_seq_start: 0,
            global_seq_end: half as usize,
            k: k0.shallow_clone(),
            v: v0.shallow_clone(),
        });

        let mut backend1 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport1)),
            local_domain_id: 1,
            seq_offset: half as usize,
            prefill_kv_len: 0,
            is_prefill_done: false,
        };
        let out1 = backend1.ring_attention(&q1, &k1, &v1, Some(&mask), half as usize);

        // Compare each worker against expected slice before concatenation
        let expected0 = expected.narrow(2, 0, half);
        let expected1 = expected.narrow(2, half, half);
        let diff0 = (&expected0 - &out0).abs().mean(Kind::Float).double_value(&[]);
        let diff1 = (&expected1 - &out1).abs().mean(Kind::Float).double_value(&[]);
        println!("worker0 diff={}, worker1 diff={}", diff0, diff1);

        // Concatenate outputs: out0 [1, num_heads, half, head_dim], out1 [1, num_heads, half, head_dim]
        let actual = Tensor::cat(&[out0, out1], 2);

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Distributed ring attention diff = {}", diff_val);
        assert!(diff_val < 1e-4, "Distributed ring attention differs from local causal: {}", diff_val);
    }
}
