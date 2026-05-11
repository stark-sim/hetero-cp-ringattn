use crate::model::{ModelConfig, ModelError, ModelWeights, WeightNames};
use super::rotary::RotaryEmbedding;

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};


// ==================== GQA Attention ====================

/// 【GQA Attention：分组查询注意力】
///
/// 这是标准 Transformer Attention 的变体，通过让多个 Query head 共享同一个 Key/Value head
/// 来大幅减少 KV cache 的显存占用。
///
/// 对比：
/// - MHA（Multi-Head Attention）: num_kv_heads == num_heads，每个 Q head 对应独立的 K/V head
///   显存占用高，但表达能力最强。
/// - GQA（Grouped Query Attention）: num_kv_heads < num_heads，多个 Q head 共享 K/V head
///   显存占用低，速度更快，现代 LLM（Llama-2/3、Qwen2、Mistral）都使用 GQA。
/// - MQA（Multi-Query Attention）: num_kv_heads == 1，所有 Q head 共享同一个 K/V head
///   显存最低，但可能影响质量。
///
/// 例如 Qwen2-0.5B：num_heads=14, num_kv_heads=2
/// - 14 个 query head 分成 2 组，每组 7 个 query head 共享 1 个 key head 和 1 个 value head。
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct GqaAttention {
    pub q_proj: Tensor,       // Q 投影权重 [num_heads * head_dim, hidden_size]
    pub k_proj: Tensor,       // K 投影权重 [num_kv_heads * head_dim, hidden_size]
    pub v_proj: Tensor,       // V 投影权重 [num_kv_heads * head_dim, hidden_size]
    pub o_proj: Tensor,       // O 投影权重 [hidden_size, num_heads * head_dim]
    pub q_bias: Option<Tensor>,  // Q 投影偏置（可选，Qwen 系列有）
    pub k_bias: Option<Tensor>,  // K 投影偏置
    pub v_bias: Option<Tensor>,  // V 投影偏置
    pub num_heads: usize,     // Query head 数量
    pub num_kv_heads: usize,  // Key/Value head 数量（GQA 时小于 num_heads）
    pub head_dim: usize,      // 每个 head 的维度
    pub rope: RotaryEmbedding, // 旋转位置编码器
    pub scale: f64,           // Attention score 缩放因子：1/sqrt(head_dim)
}

#[cfg(feature = "tch-backend")]
impl GqaAttention {
    pub fn from_weights(
        weights: &ModelWeights,
        layer: usize,
        config: &ModelConfig,
        rope: &RotaryEmbedding,
    ) -> Result<Self, ModelError> {
        let q_bias = weights.get(&WeightNames::q_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        let k_bias = weights.get(&WeightNames::k_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        let v_bias = weights.get(&WeightNames::v_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        Ok(Self {
            q_proj: weights.get(&WeightNames::q_proj_weight(layer))?.shallow_clone(),
            k_proj: weights.get(&WeightNames::k_proj_weight(layer))?.shallow_clone(),
            v_proj: weights.get(&WeightNames::v_proj_weight(layer))?.shallow_clone(),
            o_proj: weights.get(&WeightNames::o_proj_weight(layer))?.shallow_clone(),
            q_bias,
            k_bias,
            v_bias,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads(),
            head_dim: config.head_dim(),
            rope: rope.clone(),
            scale: 1.0 / (config.head_dim() as f64).sqrt(),
        })
    }

    /// 【前向传播】标准 GQA Attention 计算。
    ///
    /// 参数：
    /// - hidden_states: [batch, seq_len, hidden_size]，上一层的输出
    /// - position_ids: [batch, seq_len]，每个 token 的绝对位置
    /// - kv_cache: 可选的 KV 缓存，用于自回归生成
    /// - attention_mask: 可选的注意力掩码，prefill 阶段用因果掩码
    ///
    /// 返回值：
    /// - [batch, seq_len, hidden_size]，Attention 后的输出
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        let batch = hidden_states.size()[0];
        let seq_len = hidden_states.size()[1];
        let hidden_size = hidden_states.size()[2];

        // Chunked projection for long sequences to avoid cuBLAS execution failure
        // on very large M dimensions (e.g. M=131072 for 128K seq).
        const PROJ_CHUNK_SIZE: i64 = 8192;
        let project = |x: &Tensor, w: &Tensor| -> Tensor {
            let sl = x.size()[1];
            if sl > PROJ_CHUNK_SIZE {
                let mut parts = Vec::new();
                for start in (0..sl).step_by(PROJ_CHUNK_SIZE as usize) {
                    let chunk_len = (start + PROJ_CHUNK_SIZE).min(sl) - start;
                    let chunk = x.narrow(1, start, chunk_len);
                    parts.push(chunk.matmul(&w.transpose(0, 1)));
                }
                Tensor::cat(&parts, 1)
            } else {
                x.matmul(&w.transpose(0, 1))
            }
        };

        // ====== 第一步：线性投影 ======
        // 用三个权重矩阵把 hidden_states 映射成 Q、K、V。
        let mut q = project(hidden_states, &self.q_proj);
        if let Some(ref bias) = self.q_bias { q += bias; }
        let mut k = project(hidden_states, &self.k_proj);
        if let Some(ref bias) = self.k_bias { k += bias; }
        let mut v = project(hidden_states, &self.v_proj);
        if let Some(ref bias) = self.v_bias { v += bias; }

        // ====== 第二步：reshape 成多头格式 ======
        // 从 [batch, seq_len, num_heads * head_dim] → [batch, num_heads, seq_len, head_dim]
        let q = q.view([batch, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let k = k.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let v = v.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // ====== 第三步：应用 RoPE ======
        let (q, k) = self.rope.apply(&q, &k, Some(position_ids));

        // ====== 第四步：更新 KV Cache ======
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k.shallow_clone(), v.shallow_clone())
        };

        // ====== 第五步：GQA 头重复 ======
        // 如果 num_kv_heads < num_heads，需要把 K/V 在 head 维度上重复，
        // 让每个 query head 都能找到对应的 key/value head。
        let num_rep = self.num_heads / self.num_kv_heads;
        let k = Self::repeat_kv(&k, num_rep);
        let v = Self::repeat_kv(&v, num_rep);

        // ====== 第六步：计算 Attention Scores ======
        // score = Q @ K^T / sqrt(head_dim)
        // shape: [batch, num_heads, seq_len, kv_len]
        let scores: Tensor = q.matmul(&k.transpose(2, 3)) * self.scale;

        // ====== 第七步：应用 Attention Mask ======
        // causal mask: 当前 token 看不到未来的 token（mask 值为 -inf）
        let scores = if let Some(mask) = attention_mask {
            scores + mask
        } else {
            scores
        };

        // ====== 第八步：Softmax ======
        // 把 scores 转换成概率分布（所有权重之和为 1）。
        let attn_weights = scores.softmax(-1, Kind::Float);

        // ====== 第九步：加权求和 ======
        // output = attn_weights @ V
        // shape: [batch, num_heads, seq_len, head_dim]
        let attn_output = attn_weights.matmul(&v);

        // ====== 第十步：reshape 并 O-projection ======
        // 把多头的输出拼接起来，再乘 o_proj 映射回 hidden_size。
        let attn_output = attn_output.transpose(1, 2).contiguous().view([batch, seq_len, hidden_size]);
        Ok(project(&attn_output, &self.o_proj))
    }

    /// 【重复 KV head】
    /// GQA 中 num_kv_heads < num_heads，为了让矩阵乘法维度匹配，
    /// 需要在 head 维度上重复 K/V。
    ///
    /// 例如 num_heads=4, num_kv_heads=1：
    /// - K 原始形状: [batch, 1, seq_len, head_dim]
    /// - repeat(&[1, 4, 1, 1]) → [batch, 4, seq_len, head_dim]
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
        if n_rep == 1 {
            return x.shallow_clone();
        }
        let shape = x.size();
        let batch = shape[0];
        let num_kv_heads = shape[1];
        let slen = shape[2];
        let head_dim = shape[3];
        x.unsqueeze(2)
            .expand([batch, num_kv_heads, n_rep as i64, slen, head_dim], false)
            .reshape([batch, num_kv_heads * n_rep as i64, slen, head_dim])
    }
}
