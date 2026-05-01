use crate::model::{ModelConfig, ModelError, ModelWeights, WeightNames};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

// ==================== RMSNorm ====================

/// 【RMSNorm：均方根归一化】
///
/// 传统的 LayerNorm 需要计算均值和方差：
///   LayerNorm(x) = (x - mean) / sqrt(variance + eps) * gamma
///
/// RMSNorm 是一种简化版本，不计算均值，只计算均方根（RMS）：
///   RMSNorm(x) = x / RMS(x) * gamma
///   其中 RMS(x) = sqrt(mean(x^2) + eps)
///
/// 这样做的好处：
/// - 少一次均值计算，更快
/// - 实验表明在 LLM 上效果和 LayerNorm 相当甚至更好（Llama、Qwen2 都用 RMSNorm）
/// - gamma（weight）是可学习的缩放参数
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,  // 可学习的缩放参数 gamma，shape [hidden_size]
    eps: f64,        // 防止除零的小常数，默认 1e-6
}

#[cfg(feature = "tch-backend")]
impl RmsNorm {
    /// 【从权重加载】
    pub fn from_weights(weights: &ModelWeights, name: &str, eps: f64) -> Result<Self, ModelError> {
        let weight = weights.get(name)?.shallow_clone();
        Ok(Self { weight, eps })
    }

    /// 【前向传播】
    ///
    /// 数学公式：
    ///   variance = mean(x^2, dim=-1, keepdim=True)
    ///   rms = sqrt(variance + eps)
    ///   output = x / rms * weight
    ///
    /// 代码解释：
    /// - pow_tensor_scalar(2): 每个元素平方
    /// - mean_dim(&[-1][..], true, ...): 在最后一个维度（特征维）上求均值，keepdim=true 保留维度以便广播
    /// - rsqrt(): 平方根后取倒数（1/sqrt）
    /// - 最后乘 weight（gamma）做缩放
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let variance = x.pow_tensor_scalar(2i64).mean_dim(&[-1i64][..], true, Kind::Float);
        x * (variance + self.eps).rsqrt() * &self.weight
    }
}

// ==================== Rotary Embedding ====================

/// 【RoPE：旋转位置编码】
///
/// 传统位置编码（如正弦/余弦绝对位置编码）把位置信息加到输入 embedding 上。
/// RoPE 的独特之处：它把位置信息编码到 Attention 的 Q 和 K 中，通过旋转向量实现。
///
/// 核心思想：
/// - 把 Q/K 向量的每对维度 (x_{2i}, x_{2i+1}) 看作一个 2D 向量
/// - 根据 token 的位置 m，把这个 2D 向量旋转一个角度 m * theta_i
/// - 旋转角度 theta_i 取决于维度索引 i：theta_i = base^{-2i/dim}
///
/// 优点：
/// - 相对位置编码：旋转后的 dot product 自动包含相对位置信息
/// - 外推性好：训练时没见过长序列，也能通过旋转角度外推
/// - Llama、Qwen2、Mistral 等主流模型都用 RoPE
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct RotaryEmbedding {
    dim: usize,           // 旋转维度（通常是 head_dim）
    max_seq_len: usize,   // 预计算的最大序列长度
    base: f64,            // 旋转角度基数（如 10000.0、1000000.0）
    cos_cache: Tensor,    // 预计算的余弦缓存 [max_seq_len, dim/2]
    sin_cache: Tensor,    // 预计算的正弦缓存 [max_seq_len, dim/2]
    device: Device,       // 设备（CPU/MPS/CUDA）
}

#[cfg(feature = "tch-backend")]
impl Clone for RotaryEmbedding {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            max_seq_len: self.max_seq_len,
            base: self.base,
            cos_cache: self.cos_cache.shallow_clone(),
            sin_cache: self.sin_cache.shallow_clone(),
            device: self.device,
        }
    }
}

#[cfg(feature = "tch-backend")]
impl RotaryEmbedding {
    /// 【创建 RoPE 实例】
    ///
    /// 参数：
    /// - dim: 旋转维度（通常是 head_dim）
    /// - max_seq_len: 预计算的最大序列长度（如 4096、131072）
    /// - base: 旋转角度基数（如 10000.0）
    /// - device: 目标设备
    ///
    /// 构造函数会预计算所有位置的 cos/sin 缓存，避免每次 forward 重复计算。
    pub fn new(dim: usize, max_seq_len: usize, base: f64, device: Device) -> Self {
        let inv_freq = Self::compute_inv_freq(dim, base, device);
        let (cos_cache, sin_cache) = Self::compute_caches(max_seq_len, &inv_freq, device);
        Self { dim, max_seq_len, base, cos_cache, sin_cache, device }
    }

    /// 【计算逆频率】
    ///
    /// 公式：inv_freq[i] = 1.0 / base^(2i / dim)
    ///
    /// 其中 i = 0, 1, ..., dim/2 - 1
    ///
    /// 这个频率决定了不同维度对的旋转速度：
    /// - 低维度（i 小）：旋转慢（频率低）
    /// - 高维度（i 大）：旋转快（频率高）
    fn compute_inv_freq(dim: usize, base: f64, device: Device) -> Tensor {
        let half_dim = (dim / 2) as i64;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf((2.0 * i as f64) / dim as f64) as f32)
            .collect();
        Tensor::from_slice(&inv_freq).to_device(device)
    }

    /// 【预计算 cos/sin 缓存】
    ///
    /// 对于每个位置 m 和每个频率 inv_freq[i]，计算：
    ///   angle = m * inv_freq[i]
    ///   cos_cache[m, i] = cos(angle)
    ///   sin_cache[m, i] = sin(angle)
    ///
    /// 缓存形状：[max_seq_len, dim/2]
    fn compute_caches(max_seq_len: usize, inv_freq: &Tensor, device: Device) -> (Tensor, Tensor) {
        // positions: [0, 1, 2, ..., max_seq_len-1]
        let positions = Tensor::arange(max_seq_len as i64, (Kind::Float, device));
        // angles[m, i] = positions[m] * inv_freq[i]
        // unsqueeze(1) 把 positions 变成 [max_seq_len, 1]
        // unsqueeze(0) 把 inv_freq 变成 [1, dim/2]
        // 广播后相乘得到 [max_seq_len, dim/2]
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0);
        (angles.cos(), angles.sin())
    }

    /// 【对 Q 和 K 应用 RoPE】
    ///
    /// 参数：
    /// - q / k: shape [batch, num_heads, seq_len, head_dim]
    /// - position_ids: 每个 token 的绝对位置 [batch, seq_len]；如果为 None 则使用 0..seq_len
    ///
    /// 返回值：旋转后的 (q_rot, k_rot)，形状不变。
    pub fn apply(&self, q: &Tensor, k: &Tensor, position_ids: Option<&Tensor>) -> (Tensor, Tensor) {
        let seq_len = q.size()[2] as usize;
        let num_heads_q = q.size()[1];
        let num_heads_k = k.size()[1];

        // 根据 position_ids 从缓存中取出对应的 cos/sin。
        let (cos, sin) = if let Some(pos_ids) = position_ids {
            // position_ids 包含每个 token 的绝对位置（分布式场景下可能不是 0..seq_len）。
            let batch = pos_ids.size()[0];
            let seq = pos_ids.size()[1];
            // view(-1) 把 [batch, seq] 展平成一维 [batch*seq]
            let pos_flat = pos_ids.view(-1);
            // index_select(0, &pos_flat): 从 cos_cache 中按位置索引选取对应的 cos 值
            // view([batch, seq, dim/2]): 恢复成 [batch, seq, dim/2]
            // unsqueeze(1): 在 head 维度插入一维，变成 [batch, 1, seq, dim/2]，方便广播到 num_heads
            let cos = self.cos_cache.index_select(0, &pos_flat)
                .view([batch, seq, (self.dim / 2) as i64])
                .unsqueeze(1);
            let sin = self.sin_cache.index_select(0, &pos_flat)
                .view([batch, seq, (self.dim / 2) as i64])
                .unsqueeze(1);
            (cos, sin)
        } else {
            // 没有 position_ids 时，默认使用位置 0..seq_len
            let cos = self.cos_cache.narrow(0, 0, seq_len as i64).unsqueeze(0).unsqueeze(1);
            let sin = self.sin_cache.narrow(0, 0, seq_len as i64).unsqueeze(0).unsqueeze(1);
            (cos, sin)
        };

        let q_rot = Self::rotate_pairs(q, &cos, &sin, num_heads_q);
        let k_rot = Self::rotate_pairs(k, &cos, &sin, num_heads_k);
        (q_rot, k_rot)
    }

    /// 【旋转向量的维度对】
    ///
    /// 把 x 的 head_dim 维度分成两半：x1 = x[..., :half_dim], x2 = x[..., half_dim:]
    /// 对每一对 (x1, x2) 做 2D 旋转：
    ///   r1 = x1 * cos - x2 * sin
    ///   r2 = x1 * sin + x2 * cos
    /// 最后把 r1 和 r2 拼接回去。
    fn rotate_pairs(x: &Tensor, cos: &Tensor, sin: &Tensor, _num_heads: i64) -> Tensor {
        let dim = x.size()[3] as usize;
        let half_dim = dim / 2;
        let x1 = x.narrow(3, 0, half_dim as i64);          // 前一半维度
        let x2 = x.narrow(3, half_dim as i64, half_dim as i64);  // 后一半维度
        let r1 = &x1 * cos - &x2 * sin;
        let r2 = &x1 * sin + &x2 * cos;
        Tensor::cat(&[r1, r2], 3)  // 在第 3 维（head_dim）上拼接
    }
}

// ==================== MLP (SwiGLU) ====================

/// 【MLP：前馈神经网络】
///
/// 传统 Transformer 使用 ReLU 或 GELU 作为 FFN 的激活函数。
/// Llama/Qwen2 使用 SwiGLU，它是一种门控结构：
///
///   gate = x @ W_gate^T
///   up   = x @ W_up^T
///   output = silu(gate) * up @ W_down^T
///
/// 其中 silu(x) = x * sigmoid(x)，也叫 Swish 激活函数。
///
/// SwiGLU 的效果：
/// - gate 控制信息流通的"门"，决定哪些信息通过
/// - up 提供升维后的特征表示
/// - silu(gate) * up 是逐元素乘法（Hadamard 积），实现门控效果
/// - down 把维度降回 hidden_size
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Tensor,  // [intermediate_size, hidden_size]
    up_proj: Tensor,    // [intermediate_size, hidden_size]
    down_proj: Tensor,  // [hidden_size, intermediate_size]
}

#[cfg(feature = "tch-backend")]
impl Mlp {
    pub fn from_weights(weights: &ModelWeights, layer: usize) -> Result<Self, ModelError> {
        Ok(Self {
            gate_proj: weights.get(&WeightNames::gate_proj_weight(layer))?.shallow_clone(),
            up_proj: weights.get(&WeightNames::up_proj_weight(layer))?.shallow_clone(),
            down_proj: weights.get(&WeightNames::down_proj_weight(layer))?.shallow_clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // gate: 生成门控信号，shape [batch, seq_len, intermediate_size]
        let gate = x.matmul(&self.gate_proj.transpose(0, 1));
        // up: 升维特征，shape [batch, seq_len, intermediate_size]
        let up = x.matmul(&self.up_proj.transpose(0, 1));
        // silu(gate) * up: 门控后的激活，shape [batch, seq_len, intermediate_size]
        let activated = gate.silu() * up;
        // down: 降维回 hidden_size，shape [batch, seq_len, hidden_size]
        activated.matmul(&self.down_proj.transpose(0, 1))
    }
}

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

        // ====== 第一步：线性投影 ======
        // 用三个权重矩阵把 hidden_states 映射成 Q、K、V。
        let mut q = hidden_states.matmul(&self.q_proj.transpose(0, 1));
        if let Some(ref bias) = self.q_bias { q = q + bias; }
        let mut k = hidden_states.matmul(&self.k_proj.transpose(0, 1));
        if let Some(ref bias) = self.k_bias { k = k + bias; }
        let mut v = hidden_states.matmul(&self.v_proj.transpose(0, 1));
        if let Some(ref bias) = self.v_bias { v = v + bias; }

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
        let scores = q.matmul(&k.transpose(2, 3)) * self.scale;

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
        Ok(attn_output.matmul(&self.o_proj.transpose(0, 1)))
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
        x.repeat(&[1, n_rep as i64, 1, 1])
    }
}

// ==================== Decoder Layer ====================

/// 【Decoder Layer：解码器层】
///
/// 这是 Transformer 的核心 building block，每个 layer 包含两个子层：
/// 1. Attention 子层：计算 self-attention，让 token 之间互相"交流"
/// 2. MLP 子层：前馈网络，对每个 token 独立做非线性变换
///
/// 每个子层都用 Pre-Norm + Residual Connection：
/// - Pre-Norm：先归一化再计算，训练更稳定
/// - Residual Connection：x = sublayer(norm(x)) + x，防止梯度消失，允许构建深层网络
///
/// 现代 LLM（Llama、Qwen2）通常堆叠 24~128 个这样的 layer。
#[cfg(feature = "tch-backend")]
pub struct DecoderLayer {
    pub input_layernorm: RmsNorm,        // Attention 之前的归一化
    pub post_attention_layernorm: RmsNorm, // MLP 之前的归一化
    pub attention: Box<dyn crate::model::backend::AttentionBackend>, // Attention 计算后端（可切换）
    pub mlp: Mlp,                        // 前馈网络
}

#[cfg(feature = "tch-backend")]
impl DecoderLayer {
    /// 【前向传播】
    ///
    /// 数据流：
    /// 1. 保存输入 residual = x
    /// 2. x = RMSNorm(x)          ← Pre-Norm
    /// 3. x = Attention(x)        ← 注意力计算
    /// 4. x = x + residual        ← Residual Connection
    /// 5. 保存新的 residual = x
    /// 6. x = RMSNorm(x)          ← 第二次 Pre-Norm
    /// 7. x = MLP(x)              ← 前馈网络
    /// 8. x = x + residual        ← 第二次 Residual Connection
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        // ====== Attention 子层（带 Residual）======
        let residual = hidden_states.shallow_clone();  // 保存原始输入
        let hidden_states = self.input_layernorm.forward(hidden_states);  // Pre-Norm
        let hidden_states = self.attention.forward(&hidden_states, position_ids, kv_cache, attention_mask)?;
        let hidden_states = &hidden_states + &residual;  // Residual: 输出 + 原始输入

        // ====== MLP 子层（带 Residual）======
        let residual = hidden_states.shallow_clone();  // 保存 Attention 后的输出
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states);  // Pre-Norm
        let hidden_states = self.mlp.forward(&hidden_states);  // 前馈网络
        Ok(&hidden_states + &residual)  // Residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_rmsnorm_shape() {
        let device = Device::Cpu;
        let weight = Tensor::ones(&[8], (Kind::Float, device));
        let rms = RmsNorm { weight, eps: 1e-6 };
        let x = Tensor::randn(&[2, 4, 8], (Kind::Float, device));
        let out = rms.forward(&x);
        assert_eq!(out.size(), vec![2, 4, 8]);
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_rope_shape() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 128, 10000.0, device);
        let q = Tensor::randn(&[1, 14, 10, 64], (Kind::Float, device));
        let k = Tensor::randn(&[1, 2, 10, 64], (Kind::Float, device));
        let (q_rot, k_rot) = rope.apply(&q, &k, None);
        assert_eq!(q_rot.size(), q.size());
        assert_eq!(k_rot.size(), k.size());
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_mlp_shape() {
        let device = Device::Cpu;
        let gate = Tensor::randn(&[32, 8], (Kind::Float, device));
        let up = Tensor::randn(&[32, 8], (Kind::Float, device));
        let down = Tensor::randn(&[8, 32], (Kind::Float, device));
        let mlp = Mlp { gate_proj: gate, up_proj: up, down_proj: down };
        let x = Tensor::randn(&[1, 4, 8], (Kind::Float, device));
        let out = mlp.forward(&x);
        assert_eq!(out.size(), vec![1, 4, 8]);
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_gqa_attention_shape() {
        let device = Device::Cpu;
        let hidden_size = 64i64;
        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;

        let attn = GqaAttention {
            q_proj: Tensor::randn(&[(num_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            k_proj: Tensor::randn(&[(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            v_proj: Tensor::randn(&[(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            o_proj: Tensor::randn(&[hidden_size, (num_heads * head_dim) as i64], (Kind::Float, device)),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            num_heads,
            num_kv_heads,
            head_dim,
            rope: RotaryEmbedding::new(head_dim, 128, 10000.0, device),
            scale: 1.0 / (head_dim as f64).sqrt(),
        };

        let batch = 1i64;
        let seq_len = 5i64;
        let hidden = Tensor::randn(&[batch, seq_len, hidden_size], (Kind::Float, device));
        let pos_ids = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);

        let out = attn.forward(&hidden, &pos_ids, None, None).unwrap();
        assert_eq!(out.size(), vec![batch, seq_len, hidden_size]);
    }
}

