use crate::model::{ModelConfig, ModelError, ModelWeights, WeightNames};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

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
