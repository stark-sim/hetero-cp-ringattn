#[cfg(feature = "tch-backend")]
pub mod norm;
#[cfg(feature = "tch-backend")]
pub mod rotary;
#[cfg(feature = "tch-backend")]
pub mod mlp;
#[cfg(feature = "tch-backend")]
pub mod attention;

#[cfg(feature = "tch-backend")]
pub use norm::RmsNorm;
#[cfg(feature = "tch-backend")]
pub use rotary::RotaryEmbedding;
#[cfg(feature = "tch-backend")]
pub use mlp::Mlp;
#[cfg(feature = "tch-backend")]
pub use attention::GqaAttention;

#[cfg(feature = "tch-backend")]
use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::Tensor;

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
    pub attention: Box<dyn crate::model::attention::AttentionBackend>, // Attention 计算后端（可切换）
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
        kv_cache: Option<&mut dyn crate::model::cache::KvCache>,
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
    #[cfg(feature = "tch-backend")]
    use tch::{Device, Kind, Tensor};

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_rmsnorm_shape() {
        let device = Device::Cpu;
        let weight = Tensor::ones([8], (Kind::Float, device));
        let rms = RmsNorm { weight, eps: 1e-6 };
        let x = Tensor::randn([2, 4, 8], (Kind::Float, device));
        let out = rms.forward(&x);
        assert_eq!(out.size(), vec![2, 4, 8]);
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_rope_shape() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 128, 10000.0, device);
        let q = Tensor::randn([1, 14, 10, 64], (Kind::Float, device));
        let k = Tensor::randn([1, 2, 10, 64], (Kind::Float, device));
        let (q_rot, k_rot) = rope.apply(&q, &k, None);
        assert_eq!(q_rot.size(), q.size());
        assert_eq!(k_rot.size(), k.size());
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_mlp_shape() {
        let device = Device::Cpu;
        let gate = Tensor::randn([32, 8], (Kind::Float, device));
        let up = Tensor::randn([32, 8], (Kind::Float, device));
        let down = Tensor::randn([8, 32], (Kind::Float, device));
        let mlp = Mlp { gate_proj: gate, up_proj: up, down_proj: down };
        let x = Tensor::randn([1, 4, 8], (Kind::Float, device));
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
            rope: RotaryEmbedding::new(head_dim, 128, 10000.0, device),
            scale: 1.0 / (head_dim as f64).sqrt(),
        };

        let batch = 1i64;
        let seq_len = 5i64;
        let hidden = Tensor::randn([batch, seq_len, hidden_size], (Kind::Float, device));
        let pos_ids = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);

        let out = attn.forward(&hidden, &pos_ids, None, None).unwrap();
        assert_eq!(out.size(), vec![batch, seq_len, hidden_size]);
    }
}
