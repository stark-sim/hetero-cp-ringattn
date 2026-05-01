use crate::model::{ModelConfig, ModelError, ModelWeights, WeightNames};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

// ==================== RMSNorm ====================

#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

#[cfg(feature = "tch-backend")]
impl RmsNorm {
    pub fn from_weights(weights: &ModelWeights, name: &str, eps: f64) -> Result<Self, ModelError> {
        let weight = weights.get(name)?.shallow_clone();
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let variance = x.pow_tensor_scalar(2i64).mean_dim(&[-1i64][..], true, Kind::Float);
        x * (variance + self.eps).rsqrt() * &self.weight
    }
}

// ==================== Rotary Embedding ====================

#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    base: f64,
    cos_cache: Tensor,
    sin_cache: Tensor,
    device: Device,
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
    pub fn new(dim: usize, max_seq_len: usize, base: f64, device: Device) -> Self {
        let inv_freq = Self::compute_inv_freq(dim, base, device);
        let (cos_cache, sin_cache) = Self::compute_caches(max_seq_len, &inv_freq, device);
        Self { dim, max_seq_len, base, cos_cache, sin_cache, device }
    }

    fn compute_inv_freq(dim: usize, base: f64, device: Device) -> Tensor {
        let half_dim = (dim / 2) as i64;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf((2.0 * i as f64) / dim as f64) as f32)
            .collect();
        Tensor::from_slice(&inv_freq).to_device(device)
    }

    fn compute_caches(max_seq_len: usize, inv_freq: &Tensor, device: Device) -> (Tensor, Tensor) {
        let positions = Tensor::arange(max_seq_len as i64, (Kind::Float, device));
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0);
        (angles.cos(), angles.sin())
    }

    /// Apply RoPE to Q and K.
    ///
    /// Q/K shapes: `[batch, num_heads, seq_len, head_dim]`
    pub fn apply(&self, q: &Tensor, k: &Tensor, position_ids: Option<&Tensor>) -> (Tensor, Tensor) {
        let seq_len = q.size()[2] as usize;
        let num_heads_q = q.size()[1];
        let num_heads_k = k.size()[1];

        let (cos, sin) = if let Some(pos_ids) = position_ids {
            let batch = pos_ids.size()[0];
            let seq = pos_ids.size()[1];
            let batch = pos_ids.size()[0];
            let seq = pos_ids.size()[1];
            let pos_flat = pos_ids.view(-1);
            let cos = self.cos_cache.index_select(0, &pos_flat)
                .view([batch, seq, (self.dim / 2) as i64])
                .unsqueeze(1);
            let sin = self.sin_cache.index_select(0, &pos_flat)
                .view([batch, seq, (self.dim / 2) as i64])
                .unsqueeze(1);
            (cos, sin)
        } else {
            let cos = self.cos_cache.narrow(0, 0, seq_len as i64).unsqueeze(0).unsqueeze(1);
            let sin = self.sin_cache.narrow(0, 0, seq_len as i64).unsqueeze(0).unsqueeze(1);
            (cos, sin)
        };

        let q_rot = Self::rotate_pairs(q, &cos, &sin, num_heads_q);
        let k_rot = Self::rotate_pairs(k, &cos, &sin, num_heads_k);
        (q_rot, k_rot)
    }

    fn rotate_pairs(x: &Tensor, cos: &Tensor, sin: &Tensor, _num_heads: i64) -> Tensor {
        let dim = x.size()[3] as usize;
        let half_dim = dim / 2;
        let x1 = x.narrow(3, 0, half_dim as i64);
        let x2 = x.narrow(3, half_dim as i64, half_dim as i64);
        let r1 = &x1 * cos - &x2 * sin;
        let r2 = &x1 * sin + &x2 * cos;
        Tensor::cat(&[r1, r2], 3)
    }
}

// ==================== MLP (SwiGLU) ====================

#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
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
        let gate = x.matmul(&self.gate_proj.transpose(0, 1));
        let up = x.matmul(&self.up_proj.transpose(0, 1));
        let activated = gate.silu() * up;
        activated.matmul(&self.down_proj.transpose(0, 1))
    }
}

// ==================== GQA Attention ====================

#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct GqaAttention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope: RotaryEmbedding,
    pub scale: f64,
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

    /// Forward pass with optional KV cache and attention mask.
    ///
    /// hidden_states: `[batch, seq_len, hidden_size]`
    /// position_ids: `[batch, seq_len]`
    /// Returns: `[batch, seq_len, hidden_size]`
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

        // Projections (with optional bias)
        let mut q = hidden_states.matmul(&self.q_proj.transpose(0, 1));
        if let Some(ref bias) = self.q_bias {
            q = q + bias;
        }
        let mut k = hidden_states.matmul(&self.k_proj.transpose(0, 1));
        if let Some(ref bias) = self.k_bias {
            k = k + bias;
        }
        let mut v = hidden_states.matmul(&self.v_proj.transpose(0, 1));
        if let Some(ref bias) = self.v_bias {
            v = v + bias;
        }

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q.view([batch, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let k = k.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let v = v.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // Apply RoPE
        let (q, k) = self.rope.apply(&q, &k, Some(position_ids));

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k.shallow_clone(), v.shallow_clone())
        };

        // Repeat K/V heads for GQA
        let num_rep = self.num_heads / self.num_kv_heads;
        let k = Self::repeat_kv(&k, num_rep);
        let v = Self::repeat_kv(&v, num_rep);

        // Attention scores: [batch, num_heads, seq_len, kv_len]
        let scores = q.matmul(&k.transpose(2, 3)) * self.scale;

        // Apply attention mask (causal or padding)
        let scores = if let Some(mask) = attention_mask {
            scores + mask
        } else {
            scores
        };

        // Softmax
        let attn_weights = scores.softmax(-1, Kind::Float);

        // Attention output
        let attn_output = attn_weights.matmul(&v);

        // Reshape back and O-projection
        let attn_output = attn_output.transpose(1, 2).contiguous().view([batch, seq_len, hidden_size]);
        Ok(attn_output.matmul(&self.o_proj.transpose(0, 1)))
    }

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
        if n_rep == 1 {
            return x.shallow_clone();
        }
        x.repeat(&[1, n_rep as i64, 1, 1])
    }
}

// ==================== Decoder Layer ====================

#[cfg(feature = "tch-backend")]
pub struct DecoderLayer {
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub attention: Box<dyn crate::model::backend::AttentionBackend>,
    pub mlp: Mlp,
}

#[cfg(feature = "tch-backend")]
impl DecoderLayer {
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        // Pre-attention residual
        let residual = hidden_states.shallow_clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.attention.forward(&hidden_states, position_ids, kv_cache, attention_mask)?;
        let hidden_states = &hidden_states + &residual;

        // Pre-FFN residual
        let residual = hidden_states.shallow_clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states);
        let hidden_states = self.mlp.forward(&hidden_states);
        Ok(&hidden_states + &residual)
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

