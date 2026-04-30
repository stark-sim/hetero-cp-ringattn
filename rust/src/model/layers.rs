use crate::model::{ModelError, ModelWeights, WeightNames};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

/// Root-Mean-Square Layer Normalization (RMSNorm).
///
/// Formula: `x * rsqrt(mean(x^2) + eps) * weight`
///
/// Used by Llama, Mistral, Qwen2, and most modern decoder-only LLMs.
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

#[cfg(feature = "tch-backend")]
impl RmsNorm {
    /// Load from safetensors weight name (e.g. `model.layers.0.input_layernorm.weight`).
    pub fn from_weights(weights: &ModelWeights, name: &str, eps: f64) -> Result<Self, ModelError> {
        let weight = weights.get(name)?.shallow_clone();
        Ok(Self { weight, eps })
    }

    /// Forward pass.
    ///
    /// Input: `[batch, seq_len, hidden_size]`
    /// Output: `[batch, seq_len, hidden_size]`
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Compute variance along last dimension
        let variance = x.pow_tensor_scalar(2i64).mean_dim(&[-1i64][..], true, Kind::Float);
        // Normalize and scale
        x * (variance + self.eps).rsqrt() * &self.weight
    }
}

/// Rotary Position Embedding (RoPE).
///
/// Pre-computes cos/sin caches for positions up to `max_seq_len`.
/// Applies rotation to Q and K head-dimension pairs.
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
impl RotaryEmbedding {
    /// Create RoPE with given head dimension and config.
    pub fn new(dim: usize, max_seq_len: usize, base: f64, device: Device) -> Self {
        let inv_freq = Self::compute_inv_freq(dim, base, device);
        let (cos_cache, sin_cache) = Self::compute_caches(max_seq_len, &inv_freq, device);
        Self {
            dim,
            max_seq_len,
            base,
            cos_cache,
            sin_cache,
            device,
        }
    }

    /// Compute inverse frequencies: `1.0 / (base^(2i / dim))` for i in [0, dim/2).
    fn compute_inv_freq(dim: usize, base: f64, device: Device) -> Tensor {
        let half_dim = (dim / 2) as i64;
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / base.powf((2.0 * i as f64) / dim as f64))
            .collect();
        Tensor::from_slice(&inv_freq).to_device(device)
    }

    /// Compute cos/sin caches for all positions.
    fn compute_caches(max_seq_len: usize, inv_freq: &Tensor, device: Device) -> (Tensor, Tensor) {
        let positions = Tensor::arange(max_seq_len as i64, (Kind::Float, device));
        // angles: [max_seq_len, dim/2]
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0);
        let cos_cache = angles.cos();
        let sin_cache = angles.sin();
        (cos_cache, sin_cache)
    }

    /// Apply RoPE to Q and K tensors.
    ///
    /// Input Q/K shapes: `[batch, num_heads, seq_len, head_dim]`
    /// Output shapes: same as input
    pub fn apply(&self, q: &Tensor, k: &Tensor, position_ids: Option<&Tensor>) -> (Tensor, Tensor) {
        let seq_len = q.size()[2] as usize;
        let _batch = q.size()[0];
        let num_heads_q = q.size()[1];
        let num_heads_k = k.size()[1];

        // Gather cos/sin for the relevant positions
        let (cos, sin) = if let Some(pos_ids) = position_ids {
            // pos_ids: [batch, seq_len]
            let cos = self.cos_cache.index_select(0, pos_ids);
            let sin = self.sin_cache.index_select(0, pos_ids);
            // Expand to [batch, 1, seq_len, dim]
            let cos = cos.unsqueeze(1);
            let sin = sin.unsqueeze(1);
            (cos, sin)
        } else {
            // Use sequential positions [0, seq_len)
            let cos = self.cos_cache.narrow(0, 0, seq_len as i64).unsqueeze(0).unsqueeze(1);
            let sin = self.sin_cache.narrow(0, 0, seq_len as i64).unsqueeze(0).unsqueeze(1);
            (cos, sin)
        };

        let q_rot = Self::apply_rotary(q, &cos, &sin, num_heads_q);
        let k_rot = Self::apply_rotary(k, &cos, &sin, num_heads_k);
        (q_rot, k_rot)
    }

    /// Apply 2D rotation to each pair of dimensions.
    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor, _num_heads: i64) -> Tensor {
        // x: [batch, num_heads, seq_len, head_dim]
        // cos/sin: [batch, 1, seq_len, dim/2] or [1, 1, seq_len, dim/2]
        let dim = x.size()[3] as usize;
        let half_dim = dim / 2;

        // Split x into two halves along last dimension
        let x1 = x.narrow(3, 0, half_dim as i64);   // [batch, heads, seq, dim/2]
        let x2 = x.narrow(3, half_dim as i64, half_dim as i64);

        // Apply rotation: [x1, x2] * [cos, -sin; sin, cos]
        let rotated1 = &x1 * cos - &x2 * sin;
        let rotated2 = &x1 * sin + &x2 * cos;

        Tensor::cat(&[rotated1, rotated2], 3)
    }
}

/// SwiGLU MLP (Feed-Forward Network).
///
/// Formula: `down_proj(silu(gate_proj(x)) * up_proj(x))`
///
/// Used by Llama, Mistral, Qwen2.
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

#[cfg(feature = "tch-backend")]
impl Mlp {
    /// Load MLP weights for a given layer.
    pub fn from_weights(weights: &ModelWeights, layer: usize) -> Result<Self, ModelError> {
        let gate_proj = weights.get(&WeightNames::gate_proj_weight(layer))?.shallow_clone();
        let up_proj = weights.get(&WeightNames::up_proj_weight(layer))?.shallow_clone();
        let down_proj = weights.get(&WeightNames::down_proj_weight(layer))?.shallow_clone();
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass.
    ///
    /// Input: `[batch, seq_len, hidden_size]`
    /// Output: `[batch, seq_len, hidden_size]`
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // gate_proj and up_proj: [intermediate_size, hidden_size]
        // We need x @ W^T, so transpose to [hidden_size, intermediate_size]
        let gate = x.matmul(&self.gate_proj.transpose(0, 1));
        let up = x.matmul(&self.up_proj.transpose(0, 1));
        let activated = gate.silu() * up;
        activated.matmul(&self.down_proj.transpose(0, 1))
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
        let mlp = Mlp {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        };
        let x = Tensor::randn(&[1, 4, 8], (Kind::Float, device));
        let out = mlp.forward(&x);
        assert_eq!(out.size(), vec![1, 4, 8]);
    }
}
