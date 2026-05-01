use crate::model::{ModelConfig, ModelError};

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

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
        })
    }

    /// Compute attention by splitting Q into chunks and K/V into blocks,
    /// applying online softmax across blocks.
    fn ring_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let batch = q.size()[0];
        let num_heads = q.size()[1];
        let seq_len = q.size()[2];
        let head_dim = q.size()[3];

        // For very short sequences, just do local attention
        if seq_len <= 1 || self.num_domains == 1 {
            return self.local_attention_scores(q, k, v, attention_mask);
        }

        let q_chunk_size = ((seq_len as usize + self.num_domains - 1) / self.num_domains).max(1);
        let kv_chunk_size = q_chunk_size;

        let mut outputs = Vec::new();

        for q_start in (0..seq_len as usize).step_by(q_chunk_size) {
            let q_end = (q_start + q_chunk_size).min(seq_len as usize);
            let q_chunk_len = (q_end - q_start) as i64;
            let q_chunk = q.narrow(2, q_start as i64, q_chunk_len);

            let kv_chunks: Vec<(usize, usize)> = (0..seq_len as usize)
                .step_by(kv_chunk_size)
                .map(|start| (start, (start + kv_chunk_size).min(seq_len as usize)))
                .collect();

            if attention_mask.is_some() {
                // Causal prefill path: pure-tensor online softmax on device.
                // No CPU buffer round-trip.
                let mut rm = Tensor::full(
                    &[batch, num_heads, q_chunk_len],
                    f64::NEG_INFINITY,
                    (Kind::Float, q.device()),
                );
                let mut rs = Tensor::zeros(&[batch, num_heads, q_chunk_len], (Kind::Float, q.device()));
                let mut obh = Tensor::zeros(
                    &[batch, num_heads, q_chunk_len, head_dim],
                    (Kind::Float, q.device()),
                );

                for (kv_start, kv_end) in &kv_chunks {
                    let kv_chunk_len = (*kv_end - *kv_start) as i64;

                    // Apply causal mask: skip blocks that are entirely after this Q chunk
                    if *kv_start >= q_end {
                        continue;
                    }

                    let k_chunk = k.narrow(2, *kv_start as i64, kv_chunk_len);
                    let v_chunk = v.narrow(2, *kv_start as i64, kv_chunk_len);

                    // Build scores for this Q chunk vs K/V chunk
                    let scores = q_chunk.matmul(&k_chunk.transpose(2, 3)) * self.scale;

                    // Apply causal mask
                    let q_pos = Tensor::arange_start(
                        q_start as i64,
                        q_end as i64,
                        (Kind::Int64, q.device()),
                    )
                    .unsqueeze(1)
                    .unsqueeze(0)
                    .unsqueeze(0);
                    let k_pos = Tensor::arange_start(
                        *kv_start as i64,
                        *kv_end as i64,
                        (Kind::Int64, q.device()),
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0);
                    let causal = q_pos.ge_tensor(&k_pos);
                    let scores = scores.masked_fill(&causal.logical_not(), f64::NEG_INFINITY);

                    // Online softmax update
                    let (local_max, _) = scores.max_dim(3, false); // [batch, num_heads, q_chunk_len]
                    let weights = (&scores - local_max.unsqueeze(3)).exp();
                    let local_sum = weights.sum_dim_intlist(&[3i64][..], false, Kind::Float);
                    let local_pv = weights.matmul(&v_chunk); // [batch, num_heads, q_chunk_len, head_dim]

                    let new_max = rm.max_other(&local_max);
                    let exp_prev = (&rm - &new_max).exp();
                    let exp_local = (&local_max - &new_max).exp();
                    let new_sum = &exp_prev * &rs + &exp_local * &local_sum;

                    obh = (&exp_prev.unsqueeze(3) * &rs.unsqueeze(3) * &obh
                        + &exp_local.unsqueeze(3) * &local_pv)
                        / &new_sum.unsqueeze(3);
                    rm = new_max;
                    rs = new_sum;
                }

                outputs.push(obh); // [batch, num_heads, q_chunk_len, head_dim]
            } else {
                // Non-causal path (protocol smoke): use CPU-buffer-based block update
                // for compatibility with compute_chunk_attention_step payload interface.
                let qh = (q_chunk_len * num_heads) as usize;
                let mut running_max = vec![f32::NEG_INFINITY; qh];
                let mut running_sum = vec![0.0_f32; qh];
                let mut output_acc = vec![0.0_f32; qh * head_dim as usize];

                for (kv_start, kv_end) in &kv_chunks {
                    let kv_chunk_len = (*kv_end - *kv_start) as i64;

                    let k_chunk = k.narrow(2, *kv_start as i64, kv_chunk_len);
                    let v_chunk = v.narrow(2, *kv_start as i64, kv_chunk_len);

                    let q_payload = Self::tensor_to_q_payload(&q_chunk);
                    let kv_payload = Self::tensor_to_kv_payload(&k_chunk, &v_chunk);

                    crate::tch_backend::backend::compute_chunk_attention_step(
                        &q_payload,
                        &kv_payload,
                        kv_chunk_len as i32,
                        q_chunk_len as i32,
                        num_heads as i32,
                        head_dim as i32,
                        &mut running_max,
                        &mut running_sum,
                        &mut output_acc,
                    )
                    .expect("compute_chunk_attention_step failed");
                }

                let out_tensor = Tensor::from_slice(&output_acc)
                    .reshape([num_heads as i64, q_chunk_len, head_dim])
                    .permute(&[1, 0, 2])
                    .unsqueeze(0)
                    .to(q.device()); // [1, q_chunk_len, num_heads, head_dim]
                outputs.push(out_tensor);
            }
        }

        if attention_mask.is_some() {
            // outputs are [batch, num_heads, q_chunk_len, head_dim]; cat on dim 2
            Tensor::cat(&outputs, 2)
        } else {
            // outputs are [1, q_chunk_len, num_heads, head_dim]; cat on dim 1, then permute
            Tensor::cat(&outputs, 1).permute(&[0, 2, 1, 3]).to(q.device())
        }
    }

    /// Convert Q tensor to payload bytes: [query_len, num_heads, head_dim] row-major.
    #[cfg(feature = "tch-backend")]
    fn tensor_to_q_payload(q: &Tensor) -> Vec<u8> {
        // q shape: [batch(1), num_heads, query_len, head_dim]
        let q_perm = q.permute(&[0, 2, 1, 3]).contiguous(); // [batch, query_len, num_heads, head_dim]
        let flat = q_perm.view(-1);
        let values: Vec<f32> = Vec::try_from(&flat).unwrap();
        values.iter().flat_map(|&v| v.to_le_bytes()).collect()
    }

    /// Convert K/V tensors to payload bytes: [2, block_len, num_heads, head_dim] row-major.
    #[cfg(feature = "tch-backend")]
    fn tensor_to_kv_payload(k: &Tensor, v: &Tensor) -> Vec<u8> {
        // k/v shape: [batch(1), num_heads, block_len, head_dim]
        let k_perm = k.permute(&[0, 2, 1, 3]).contiguous(); // [batch, block_len, num_heads, head_dim]
        let v_perm = v.permute(&[0, 2, 1, 3]).contiguous();
        let k_flat = k_perm.view(-1);
        let v_flat = v_perm.view(-1);
        let k_values: Vec<f32> = Vec::try_from(&k_flat).unwrap();
        let v_values: Vec<f32> = Vec::try_from(&v_flat).unwrap();
        let mut payload = Vec::with_capacity((k_values.len() + v_values.len()) * 4);
        for &v in &k_values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &v_values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        payload
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
        attn_weights.matmul(&v)
    }
}

#[cfg(feature = "tch-backend")]
impl AttentionBackend for HcpRingAttentionBackend {
    fn forward(
        &mut self,
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
        let k = if num_rep > 1 { k.repeat(&[1, num_rep as i64, 1, 1]) } else { k };
        let v = if num_rep > 1 { v.repeat(&[1, num_rep as i64, 1, 1]) } else { v };

        // Ring attention
        let attn_output = self.ring_attention(&q, &k, &v, attention_mask);

        // Reshape back and O-projection
        let attn_output = attn_output.transpose(1, 2).contiguous().view([batch, seq_len, hidden_size]);
        Ok(attn_output.matmul(&self.o_proj.transpose(0, 1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_attention(device: tch::Device) -> super::super::layers::GqaAttention {
        let hidden_size = 64i64;
        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;

        super::super::layers::GqaAttention {
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
            rope: super::super::layers::RotaryEmbedding::new(head_dim, 128, 10000.0, device),
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Create a causal mask for local attention testing.
    fn make_causal_mask(seq_len: i64, device: tch::Device) -> Tensor {
        let mask = Tensor::ones(&[seq_len, seq_len], (Kind::Float, device))
            .triu(1)
            .to_kind(Kind::Bool);
        Tensor::zeros(&[seq_len, seq_len], (Kind::Float, device))
            .masked_fill(&mask, f64::NEG_INFINITY)
            .unsqueeze(0)
            .unsqueeze(0)
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_chunk_step_vs_softmax_single_block() {
        let device = tch::Device::Cpu;
        let query_len = 4i64;
        let block_len = 4i64;
        let num_heads = 2i64;
        let head_dim = 8i64;

        tch::manual_seed(42);
        let q = Tensor::randn(&[1, num_heads, query_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn(&[1, num_heads, block_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn(&[1, num_heads, block_len, head_dim], (Kind::Float, device));

        // Local attention
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // Chunk attention step
        let q_payload = HcpRingAttentionBackend::tensor_to_q_payload(&q);
        let kv_payload = HcpRingAttentionBackend::tensor_to_kv_payload(&k, &v);

        println!("q_payload len={}, kv_payload len={}", q_payload.len(), kv_payload.len());

        let mut running_max = vec![f32::NEG_INFINITY; (query_len * num_heads) as usize];
        let mut running_sum = vec![0.0_f32; (query_len * num_heads) as usize];
        let mut output_acc = vec![0.0_f32; (query_len * num_heads * head_dim) as usize];

        crate::tch_backend::backend::compute_chunk_attention_step(
            &q_payload, &kv_payload,
            block_len as i32, query_len as i32, num_heads as i32, head_dim as i32,
            &mut running_max, &mut running_sum, &mut output_acc,
        ).unwrap();

        let actual = Tensor::from_slice(&output_acc)
            .reshape([num_heads, query_len, head_dim])
            .unsqueeze(0);

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
        let hidden = Tensor::randn(&[batch, seq_len, hidden_size], (Kind::Float, device));
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
        let q = Tensor::randn(&[1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn(&[1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn(&[1, num_heads, seq_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // Create a minimal backend (only needs q/k/v/o_proj for local_attention_scores)
        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
        };

        let actual = backend.ring_attention(&q, &k, &v, None);

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
        let q = Tensor::randn(&[1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn(&[1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn(&[1, num_heads, seq_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;

        let mask = make_causal_mask(seq_len, device);
        let scores_masked = scores + mask.shallow_clone();
        let attn = scores_masked.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        let rope = super::super::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn(&[1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
        };

        let actual = backend.ring_attention(&q, &k, &v, Some(&mask));

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Ring vs local causal diff = {}", diff_val);
        assert!(diff_val < 1e-4, "Ring attention differs from local causal: {}", diff_val);
    }
}
