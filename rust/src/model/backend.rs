use crate::model::{ModelError, KvCache};

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

/// Abstraction over attention computation.
///
/// This is the key decoupling layer between the model and HCP Core.
/// - `LocalAttentionBackend` runs attention on a single device.
/// - `HcpRingAttentionBackend` (Phase 2) distributes attention across domains.
///
/// Future inference engines like vLLM can plug in their own backend
/// or use `HcpRingAttentionBackend` to offload distributed attention.
#[cfg(feature = "tch-backend")]
pub trait AttentionBackend {
    /// Run attention forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - `[batch, seq_len, hidden_size]`
    /// * `position_ids` - `[batch, seq_len]` absolute positions for RoPE
    /// * `kv_cache` - Optional KV cache for incremental decoding
    /// * `attention_mask` - Optional causal/padding mask to add to scores
    ///
    /// # Returns
    /// * `[batch, seq_len, hidden_size]` attention output
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError>;
}

/// Single-device attention backend.
///
/// Wraps `GqaAttention` and runs full attention locally on CPU/MPS/CUDA.
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
        kv_cache: Option<&mut KvCache>,
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

            // Online softmax state for this Q chunk
            let qh = (q_chunk_len * num_heads) as usize;
            let mut running_max = vec![f32::NEG_INFINITY; qh];
            let mut running_sum = vec![0.0_f32; qh];
            let mut output_acc = vec![0.0_f32; qh * head_dim as usize];

            let kv_chunks: Vec<(usize, usize)> = (0..seq_len as usize)
                .step_by(kv_chunk_size)
                .map(|start| (start, (start + kv_chunk_size).min(seq_len as usize)))
                .collect();

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

                // Apply causal mask only when requested (prefill phase)
                let scores = if attention_mask.is_some() {
                    let q_pos = Tensor::arange_start(q_start as i64, q_end as i64, (Kind::Int64, q.device()))
                        .unsqueeze(1)
                        .unsqueeze(0)
                        .unsqueeze(0);
                    let k_pos = Tensor::arange_start(*kv_start as i64, *kv_end as i64, (Kind::Int64, q.device()))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0);
                    let causal = q_pos.ge_tensor(&k_pos);
                    scores.masked_fill(&causal.logical_not(), f64::NEG_INFINITY)
                } else {
                    scores
                };

                if attention_mask.is_some() {
                    // For causal prefill, use the already-masked scores directly
                    // to ensure correct causal masking within each block.
                    let scores_nh = scores.squeeze_dim(0); // [num_heads, query_len, block_len]
                    let v_nh = v_chunk.squeeze_dim(0); // [num_heads, block_len, head_dim]

                    let (local_max, _) = scores_nh.max_dim(2, false);
                    let weights = (&scores_nh - local_max.unsqueeze(2i64)).exp();
                    let local_sum = weights.sum_dim_intlist(&[2i64][..], false, Kind::Float);
                    let local_pv = weights.matmul(&v_nh); // [num_heads, query_len, head_dim]

                    let device = q.device();
                    let mut rm_t = Tensor::from_slice(&running_max)
                        .reshape([num_heads as i64, q_chunk_len])
                        .to(device);
                    let mut rs_t = Tensor::from_slice(&running_sum)
                        .reshape([num_heads as i64, q_chunk_len])
                        .to(device);
                    let mut obh_t = Tensor::from_slice(&output_acc)
                        .reshape([num_heads as i64, q_chunk_len, head_dim])
                        .to(device);

                    let new_max = rm_t.max_other(&local_max);
                    let exp_prev = (&rm_t - &new_max).exp();
                    let exp_local = (&local_max - &new_max).exp();
                    let new_sum = &exp_prev * &rs_t + &exp_local * &local_sum;
                    obh_t = (&exp_prev.unsqueeze(2i64) * &rs_t.unsqueeze(2i64) * &obh_t
                        + &exp_local.unsqueeze(2i64) * &local_pv)
                        / &new_sum.unsqueeze(2i64);
                    rm_t = new_max;
                    rs_t = new_sum;

                    let rm_cpu = rm_t.to(tch::Device::Cpu);
                    let rs_cpu = rs_t.to(tch::Device::Cpu);
                    let obh_cpu = obh_t.to(tch::Device::Cpu);
                    running_max.copy_from_slice(
                        &Vec::<f32>::try_from(&rm_cpu.contiguous().view(-1)).unwrap(),
                    );
                    running_sum.copy_from_slice(
                        &Vec::<f32>::try_from(&rs_cpu.contiguous().view(-1)).unwrap(),
                    );
                    output_acc.copy_from_slice(
                        &Vec::<f32>::try_from(&obh_cpu.contiguous().view(-1)).unwrap(),
                    );
                } else {
                    // Use compute_chunk_attention_step for numerically stable block update
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
                    ).expect("compute_chunk_attention_step failed");
                }
            }

            // Reconstruct output tensor from output_acc
            // output_acc layout: [num_heads, query_len, head_dim] (from compute_chunk_attention_step)
            // We need [1, q_chunk_len, num_heads, head_dim] for cat along dim 1
            let out_tensor = Tensor::from_slice(&output_acc)
                .reshape([num_heads as i64, q_chunk_len, head_dim])
                .permute(&[1, 0, 2])
                .unsqueeze(0); // [1, q_chunk_len, num_heads, head_dim]
            outputs.push(out_tensor);
        }

        // Concatenate along q_chunk_len (dim 1), then permute to [batch, num_heads, seq_len, head_dim]
        Tensor::cat(&outputs, 1).permute(&[0, 2, 1, 3])
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
        let scale = Tensor::from(self.scale).to_kind(Kind::Float);
        let scores = q.matmul(&k.transpose(2, 3)) * scale;

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
        kv_cache: Option<&mut KvCache>,
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

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_matches_local_full() {
        let device = tch::Device::Cpu;
        let attn = create_test_attention(device);
        let hidden_size = attn.q_proj.size()[1];
        let num_heads = attn.num_heads;
        let num_kv_heads = attn.num_kv_heads;
        let head_dim = attn.head_dim;
        let scale = attn.scale;
        let rope = attn.rope.clone();

        let mut local_backend = LocalAttentionBackend {
            attention: super::super::layers::GqaAttention {
                q_proj: attn.q_proj.shallow_clone(),
                k_proj: attn.k_proj.shallow_clone(),
                v_proj: attn.v_proj.shallow_clone(),
                o_proj: attn.o_proj.shallow_clone(),
                q_bias: None, k_bias: None, v_bias: None,
                num_heads, num_kv_heads, head_dim,
                rope: rope.clone(), scale,
            }
        };

        let mut ring_backend = HcpRingAttentionBackend {
            q_proj: attn.q_proj.shallow_clone(),
            k_proj: attn.k_proj.shallow_clone(),
            v_proj: attn.v_proj.shallow_clone(),
            o_proj: attn.o_proj.shallow_clone(),
            q_bias: None, k_bias: None, v_bias: None,
            rope: rope.clone(),
            num_heads, num_kv_heads, head_dim, scale,
            num_domains: 1,
        };

        // Test with a sequence length that splits into 2 chunks
        let batch = 1i64;
        let seq_len = 8i64;
        tch::manual_seed(42);
        let hidden = Tensor::randn(&[batch, seq_len, hidden_size], (Kind::Float, device));
        let pos_ids = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);

        let local_out = local_backend.forward(&hidden, &pos_ids, None, None).unwrap();
        let ring_out = ring_backend.forward(&hidden, &pos_ids, None, None).unwrap();

        assert_eq!(local_out.size(), ring_out.size());

        let local_has_nan = local_out.isnan().any().int64_value(&[]) != 0;
        let ring_has_nan = ring_out.isnan().any().int64_value(&[]) != 0;
        println!("local_has_nan={}, ring_has_nan={}", local_has_nan, ring_has_nan);

        // Compare a single position
        let local_first = local_out.narrow(1, 0, 1).squeeze();
        let ring_first = ring_out.narrow(1, 0, 1).squeeze();
        println!("local_first mean={}", local_first.mean(Kind::Float).double_value(&[]));
        println!("ring_first mean={}", ring_first.mean(Kind::Float).double_value(&[]));

        let diff = (&local_out - &ring_out).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Mean absolute diff between local and ring attention: {}", diff_val);
        assert!(diff_val < 1e-4, "Local and ring attention outputs differ too much: {}", diff_val);
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_matches_local_causal() {
        let device = tch::Device::Cpu;
        let attn = create_test_attention(device);
        let hidden_size = attn.q_proj.size()[1];
        let num_heads = attn.num_heads;
        let num_kv_heads = attn.num_kv_heads;
        let head_dim = attn.head_dim;
        let scale = attn.scale;
        let rope = attn.rope.clone();

        let mut local_backend = LocalAttentionBackend {
            attention: super::super::layers::GqaAttention {
                q_proj: attn.q_proj.shallow_clone(),
                k_proj: attn.k_proj.shallow_clone(),
                v_proj: attn.v_proj.shallow_clone(),
                o_proj: attn.o_proj.shallow_clone(),
                q_bias: None, k_bias: None, v_bias: None,
                num_heads, num_kv_heads, head_dim,
                rope: rope.clone(), scale,
            }
        };

        let mut ring_backend = HcpRingAttentionBackend {
            q_proj: attn.q_proj.shallow_clone(),
            k_proj: attn.k_proj.shallow_clone(),
            v_proj: attn.v_proj.shallow_clone(),
            o_proj: attn.o_proj.shallow_clone(),
            q_bias: None, k_bias: None, v_bias: None,
            rope: rope.clone(),
            num_heads, num_kv_heads, head_dim, scale,
            num_domains: 2,
        };

        let batch = 1i64;
        let seq_len = 6i64;
        tch::manual_seed(123);
        let hidden = Tensor::randn(&[batch, seq_len, hidden_size], (Kind::Float, device));
        let pos_ids = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);

        let mask = make_causal_mask(seq_len, device);

        let local_out = local_backend.forward(&hidden, &pos_ids, None, Some(&mask)).unwrap();
        let ring_out = ring_backend.forward(&hidden, &pos_ids, None, Some(&mask)).unwrap();

        let diff = (&local_out - &ring_out).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Mean absolute diff with causal mask: {}", diff_val);
        assert!(diff_val < 1e-4, "Local and ring attention outputs differ too much: {}", diff_val);
    }
}
