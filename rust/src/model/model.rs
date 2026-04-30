use crate::model::{
    backend::LocalAttentionBackend,
    cache::{create_kv_caches, KvCaches},
    config::ModelConfig,
    layers::{DecoderLayer, GqaAttention, Mlp, RmsNorm, RotaryEmbedding},
    ModelError, ModelWeights, WeightNames,
};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

/// A Llama-family causal language model.
///
/// Supports Llama, Mistral, Qwen2, and compatible architectures.
#[cfg(feature = "tch-backend")]
pub struct LlamaModel {
    pub config: ModelConfig,
    pub embedding: Tensor,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Option<Tensor>,
}

#[cfg(feature = "tch-backend")]
impl LlamaModel {
    /// Build model from loaded safetensors weights and config.
    pub fn from_weights(config: ModelConfig, weights: &ModelWeights, device: Device) -> Result<Self, ModelError> {
        let embedding = weights.get(WeightNames::embedding())?.shallow_clone();

        let norm = RmsNorm::from_weights(weights, WeightNames::layer_norm(), config.rms_norm_eps)?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(weights.get_lm_head(&config)?.shallow_clone())
        };

        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings.unwrap_or(4096),
            config.rope_theta,
            device,
        );

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let input_ln = RmsNorm::from_weights(
                weights,
                &WeightNames::rms_norm_weight(layer_idx),
                config.rms_norm_eps,
            )?;
            let post_attn_ln = RmsNorm::from_weights(
                weights,
                &WeightNames::post_attn_norm_weight(layer_idx),
                config.rms_norm_eps,
            )?;

            let attention = GqaAttention::from_weights(weights, layer_idx, &config, &rope)?;
            let mlp = Mlp::from_weights(weights, layer_idx)?;

            layers.push(DecoderLayer {
                input_layernorm: input_ln,
                post_attention_layernorm: post_attn_ln,
                attention: Box::new(LocalAttentionBackend { attention }),
                mlp,
            });
        }

        Ok(Self { config, embedding, layers, norm, lm_head })
    }

    /// Full forward pass (prefill or single step).
    ///
    /// `input_ids`: `[batch, seq_len]` (Int64)
    /// `kv_caches`: per-layer KV caches; `None` means no caching for that layer
    ///
    /// Returns logits: `[batch, seq_len, vocab_size]`
    pub fn forward(&mut self, input_ids: &Tensor, kv_caches: &mut KvCaches) -> Result<Tensor, ModelError> {
        let batch = input_ids.size()[0];
        let seq_len = input_ids.size()[1];
        let device = input_ids.device();

        // Embedding lookup
        let mut hidden_states = Tensor::embedding(input_ids, &self.embedding, -1, false, false);

        // Position IDs: [batch, seq_len]
        let position_ids = if seq_len > 1 {
            // Prefill: sequential positions
            Tensor::arange(seq_len, (Kind::Int64, device))
                .unsqueeze(0)
                .repeat(&[batch, 1])
        } else {
            // Decode: each sample uses its current cache length as position
            // For simplicity, assume single sample or all samples share cache length
            Tensor::arange(seq_len, (Kind::Int64, device))
                .unsqueeze(0)
                .repeat(&[batch, 1])
        };

        // Causal mask for prefill (not needed for single-token decode)
        let attention_mask = if seq_len > 1 {
            Some(Self::create_causal_mask(seq_len, device))
        } else {
            None
        };

        // Layer stack
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv_cache = kv_caches.get_mut(layer_idx).and_then(|c| c.as_mut());

            // Pass causal mask only to attention backend
            let hidden = if let Some(ref mask) = attention_mask {
                layer.attention.forward(&hidden_states, &position_ids, kv_cache, Some(mask))?
            } else {
                layer.forward(&hidden_states, &position_ids, kv_cache)?
            };

            hidden_states = hidden;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states);

        // LM Head
        let logits = if let Some(ref lm_head) = self.lm_head {
            hidden_states.matmul(&lm_head.transpose(0, 1))
        } else {
            hidden_states.matmul(&self.embedding.transpose(0, 1))
        };

        Ok(logits)
    }

    /// Create a causal attention mask for prefill.
    ///
    /// Shape: `[1, 1, seq_len, seq_len]` — broadcasts over batch and heads.
    fn create_causal_mask(seq_len: i64, device: Device) -> Tensor {
        let mask = Tensor::ones(&[seq_len, seq_len], (Kind::Float, device))
            .triu(1)
            .to_kind(Kind::Bool);
        let additive = mask.to_kind(Kind::Float) * f64::NEG_INFINITY;
        additive.unsqueeze(0).unsqueeze(0)
    }

    /// Create fresh KV caches for all layers.
    pub fn create_kv_caches(&self) -> KvCaches {
        create_kv_caches(self.config.num_layers)
    }
}
