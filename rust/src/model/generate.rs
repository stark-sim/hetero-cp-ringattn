use crate::model::{LlamaModel, ModelError};
use tokenizers::Tokenizer;

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

/// Sample a single token from logits.
///
/// - `temperature == 0.0`: greedy argmax.
/// - `temperature > 0.0`: temperature scaling + optional top-p filtering + multinomial sampling.
#[cfg(feature = "tch-backend")]
pub(crate) fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32, ModelError> {
    // Greedy decoding
    if temperature <= 0.0 {
        return Ok(logits.argmax(-1, false).int64_value(&[]) as u32);
    }

    // Temperature scaling
    let scaled_logits = logits / temperature;

    // Top-p (nucleus) filtering
    let filtered_logits = if top_p > 0.0 && top_p < 1.0 {
        let probs = scaled_logits.softmax(-1, Kind::Float);
        // Sort descending: (values, indices)
        let sorted = probs.sort(-1, true);
        let sorted_probs = sorted.0;
        let sorted_indices = sorted.1;
        // Cumulative sum
        let cumsum = sorted_probs.cumsum(-1, Kind::Float);
        // Find tokens to remove: cumsum > top_p
        let remove_mask = cumsum.gt(top_p);
        // Set removed probabilities to 0
        let filtered_sorted = sorted_probs * remove_mask.logical_not().to_kind(Kind::Float);
        // Scatter back to original positions and renormalize
        let filtered = Tensor::zeros_like(&probs)
            .scatter_add(-1, &sorted_indices, &filtered_sorted);
        // Avoid log(0) by adding epsilon
        let epsilon = 1e-10;
        (filtered.clamp_min(epsilon)).log()
    } else {
        scaled_logits.shallow_clone()
    };

    // Convert to probabilities and sample
    let probs = filtered_logits.softmax(-1, Kind::Float);
    // Clamp to avoid numerical issues
    let safe_probs = probs.clamp_min(1e-10);
    let sampled = safe_probs.multinomial(1, true);
    Ok(sampled.int64_value(&[]) as u32)
}

/// Autoregressive text generator (single-node).
///
/// Handles tokenization, prefill, decode loop, and sampling.
#[cfg(feature = "tch-backend")]
pub struct Generator {
    model: LlamaModel,
    tokenizer: Tokenizer,
    device: Device,
}

#[cfg(feature = "tch-backend")]
impl Generator {
    /// Create a generator from a loaded model and tokenizer file path.
    pub fn new(model: LlamaModel, tokenizer_path: &str, device: Device) -> Result<Self, ModelError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(Self { model, tokenizer, device })
    }

    /// Generate text from a prompt.
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize, temperature: f64, top_p: f64) -> Result<String, ModelError> {
        let prompt_ids = self.tokenize(prompt)?;
        let generated_ids = self.generate_tokens(&prompt_ids, max_new_tokens, temperature, top_p)?;
        self.decode_tokens(&generated_ids)
    }

    /// Tokenize prompt text into IDs.
    fn tokenize(&self, prompt: &str) -> Result<Vec<i64>, ModelError> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Decode generated IDs back to text.
    fn decode_tokens(&self, ids: &[u32]) -> Result<String, ModelError> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))
    }

    /// Core autoregressive generation from token IDs.
    fn generate_tokens(&mut self, prompt_ids: &[i64], max_new_tokens: usize, temperature: f64, top_p: f64) -> Result<Vec<u32>, ModelError> {
        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut kv_caches = self.model.create_kv_caches();

        // Prefill
        let input_tensor = Tensor::from_slice(prompt_ids)
            .unsqueeze(0)
            .to_device(self.device);
        let mut logits = self.model.forward(&input_tensor, &mut kv_caches)?;

        // Decode loop
        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_token = self.model.config.eos_token_id();

        for _ in 0..max_new_tokens {
            let seq_len = logits.size()[1];
            let last_logits = logits.narrow(1, seq_len - 1, 1).squeeze();

            let next_token_id = sample_token(&last_logits, temperature, top_p)?;
            generated_ids.push(next_token_id);

            if Some(next_token_id) == eos_token {
                break;
            }

            let next_input = Tensor::from_slice(&[next_token_id as i64])
                .unsqueeze(0)
                .to_device(self.device);
            logits = self.model.forward(&next_input, &mut kv_caches)?;
        }

        Ok(generated_ids)
    }
}

/// Distributed autoregressive text generator.
///
/// Simulates multi-domain CP inference in a single process.
/// Each domain holds a `LlamaModel` with its own KV cache; KV blocks are
/// exchanged via `LinkedMockKvTransport` during decode.
#[cfg(feature = "tch-backend")]
#[allow(dead_code)]
pub struct DistributedGenerator {
    models: Vec<LlamaModel>,
    tokenizer: Tokenizer,
    device: Device,
    num_domains: usize,
}

#[cfg(feature = "tch-backend")]
#[allow(dead_code)]
impl DistributedGenerator {
    /// Create a distributed generator from pre-configured models.
    ///
    /// Caller must have already called `setup_distributed_domain` on each model
    /// with appropriate transports and `seq_offset`.
    pub fn new(models: Vec<LlamaModel>, tokenizer_path: &str, device: Device) -> Result<Self, ModelError> {
        let num_domains = models.len();
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(Self { models, tokenizer, device, num_domains })
    }

    /// Generate text from a prompt.
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize, temperature: f64, top_p: f64) -> Result<String, ModelError> {
        let prompt_ids = self.tokenize(prompt)?;
        let generated_ids = self.generate_tokens(&prompt_ids, max_new_tokens, temperature, top_p)?;
        self.decode_tokens(&generated_ids)
    }

    /// Tokenize prompt text into IDs.
    fn tokenize(&self, prompt: &str) -> Result<Vec<i64>, ModelError> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Decode generated IDs back to text.
    fn decode_tokens(&self, ids: &[u32]) -> Result<String, ModelError> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))
    }

    /// Core distributed autoregressive generation from token IDs.
    ///
    /// Prefill is split across domains; decode broadcasts the sampled token to
    /// all domains and collects logits from any domain (they should be identical).
    pub fn generate_tokens(&mut self, prompt_ids: &[i64], max_new_tokens: usize, temperature: f64, top_p: f64) -> Result<Vec<u32>, ModelError> {
        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut kv_caches_list: Vec<_> = self.models.iter().map(|m| m.create_kv_caches()).collect();

        // ====== Prefill: split prompt into domain chunks ======
        let seq_len = prompt_ids.len() as i64;
        let chunk_size = (seq_len as usize).div_ceil(self.num_domains).max(1) as i64;
        let mut prefill_logits = Vec::with_capacity(self.num_domains);

        for (domain_id, model) in self.models.iter_mut().enumerate() {
            let start = (domain_id as i64 * chunk_size).min(seq_len);
            let end = ((domain_id as i64 + 1) * chunk_size).min(seq_len);
            let chunk_len = end - start;
            if chunk_len <= 0 {
                continue;
            }
            let chunk = Tensor::from_slice(&prompt_ids[start as usize..end as usize])
                .unsqueeze(0)
                .to_device(self.device);
            let logits = model.forward(&chunk, &mut kv_caches_list[domain_id])?;
            prefill_logits.push(logits);
        }

        // Synchronize global_seq_len across all domains for decode.
        // In a real multi-process setup, the coordinator broadcasts the global prompt length.
        let max_global_seq_len = self.models.iter().map(|m| m.global_seq_len).max().unwrap_or(0);
        for model in self.models.iter_mut() {
            model.global_seq_len = max_global_seq_len;
        }

        // ====== Decode loop ======
        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_token = self.models[0].config.eos_token_id();

        // Sample first token from the last domain's prefill output
        let last_domain = self.num_domains.saturating_sub(1);
        let first_logits = prefill_logits[last_domain]
            .narrow(1, prefill_logits[last_domain].size()[1] - 1, 1)
            .squeeze();
        let mut next_token_id = sample_token(&first_logits, temperature, top_p)? as i64;

        for _ in 0..max_new_tokens {
            let token = next_token_id as u32;
            generated_ids.push(token);

            if Some(token) == eos_token {
                break;
            }

            let decode_input = Tensor::from_slice(&[next_token_id])
                .unsqueeze(0)
                .to_device(self.device);

            // All domains perform distributed decode
            let mut decode_logits_list = Vec::with_capacity(self.num_domains);
            for (domain_id, model) in self.models.iter_mut().enumerate() {
                let logits = model.forward(&decode_input, &mut kv_caches_list[domain_id])?;
                decode_logits_list.push(logits);
            }

            // All domains should produce identical logits; sample from domain0.
            let sample_logits = decode_logits_list[0].squeeze();
            next_token_id = sample_token(&sample_logits, temperature, top_p)? as i64;
        }

        Ok(generated_ids)
    }
}

#[cfg(test)]
#[cfg(feature = "tch-backend")]
mod tests {
    use crate::model::{
        config::ModelConfig,
        model::LlamaModel,
        weights::{ModelWeights, WeightNames},
    };
    use tch::{Device, Kind, Tensor};

    fn create_synthetic_weights(config: &ModelConfig, device: Device) -> ModelWeights {
        let mut tensors = std::collections::HashMap::new();
        let hidden = config.hidden_size as i64;
        let vocab = config.vocab_size as i64;
        let intermediate = config.intermediate_size as i64;
        let head_dim = (config.hidden_size / config.num_heads) as i64;
        let num_heads = config.num_heads as i64;
        let num_kv_heads = config.num_kv_heads.unwrap_or(config.num_heads) as i64;

        tensors.insert(WeightNames::embedding().to_string(), Tensor::randn([vocab, hidden], (Kind::Float, device)));
        tensors.insert(WeightNames::layer_norm().to_string(), Tensor::ones([hidden], (Kind::Float, device)));
        tensors.insert(WeightNames::lm_head().to_string(), Tensor::randn([vocab, hidden], (Kind::Float, device)));

        for layer in 0..config.num_layers {
            tensors.insert(WeightNames::rms_norm_weight(layer), Tensor::ones([hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::post_attn_norm_weight(layer), Tensor::ones([hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::q_proj_weight(layer), Tensor::randn([num_heads * head_dim, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::k_proj_weight(layer), Tensor::randn([num_kv_heads * head_dim, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::v_proj_weight(layer), Tensor::randn([num_kv_heads * head_dim, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::o_proj_weight(layer), Tensor::randn([hidden, num_heads * head_dim], (Kind::Float, device)));
            tensors.insert(WeightNames::gate_proj_weight(layer), Tensor::randn([intermediate, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::up_proj_weight(layer), Tensor::randn([intermediate, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::down_proj_weight(layer), Tensor::randn([hidden, intermediate], (Kind::Float, device)));
        }

        ModelWeights { tensors }
    }

    /// Build models and transports for a 2-domain distributed setup.
    fn setup_distributed_models(config: &ModelConfig, weights: &ModelWeights, device: Device) -> (LlamaModel, LlamaModel, LlamaModel) {
        let num_domains = 2usize;
        let num_layers = config.num_layers;

        let ref_model = LlamaModel::from_weights(config.clone(), weights, device, 1).unwrap();
        let mut domain0 = LlamaModel::from_weights(config.clone(), weights, device, num_domains).unwrap();
        let mut domain1 = LlamaModel::from_weights(config.clone(), weights, device, num_domains).unwrap();

        let mut transports0 = Vec::with_capacity(num_layers);
        let mut transports1 = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let (t0, t1) = crate::model::kv_transport::LinkedMockKvTransport::create_pair();
            transports0.push(t0);
            transports1.push(t1);
        }
        domain0.setup_distributed_domain(0, 0, |layer_idx| Some(Box::new(transports0[layer_idx].clone())));
        domain1.setup_distributed_domain(1, 8, |layer_idx| Some(Box::new(transports1[layer_idx].clone())));

        // Prefill sets global_seq_len per-domain; synchronize after prefill.
        // (Caller will run prefill and sync.)
        (ref_model, domain0, domain1)
    }

    /// Verify distributed token generation matches single-node reference.
    #[test]
    fn test_distributed_generator_tokens_match_reference() {
        let device = Device::Cpu;
        let config = ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: Some(1),
            intermediate_size: 64,
            vocab_size: 100,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            hidden_act: "silu".to_string(),
            max_position_embeddings: Some(128),
            attention_dropout: 0.0,
            bos_token_id: None,
            eos_token_id: None,
            use_cache: true,
            sliding_window: None,
            use_sliding_window: None,
            partial_rotary_factor: 1.0,
        };
        let weights = create_synthetic_weights(&config, device);

        let (mut ref_model, mut domain0, mut domain1) = setup_distributed_models(&config, &weights, device);

        // Prompt: 16 tokens
        let prompt_ids: Vec<i64> = (0..16i64).collect();

        // Reference: single-node generation
        let mut ref_caches = ref_model.create_kv_caches();
        let ref_input = Tensor::from_slice(&prompt_ids).unsqueeze(0).to_device(device);
        let mut ref_logits = ref_model.forward(&ref_input, &mut ref_caches).unwrap();

        // Distributed: prefill split
        let mut caches0 = domain0.create_kv_caches();
        let input0 = Tensor::from_slice(&prompt_ids[0..8]).unsqueeze(0).to_device(device);
        let _ = domain0.forward(&input0, &mut caches0).unwrap();

        let mut caches1 = domain1.create_kv_caches();
        let input1 = Tensor::from_slice(&prompt_ids[8..16]).unsqueeze(0).to_device(device);
        let _ = domain1.forward(&input1, &mut caches1).unwrap();

        // Synchronize global_seq_len
        let global_prompt_len = domain1.global_seq_len;
        domain0.global_seq_len = global_prompt_len;

        // Greedy decode 4 steps.
        // We compare logits diff rather than exact token IDs because
        // online softmax can produce ~2e-6 numerical differences that
        // occasionally flip argmax at decision boundaries.
        const MAX_STEPS: usize = 4;
        const DECODE_TOL: f64 = 1e-3;

        for step in 0..MAX_STEPS {
            let ref_last = ref_logits.narrow(1, ref_logits.size()[1] - 1, 1).squeeze();
            let ref_token = ref_last.argmax(-1, false).int64_value(&[]) as i64;

            let decode_input = Tensor::from_slice(&[ref_token]).unsqueeze(0).to_device(device);

            let d0_logits = domain0.forward(&decode_input, &mut caches0).unwrap();
            let d1_logits = domain1.forward(&decode_input, &mut caches1).unwrap();

            // Domain0 and domain1 must agree (same distributed backend)
            let d0_token = d0_logits.squeeze().argmax(-1, false).int64_value(&[]) as i64;
            let d1_token = d1_logits.squeeze().argmax(-1, false).int64_value(&[]) as i64;
            assert_eq!(d0_token, d1_token, "domain0/domain1 token mismatch at step {}", step);

            // Reference also decodes the same token for fair comparison
            let ref_decode_logits = ref_model.forward(&decode_input, &mut ref_caches).unwrap();
            let ref_decode_last = ref_decode_logits.squeeze();

            // Logits diff vs reference should be small
            let diff0 = (&ref_decode_last - &d0_logits.squeeze()).abs().mean(Kind::Float).double_value(&[]);
            let diff1 = (&ref_decode_last - &d1_logits.squeeze()).abs().mean(Kind::Float).double_value(&[]);
            println!("step {} diff ref-d0={:.2e} ref-d1={:.2e} d0_token={} d1_token={}",
                     step, diff0, diff1, d0_token, d1_token);
            assert!(diff0 < DECODE_TOL, "domain0 logits diff too large at step {}: {}", step, diff0);
            assert!(diff1 < DECODE_TOL, "domain1 logits diff too large at step {}: {}", step, diff1);

            ref_logits = ref_decode_logits;
        }
    }
}
