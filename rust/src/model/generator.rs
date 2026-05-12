use crate::model::{LlamaModel, ModelError};
use crate::model::sampling::sample_token;
use tokenizers::Tokenizer;

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

/// 【单节点自回归文本生成器】
///
/// 负责完整的文本生成流程：tokenize → prefill → decode loop → detokenize。
///
/// 使用方式：
/// ```rust,ignore
/// let mut gen = Generator::new(model, "tokenizer.json", Device::Mps)?;
/// let text = gen.generate("Hello", 100, 0.7, 0.9)?;
/// ```
///
/// 内部流程：
/// 1. tokenize: 用 HuggingFace tokenizer 把文本转成 token ID 列表
/// 2. prefill: 把完整 prompt 一次性输入模型，计算 KV cache
/// 3. decode loop: 每次取最后一个 token 的 logits，采样得到 next_token
/// 4. 把 next_token 喂回模型，重复直到 max_new_tokens 或遇到 EOS
///
/// 注意：这是单节点版本，分布式生成由 coordinator + worker 负责。
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

/// Batch autoregressive text generator (single-node).
///
/// Processes multiple prompts in parallel using the same model.
/// All prompts must have the same tokenized length; if they differ,
/// the caller must pad them to equal length before calling.
///
/// Current limitations (by design — correctness first):
/// - Static batching only: all requests start and finish together.
/// - No continuous batching (no dynamic add/remove of requests).
/// - No early stopping: every request runs for the full `max_new_tokens`.
///   Tokens after EOS are still generated but ignored. This avoids the
///   complexity of masking completed sequences in the KV cache, which
///   would require per-sample attention masks during decode.
/// - All prompts must have the same length. Unequal lengths will fail
///   with an error to prevent silent correctness bugs from padding.
///
/// These limitations do not affect correctness; they only affect
/// throughput efficiency. Once the full correctness pipeline is
/// complete, we can relax them incrementally.
#[cfg(feature = "tch-backend")]
pub struct BatchGenerator {
    model: LlamaModel,
    tokenizer: Tokenizer,
    device: Device,
}

#[cfg(feature = "tch-backend")]
impl BatchGenerator {
    /// Create a batch generator from a loaded model and tokenizer file path.
    pub fn new(model: LlamaModel, tokenizer_path: &str, device: Device) -> Result<Self, ModelError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(Self { model, tokenizer, device })
    }

    /// Create a batch generator from a model and an already-loaded tokenizer.
    #[allow(dead_code)]
    pub fn from_model(model: LlamaModel, tokenizer: Tokenizer, device: Device) -> Self {
        Self { model, tokenizer, device }
    }

    /// Generate text for multiple prompts in parallel.
    ///
    /// Returns one `Vec<u32>` of generated token IDs per prompt.
    pub fn generate_batch(
        &mut self,
        prompts: &[&str],
        max_new_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<Vec<Vec<u32>>, ModelError> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all prompts
        let prompt_id_list: Vec<Vec<i64>> = prompts
            .iter()
            .map(|p| self.tokenize(p))
            .collect::<Result<Vec<_>, _>>()?;

        self.generate_batch_from_ids(&prompt_id_list, max_new_tokens, temperature, top_p)
    }

    /// Core batch generation from token IDs.
    ///
    /// `prompt_ids_list[i]` is the token ID list for the i-th request.
    /// All lists must have the same length.
    pub fn generate_batch_from_ids(
        &mut self,
        prompt_ids_list: &[Vec<i64>],
        max_new_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<Vec<Vec<u32>>, ModelError> {
        if prompt_ids_list.is_empty() {
            return Ok(Vec::new());
        }

        // Guard: all prompts must have the same length.
        // This avoids the complexity of padding masks and per-sample
        // position_ids during decode, keeping correctness simple.
        let first_len = prompt_ids_list[0].len();
        if let Some((idx, ids)) = prompt_ids_list.iter().enumerate().find(|(_, ids)| ids.len() != first_len) {
            return Err(ModelError::Generation(format!(
                "BatchGenerator requires all prompts to have the same tokenized length \
                 (prompt 0 has {}, prompt {} has {}). \
                 Pad shorter prompts to equal length before calling.",
                first_len, idx, ids.len()
            )));
        }

        let batch_size = prompt_ids_list.len() as i64;
        let seq_len = first_len as i64;

        // Build batch input tensor: [batch, seq_len]
        let flat_ids: Vec<i64> = prompt_ids_list.iter().flatten().copied().collect();
        let input_tensor = Tensor::from_slice(&flat_ids)
            .reshape([batch_size, seq_len])
            .to_device(self.device);

        let mut kv_caches = self.model.create_kv_caches();

        // Prefill
        let logits = self.model.forward(&input_tensor, &mut kv_caches)?;

        // Decode loop
        let mut all_generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size as usize];
        let eos_token = self.model.config.eos_token_id();
        let mut finished = vec![false; batch_size as usize];

        // Extract last-token logits for each sample in the batch.
        // logits shape: [batch, seq_len, vocab_size]
        let last_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1); // [batch, vocab_size]

        // Sample first token for each request
        let mut next_tokens: Vec<i64> = Vec::with_capacity(batch_size as usize);
        for b in 0..batch_size {
            let sample_logits = last_logits.narrow(0, b, 1).squeeze();
            let token = sample_token(&sample_logits, temperature, top_p)?;
            next_tokens.push(token as i64);
            all_generated[b as usize].push(token);
            if Some(token) == eos_token {
                finished[b as usize] = true;
            }
        }

        for _ in 1..max_new_tokens {
            // Build decode input: [batch, 1]
            let decode_input = Tensor::from_slice(&next_tokens)
                .unsqueeze(1)
                .to_device(self.device); // [batch, 1]

            let decode_logits = self.model.forward(&decode_input, &mut kv_caches)?;
            // decode_logits shape: [batch, 1, vocab_size]
            let decode_last = decode_logits.squeeze_dim(1); // [batch, vocab_size]

            for b in 0..batch_size {
                if finished[b as usize] {
                    // Keep feeding 0 for finished sequences so the KV cache
                    // shape stays consistent across the batch.
                    next_tokens[b as usize] = 0;
                    continue;
                }

                let sample_logits = decode_last.narrow(0, b, 1).squeeze();
                let token = sample_token(&sample_logits, temperature, top_p)?;
                next_tokens[b as usize] = token as i64;
                all_generated[b as usize].push(token);
                if Some(token) == eos_token {
                    finished[b as usize] = true;
                }
            }
        }

        Ok(all_generated)
    }

    /// Tokenize prompt text into IDs.
    fn tokenize(&self, prompt: &str) -> Result<Vec<i64>, ModelError> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
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
    use tokenizers::Tokenizer;

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
            let (t0, t1) = crate::model::transport::LinkedMockKvTransport::create_pair();
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

    /// 【BatchGenerator correctness 验证】
    ///
    /// 验证 BatchGenerator 的 batch=2 输出与两个独立的 batch=1 生成结果一致。
    /// 这是 batching 的端到端 correctness 基线。
    #[test]
    fn test_batch_generator_correctness() {
        use crate::model::generator::BatchGenerator;

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

        // Create a minimal tokenizer in memory (BPE with 27 tokens: a-z + <pad>)
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "vocab": {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25,"<pad>":26},
                "merges": []
            }
        }"#;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json).unwrap();

        let model_batch = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let model_a = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let model_b = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();

        let mut batch_gen = BatchGenerator::from_model(model_batch, tokenizer, device);

        // Two different prompts of the same length
        let prompt_a: Vec<i64> = (0..12i64).collect();
        let prompt_b: Vec<i64> = (10..22i64).collect();

        // Batch generation
        let batch_results = batch_gen.generate_batch_from_ids(
            &[prompt_a.clone(), prompt_b.clone()],
            8,   // max_new_tokens
            0.0, // temperature=0 → greedy
            0.0, // top_p disabled
        ).unwrap();

        // Reference: single-request generation using direct model.forward + greedy sampling
        let mut ref_a = generate_single(model_a, &prompt_a, 8, device);
        let mut ref_b = generate_single(model_b, &prompt_b, 8, device);

        assert_eq!(batch_results[0], ref_a, "batch sample 0 differs from reference");
        assert_eq!(batch_results[1], ref_b, "batch sample 1 differs from reference");

        println!("BatchGenerator correctness passed: batch=2 matches two independent batch=1 runs.");
        println!("  sample0 tokens: {:?}", batch_results[0]);
        println!("  sample1 tokens: {:?}", batch_results[1]);

        // Helper: single-request greedy generation
        fn generate_single(mut model: LlamaModel, prompt_ids: &[i64], max_new: usize, device: Device) -> Vec<u32> {
            let mut caches = model.create_kv_caches();
            let input = Tensor::from_slice(prompt_ids).unsqueeze(0).to_device(device);
            let mut logits = model.forward(&input, &mut caches).unwrap();

            let mut generated = Vec::new();
            let eos = model.config.eos_token_id();

            for _ in 0..max_new {
                let last = logits.narrow(1, logits.size()[1] - 1, 1).squeeze();
                let token = last.argmax(-1, false).int64_value(&[]) as u32;
                generated.push(token);
                if Some(token) == eos {
                    break;
                }
                let next = Tensor::from_slice(&[token as i64]).unsqueeze(0).to_device(device);
                logits = model.forward(&next, &mut caches).unwrap();
            }
            generated
        }
    }
}
