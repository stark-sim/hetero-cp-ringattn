use crate::model::{LlamaModel, ModelError};
use crate::model::sampling::sample_token;
use tokenizers::Tokenizer;

#[cfg(feature = "tch-backend")]
use tch::{Device, Tensor};

/// 【分布式自回归文本生成器（单进程模拟）】
///
/// 在单个进程内模拟多 domain 的 CP 分布式推理。
/// 每个 domain 持有独立的 `LlamaModel` 和 KV cache；
/// decode 阶段通过 `LinkedMockKvTransport` 在内存中交换 KV block。
///
/// 【与真实分布式的区别】
/// - 真实分布式：domain 运行在不同机器上，通过网络（QUIC/TCP）交换 KV
/// - 这个模拟器：domain 在同一个进程内，通过共享内存队列交换 KV
///
/// 【用途】
/// - 快速验证分布式正确性（不需要启动多个进程）
/// - 调试 ring attention 算法
/// - 单元测试中的端到端验证
///
/// 【流程】
/// 1. Prefill: 把 prompt 均分到各 domain，各自独立 prefill
/// 2. Sync: 同步 global_seq_len（真实场景由 coordinator 广播）
/// 3. Decode: 每次把采样到的 token 广播给所有 domain，各自做 ring attention
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
