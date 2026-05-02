use crate::model::{LlamaModel, ModelError};
use tokenizers::Tokenizer;

#[cfg(feature = "tch-backend")]
use tch::{Device, Tensor};

/// Autoregressive text generator.
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
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy, >0 = temperature-scaled)
    /// * `top_p` - Nucleus sampling threshold (0.0 = disabled, 1.0 = no filtering)
    ///
    /// # Returns
    /// Generated text string (excluding the prompt)
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize, temperature: f64, top_p: f64) -> Result<String, ModelError> {
        // Tokenize prompt
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        if prompt_ids.is_empty() {
            return Ok(String::new());
        }

        // Create KV caches for all layers
        let mut kv_caches = self.model.create_kv_caches();

        // Prefill: process entire prompt at once
        let input_tensor = Tensor::from_slice(&prompt_ids)
            .unsqueeze(0)
            .to_device(self.device);
        let mut logits = self.model.forward(&input_tensor, &mut kv_caches)?;

        // Decode loop: generate one token at a time
        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_token = self.model.config.eos_token_id();

        for _ in 0..max_new_tokens {
            // Extract logits for the last position: [batch, vocab_size]
            let seq_len = logits.size()[1];
            let last_logits = logits.narrow(1, seq_len - 1, 1).squeeze();

            let next_token_id = self.sample(&last_logits, temperature, top_p)?;
            generated_ids.push(next_token_id);

            // Stop at EOS
            if Some(next_token_id) == eos_token {
                break;
            }

            // Forward single token
            let next_input = Tensor::from_slice(&[next_token_id as i64])
                .unsqueeze(0)
                .to_device(self.device);
            logits = self.model.forward(&next_input, &mut kv_caches)?;
        }

        // Decode tokens to text
        let text = self.tokenizer.decode(&generated_ids, true)
            .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
        Ok(text)
    }

    /// Sample next token from logits.
    ///
    /// - `temperature == 0.0`: greedy argmax.
    /// - `temperature > 0.0`: temperature scaling + optional top-p filtering + multinomial sampling.
    fn sample(&self, logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32, ModelError> {
        use tch::Kind;

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
}
