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
    ///
    /// # Returns
    /// Generated text string (excluding the prompt)
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize, temperature: f64) -> Result<String, ModelError> {
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

            let next_token_id = self.sample(&last_logits, temperature)?;
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
    fn sample(&self, logits: &Tensor, temperature: f64) -> Result<u32, ModelError> {
        let logits = if temperature > 0.0 && temperature != 1.0 {
            logits / temperature
        } else {
            logits.shallow_clone()
        };
        Ok(logits.argmax(-1, false).int64_value(&[]) as u32)
    }
}
