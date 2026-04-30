use serde::Deserialize;
use std::path::Path;

/// Configuration parsed from a HuggingFace `config.json`.
///
/// Supports Llama, Mistral, Qwen2, and other Llama-family models.
/// Fields are optional where models may differ.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Model architecture name(s), e.g. `["LlamaForCausalLM"]` or `["Qwen2ForCausalLM"]`.
    pub architectures: Option<Vec<String>>,

    /// Hidden dimension size (also called `hidden_size`).
    pub hidden_size: usize,

    /// Number of decoder layers.
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,

    /// Number of attention heads.
    #[serde(alias = "num_attention_heads")]
    pub num_heads: usize,

    /// Number of key/value heads for GQA. Defaults to `num_heads` (standard MHA).
    #[serde(default, alias = "num_key_value_heads")]
    pub num_kv_heads: Option<usize>,

    /// FFN intermediate dimension.
    pub intermediate_size: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// RoPE base theta. Common values: 10000.0 (Llama-2), 500000.0 (Llama-3), 1000000.0 (Qwen2).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// RMSNorm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Whether input and output embeddings are tied.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Torch dtype used for saving weights (e.g. "float16", "bfloat16", "float32").
    /// Inference may convert to float32 depending on backend capability.
    pub torch_dtype: Option<String>,

    /// Activation function in FFN. Llama/Qwen2 use "silu" (for SwiGLU).
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Maximum sequence length supported by position embeddings.
    pub max_position_embeddings: Option<usize>,

    /// Attention dropout rate.
    #[serde(default)]
    pub attention_dropout: f32,

    /// BOS token ID.
    pub bos_token_id: Option<u32>,

    /// EOS token ID.
    pub eos_token_id: Option<EosTokenId>,

    /// Whether to use KV cache.
    #[serde(default = "default_true")]
    pub use_cache: bool,

    /// Sliding window attention size (e.g. Mistral).
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Whether sliding window is enabled.
    #[serde(default)]
    pub use_sliding_window: Option<bool>,

    /// Partial rotary factor (e.g. some variants use partial RoPE).
    #[serde(default = "default_one_f32")]
    pub partial_rotary_factor: f32,
}

/// Some configs encode `eos_token_id` as either a single int or a list of ints.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    pub fn primary(&self) -> u32 {
        match self {
            EosTokenId::Single(id) => *id,
            EosTokenId::Multiple(ids) => ids.first().copied().unwrap_or(0),
        }
    }
}

impl ModelConfig {
    /// Load config from a HuggingFace-style `config.json` file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, crate::model::ModelError> {
        let contents = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Number of key/value heads (GQA). Falls back to `num_heads` if not specified.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    /// Head dimension: `hidden_size / num_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Whether this model uses GQA (grouped query attention).
    pub fn uses_gqa(&self) -> bool {
        self.num_kv_heads() < self.num_heads
    }

    /// Whether the hidden_act implies SwiGLU (silu + gate).
    pub fn is_swiglu(&self) -> bool {
        matches!(self.hidden_act.as_str(), "silu" | "swish")
    }

    /// Primary EOS token ID.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id.as_ref().map(|e| e.primary())
    }
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_true() -> bool {
    true
}

fn default_one_f32() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qwen2_config() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "hidden_act": "silu",
            "max_position_embeddings": 131072,
            "bos_token_id": 151643,
            "eos_token_id": 151643,
            "use_cache": true
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.num_heads, 14);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert!(cfg.uses_gqa());
        assert_eq!(cfg.head_dim(), 64);
        assert!(cfg.is_swiglu());
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.eos_token_id(), Some(151643));
    }

    #[test]
    fn test_parse_llama2_like_config() {
        let json = r#"{
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.num_kv_heads(), 32); // falls back to num_heads
        assert!(!cfg.uses_gqa());
    }
}
