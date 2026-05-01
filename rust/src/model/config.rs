use serde::Deserialize;
use std::path::Path;

/// 【模型配置】从 HuggingFace 的 `config.json` 解析得到的配置结构体。
///
/// 支持 Llama、Mistral、Qwen2 等同家族模型。
/// 不同模型的 config.json 字段可能略有差异，所以大部分字段用 Option 包装或提供默认值。
///
/// 例如 Qwen2-0.5B 的 config.json 片段：
/// {
///   "architectures": ["Qwen2ForCausalLM"],
///   "hidden_size": 896,
///   "num_hidden_layers": 24,
///   "num_attention_heads": 14,
///   "num_key_value_heads": 2,
///   "intermediate_size": 4864,
///   "vocab_size": 151936,
///   "rope_theta": 1000000.0,
///   ...
/// }
///
/// #[derive(Debug, Clone, Deserialize)] 是 Rust 的派生宏：
/// - Debug: 支持用 {:?} 打印调试信息
/// - Clone: 支持深拷贝
/// - Deserialize: 支持从 JSON 自动反序列化（靠 serde 库）
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ModelConfig {
    /// 【模型架构名称】例如 `["LlamaForCausalLM"]` 或 `["Qwen2ForCausalLM"]`。
    /// Option 表示某些模型可能不填这个字段。
    pub architectures: Option<Vec<String>>,

    /// 【隐藏层维度】也叫 `hidden_size`。
    /// 每个 token 经过 embedding 后的向量长度。例如 Qwen2-0.5B 是 896。
    pub hidden_size: usize,

    /// 【Decoder 层数】即 Transformer 有多少层。
    /// #[serde(alias = "num_hidden_layers")] 表示 JSON 中也可能叫 `num_hidden_layers`，
    /// serde 会自动映射到 `num_layers`。
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,

    /// 【Attention Head 数量】即有多少个注意力头。
    /// 例如 Llama-2-7B 是 32，Qwen2-0.5B 是 14。
    #[serde(alias = "num_attention_heads")]
    pub num_heads: usize,

    /// 【GQA 的 Key/Value Head 数量】
    /// GQA（Group Query Attention）通过让多个 Query head 共享同一个 Key/Value head 来节省显存。
    /// 如果不填，默认等于 `num_heads`（标准 MHA，不共享）。
    /// 例如 Qwen2-0.5B 的 `num_kv_heads=2`，`num_heads=14`，表示 14 个 query head 分成 2 组，
    /// 每组共享 1 个 key head 和 1 个 value head。
    #[serde(default, alias = "num_key_value_heads")]
    pub num_kv_heads: Option<usize>,

    /// 【FFN 中间层维度】Feed-Forward Network（前馈网络）中间层的宽度。
    /// 通常是 hidden_size 的 2~4 倍。例如 Qwen2-0.5B 是 4864（约 896 的 5.4 倍）。
    pub intermediate_size: usize,

    /// 【词表大小】模型能识别多少个不同的 token。
    /// 例如 Llama-2 是 32000，Qwen2 是 151936（因为支持多语言，词表更大）。
    pub vocab_size: usize,

    /// 【RoPE 基数 theta】旋转位置编码的旋转角度基数。
    /// 不同模型取值不同：Llama-2 用 10000.0，Llama-3 用 500000.0，Qwen2 用 1000000.0。
    /// 基数越大，长序列的位置区分度越好。
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// 【RMSNorm 的 epsilon】归一化时防止除零的小常数，默认 1e-6。
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// 【是否共享词嵌入】如果为 true，输出层的 lm_head 权重与输入层的 embedding 权重共享。
    /// 这样可以减少参数量。Qwen2-0.5B 设置为 true。
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// 【权重保存的数据类型】例如 "float16"、"bfloat16"、"float32"。
    /// 推理时通常会转换为 float32（或根据设备能力保持 float16），这里只是记录原始类型。
    pub torch_dtype: Option<String>,

    /// 【FFN 激活函数】Llama/Qwen2 使用 "silu"（配合 SwiGLU 门控结构）。
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// 【最大位置嵌入长度】模型支持的最大序列长度。
    /// 例如 Llama-2 是 4096，Qwen2 是 131072（支持超长上下文）。
    pub max_position_embeddings: Option<usize>,

    /// 【Attention Dropout 率】训练时的正则化参数，推理时通常为 0。
    #[serde(default)]
    pub attention_dropout: f32,

    /// 【BOS token ID】Begin Of Sequence，序列开始标记的 ID。
    pub bos_token_id: Option<u32>,

    /// 【EOS token ID】End Of Sequence，序列结束标记的 ID。
    /// 有些模型支持多个 EOS token（例如不同的停止条件），所以用枚举类型。
    pub eos_token_id: Option<EosTokenId>,

    /// 【是否使用 KV Cache】自回归生成时是否缓存历史 K/V。
    /// 默认为 true，几乎所有场景都开启。
    #[serde(default = "default_true")]
    pub use_cache: bool,

    /// 【滑动窗口 Attention 大小】Mistral 等模型使用，限制每个 token 只能看到前面 N 个 token。
    /// 不用时为 None。
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// 【是否启用滑动窗口】
    #[serde(default)]
    pub use_sliding_window: Option<bool>,

    /// 【部分旋转因子】某些变体只把 RoPE 应用到部分维度上。
    /// 默认为 1.0（全部维度都应用 RoPE）。
    #[serde(default = "default_one_f32")]
    pub partial_rotary_factor: f32,
}

/// 【EOS Token ID 的枚举】
/// 有些模型的 config.json 把 eos_token_id 写成一个整数，有些写成整数数组。
/// #[serde(untagged)] 表示 serde 在反序列化时不看标签，而是按顺序尝试每个变体：
/// 先尝试解析成 Single(u32)，如果失败再尝试 Multiple(Vec<u32>)。
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    /// 【获取主 EOS ID】
    /// 如果是 Single 就直接返回；如果是 Multiple 就返回数组中的第一个。
    pub fn primary(&self) -> u32 {
        match self {
            EosTokenId::Single(id) => *id,
            EosTokenId::Multiple(ids) => ids.first().copied().unwrap_or(0),
        }
    }
}

impl ModelConfig {
    /// 【从文件加载配置】读取 HuggingFace 风格的 `config.json`。
    /// P: AsRef<Path> 是 Rust 的泛型约束，表示 P 可以转换成文件路径（支持 String、&str、PathBuf 等）。
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, crate::model::ModelError> {
        let contents = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// 【获取 GQA 的 KV head 数量】
    /// 如果 config 中没有显式设置 num_kv_heads，就回退到 num_heads（标准 MHA）。
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    /// 【计算 head 维度】
    /// head_dim = hidden_size / num_heads
    /// 例如 hidden_size=896, num_heads=14 → head_dim=64。
    /// 注意：这个除法必须整除，否则模型配置有问题。
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// 【判断是否使用 GQA】
    /// 如果 num_kv_heads < num_heads，说明多个 query head 共享 KV head，是 GQA。
    #[allow(dead_code)]
    pub fn uses_gqa(&self) -> bool {
        self.num_kv_heads() < self.num_heads
    }

    /// 【判断是否使用 SwiGLU】
    /// SwiGLU 是一种门控激活结构，用 silu（也叫 swish）作为门控函数。
    #[allow(dead_code)]
    pub fn is_swiglu(&self) -> bool {
        matches!(self.hidden_act.as_str(), "silu" | "swish")
    }

    /// 【获取主 EOS token ID】
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id.as_ref().map(|e| e.primary())
    }
}

/// 【默认值函数】serde 在字段缺失时会调用这些函数获取默认值。
fn default_rope_theta() -> f64 {
    10000.0  // 经典 Llama-2 的默认值
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> String {
    "silu".to_string()  // SwiGLU 默认使用 silu
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
