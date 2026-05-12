use crate::model::{ModelConfig, ModelWeights};
use crate::model::model::LlamaModel;
use crate::model::generator::Generator;
use std::path::Path;

/// 【单节点推理入口】加载模型并执行完整的文本生成。
///
/// 流程：加载 config → 加载权重 → 构建模型 → 加载 tokenizer → 生成文本。
///
/// 【设备选择优先级】
/// 1. `HCP_TCH_DEVICE` / `HCP_TORCH_DEVICE` 环境变量（手动指定）
/// 2. Mac 自动检测 MPS（如果可用）
/// 3. CUDA 自动检测（如果可用）
/// 4. CPU fallback
///
/// `num_domains` 参数：
/// - 1: 单节点，不使用 KV 交换（但底层仍用 HcpRingAttentionBackend）
/// - >1: 准备分布式（但实际分布式推理由 coordinator/worker 模式处理）
#[cfg(feature = "tch-backend")]
pub fn run_inference(model_dir: &str, prompt: &str, max_tokens: usize, temperature: f64, top_p: f64, num_domains: usize) -> Result<String, String> {
    use tch::Device;

    let device = if let Ok(name) = std::env::var("HCP_TORCH_DEVICE").or_else(|_| std::env::var("HCP_TCH_DEVICE")) {
        match name.as_str() {
            "cpu" => Device::Cpu,
            "mps" => Device::Mps,
            "cuda" => Device::Cuda(0),
            _ => {
                if let Some(idx) = name.strip_prefix("cuda:") {
                    if let Ok(i) = idx.parse::<usize>() {
                        Device::Cuda(i)
                    } else {
                        Device::Cpu
                    }
                } else {
                    Device::Cpu
                }
            }
        }
    } else if cfg!(target_os = "macos") && tch::utils::has_mps() {
        Device::Mps
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    println!("[infer] device: {:?}", device);

    let config_path = Path::new(model_dir).join("config.json");
    println!("[infer] loading config from {:?}", config_path);
    let config = ModelConfig::from_file(&config_path).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[infer] loading weights from {}", model_dir);
    let weights = ModelWeights::from_dir(model_dir, device).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[infer] building model ({} layers, {} heads)", config.num_layers, config.num_heads);
    let model = LlamaModel::from_weights(config, &weights, device, num_domains).map_err(|e: crate::model::ModelError| e.to_string())?;

    let tokenizer_path = Path::new(model_dir).join("tokenizer.json");
    println!("[infer] loading tokenizer from {:?}", tokenizer_path);
    let mut generator = Generator::new(model, tokenizer_path.to_str().unwrap(), device)
        .map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[infer] generating (max_tokens={}, temperature={})...", max_tokens, temperature);
    generator.generate(prompt, max_tokens, temperature, top_p).map_err(|e: crate::model::ModelError| e.to_string())
}

#[cfg(not(feature = "tch-backend"))]
pub fn run_inference(_model_dir: &str, _prompt: &str, _max_tokens: usize, _temperature: f64) -> Result<String, String> {
    Err("tch-backend feature required".to_string())
}
