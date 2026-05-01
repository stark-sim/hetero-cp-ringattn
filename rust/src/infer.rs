use crate::model::{ModelConfig, ModelWeights};
use crate::model::model::LlamaModel;
use crate::model::generate::Generator;
use std::path::Path;

#[cfg(feature = "tch-backend")]
pub fn run_inference(model_dir: &str, prompt: &str, max_tokens: usize, temperature: f64, num_domains: usize) -> Result<String, String> {
    use tch::Device;

    let device = if cfg!(target_os = "macos") && tch::utils::has_mps() {
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
    generator.generate(prompt, max_tokens, temperature).map_err(|e: crate::model::ModelError| e.to_string())
}

#[cfg(not(feature = "tch-backend"))]
pub fn run_inference(_model_dir: &str, _prompt: &str, _max_tokens: usize, _temperature: f64) -> Result<String, String> {
    Err("tch-backend feature required".to_string())
}
