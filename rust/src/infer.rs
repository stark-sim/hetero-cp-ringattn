use crate::model::{ModelConfig, ModelWeights};
use crate::model::model::LlamaModel;
use crate::model::generator::Generator;
use crate::model::sampling::sample_token;
#[cfg(feature = "tch-backend")]
use crate::model::KvCacheImpl;
use std::path::Path;
use std::io::Write;

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

/// Single-node inference with logits export for correctness validation.
///
/// Exports raw little-endian f32 logits to a binary file for later comparison
/// with distributed outputs. File format:
///   - Header: [vocab_size: u64 LE][num_steps: u64 LE]
///   - Body: prefill last-token logits (vocab_size f32 LE) +
///     each decode step's logits (vocab_size f32 LE each)
#[cfg(feature = "tch-backend")]
pub fn run_inference_and_export_logits(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    num_domains: usize,
    export_dir: &str,
) -> Result<String, String> {
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
    println!("[infer-export] device: {:?}", device);

    let config_path = Path::new(model_dir).join("config.json");
    println!("[infer-export] loading config from {:?}", config_path);
    let config = ModelConfig::from_file(&config_path).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[infer-export] loading weights from {}", model_dir);
    let weights = ModelWeights::from_dir(model_dir, device).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[infer-export] building model ({} layers, {} heads)", config.num_layers, config.num_heads);
    println!("[infer-export] config.torch_dtype={:?}", config.torch_dtype);
    let mut model = LlamaModel::from_weights(config, &weights, device, num_domains).map_err(|e: crate::model::ModelError| e.to_string())?;
    println!("[infer-export] model.dtype={:?}", model.dtype);

    // Load tokenizer
    let tokenizer_path = Path::new(model_dir).join("tokenizer.json");
    println!("[infer-export] loading tokenizer from {:?}", tokenizer_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer load failed: {}", e))?;

    // Tokenize prompt
    let prompt_ids = tokenizer.encode(prompt, true)
        .map_err(|e| format!("tokenizer encode failed: {}", e))?
        .get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();
    println!("[infer-export] prompt tokens: {}", prompt_ids.len());

    let mut kv_caches = model.create_kv_caches();

    // Prefill
    let input_tensor = tch::Tensor::from_slice(&prompt_ids)
        .unsqueeze(0)
        .to_device(device);
    let mut logits = model.forward(&input_tensor, &mut kv_caches)
        .map_err(|e| e.to_string())?;

    let mut all_logits: Vec<Vec<f32>> = Vec::new();

    // Extract last-token logits from prefill
    let seq_len = logits.size()[1];
    let mut last_logits = logits.narrow(1, seq_len - 1, 1).squeeze();

    // Decode loop
    let mut generated_ids: Vec<u32> = Vec::new();
    let eos_token = model.config.eos_token_id();

    for step in 0..max_tokens {
        // Save logits that produced this step's token
        let logits_vec: Vec<f32> = Vec::try_from(&last_logits)
            .map_err(|e| format!("failed to convert step {} logits: {}", step, e))?;
        all_logits.push(logits_vec);

        let next_token_id = sample_token(&last_logits, temperature, top_p)
            .map_err(|e| e.to_string())?;
        generated_ids.push(next_token_id);

        if Some(next_token_id) == eos_token {
            break;
        }

        let next_input = tch::Tensor::from_slice(&[next_token_id as i64])
            .unsqueeze(0)
            .to_device(device);
        logits = model.forward(&next_input, &mut kv_caches)
            .map_err(|e| e.to_string())?;
        // After first decode step, logits shape is [batch=1, vocab_size]
        last_logits = logits.squeeze();
    }

    println!("[infer-export] generated {} tokens, total logits chunks: {}", generated_ids.len(), all_logits.len());

    // Write logits to file
    std::fs::create_dir_all(export_dir)
        .map_err(|e| format!("failed to create export dir: {}", e))?;

    let vocab_size = model.config.vocab_size;
    let num_chunks = all_logits.len() as u64;
    let out_path = Path::new(export_dir).join("logits.bin");
    let mut file = std::fs::File::create(&out_path)
        .map_err(|e| format!("failed to create logits file: {}", e))?;

    // Header: vocab_size (u64 LE), num_chunks (u64 LE)
    file.write_all(&vocab_size.to_le_bytes())
        .map_err(|e| format!("failed to write header: {}", e))?;
    file.write_all(&num_chunks.to_le_bytes())
        .map_err(|e| format!("failed to write header: {}", e))?;

    // Body: raw f32 LE
    for (i, chunk) in all_logits.iter().enumerate() {
        if chunk.len() != vocab_size {
            return Err(format!("logits chunk {} size mismatch: expected {}, got {}", i, vocab_size, chunk.len()));
        }
        for &f in chunk {
            file.write_all(&f.to_le_bytes())
                .map_err(|e| format!("failed to write logits: {}", e))?;
        }
    }
    println!("[infer-export] saved logits to {:?}", out_path);

    // Decode generated IDs to text
    let text = tokenizer.decode(&generated_ids, true)
        .map_err(|e| format!("tokenizer decode failed: {}", e))?;
    Ok(text)
}

#[cfg(not(feature = "tch-backend"))]
pub fn run_inference(_model_dir: &str, _prompt: &str, _max_tokens: usize, _temperature: f64, _top_p: f64, _num_domains: usize) -> Result<String, String> {
    Err("tch-backend feature required".to_string())
}

/// Single-node inference that exports per-layer hidden states during decode step 1.
///
/// Used for debugging Rust vs Python transformers numerical divergence.
/// Exports:
/// - `{export_dir}/layer_{i}.bin` — hidden state after each decoder layer (decode step 1)
/// - `{export_dir}/final_norm.bin` — hidden state after final norm (decode step 1)
/// - `{export_dir}/logits.bin` — logits from decode step 1
///
/// Binary format for each hidden state: `[ndims: u64 LE][dim0: u64 LE]...[dimN: u64 LE][f32 data...]`
#[cfg(feature = "tch-backend")]
pub fn run_inference_and_export_hidden_states(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    num_domains: usize,
    export_dir: &str,
) -> Result<String, String> {
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
    println!("[infer-hs] device: {:?}", device);

    let config_path = Path::new(model_dir).join("config.json");
    println!("[infer-hs] loading config from {:?}", config_path);
    let config = ModelConfig::from_file(&config_path).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[infer-hs] loading weights from {}", model_dir);
    let weights = ModelWeights::from_dir(model_dir, device).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[infer-hs] building model ({} layers, {} heads)", config.num_layers, config.num_heads);
    let mut model = LlamaModel::from_weights(config, &weights, device, num_domains).map_err(|e: crate::model::ModelError| e.to_string())?;

    let tokenizer_path = Path::new(model_dir).join("tokenizer.json");
    println!("[infer-hs] loading tokenizer from {:?}", tokenizer_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer load failed: {}", e))?;

    let prompt_ids = tokenizer.encode(prompt, true)
        .map_err(|e| format!("tokenizer encode failed: {}", e))?
        .get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();
    println!("[infer-hs] prompt tokens: {}", prompt_ids.len());

    let mut kv_caches = model.create_kv_caches();

    // Prefill
    let input_tensor = tch::Tensor::from_slice(&prompt_ids)
        .unsqueeze(0)
        .to_device(device);
    let logits = model.forward(&input_tensor, &mut kv_caches)
        .map_err(|e| e.to_string())?;

    // Export KV cache after prefill
    std::fs::create_dir_all(export_dir)
        .map_err(|e| format!("failed to create export dir: {}", e))?;
    for (layer_idx, cache_opt) in kv_caches.iter().enumerate() {
        if let Some(ref cache_impl) = cache_opt {
            if let Some((k, v)) = cache_impl.get_kv() {
                let k_path = std::path::Path::new(export_dir).join(format!("prefill_k_layer_{}.bin", layer_idx));
                let v_path = std::path::Path::new(export_dir).join(format!("prefill_v_layer_{}.bin", layer_idx));
                crate::model::model::LlamaModel::write_tensor_as_binary(&k, &k_path)
                    .map_err(|e| format!("failed to write k cache: {}", e))?;
                crate::model::model::LlamaModel::write_tensor_as_binary(&v, &v_path)
                    .map_err(|e| format!("failed to write v cache: {}", e))?;
            }
        }
    }

    let seq_len = logits.size()[1];
    let last_logits = logits.narrow(1, seq_len - 1, 1).squeeze();

    // Sample first token
    let next_token_id = sample_token(&last_logits, temperature, top_p)
        .map_err(|e| e.to_string())?;
    println!("[infer-hs] first decode token id: {}", next_token_id);

    // Decode step 1 with hidden state export
    let next_input = tch::Tensor::from_slice(&[next_token_id as i64])
        .unsqueeze(0)
        .to_device(device);

    let decode_logits = model.forward_with_hidden_state_export(&next_input, &mut kv_caches, export_dir)
        .map_err(|e| e.to_string())?;

    // Also export decode step 1 logits
    let decode_logits_vec: Vec<f32> = Vec::try_from(&decode_logits.squeeze())
        .map_err(|e| format!("failed to convert decode logits: {}", e))?;

    let vocab_size = model.config.vocab_size;
    let logits_path = Path::new(export_dir).join("logits.bin");
    let mut file = std::fs::File::create(&logits_path)
        .map_err(|e| format!("failed to create logits file: {}", e))?;
    file.write_all(&vocab_size.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&1u64.to_le_bytes()).map_err(|e| e.to_string())?; // 1 chunk
    for &f in &decode_logits_vec {
        file.write_all(&f.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    println!("[infer-hs] saved hidden states and logits to {}", export_dir);

    // Continue generation normally for the rest of max_tokens
    let mut generated_ids = vec![next_token_id];
    let eos_token = model.config.eos_token_id();
    let mut last_logits = decode_logits.squeeze();

    for _step in 1..max_tokens {
        let next_token_id = sample_token(&last_logits, temperature, top_p)
            .map_err(|e| e.to_string())?;
        generated_ids.push(next_token_id);
        if Some(next_token_id) == eos_token {
            break;
        }
        let next_input = tch::Tensor::from_slice(&[next_token_id as i64])
            .unsqueeze(0)
            .to_device(device);
        let logits = model.forward(&next_input, &mut kv_caches)
            .map_err(|e| e.to_string())?;
        last_logits = logits.squeeze();
    }

    let text = tokenizer.decode(&generated_ids, true)
        .map_err(|e| format!("tokenizer decode failed: {}", e))?;
    Ok(text)
}

#[cfg(not(feature = "tch-backend"))]
pub fn run_inference_and_export_logits(
    _model_dir: &str, _prompt: &str, _max_tokens: usize, _temperature: f64, _top_p: f64, _num_domains: usize, _export_dir: &str,
) -> Result<String, String> {
    Err("tch-backend feature required".to_string())
}

#[cfg(not(feature = "tch-backend"))]
pub fn run_inference_and_export_hidden_states(
    _model_dir: &str, _prompt: &str, _max_tokens: usize, _temperature: f64, _top_p: f64, _num_domains: usize, _export_dir: &str,
) -> Result<String, String> {
    Err("tch-backend feature required".to_string())
}

/// 【Prefill layer-0 debug export】
///
/// Runs prefill on the full prompt, but for layer 0 exports ALL attention intermediates:
/// - embedding_output.bin
/// - layer_0_input_norm.bin
/// - q_proj_layer_0.bin, k_proj_layer_0.bin, v_proj_layer_0.bin
/// - q_rope_layer_0.bin, k_rope_layer_0.bin
/// - k_cache_layer_0.bin, v_cache_layer_0.bin
/// - attn_out_layer_0.bin, attn_final_layer_0.bin
/// - layer_0_post_attn.bin, layer_0_post_mlp.bin
///
/// Also exports KV cache for ALL layers after prefill.
///
/// This is for systematic debugging of Rust vs Python transformers numerical divergence.
#[cfg(feature = "tch-backend")]
pub fn run_prefill_debug_layer_0(
    model_dir: &str,
    prompt: &str,
    export_dir: &str,
    qk_inject_dir: Option<&str>,
) -> Result<(), String> {
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
    println!("[prefill-debug] device: {:?}", device);

    let config_path = Path::new(model_dir).join("config.json");
    println!("[prefill-debug] loading config from {:?}", config_path);
    let config = ModelConfig::from_file(&config_path).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[prefill-debug] loading weights from {}", model_dir);
    let weights = ModelWeights::from_dir(model_dir, device).map_err(|e: crate::model::ModelError| e.to_string())?;

    println!("[prefill-debug] building model ({} layers, {} heads)", config.num_layers, config.num_heads);
    let mut model = LlamaModel::from_weights(config, &weights, device, 1)
        .map_err(|e: crate::model::ModelError| e.to_string())?;

    let tokenizer_path = Path::new(model_dir).join("tokenizer.json");
    println!("[prefill-debug] loading tokenizer from {:?}", tokenizer_path);
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer load failed: {}", e))?;

    let prompt_ids = tokenizer.encode(prompt, true)
        .map_err(|e| format!("tokenizer encode failed: {}", e))?
        .get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();
    println!("[prefill-debug] prompt tokens: {}", prompt_ids.len());

    let mut kv_caches = model.create_kv_caches();

    let input_tensor = tch::Tensor::from_slice(&prompt_ids)
        .unsqueeze(0)
        .to_device(device);

    let _logits = model.forward_prefill_debug_layer_0(&input_tensor, &mut kv_caches, export_dir, qk_inject_dir)
        .map_err(|e| e.to_string())?;

    // Export KV cache for all layers
    for (layer_idx, cache_opt) in kv_caches.iter().enumerate() {
        if let Some(ref cache_impl) = cache_opt {
            if let Some((k, v)) = cache_impl.get_kv() {
                let k_path = std::path::Path::new(export_dir).join(format!("prefill_k_layer_{}.bin", layer_idx));
                let v_path = std::path::Path::new(export_dir).join(format!("prefill_v_layer_{}.bin", layer_idx));
                crate::model::model::LlamaModel::write_tensor_as_binary(&k, &k_path)
                    .map_err(|e| format!("failed to write k cache: {}", e))?;
                crate::model::model::LlamaModel::write_tensor_as_binary(&v, &v_path)
                    .map_err(|e| format!("failed to write v cache: {}", e))?;
            }
        }
    }

    println!("[prefill-debug] exported all intermediates to {}", export_dir);
    Ok(())
}

#[cfg(not(feature = "tch-backend"))]
pub fn run_prefill_debug_layer_0(
    _model_dir: &str, _prompt: &str, _export_dir: &str, _qk_inject_dir: Option<&str>,
) -> Result<(), String> {
    Err("tch-backend feature required".to_string())
}
