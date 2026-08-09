//! Diagnostic: run the dense reference prefill forward on a given device and
//! print the top-8 logits of the last position, plus a CPU-vs-device diff.
//! Usage: dense_forward_probe <cpu|cuda:N> [model_dir]

use hcp_ringattn_rust::{LlamaModel, ModelConfig, ModelWeights};
use tch::{Device, Kind, Tensor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let device = match args.get(1).map(|s| s.as_str()) {
        Some("cpu") | None => Device::Cpu,
        Some(s) if s.starts_with("cuda:") => Device::Cuda(s[5..].parse().expect("cuda index")),
        Some("mps") => Device::Mps,
        Some(other) => panic!("unknown device {other}"),
    };
    let model_dir = args.get(2).cloned().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen2-0.5B").to_string()
    });
    let model_dir = std::path::Path::new(&model_dir);
    let config = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
    let weights = ModelWeights::from_dir(model_dir, device).unwrap();
    let mut model = LlamaModel::from_weights(config, &weights, device, 1).unwrap();

    let prompt = [151644_i64, 9707, 0, 16];
    let input = Tensor::from_slice(&prompt).unsqueeze(0).to_device(device);
    let mut caches = model.create_kv_caches();
    let logits = model
        .forward(&input, &mut caches)
        .unwrap()
        .select(1, prompt.len() as i64 - 1)
        .squeeze()
        .to_kind(Kind::Float)
        .to_device(Device::Cpu);
    let values = Vec::<f32>::try_from(&logits).unwrap();
    let mut top: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("device={device:?} top-8 last-position logits:");
    for (idx, val) in top.iter().take(8) {
        println!("  token {idx}: {val:.6}");
    }
    if device != Device::Cpu {
        let weights_cpu = ModelWeights::from_dir(model_dir, Device::Cpu).unwrap();
        let config = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
        let mut cpu_model = LlamaModel::from_weights(config, &weights_cpu, Device::Cpu, 1).unwrap();
        let input_cpu = Tensor::from_slice(&prompt).unsqueeze(0);
        let mut cpu_caches = cpu_model.create_kv_caches();
        let cpu_logits = cpu_model
            .forward(&input_cpu, &mut cpu_caches)
            .unwrap()
            .select(1, prompt.len() as i64 - 1)
            .squeeze()
            .to_kind(Kind::Float);
        let diff = (&logits - &cpu_logits).abs();
        println!(
            "vs cpu: mean_diff={:.6} max_diff={:.6}",
            diff.mean(Kind::Float).double_value(&[]),
            diff.max().double_value(&[])
        );
    }
}
