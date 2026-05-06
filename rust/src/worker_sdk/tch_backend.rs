//! 默认后端：`TchWorkerBackend`
//!
//! 包装现有的 `LlamaModel`（tch-rs），使其适配 `WorkerBackend` trait。
//! 这是 HCP 的默认分布式 Worker 后端；在同构 tch-rs 环境下无需任何改动即可使用。
//!
//! 对于外部框架（vLLM、TensorRT-LLM、MLX 等），需要实现自定义的 `WorkerBackend`。

use crate::model::cache::KvCaches;
use crate::model::kv_transport::KvTransport;
use crate::model::model::LlamaModel;
use crate::model::{ModelConfig, ModelWeights};
use crate::worker_sdk::backend::WorkerBackend;
use std::path::Path;
use tch::{Device, Tensor};

/// 默认的 tch-rs Worker 后端。
///
/// 包装 `LlamaModel`，负责：
/// - 从 HuggingFace 格式目录加载模型权重
/// - 执行 prefill / decode forward
/// - 在 forward 过程中通过 per-layer `KvTransport` 完成 KV ring 交换
///
/// 使用方式：
/// ```rust,ignore
/// let backend = TchWorkerBackend::load("/path/to/model", Device::Mps, domain_id, num_domains)?;
/// ```
pub struct TchWorkerBackend {
    model: LlamaModel,
    device: Device,
    kv_caches: KvCaches,
    domain_id: usize,
}

impl TchWorkerBackend {
    #[allow(dead_code)]
    /// 从模型目录加载权重并创建后端。
    ///
    /// # Arguments
    /// - `model_dir`: HuggingFace 格式目录（`config.json` + `model.safetensors` + `tokenizer.json`）
    /// - `device`: 目标设备
    /// - `domain_id`: 本 domain 的 ID
    /// - `num_domains`: 总 domain 数
    pub fn load(
        model_dir: &str,
        device: Device,
        domain_id: usize,
        num_domains: usize,
    ) -> Result<Self, String> {
        let config_path = Path::new(model_dir).join("config.json");
        let config = ModelConfig::from_file(&config_path)
            .map_err(|e| format!("load config failed: {e}"))?;
        let weights = ModelWeights::from_dir(model_dir, device)
            .map_err(|e| format!("load weights failed: {e}"))?;

        let model =
            LlamaModel::from_weights(config, &weights, device, num_domains)
                .map_err(|e| format!("build model failed: {e}"))?;

        let kv_caches = model.create_kv_caches();

        println!("[TchWorkerBackend] loaded model, device={device:?}, domain_id={domain_id}, num_domains={num_domains}");

        Ok(Self {
            model,
            device,
            kv_caches,
            domain_id,
        })
    }

    /// 从已有的 `LlamaModel` 和 `KvCaches` 创建后端（用于多 domain 权重共享场景）。
    pub fn from_model(model: LlamaModel, device: Device, domain_id: usize) -> Self {
        let kv_caches = model.create_kv_caches();
        Self {
            model,
            device,
            kv_caches,
            domain_id,
        }
    }
}

impl WorkerBackend for TchWorkerBackend {
    fn setup_kv_transports(&mut self, transports: Vec<Box<dyn KvTransport>>) {
        let domain_id = self.domain_id;
        for (layer_idx, transport) in transports.into_iter().enumerate() {
            if let Some(layer) = self.model.layers.get_mut(layer_idx) {
                layer.attention.set_distributed(domain_id, 0, Some(transport));
            }
        }
    }

    fn prefill(
        &mut self,
        chunk: &[i64],
        seq_offset: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        self.model.seq_offset = seq_offset as i64;
        // 更新每层的 seq_offset（用于 causal mask 的全局位置计算）
        for layer in self.model.layers.iter_mut() {
            layer
                .attention
                .set_distributed(self.domain_id, seq_offset, None);
        }

        let input = Tensor::from_slice(chunk)
            .unsqueeze(0)
            .to_device(self.device);
        let logits = self
            .model
            .forward(&input, &mut self.kv_caches)
            .map_err(|e| format!("prefill forward failed: {e}"))?;

        let last_logits = logits.narrow(1, logits.size()[1] - 1, 1).squeeze();
        let logits_vec: Vec<f32> =
            Vec::try_from(&last_logits).map_err(|e| format!("logits to vec failed: {e}"))?;

        Ok((logits_vec, self.model.global_seq_len))
    }

    fn decode(&mut self, token: i64) -> Result<Vec<f32>, String> {
        let input = Tensor::from_slice(&[token])
            .unsqueeze(0)
            .to_device(self.device);
        let logits = self
            .model
            .forward(&input, &mut self.kv_caches)
            .map_err(|e| format!("decode forward failed: {e}"))?;

        let logits_vec: Vec<f32> = Vec::try_from(&logits.squeeze())
            .map_err(|e| format!("logits to vec failed: {e}"))?;

        Ok(logits_vec)
    }

    fn sync_global_seq_len(&mut self, len: usize) {
        self.model.global_seq_len = len;
    }

    fn capacity_mb(&self) -> u64 {
        crate::capacity::query_device_capacity_mb(self.device)
    }

    fn num_layers(&self) -> usize {
        self.model.config.num_layers
    }

    fn device(&self) -> Device {
        self.device
    }
}
