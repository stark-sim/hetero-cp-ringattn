//! 【默认后端：`TchWorkerBackend`】
//!
//! 包装现有的 `LlamaModel`（tch-rs），使其适配 `WorkerBackend` trait。
//! 这是 HCP 的默认分布式 Worker 后端；在同构 tch-rs 环境下无需任何改动即可使用。
//!
//! 【生命周期】
//! 1. `TchWorkerBackend::load()` / `from_model()`: 加载权重，创建 KV cache
//! 2. `setup_kv_transports()`: 把 per-layer QUIC transports 绑定到 attention layers
//! 3. `prefill()`: 处理 prompt chunk，计算 logits，更新 KV cache（单请求 backward-compatible）
//! 4. `decode()`: 单 token forward，复用 KV cache（单请求 backward-compatible）
//! 5. `prefill_request() / decode_request()`: 多请求隔离版本，每个 request_id 有独立 KV cache
//! 6. `sync_global_seq_len()`: coordinator 广播后同步全局序列长度
//!
//! 【与真实分布式的关系】
//! `prefill` 和 `decode` 内部调用 `LlamaModel::forward()`，
//! 而 `LlamaModel` 内的 `HcpRingAttentionBackend` 会在 forward 过程中
//! 通过已设置的 `KvTransport` 自动完成 KV ring 交换。
//! 所以 `TchWorkerBackend` 本身不需要关心网络细节。

use crate::model::cache::KvCaches;
use crate::model::transport::KvTransport;
use crate::model::model::LlamaModel;
use crate::model::{ModelConfig, ModelWeights};
use crate::worker_sdk::backend::WorkerBackend;
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Tensor};

/// Per-request context holding the KV cache and model state.
///
/// When a request arrives, `prefill_request()` creates a new `RequestContext`.
/// Each subsequent `decode_request()` uses this context's KV cache and restores
/// the model state (`global_seq_len`, `is_prefill_done`) before forward.
pub struct RequestContext {
    pub kv_caches: KvCaches,
    pub global_seq_len: usize,
    pub is_prefill_done: bool,
}

/// 默认的 tch-rs Worker 后端。
///
/// 包装 `LlamaModel`，负责：
/// - 从 HuggingFace 格式目录加载模型权重
/// - 执行 prefill / decode forward
/// - 在 forward 过程中通过 per-layer `KvTransport` 完成 KV ring 交换
///
/// **多请求支持（M13）**：
/// `request_contexts` 为每个 `request_id` 维护独立的 KV cache 和模型状态。
/// 单请求接口（`prefill()` / `decode()`）继续使用 `self.kv_caches`，保持 backward compatible。
///
/// 使用方式：
/// ```rust,ignore
/// let backend = TchWorkerBackend::load("/path/to/model", Device::Mps, domain_id, num_domains)?;
/// ```
pub struct TchWorkerBackend {
    model: LlamaModel,
    device: Device,
    /// Backward-compatible single-request KV cache.
    kv_caches: KvCaches,
    domain_id: usize,
    /// Per-request KV cache and model state (M13 continuous batching).
    request_contexts: HashMap<u64, RequestContext>,
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
            request_contexts: HashMap::new(),
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
            request_contexts: HashMap::new(),
        }
    }

    /// Shared prefill logic used by both `prefill()` and `prefill_request()`.
    ///
    /// Operates on `self.kv_caches` and updates `self.model` state.
    /// The caller is responsible for saving/restoring state if needed.
    fn do_prefill(&mut self, chunk: &[i64], seq_offset: usize) -> Result<(Vec<f32>, usize), String> {
        // Reset KV cache for a new request.
        self.kv_caches = self.model.create_kv_caches();
        self.model.is_prefill_done = false;
        self.model.global_seq_len = 0;

        self.model.seq_offset = seq_offset as i64;
        // Update per-layer seq_offset for causal mask global position computation
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

    // Note: do_decode removed to avoid borrow checker issues.
    // decode() and decode_request() inline the small forward logic directly.
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
        self.do_prefill(chunk, seq_offset)
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

    /// Request-aware prefill: creates an isolated `RequestContext` for the given request_id.
    fn prefill_request(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        let (logits_vec, global_seq_len) = self.do_prefill(chunk, seq_offset)?;

        // Save the freshly computed KV cache and model state into per-request context.
        self.request_contexts.insert(request_id, RequestContext {
            kv_caches: std::mem::replace(&mut self.kv_caches, self.model.create_kv_caches()),
            global_seq_len: self.model.global_seq_len,
            is_prefill_done: self.model.is_prefill_done,
        });

        Ok((logits_vec, global_seq_len))
    }

    /// Request-aware decode: uses the request's isolated KV cache.
    fn decode_request(&mut self, request_id: u64, token: i64) -> Result<Vec<f32>, String> {
        let ctx = self.request_contexts.get_mut(&request_id)
            .ok_or_else(|| format!("request {request_id} not found"))?;

        // Restore model state from the request's context before forward.
        self.model.global_seq_len = ctx.global_seq_len;
        self.model.is_prefill_done = ctx.is_prefill_done;

        let input = Tensor::from_slice(&[token])
            .unsqueeze(0)
            .to_device(self.device);
        let logits = self
            .model
            .forward(&input, &mut ctx.kv_caches)
            .map_err(|e| format!("decode forward failed: {e}"))?;

        // Save model state back to the request's context after forward.
        ctx.global_seq_len = self.model.global_seq_len;
        ctx.is_prefill_done = self.model.is_prefill_done;

        let logits_vec: Vec<f32> = Vec::try_from(&logits.squeeze())
            .map_err(|e| format!("logits to vec failed: {e}"))?;

        Ok(logits_vec)
    }

    fn sync_global_seq_len(&mut self, len: usize) {
        self.model.global_seq_len = len;
    }

    /// Request-aware sync: updates the per-request context.
    fn sync_global_seq_len_for_request(&mut self, request_id: u64, len: usize) {
        if let Some(ctx) = self.request_contexts.get_mut(&request_id) {
            ctx.global_seq_len = len;
        }
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
