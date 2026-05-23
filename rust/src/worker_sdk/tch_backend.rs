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


#[cfg(test)]
#[cfg(feature = "tch-backend")]
mod tests {
    use super::*;
    use crate::model::config::ModelConfig;
    use crate::model::model::{LlamaModel, create_synthetic_weights};
    use crate::worker_sdk::backend::WorkerBackend;
    use tch::{Device, Kind, Tensor};

    /// Verify that `decode_batch` produces identical logits to individual `decode_request`
    /// calls, and that per-request KV caches remain isolated (no cross-contamination).
    #[test]
    fn test_decode_batch_isolation() {
        let device = Device::Cpu;

        let config = ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: Some(1),
            intermediate_size: 64,
            vocab_size: 100,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            hidden_act: "silu".to_string(),
            max_position_embeddings: Some(128),
            attention_dropout: 0.0,
            bos_token_id: None,
            eos_token_id: None,
            use_cache: true,
            sliding_window: None,
            use_sliding_window: None,
            partial_rotary_factor: 1.0,
        };

        let weights = create_synthetic_weights(&config, device);

        // Create two independent backends from identical weights.
        let model_batch = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let model_ref = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();

        let mut backend_batch = TchWorkerBackend::from_model(model_batch, device, 0);
        let mut backend_ref = TchWorkerBackend::from_model(model_ref, device, 0);

        // Two different prompts (same length to keep things simple)
        let seq_len = 12i64;
        let prompt_a: Vec<i64> = (0..seq_len).collect();
        let prompt_b: Vec<i64> = (10..10 + seq_len).collect();

        // Prefill both requests on both backends
        let (logits_a_batch, _) = backend_batch.prefill_request(1, &prompt_a, 0).unwrap();
        let (logits_b_batch, _) = backend_batch.prefill_request(2, &prompt_b, 0).unwrap();

        let (logits_a_ref, _) = backend_ref.prefill_request(1, &prompt_a, 0).unwrap();
        let (logits_b_ref, _) = backend_ref.prefill_request(2, &prompt_b, 0).unwrap();

        // Verify prefill logits match (sanity check)
        let prefill_diff_a = Tensor::from_slice(&logits_a_batch)
            .f_sub(&Tensor::from_slice(&logits_a_ref)).unwrap()
            .abs().mean(Kind::Float).double_value(&[]);
        let prefill_diff_b = Tensor::from_slice(&logits_b_batch)
            .f_sub(&Tensor::from_slice(&logits_b_ref)).unwrap()
            .abs().mean(Kind::Float).double_value(&[]);
        assert!(prefill_diff_a < 1e-6, "prefill logits mismatch for request A: {}", prefill_diff_a);
        assert!(prefill_diff_b < 1e-6, "prefill logits mismatch for request B: {}", prefill_diff_b);

        // Sample deterministic tokens (argmax, temperature=0)
        let token_a = Tensor::from_slice(&logits_a_batch).argmax(-1, false).int64_value(&[]) as i64;
        let token_b = Tensor::from_slice(&logits_b_batch).argmax(-1, false).int64_value(&[]) as i64;

        // ====== Single-step decode: batch vs individual ======
        let ref_logits_a = backend_ref.decode_request(1, token_a).unwrap();
        let ref_logits_b = backend_ref.decode_request(2, token_b).unwrap();

        let batch_results = backend_batch
            .decode_batch(&[(1, token_a), (2, token_b)])
            .unwrap();

        let batch_logits_a = batch_results.iter().find(|(id, _)| *id == 1).unwrap().1.clone();
        let batch_logits_b = batch_results.iter().find(|(id, _)| *id == 2).unwrap().1.clone();

        let diff_a = Tensor::from_slice(&batch_logits_a)
            .f_sub(&Tensor::from_slice(&ref_logits_a)).unwrap()
            .abs().mean(Kind::Float).double_value(&[]);
        let diff_b = Tensor::from_slice(&batch_logits_b)
            .f_sub(&Tensor::from_slice(&ref_logits_b)).unwrap()
            .abs().mean(Kind::Float).double_value(&[]);

        println!("decode_batch isolation: diff_a={:.2e}, diff_b={:.2e}", diff_a, diff_b);

        const TOL: f64 = 1e-5;
        assert!(diff_a < TOL, "decode_batch logits differ for request A: {}", diff_a);
        assert!(diff_b < TOL, "decode_batch logits differ for request B: {}", diff_b);

        // ====== Multi-step decode: ensure no cross-contamination over 4 steps ======
        const NUM_DECODE_STEPS: usize = 4;
        const LOGITS_TOL: f64 = 1e-3;

        let mut next_token_a = token_a;
        let mut next_token_b = token_b;

        for step in 0..NUM_DECODE_STEPS {
            // Reference: individual decode requests
            let ref_la = backend_ref.decode_request(1, next_token_a).unwrap();
            let ref_lb = backend_ref.decode_request(2, next_token_b).unwrap();

            // Batch decode
            let batch_res = backend_batch
                .decode_batch(&[(1, next_token_a), (2, next_token_b)])
                .unwrap();
            let batch_la = batch_res.iter().find(|(id, _)| *id == 1).unwrap().1.clone();
            let batch_lb = batch_res.iter().find(|(id, _)| *id == 2).unwrap().1.clone();

            let step_diff_a = Tensor::from_slice(&batch_la)
                .f_sub(&Tensor::from_slice(&ref_la)).unwrap()
                .abs().mean(Kind::Float).double_value(&[]);
            let step_diff_b = Tensor::from_slice(&batch_lb)
                .f_sub(&Tensor::from_slice(&ref_lb)).unwrap()
                .abs().mean(Kind::Float).double_value(&[]);

            let token_batch_a = Tensor::from_slice(&batch_la).argmax(-1, false).int64_value(&[]);
            let token_batch_b = Tensor::from_slice(&batch_lb).argmax(-1, false).int64_value(&[]);
            let token_ref_a = Tensor::from_slice(&ref_la).argmax(-1, false).int64_value(&[]);
            let token_ref_b = Tensor::from_slice(&ref_lb).argmax(-1, false).int64_value(&[]);

            println!(
                "Multi-step decode step {}: diff_a={:.2e}, diff_b={:.2e}, tokens=[{},{}] vs ref=[{},{}]",
                step, step_diff_a, step_diff_b, token_batch_a, token_batch_b, token_ref_a, token_ref_b
            );

            assert!(step_diff_a < LOGITS_TOL, "step {} request A logits diff too large: {}", step, step_diff_a);
            assert!(step_diff_b < LOGITS_TOL, "step {} request B logits diff too large: {}", step, step_diff_b);
            assert_eq!(token_batch_a, token_ref_a, "step {} request A token mismatch", step);
            assert_eq!(token_batch_b, token_ref_b, "step {} request B token mismatch", step);

            next_token_a = token_ref_a as i64;
            next_token_b = token_ref_b as i64;
        }
    }
}
