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

use crate::model::cache::{KvCacheImpl, KvCaches};
use crate::model::model::LlamaModel;
use crate::model::self_driving::ReservedPositionedKvShard;
use crate::model::transport::KvTransport;
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
        let config =
            ModelConfig::from_file(&config_path).map_err(|e| format!("load config failed: {e}"))?;
        let weights = ModelWeights::from_dir(model_dir, device)
            .map_err(|e| format!("load weights failed: {e}"))?;

        let model = LlamaModel::from_weights(config, &weights, device, num_domains)
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
    fn do_prefill(
        &mut self,
        chunk: &[i64],
        seq_offset: usize,
        position_ids: Option<&[i64]>,
        layer_kv_capacities: Option<&[usize]>,
    ) -> Result<(Vec<f32>, usize), String> {
        let next_kv_caches = if let Some(capacities) = layer_kv_capacities {
            if capacities.len() != self.model.config.num_layers {
                return Err(format!(
                    "reserved prefill requires {} layer capacities, got {}",
                    self.model.config.num_layers,
                    capacities.len()
                ));
            }
            for (layer_idx, &capacity) in capacities.iter().enumerate() {
                if capacity < chunk.len() {
                    return Err(format!(
                        "reserved prefill layer {layer_idx} capacity {capacity} is smaller than local prompt length {}",
                        chunk.len()
                    ));
                }
            }
            capacities
                .iter()
                .map(|&capacity| {
                    Some(KvCacheImpl::ReservedPositioned(
                        ReservedPositionedKvShard::new_with_kind(
                            &self.model.config,
                            capacity,
                            self.device,
                            self.model.dtype,
                        ),
                    ))
                })
                .collect()
        } else {
            self.model.create_kv_caches()
        };

        // Reset KV cache for a new request.
        self.kv_caches = next_kv_caches;
        self.model.is_prefill_done = false;
        self.model.global_seq_len = 0;
        self.model.prefill_position_ids = None;

        self.model.seq_offset = seq_offset as i64;
        // Update per-layer seq_offset and scheduling strategy state.
        for layer in self.model.layers.iter_mut() {
            layer
                .attention
                .set_distributed(self.domain_id, seq_offset, None);
        }

        if let Some(pos) = position_ids {
            self.model.set_prefill_position_ids(pos, self.device);
        }

        let input = Tensor::from_slice(chunk)
            .unsqueeze(0)
            .to_device(self.device);
        let logits = self
            .model
            .forward(&input, &mut self.kv_caches)
            .map_err(|e| format!("prefill forward failed: {e}"))?;

        // Clear one-shot position ids so they are not reused by decode.
        self.model.prefill_position_ids = None;

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
                layer
                    .attention
                    .set_distributed(domain_id, 0, Some(transport));
            }
        }
    }

    fn prefill(
        &mut self,
        chunk: &[i64],
        seq_offset: usize,
        position_ids: Option<&[i64]>,
    ) -> Result<(Vec<f32>, usize), String> {
        self.do_prefill(chunk, seq_offset, position_ids, None)
    }

    fn decode(&mut self, token: i64) -> Result<Vec<f32>, String> {
        let input = Tensor::from_slice(&[token])
            .unsqueeze(0)
            .to_device(self.device);
        let logits = self
            .model
            .forward(&input, &mut self.kv_caches)
            .map_err(|e| format!("decode forward failed: {e}"))?;

        let logits_vec: Vec<f32> =
            Vec::try_from(&logits.squeeze()).map_err(|e| format!("logits to vec failed: {e}"))?;

        Ok(logits_vec)
    }

    /// Request-aware prefill: creates an isolated `RequestContext` for the given request_id.
    fn prefill_request(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
        position_ids: Option<&[i64]>,
    ) -> Result<(Vec<f32>, usize), String> {
        let (logits_vec, global_seq_len) =
            self.do_prefill(chunk, seq_offset, position_ids, None)?;

        // Save the freshly computed KV cache and model state into per-request context.
        self.request_contexts.insert(
            request_id,
            RequestContext {
                kv_caches: std::mem::replace(&mut self.kv_caches, self.model.create_kv_caches()),
                global_seq_len: self.model.global_seq_len,
                is_prefill_done: self.model.is_prefill_done,
            },
        );

        Ok((logits_vec, global_seq_len))
    }

    fn prefill_request_with_reservation(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
        position_ids: Option<&[i64]>,
        layer_kv_capacities: Option<&[usize]>,
    ) -> Result<(Vec<f32>, usize), String> {
        let (logits_vec, global_seq_len) =
            self.do_prefill(chunk, seq_offset, position_ids, layer_kv_capacities)?;

        self.request_contexts.insert(
            request_id,
            RequestContext {
                kv_caches: std::mem::replace(&mut self.kv_caches, self.model.create_kv_caches()),
                global_seq_len: self.model.global_seq_len,
                is_prefill_done: self.model.is_prefill_done,
            },
        );

        Ok((logits_vec, global_seq_len))
    }

    /// Request-aware decode: uses the request's isolated KV cache.
    fn decode_request(&mut self, request_id: u64, token: i64) -> Result<Vec<f32>, String> {
        let ctx = self
            .request_contexts
            .get_mut(&request_id)
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

        let logits_vec: Vec<f32> =
            Vec::try_from(&logits.squeeze()).map_err(|e| format!("logits to vec failed: {e}"))?;

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

    /// Release per-request state to prevent memory leak.
    fn release_request(&mut self, request_id: u64) {
        if self.request_contexts.remove(&request_id).is_some() {
            println!("[TchWorkerBackend] released request {request_id}");
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
    use crate::model::cache::KvCacheImpl;
    use crate::model::config::ModelConfig;
    use crate::model::model::{create_synthetic_weights, LlamaModel};
    use crate::model::self_driving::{
        process_layer_packet_with_reserved_history, project_final_logits, FrozenKvAssigneeSchedule,
        LayerPacket, LayerStepOutcome,
    };
    use crate::worker_sdk::backend::WorkerBackend;
    use tch::{Device, Kind, Tensor};

    fn run_two_backend_reserved_tcp_decode(
        backends: &mut [TchWorkerBackend],
        transports: &mut [crate::model::transport::TcpKvTransport],
        request_id: u64,
        token: i64,
        position: i64,
        starter: usize,
        assignees: &[usize],
    ) -> (Tensor, usize, usize, usize) {
        let domains = backends.len();
        assert_eq!(domains, 2);
        assert_eq!(transports.len(), domains);
        assert_eq!(assignees.len(), backends[0].model.layers.len());

        let input_ids = Tensor::from_slice(&[token])
            .unsqueeze(0)
            .to_device(backends[starter].device);
        let mut hidden_states = Tensor::embedding(
            &backends[starter].model.embedding,
            &input_ids,
            -1,
            false,
            false,
        );
        let position_ids = Tensor::from_slice(&[position])
            .unsqueeze(0)
            .to_device(backends[starter].device);
        let mut current_starter = starter;
        let mut total_hops = 0_usize;
        let mut total_wire_bytes = 0_usize;

        for (layer_idx, &assignee) in assignees.iter().enumerate() {
            let mut packet = Some(
                LayerPacket::start(
                    &mut backends[current_starter].model.layers[layer_idx],
                    &hidden_states,
                    &position_ids,
                    current_starter,
                    assignee,
                    domains,
                )
                .unwrap(),
            );
            let mut next_hidden = None;

            for visit_index in 0..domains {
                let domain = (current_starter + visit_index) % domains;
                let backend = &mut backends[domain];
                let context = backend.request_contexts.get_mut(&request_id).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                };
                let outcome = process_layer_packet_with_reserved_history(
                    &mut backend.model.layers[layer_idx],
                    packet.take().unwrap(),
                    shard,
                )
                .unwrap();
                match outcome {
                    LayerStepOutcome::Forward(next_packet) => {
                        let wire = next_packet.into_self_driving_packet(layer_idx).unwrap();
                        let receiver = wire.current_domain;
                        total_wire_bytes +=
                            transports[domain].send_self_driving_packet(&wire).unwrap();
                        total_hops += 1;
                        let received = transports[receiver]
                            .recv_self_driving_packet()
                            .unwrap()
                            .expect("TCP peer closed before the next self-driving packet");
                        assert_eq!(received.layer_idx, layer_idx);
                        assert_eq!(received.current_domain, receiver);
                        packet = Some(LayerPacket::from_self_driving_packet(received).unwrap());
                    }
                    LayerStepOutcome::Finished { hidden_states, .. } => {
                        assert_eq!(visit_index + 1, domains);
                        next_hidden = Some(hidden_states);
                    }
                }
            }

            hidden_states = next_hidden.expect("the final worker must finish the layer");
            current_starter = (current_starter + domains - 1) % domains;
        }

        let logits = project_final_logits(&backends[current_starter].model, &hidden_states);
        for backend in backends {
            let context = backend.request_contexts.get_mut(&request_id).unwrap();
            context.global_seq_len = position as usize + 1;
        }
        (logits, current_starter, total_hops, total_wire_bytes)
    }

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
        let (logits_a_batch, _) = backend_batch
            .prefill_request(1, &prompt_a, 0, None)
            .unwrap();
        let (logits_b_batch, _) = backend_batch
            .prefill_request(2, &prompt_b, 0, None)
            .unwrap();

        let (logits_a_ref, _) = backend_ref.prefill_request(1, &prompt_a, 0, None).unwrap();
        let (logits_b_ref, _) = backend_ref.prefill_request(2, &prompt_b, 0, None).unwrap();

        // Verify prefill logits match (sanity check)
        let prefill_diff_a = Tensor::from_slice(&logits_a_batch)
            .f_sub(&Tensor::from_slice(&logits_a_ref))
            .unwrap()
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        let prefill_diff_b = Tensor::from_slice(&logits_b_batch)
            .f_sub(&Tensor::from_slice(&logits_b_ref))
            .unwrap()
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        assert!(
            prefill_diff_a < 1e-6,
            "prefill logits mismatch for request A: {}",
            prefill_diff_a
        );
        assert!(
            prefill_diff_b < 1e-6,
            "prefill logits mismatch for request B: {}",
            prefill_diff_b
        );

        // Sample deterministic tokens (argmax, temperature=0)
        let token_a = Tensor::from_slice(&logits_a_batch)
            .argmax(-1, false)
            .int64_value(&[]) as i64;
        let token_b = Tensor::from_slice(&logits_b_batch)
            .argmax(-1, false)
            .int64_value(&[]) as i64;

        // ====== Single-step decode: batch vs individual ======
        let ref_logits_a = backend_ref.decode_request(1, token_a).unwrap();
        let ref_logits_b = backend_ref.decode_request(2, token_b).unwrap();

        let batch_results = backend_batch
            .decode_batch(&[(1, token_a), (2, token_b)])
            .unwrap();

        let batch_logits_a = batch_results
            .iter()
            .find(|(id, _)| *id == 1)
            .unwrap()
            .1
            .clone();
        let batch_logits_b = batch_results
            .iter()
            .find(|(id, _)| *id == 2)
            .unwrap()
            .1
            .clone();

        let diff_a = Tensor::from_slice(&batch_logits_a)
            .f_sub(&Tensor::from_slice(&ref_logits_a))
            .unwrap()
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        let diff_b = Tensor::from_slice(&batch_logits_b)
            .f_sub(&Tensor::from_slice(&ref_logits_b))
            .unwrap()
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);

        println!(
            "decode_batch isolation: diff_a={:.2e}, diff_b={:.2e}",
            diff_a, diff_b
        );

        const TOL: f64 = 1e-5;
        assert!(
            diff_a < TOL,
            "decode_batch logits differ for request A: {}",
            diff_a
        );
        assert!(
            diff_b < TOL,
            "decode_batch logits differ for request B: {}",
            diff_b
        );

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
                .f_sub(&Tensor::from_slice(&ref_la))
                .unwrap()
                .abs()
                .mean(Kind::Float)
                .double_value(&[]);
            let step_diff_b = Tensor::from_slice(&batch_lb)
                .f_sub(&Tensor::from_slice(&ref_lb))
                .unwrap()
                .abs()
                .mean(Kind::Float)
                .double_value(&[]);

            let token_batch_a = Tensor::from_slice(&batch_la)
                .argmax(-1, false)
                .int64_value(&[]);
            let token_batch_b = Tensor::from_slice(&batch_lb)
                .argmax(-1, false)
                .int64_value(&[]);
            let token_ref_a = Tensor::from_slice(&ref_la)
                .argmax(-1, false)
                .int64_value(&[]);
            let token_ref_b = Tensor::from_slice(&ref_lb)
                .argmax(-1, false)
                .int64_value(&[]);

            println!(
                "Multi-step decode step {}: diff_a={:.2e}, diff_b={:.2e}, tokens=[{},{}] vs ref=[{},{}]",
                step, step_diff_a, step_diff_b, token_batch_a, token_batch_b, token_ref_a, token_ref_b
            );

            assert!(
                step_diff_a < LOGITS_TOL,
                "step {} request A logits diff too large: {}",
                step,
                step_diff_a
            );
            assert!(
                step_diff_b < LOGITS_TOL,
                "step {} request B logits diff too large: {}",
                step,
                step_diff_b
            );
            assert_eq!(
                token_batch_a, token_ref_a,
                "step {} request A token mismatch",
                step
            );
            assert_eq!(
                token_batch_b, token_ref_b,
                "step {} request B token mismatch",
                step
            );

            next_token_a = token_ref_a as i64;
            next_token_b = token_ref_b as i64;
        }
    }

    #[test]
    fn reserved_prefill_uses_explicit_per_layer_capacities() {
        let device = Device::Cpu;
        let config = ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 24,
            num_heads: 4,
            num_kv_heads: Some(1),
            intermediate_size: 64,
            vocab_size: 100,
            rope_theta: 10_000.0,
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
        let model = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut backend = TchWorkerBackend::from_model(model, device, 0);
        let prompt = [3_i64, 5];
        let capacities = (0..config.num_layers)
            .map(|layer_idx| prompt.len() + layer_idx % 3)
            .collect::<Vec<_>>();

        let mut under_capacity = capacities.clone();
        under_capacity[7] = prompt.len() - 1;
        let error = backend
            .prefill_request_with_reservation(8, &prompt, 0, None, Some(&under_capacity))
            .unwrap_err();
        assert!(error.contains("layer 7"));
        assert!(!backend.request_contexts.contains_key(&8));

        let (logits, _) = backend
            .prefill_request_with_reservation(9, &prompt, 0, None, Some(&capacities))
            .unwrap();
        assert_eq!(logits.len(), config.vocab_size);

        let context = backend.request_contexts.get(&9).unwrap();
        assert_eq!(context.kv_caches.len(), config.num_layers);
        for (layer_idx, cache) in context.kv_caches.iter().enumerate() {
            let Some(KvCacheImpl::ReservedPositioned(shard)) = cache else {
                panic!("layer {layer_idx} did not use reserved positioned KV");
            };
            assert_eq!(shard.reserved_capacity(), capacities[layer_idx]);
            assert_eq!(shard.committed_len(), prompt.len());
            assert_eq!(shard.positions(), &[0, 1]);
        }
    }

    #[test]
    #[ignore = "requires the local Qwen2-0.5B model weights"]
    fn real_qwen_two_worker_reserved_prefill_matches_reference() {
        let device = Device::Cpu;
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("Qwen2-0.5B");
        let config = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
        assert_eq!(config.num_layers, 24);
        let weights = ModelWeights::from_dir(&model_dir, device).unwrap();

        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let model0 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let model1 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let mut backend0 = TchWorkerBackend::from_model(model0, device, 0);
        let mut backend1 = TchWorkerBackend::from_model(model1, device, 1);

        let mut transports0: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        let mut transports1: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let (transport0, transport1) =
                crate::model::transport::LinkedMockKvTransport::create_pair();
            transports0.push(Box::new(transport0));
            transports1.push(Box::new(transport1));
        }
        backend0.setup_kv_transports(transports0);
        backend1.setup_kv_transports(transports1);

        let request_id = 73;
        let prompt = [151644_i64, 9707, 0, 16];
        let input_ids = Tensor::from_slice(&prompt).unsqueeze(0);
        let mut reference_caches = reference.create_kv_caches();
        let reference_logits = reference
            .forward(&input_ids, &mut reference_caches)
            .unwrap()
            .narrow(1, prompt.len() as i64 - 1, 1)
            .squeeze();

        let capacities0 = (0..config.num_layers)
            .map(|layer_idx| 1 + layer_idx % 2)
            .collect::<Vec<_>>();
        let capacities1 = (0..config.num_layers)
            .map(|layer_idx| 3 + (layer_idx + 1) % 2)
            .collect::<Vec<_>>();
        let (_, global_len0) = backend0
            .prefill_request_with_reservation(
                request_id,
                &prompt[..1],
                0,
                Some(&[0]),
                Some(&capacities0),
            )
            .unwrap();
        let (last_logits1, global_len1) = backend1
            .prefill_request_with_reservation(
                request_id,
                &prompt[1..],
                1,
                Some(&[1, 2, 3]),
                Some(&capacities1),
            )
            .unwrap();
        assert_eq!(global_len0, 1);
        assert_eq!(global_len1, prompt.len());

        let distributed_logits = Tensor::from_slice(&last_logits1);
        let max_diff = (&reference_logits - &distributed_logits)
            .abs()
            .max()
            .double_value(&[]);
        let mean_diff = (&reference_logits - &distributed_logits)
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        let reference_token = reference_logits.argmax(-1, false).int64_value(&[]);
        let distributed_token = distributed_logits.argmax(-1, false).int64_value(&[]);
        println!(
            "two-worker real prefill: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, tokens={reference_token}/{distributed_token}"
        );
        assert_eq!(reference_token, distributed_token);
        assert!(max_diff < 0.5, "distributed max logits diff: {max_diff}");
        assert!(mean_diff < 0.1, "distributed mean logits diff: {mean_diff}");

        let context0 = backend0.request_contexts.get(&request_id).unwrap();
        let context1 = backend1.request_contexts.get(&request_id).unwrap();
        for layer_idx in 0..config.num_layers {
            let Some(KvCacheImpl::ReservedPositioned(shard0)) = &context0.kv_caches[layer_idx]
            else {
                panic!("worker 0 layer {layer_idx} did not use reserved KV");
            };
            let Some(KvCacheImpl::ReservedPositioned(shard1)) = &context1.kv_caches[layer_idx]
            else {
                panic!("worker 1 layer {layer_idx} did not use reserved KV");
            };
            assert_eq!(shard0.reserved_capacity(), capacities0[layer_idx]);
            assert_eq!(shard1.reserved_capacity(), capacities1[layer_idx]);
            assert_eq!(shard0.positions(), &[0]);
            assert_eq!(shard1.positions(), &[1, 2, 3]);
            assert_eq!(
                shard0.committed_len() + shard1.committed_len(),
                prompt.len()
            );
            assert_eq!(shard0.active_k().kind(), Kind::BFloat16);
            assert_eq!(shard1.active_k().kind(), Kind::BFloat16);
        }
    }

    #[test]
    #[ignore = "requires the local Qwen2-0.5B model weights"]
    fn real_qwen_two_worker_reserved_prefill_decodes_two_self_driving_tokens_over_tcp() {
        let device = Device::Cpu;
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("Qwen2-0.5B");
        let config = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
        assert_eq!(config.num_layers, 24);
        let weights = ModelWeights::from_dir(&model_dir, device).unwrap();

        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let model0 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let model1 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let mut backend0 = TchWorkerBackend::from_model(model0, device, 0);
        let mut backend1 = TchWorkerBackend::from_model(model1, device, 1);

        let mut transports0: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        let mut transports1: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let (transport0, transport1) =
                crate::model::transport::LinkedMockKvTransport::create_pair();
            transports0.push(Box::new(transport0));
            transports1.push(Box::new(transport1));
        }
        backend0.setup_kv_transports(transports0);
        backend1.setup_kv_transports(transports1);

        let request_id = 74;
        let prompt = [151644_i64, 9707, 0, 16];
        let decode_tokens = 2_usize;
        let schedule =
            FrozenKvAssigneeSchedule::new(&[1, 3], request_id, decode_tokens * config.num_layers)
                .unwrap();
        assert_eq!(schedule.counts(), &[12, 36]);
        let assignees = (0..decode_tokens)
            .map(|token_offset| {
                (0..config.num_layers)
                    .map(|layer_idx| {
                        schedule
                            .assignee_for(token_offset, layer_idx, config.num_layers)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let capacities = (0..2)
            .map(|domain| {
                (0..config.num_layers)
                    .map(|layer_idx| {
                        let initial = if domain == 0 { 1 } else { 3 };
                        initial
                            + assignees
                                .iter()
                                .filter(|token_assignees| token_assignees[layer_idx] == domain)
                                .count()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let input_ids = Tensor::from_slice(&prompt).unsqueeze(0);
        let mut reference_caches = reference.create_kv_caches();
        let reference_prefill_logits = reference
            .forward(&input_ids, &mut reference_caches)
            .unwrap()
            .narrow(1, prompt.len() as i64 - 1, 1)
            .squeeze();
        backend0
            .prefill_request_with_reservation(
                request_id,
                &prompt[..1],
                0,
                Some(&[0]),
                Some(&capacities[0]),
            )
            .unwrap();
        let (distributed_prefill_logits, _) = backend1
            .prefill_request_with_reservation(
                request_id,
                &prompt[1..],
                1,
                Some(&[1, 2, 3]),
                Some(&capacities[1]),
            )
            .unwrap();
        let distributed_prefill_logits = Tensor::from_slice(&distributed_prefill_logits);
        let mut token = distributed_prefill_logits
            .argmax(-1, false)
            .int64_value(&[]);
        assert_eq!(
            token,
            reference_prefill_logits.argmax(-1, false).int64_value(&[])
        );

        let mut backends = vec![backend0, backend1];
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let stream0 = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let stream1 = listener.accept().unwrap().0;
        let mut decode_transports = vec![
            crate::model::transport::TcpKvTransport::new(stream0, device).unwrap(),
            crate::model::transport::TcpKvTransport::new(stream1, device).unwrap(),
        ];
        let mut starter = 1_usize;
        let mut tcp_hops = 0_usize;
        let mut tcp_bytes = 0_usize;
        for token_offset in 0..decode_tokens {
            let position = (prompt.len() + token_offset) as i64;
            let committed_before = (0..config.num_layers)
                .map(|layer_idx| {
                    (0..2)
                        .map(|domain| {
                            let context =
                                backends[domain].request_contexts.get(&request_id).unwrap();
                            let Some(KvCacheImpl::ReservedPositioned(shard)) =
                                &context.kv_caches[layer_idx]
                            else {
                                panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                            };
                            shard.committed_len()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let (distributed_logits, producer, hops, wire_bytes) =
                run_two_backend_reserved_tcp_decode(
                    &mut backends,
                    &mut decode_transports,
                    request_id,
                    token,
                    position,
                    starter,
                    &assignees[token_offset],
                );
            assert_eq!(hops, config.num_layers);
            assert!(wire_bytes > 0);
            tcp_hops += hops;
            tcp_bytes += wire_bytes;
            let reference_input = Tensor::from_slice(&[token]).unsqueeze(0);
            let reference_logits = reference
                .forward(&reference_input, &mut reference_caches)
                .unwrap()
                .squeeze();
            let distributed_logits = distributed_logits.squeeze();
            let max_diff = (&distributed_logits - &reference_logits)
                .abs()
                .max()
                .double_value(&[]);
            let mean_diff = (&distributed_logits - &reference_logits)
                .abs()
                .mean(Kind::Float)
                .double_value(&[]);
            let distributed_token = distributed_logits.argmax(-1, false).int64_value(&[]);
            let reference_token = reference_logits.argmax(-1, false).int64_value(&[]);
            println!(
                "real two-worker self-driving decode {token_offset}: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, tokens={distributed_token}/{reference_token}"
            );
            assert_eq!(distributed_token, reference_token);
            assert!(
                max_diff < 0.75,
                "decode {token_offset} max logits diff: {max_diff}"
            );
            assert!(
                mean_diff < 0.1,
                "decode {token_offset} mean logits diff: {mean_diff}"
            );

            for layer_idx in 0..config.num_layers {
                let mut positions = Vec::new();
                for domain in 0..2 {
                    let context = backends[domain].request_contexts.get(&request_id).unwrap();
                    let Some(KvCacheImpl::ReservedPositioned(shard)) =
                        &context.kv_caches[layer_idx]
                    else {
                        panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                    };
                    let expected_growth = usize::from(assignees[token_offset][layer_idx] == domain);
                    assert_eq!(
                        shard.committed_len(),
                        committed_before[layer_idx][domain] + expected_growth
                    );
                    assert!(shard.committed_len() <= shard.reserved_capacity());
                    assert_eq!(shard.active_k().kind(), Kind::BFloat16);
                    positions.extend_from_slice(shard.positions());
                }
                positions.sort_unstable();
                assert_eq!(positions, (0..=position).collect::<Vec<_>>());
            }

            starter = producer;
            token = distributed_token;
        }

        assert_eq!(tcp_hops, decode_tokens * config.num_layers);
        assert!(tcp_bytes > 0);
        println!("real two-worker self-driving TCP: hops={tcp_hops}, bytes={tcp_bytes}");

        let domain_totals = (0..2)
            .map(|domain| {
                (0..config.num_layers)
                    .map(|layer_idx| {
                        let context = backends[domain].request_contexts.get(&request_id).unwrap();
                        let Some(KvCacheImpl::ReservedPositioned(shard)) =
                            &context.kv_caches[layer_idx]
                        else {
                            panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                        };
                        assert_eq!(shard.committed_len(), shard.reserved_capacity());
                        shard.committed_len()
                    })
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        assert_eq!(domain_totals, vec![36, 108]);
    }
}
