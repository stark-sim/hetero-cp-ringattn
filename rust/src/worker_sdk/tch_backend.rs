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
use crate::model::self_driving::{
    process_layer_packet_with_reserved_history_for_positions, project_final_logits,
    stationary_layer_starters, FrozenKvAssigneeSchedule, LayerPacket, LayerStepOutcome,
    ReservedPositionedKvShard,
};
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
    // experimental: raised for route_b_cross_node_smoke
    pub model: LlamaModel,
    device: Device,
    /// Backward-compatible single-request KV cache.
    kv_caches: KvCaches,
    domain_id: usize,
    /// Per-request KV cache and model state (M13 continuous batching).
    // experimental: raised for route_b_cross_node_smoke
    pub request_contexts: HashMap<u64, RequestContext>,
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

    fn do_request_prefill(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
        position_ids: Option<&[i64]>,
        layer_kv_capacities: Option<&[usize]>,
    ) -> Result<(Vec<f32>, usize), String> {
        if self.request_contexts.contains_key(&request_id) {
            let positions = position_ids.ok_or_else(|| {
                format!(
                    "request {request_id} already exists; positioned continuation requires explicit position_ids"
                )
            })?;
            return self.do_positioned_continuation(
                request_id,
                chunk,
                positions,
                layer_kv_capacities,
            );
        }

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

    fn do_positioned_continuation(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        position_ids: &[i64],
        layer_kv_capacities: Option<&[usize]>,
    ) -> Result<(Vec<f32>, usize), String> {
        if chunk.is_empty() || chunk.len() != position_ids.len() {
            return Err(format!(
                "request {request_id} positioned continuation requires one position per token"
            ));
        }

        let context = self
            .request_contexts
            .get(&request_id)
            .ok_or_else(|| format!("request {request_id} not found"))?;
        if !context.is_prefill_done {
            return Err(format!(
                "request {request_id} cannot continue before initial prefill completes"
            ));
        }
        if let Some(capacities) = layer_kv_capacities {
            if capacities.len() != context.kv_caches.len() {
                return Err(format!(
                    "positioned continuation requires {} layer capacities, got {}",
                    context.kv_caches.len(),
                    capacities.len()
                ));
            }
        }
        for (layer_idx, cache) in context.kv_caches.iter().enumerate() {
            let Some(KvCacheImpl::ReservedPositioned(shard)) = cache else {
                return Err(format!(
                    "request {request_id} layer {layer_idx} does not have reserved positioned KV"
                ));
            };
            if shard.committed_len() + chunk.len() > shard.reserved_capacity() {
                return Err(format!(
                    "request {request_id} layer {layer_idx} continuation exceeds reserved capacity"
                ));
            }
            if position_ids
                .iter()
                .any(|position| shard.positions().contains(position))
            {
                return Err(format!(
                    "request {request_id} layer {layer_idx} continuation reuses a committed position"
                ));
            }
            if let Some(capacities) = layer_kv_capacities {
                if shard.reserved_capacity() != capacities[layer_idx] {
                    return Err(format!(
                        "request {request_id} layer {layer_idx} reservation is {}, got continuation capacity {}",
                        shard.reserved_capacity(),
                        capacities[layer_idx]
                    ));
                }
            }
        }

        let context = self
            .request_contexts
            .get_mut(&request_id)
            .ok_or_else(|| format!("request {request_id} not found"))?;
        self.model.global_seq_len = context.global_seq_len;
        self.model.is_prefill_done = context.is_prefill_done;
        self.model.prefill_position_ids = None;
        self.model
            .set_prefill_position_ids(position_ids, self.device);

        let input = Tensor::from_slice(chunk)
            .unsqueeze(0)
            .to_device(self.device);
        let logits_result = self
            .model
            .forward(&input, &mut context.kv_caches)
            .map_err(|e| format!("positioned continuation forward failed: {e}"));
        self.model.prefill_position_ids = None;
        context.global_seq_len = self.model.global_seq_len;
        context.is_prefill_done = self.model.is_prefill_done;

        let logits = logits_result?;
        let last_logits = logits.narrow(1, logits.size()[1] - 1, 1).squeeze();
        let logits_vec: Vec<f32> = Vec::try_from(&last_logits)
            .map_err(|e| format!("continuation logits to vec failed: {e}"))?;
        Ok((logits_vec, context.global_seq_len))
    }

    // Note: do_decode removed to avoid borrow checker issues.
    // decode() and decode_request() inline the small forward logic directly.

    /// 【Stationary continuation 生产驱动】(route-B 2b)
    ///
    /// Drives one m>1 stationary continuation segment through the self-driving
    /// ring: per layer the packet starts at the layer's starter domain, visits
    /// every domain in successor order, and finishes one hop before the
    /// starter. Every domain projects and appends only its own frozen position
    /// offsets; historical KV never enters the packet. Transport I/O uses the
    /// per-layer transports installed by `setup_kv_transports`.
    ///
    /// Returns the last-position logits as `Some(Vec<f32>)` on the final
    /// finisher domain, `None` on every other domain. The request's
    /// `global_seq_len` is advanced to the segment end on every domain.
    pub fn run_stationary_continuation(
        &mut self,
        request_id: u64,
        tokens: &[i64],
        position_ids: &[i64],
        capacity_tickets: &[u64],
        starter_domain: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        let domains = self.model.num_domains;
        crate::distributed::protocol::validate_stationary_continuation(
            domains,
            tokens,
            position_ids,
            capacity_tickets,
            starter_domain,
        )?;
        let layers = self.model.config.num_layers;
        let sc_start = std::time::Instant::now();

        // Frozen plan: this domain's position offsets and the per-layer starters.
        let schedule = FrozenKvAssigneeSchedule::new(capacity_tickets, request_id, tokens.len())?;
        let my_offsets = (0..tokens.len())
            .filter(|&offset| schedule.assignee_for(offset, 0, 1) == Some(self.domain_id))
            .collect::<Vec<_>>();
        let starters = stationary_layer_starters(starter_domain, layers, domains)?;

        // Admission: every layer shard must be reserved positioned KV with
        // enough headroom for this domain's new offsets (byte-level admission
        // is a later work item).
        {
            let context = self
                .request_contexts
                .get(&request_id)
                .ok_or_else(|| format!("request {request_id} not found"))?;
            for (layer_idx, cache) in context.kv_caches.iter().enumerate() {
                let Some(KvCacheImpl::ReservedPositioned(shard)) = cache else {
                    return Err(format!(
                        "request {request_id} layer {layer_idx} does not have reserved positioned KV"
                    ));
                };
                if shard.committed_len() + my_offsets.len() > shard.reserved_capacity() {
                    return Err(format!(
                        "request {request_id} layer {layer_idx} stationary continuation exceeds reserved capacity: committed {} + new {} > {}",
                        shard.committed_len(),
                        my_offsets.len(),
                        shard.reserved_capacity()
                    ));
                }
            }
        }

        let position_tensor = Tensor::from_slice(position_ids)
            .unsqueeze(0)
            .to_device(self.device);
        let mut hidden_states = if starters[0] == self.domain_id {
            let input = Tensor::from_slice(tokens)
                .unsqueeze(0)
                .to_device(self.device);
            Some(Tensor::embedding(
                &self.model.embedding,
                &input,
                -1,
                false,
                false,
            ))
        } else {
            None
        };

        let mut final_finisher = starter_domain;
        let mut starter_layers = 0_usize;
        let mut middle_layers = 0_usize;
        let mut finisher_layers = 0_usize;
        let mut sends = 0_usize;
        let mut recvs = 0_usize;
        let mut sc_recv_wait_ms = 0.0_f64;
        let mut sc_process_ms = 0.0_f64;
        let mut sc_send_ms = 0.0_f64;
        for (layer_idx, &starter) in starters.iter().enumerate() {
            let finisher = (starter + domains - 1) % domains;
            final_finisher = finisher;
            if starter == self.domain_id {
                let packet = LayerPacket::start(
                    &mut self.model.layers[layer_idx],
                    hidden_states
                        .as_ref()
                        .ok_or("stationary continuation starter is missing hidden states")?,
                    &position_tensor,
                    self.domain_id,
                    self.domain_id,
                    domains,
                )
                .map_err(|e| {
                    format!("stationary continuation layer {layer_idx} start failed: {e}")
                })?;
                let process_start = std::time::Instant::now();
                let outcome = {
                    let context = self.request_contexts.get_mut(&request_id).unwrap();
                    let Some(KvCacheImpl::ReservedPositioned(shard)) =
                        &mut context.kv_caches[layer_idx]
                    else {
                        return Err(format!(
                            "request {request_id} layer {layer_idx} does not have reserved positioned KV"
                        ));
                    };
                    process_layer_packet_with_reserved_history_for_positions(
                        &mut self.model.layers[layer_idx],
                        packet,
                        shard,
                        &my_offsets,
                    )
                    .map_err(|e| {
                        format!(
                            "stationary continuation layer {layer_idx} starter step failed: {e}"
                        )
                    })?
                };
                sc_process_ms += process_start.elapsed().as_secs_f64() * 1000.0;
                let LayerStepOutcome::Forward(next_packet) = outcome else {
                    return Err(format!(
                        "stationary continuation layer {layer_idx} starter finished a {domains}-domain route"
                    ));
                };
                let wire = next_packet.into_self_driving_packet(0).map_err(|e| {
                    format!("stationary continuation layer {layer_idx} wire encode failed: {e}")
                })?;
                let transport =
                    self.model.layers[layer_idx]
                        .kv_transport_mut()
                        .ok_or_else(|| {
                            format!("stationary continuation layer {layer_idx} has no KV transport")
                        })?;
                let send_start = std::time::Instant::now();
                transport.submit_send_self_driving_packet(&wire)?;
                transport.flush_send()?;
                sc_send_ms += send_start.elapsed().as_secs_f64() * 1000.0;
                starter_layers += 1;
                sends += 1;
            } else {
                let recv_start = std::time::Instant::now();
                let wire = {
                    let transport =
                        self.model.layers[layer_idx]
                            .kv_transport_mut()
                            .ok_or_else(|| {
                                format!(
                                    "stationary continuation layer {layer_idx} has no KV transport"
                                )
                            })?;
                    transport.recv_self_driving_packet()?.ok_or_else(|| {
                        format!("stationary continuation layer {layer_idx} predecessor closed")
                    })?
                };
                sc_recv_wait_ms += recv_start.elapsed().as_secs_f64() * 1000.0;
                recvs += 1;
                let packet = LayerPacket::from_self_driving_packet(wire).map_err(|e| {
                    format!("stationary continuation layer {layer_idx} wire decode failed: {e}")
                })?;
                let process_start = std::time::Instant::now();
                let outcome = {
                    let context = self.request_contexts.get_mut(&request_id).unwrap();
                    let Some(KvCacheImpl::ReservedPositioned(shard)) =
                        &mut context.kv_caches[layer_idx]
                    else {
                        return Err(format!(
                            "request {request_id} layer {layer_idx} does not have reserved positioned KV"
                        ));
                    };
                    process_layer_packet_with_reserved_history_for_positions(
                        &mut self.model.layers[layer_idx],
                        packet,
                        shard,
                        &my_offsets,
                    )
                    .map_err(|e| {
                        format!("stationary continuation layer {layer_idx} ring step failed: {e}")
                    })?
                };
                sc_process_ms += process_start.elapsed().as_secs_f64() * 1000.0;
                if finisher == self.domain_id {
                    let LayerStepOutcome::Finished {
                        hidden_states: next_hidden,
                        ..
                    } = outcome
                    else {
                        return Err(format!(
                            "stationary continuation layer {layer_idx} finisher forwarded a {domains}-domain route"
                        ));
                    };
                    hidden_states = Some(next_hidden);
                    finisher_layers += 1;
                } else {
                    let LayerStepOutcome::Forward(next_packet) = outcome else {
                        return Err(format!(
                            "stationary continuation layer {layer_idx} middle domain finished a {domains}-domain route"
                        ));
                    };
                    let wire = next_packet.into_self_driving_packet(0).map_err(|e| {
                        format!("stationary continuation layer {layer_idx} wire encode failed: {e}")
                    })?;
                    let transport =
                        self.model.layers[layer_idx]
                            .kv_transport_mut()
                            .ok_or_else(|| {
                                format!(
                                    "stationary continuation layer {layer_idx} has no KV transport"
                                )
                            })?;
                    let send_start = std::time::Instant::now();
                    transport.submit_send_self_driving_packet(&wire)?;
                    transport.flush_send()?;
                    sc_send_ms += send_start.elapsed().as_secs_f64() * 1000.0;
                    middle_layers += 1;
                    sends += 1;
                }
            }
        }

        // HCP_PERF_LOG timing event (same JSONL shape as ring_decode) so a single
        // N=2/N=3 run can compare Q-ring decode vs stationary continuation.
        if let Ok(path) = std::env::var("HCP_PERF_LOG") {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let ts = format!("{}.{:03}Z", now.as_secs(), now.subsec_millis());
            let line = format!(
                "{{\"ts\":\"{ts}\",\"event\":\"stationary_continuation\",\"domain\":{},\"request_id\":{request_id},\"layers\":{layers},\"domains\":{domains},\"tokens\":{},\"sends\":{sends},\"recvs\":{recvs},\"hops_per_layer\":{},\"recv_wait_ms\":{:.3},\"process_ms\":{:.3},\"send_ms\":{:.3},\"total_ms\":{:.3}}}\n",
                self.domain_id,
                tokens.len(),
                domains - 1,
                sc_recv_wait_ms,
                sc_process_ms,
                sc_send_ms,
                sc_start.elapsed().as_secs_f64() * 1000.0
            );
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = f.write_all(line.as_bytes());
            }
        }

        println!(
            "[TchWorkerBackend] stationary continuation stats: request_id={request_id} domain={} layers={} domains={domains} starter_layers={starter_layers} middle_layers={middle_layers} finisher_layers={finisher_layers} sends={sends} recvs={recvs} expected_hops_per_layer={}",
            self.domain_id,
            starters.len(),
            domains - 1,
        );

        // Every domain resyncs the request horizon to the segment end.
        {
            let context = self.request_contexts.get_mut(&request_id).unwrap();
            context.global_seq_len = position_ids[position_ids.len() - 1] as usize + 1;
        }

        if final_finisher != self.domain_id {
            return Ok(None);
        }
        let hidden =
            hidden_states.ok_or("stationary continuation finisher has no hidden states")?;
        let logits = project_final_logits(&self.model, &hidden);
        let last = logits.select(1, tokens.len() as i64 - 1).squeeze();
        let values: Vec<f32> = Vec::try_from(&last.contiguous())
            .map_err(|e| format!("stationary continuation logits to vec failed: {e}"))?;
        Ok(Some(values))
    }
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

    /// Request-aware prefill creates a context, or extends an existing reserved context
    /// when explicit continuation positions are supplied.
    fn prefill_request(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
        position_ids: Option<&[i64]>,
    ) -> Result<(Vec<f32>, usize), String> {
        self.do_request_prefill(request_id, chunk, seq_offset, position_ids, None)
    }

    fn prefill_request_with_reservation(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
        position_ids: Option<&[i64]>,
        layer_kv_capacities: Option<&[usize]>,
    ) -> Result<(Vec<f32>, usize), String> {
        self.do_request_prefill(
            request_id,
            chunk,
            seq_offset,
            position_ids,
            layer_kv_capacities,
        )
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
        let decode_fwd_start = std::time::Instant::now();
        let logits = self
            .model
            .forward(&input, &mut ctx.kv_caches)
            .map_err(|e| format!("decode forward failed: {e}"))?;
        // Full Q-ring decode forward timing (embed + 24x(norm+attn+mlp) + lm head).
        if let Ok(path) = std::env::var("HCP_PERF_LOG") {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let ts = format!("{}.{:03}Z", now.as_secs(), now.subsec_millis());
            let line = format!(
                "{{\"ts\":\"{ts}\",\"event\":\"decode_forward_full\",\"domain\":{},\"request_id\":{request_id},\"token\":{token},\"total_ms\":{:.3}}}\n",
                self.domain_id,
                decode_fwd_start.elapsed().as_secs_f64() * 1000.0
            );
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = f.write_all(line.as_bytes());
            }
        }

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

    // experimental: route-B stationary continuation driver (phase-2 node 2c);
    // delegates to the inherent implementation.
    fn run_stationary_continuation(
        &mut self,
        request_id: u64,
        tokens: &[i64],
        position_ids: &[i64],
        capacity_tickets: &[u64],
        starter_domain: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        TchWorkerBackend::run_stationary_continuation(
            self,
            request_id,
            tokens,
            position_ids,
            capacity_tickets,
            starter_domain,
        )
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
        process_layer_packet_with_reserved_history,
        process_layer_packet_with_reserved_history_for_positions, project_final_logits,
        FrozenKvAssigneeSchedule, LayerPacket, LayerStepOutcome,
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

    /// Verify unequal-prefix request isolation against independent references.
    #[test]
    fn test_decode_batch_isolation_with_unequal_prefixes() {
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
        let model_ref_a = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let model_ref_b = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();

        let mut backend_batch = TchWorkerBackend::from_model(model_batch, device, 0);
        let mut backend_ref_a = TchWorkerBackend::from_model(model_ref_a, device, 0);
        let mut backend_ref_b = TchWorkerBackend::from_model(model_ref_b, device, 0);

        // Two different prompts with different prefix lengths.
        let prompt_a: Vec<i64> = (0..8).collect();
        let prompt_b: Vec<i64> = (10..22).collect();

        // Prefill both requests on the mixed backend and on independent references.
        let (logits_a_batch, len_a_batch) = backend_batch
            .prefill_request(1, &prompt_a, 0, None)
            .unwrap();
        let (logits_b_batch, len_b_batch) = backend_batch
            .prefill_request(2, &prompt_b, 0, None)
            .unwrap();

        let (logits_a_ref, len_a_ref) = backend_ref_a
            .prefill_request(1, &prompt_a, 0, None)
            .unwrap();
        let (logits_b_ref, len_b_ref) = backend_ref_b
            .prefill_request(2, &prompt_b, 0, None)
            .unwrap();

        assert_eq!(prompt_a.len(), 8);
        assert_eq!(prompt_b.len(), 12);
        assert_ne!(prompt_a.len(), prompt_b.len());
        assert_eq!(len_a_batch, len_a_ref);
        assert_eq!(len_b_batch, len_b_ref);

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
        let ref_logits_a = backend_ref_a.decode_request(1, token_a).unwrap();
        let ref_logits_b = backend_ref_b.decode_request(2, token_b).unwrap();

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
            // References remain fully independent so shared state cannot mask a mismatch.
            let ref_la = backend_ref_a.decode_request(1, next_token_a).unwrap();
            let ref_lb = backend_ref_b.decode_request(2, next_token_b).unwrap();

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

    /// Verify two unequal-prefix requests remain isolated while both workers
    /// execute the request-aware decode path over a shared Q-ring transport.
    ///
    /// Prefill uses the real two-worker TCP KV exchange so each request owns
    /// only its capacity-weighted local prefix.  The same transports expose a
    /// packet-only rendezvous channel for decode, and A/B advance in the same
    /// order on both workers, matching the coordinator's batch contract.
    #[test]
    fn test_decode_qring_request_isolation_with_unequal_prefixes() {
        use crate::model::transport::{KvBlock, RingPacket};
        use std::sync::mpsc;

        struct ChanPacketTransport {
            kv: crate::model::transport::TcpKvTransport,
            tx: mpsc::Sender<RingPacket>,
            rx: mpsc::Receiver<RingPacket>,
        }

        impl KvTransport for ChanPacketTransport {
            fn submit_send(&mut self, block: &KvBlock) -> Result<(), String> {
                self.kv.submit_send(block)
            }

            fn poll_recv(&mut self) -> Result<Option<KvBlock>, String> {
                self.kv.poll_recv()
            }

            fn flush_send(&mut self) -> Result<(), String> {
                self.kv.flush_send()
            }

            fn supports_ring_packets(&self) -> bool {
                true
            }

            fn submit_send_packet(&mut self, packet: &RingPacket) -> Result<(), String> {
                self.tx.send(packet.clone()).map_err(|e| e.to_string())
            }

            fn poll_recv_packet(&mut self) -> Result<Option<RingPacket>, String> {
                Ok(self.rx.try_recv().ok())
            }

            fn recv_packet(&mut self) -> Result<Option<RingPacket>, String> {
                self.rx
                    .recv_timeout(std::time::Duration::from_secs(30))
                    .map(Some)
                    .map_err(|e| format!("recv Q-ring packet timeout: {e}"))
            }
        }

        fn make_config() -> ModelConfig {
            ModelConfig {
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
            }
        }

        let device = Device::Cpu;
        let config = make_config();
        let weights = create_synthetic_weights(&config, device);
        let mut backend0 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            0,
        );
        let mut backend1 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            1,
        );
        let mut reference_a = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap(),
            device,
            0,
        );
        let mut reference_b = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap(),
            device,
            0,
        );

        let mut transports0: Vec<Box<dyn KvTransport>> = Vec::new();
        let mut transports1: Vec<Box<dyn KvTransport>> = Vec::new();
        for _ in 0..config.num_layers {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let stream0 = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let stream1 = listener.accept().unwrap().0;
            let (tx01, rx01) = mpsc::channel::<RingPacket>();
            let (tx10, rx10) = mpsc::channel::<RingPacket>();
            transports0.push(Box::new(ChanPacketTransport {
                kv: crate::model::transport::TcpKvTransport::new(stream0, device).unwrap(),
                tx: tx01,
                rx: rx10,
            }));
            transports1.push(Box::new(ChanPacketTransport {
                kv: crate::model::transport::TcpKvTransport::new(stream1, device).unwrap(),
                tx: tx10,
                rx: rx01,
            }));
        }
        backend0.setup_kv_transports(transports0);
        backend1.setup_kv_transports(transports1);

        let prompt_a: Vec<i64> = (0..4).collect();
        let prompt_b: Vec<i64> = (10..16).collect();
        let capacities_a0 = vec![5; config.num_layers];
        let capacities_a1 = vec![7; config.num_layers];
        let capacities_b0 = vec![6; config.num_layers];
        let capacities_b1 = vec![8; config.num_layers];

        // Every worker processes requests in the same order.  This is the
        // current coordinator batch contract because RingPacket has no request
        // identifier and the per-layer channel is FIFO.
        let ((_logits_a0, _), (_logits_b0, _), (logits_a1, _), (logits_b1, _)) =
            std::thread::scope(|scope| {
                let worker0 = scope.spawn(|| {
                    let a = backend0.prefill_request_with_reservation(
                        101,
                        &prompt_a[..1],
                        0,
                        Some(&[0]),
                        Some(&capacities_a0),
                    );
                    let b = backend0.prefill_request_with_reservation(
                        202,
                        &prompt_b[..2],
                        0,
                        Some(&[0, 1]),
                        Some(&capacities_b0),
                    );
                    (a, b)
                });
                let worker1 = scope.spawn(|| {
                    let a = backend1.prefill_request_with_reservation(
                        101,
                        &prompt_a[1..],
                        1,
                        Some(&[1, 2, 3]),
                        Some(&capacities_a1),
                    );
                    let b = backend1.prefill_request_with_reservation(
                        202,
                        &prompt_b[2..],
                        2,
                        Some(&[2, 3, 4, 5]),
                        Some(&capacities_b1),
                    );
                    (a, b)
                });
                let (a0, b0) = worker0.join().unwrap();
                let (a1, b1) = worker1.join().unwrap();
                (a0.unwrap(), b0.unwrap(), a1.unwrap(), b1.unwrap())
            });
        for backend in [&mut backend0, &mut backend1] {
            backend.sync_global_seq_len_for_request(101, prompt_a.len());
            backend.sync_global_seq_len_for_request(202, prompt_b.len());
        }

        let (ref_a, _) = reference_a
            .prefill_request(101, &prompt_a, 0, None)
            .unwrap();
        let (ref_b, _) = reference_b
            .prefill_request(202, &prompt_b, 0, None)
            .unwrap();
        let prefill_a = Tensor::from_slice(&logits_a1);
        let prefill_b = Tensor::from_slice(&logits_b1);
        let ref_a = Tensor::from_slice(&ref_a);
        let ref_b = Tensor::from_slice(&ref_b);
        assert!((&prefill_a - &ref_a).abs().max().double_value(&[]) < 1e-3);
        assert!((&prefill_b - &ref_b).abs().max().double_value(&[]) < 1e-3);

        let mut next_a = prefill_a.argmax(-1, false).int64_value(&[]);
        let mut next_b = prefill_b.argmax(-1, false).int64_value(&[]);
        const DECODE_STEPS: usize = 3;
        for step in 0..DECODE_STEPS {
            let ref_logits_a = reference_a.decode_request(101, next_a).unwrap();
            let ref_logits_b = reference_b.decode_request(202, next_b).unwrap();
            let (batch0, batch1) = std::thread::scope(|scope| {
                let worker0 =
                    scope.spawn(|| backend0.decode_batch(&[(101, next_a), (202, next_b)]));
                let worker1 =
                    scope.spawn(|| backend1.decode_batch(&[(101, next_a), (202, next_b)]));
                (
                    worker0.join().unwrap().unwrap(),
                    worker1.join().unwrap().unwrap(),
                )
            });
            for (worker, batch) in [(0_usize, &batch0), (1, &batch1)] {
                let logits_a = batch.iter().find(|(id, _)| *id == 101).unwrap().1.clone();
                let logits_b = batch.iter().find(|(id, _)| *id == 202).unwrap().1.clone();
                let diff_a = (&Tensor::from_slice(&logits_a) - &Tensor::from_slice(&ref_logits_a))
                    .abs()
                    .max()
                    .double_value(&[]);
                let diff_b = (&Tensor::from_slice(&logits_b) - &Tensor::from_slice(&ref_logits_b))
                    .abs()
                    .max()
                    .double_value(&[]);
                assert!(
                    diff_a < 1e-3,
                    "step {step} worker {worker} request A diff {diff_a}"
                );
                assert!(
                    diff_b < 1e-3,
                    "step {step} worker {worker} request B diff {diff_b}"
                );
                assert_eq!(
                    Tensor::from_slice(&logits_a)
                        .argmax(-1, false)
                        .int64_value(&[]),
                    Tensor::from_slice(&ref_logits_a)
                        .argmax(-1, false)
                        .int64_value(&[]),
                    "step {step} worker {worker} request A token"
                );
                assert_eq!(
                    Tensor::from_slice(&logits_b)
                        .argmax(-1, false)
                        .int64_value(&[]),
                    Tensor::from_slice(&ref_logits_b)
                        .argmax(-1, false)
                        .int64_value(&[]),
                    "step {step} worker {worker} request B token"
                );
            }
            next_a = Tensor::from_slice(&ref_logits_a)
                .argmax(-1, false)
                .int64_value(&[]);
            next_b = Tensor::from_slice(&ref_logits_b)
                .argmax(-1, false)
                .int64_value(&[]);
        }

        let split_points = [(101_u64, prompt_a.len(), 1_usize), (202, prompt_b.len(), 2)];
        for (domain, backend) in [&backend0, &backend1].into_iter().enumerate() {
            for (request_id, prompt_len, split) in split_points {
                let context = backend.request_contexts.get(&request_id).unwrap();
                assert_eq!(context.global_seq_len, prompt_len + DECODE_STEPS);
                let mut expected_positions = if domain == 0 {
                    (0..split as i64).collect::<Vec<_>>()
                } else {
                    (split as i64..prompt_len as i64).collect::<Vec<_>>()
                };
                expected_positions.extend(
                    (prompt_len as i64..(prompt_len + DECODE_STEPS) as i64)
                        .filter(|position| *position as usize % 2 == domain),
                );
                for cache in &context.kv_caches {
                    let Some(KvCacheImpl::ReservedPositioned(shard)) = cache else {
                        panic!("request {request_id} did not use reserved positioned KV");
                    };
                    assert_eq!(shard.positions(), expected_positions);
                }
            }
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
    fn release_request_frees_context_and_further_decode_fails() {
        let device = Device::Cpu;
        let config = ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 2,
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

        backend.prefill_request(1, &[3, 5], 0, None).unwrap();
        assert!(backend.request_contexts.contains_key(&1));
        let logits = backend.decode_request(1, 7).unwrap();
        assert_eq!(logits.len(), config.vocab_size);

        // Release must free the per-request KV cache exactly like the
        // coordinator's ReleaseRequest command after completion.
        backend.release_request(1);
        assert!(!backend.request_contexts.contains_key(&1));
        // Idempotent release is a no-op, not an error.
        backend.release_request(1);
        // Decoding a released request must fail, not reuse stale state.
        assert!(backend.decode_request(1, 7).is_err());
    }

    #[test]
    fn single_local_token_prefill_remains_causal_with_future_peer_kv() {
        let device = Device::Cpu;
        let config = ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 2,
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
        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut backend0 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            0,
        );
        let mut backend1 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            1,
        );
        let mut transports0: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        let mut transports1: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let stream0 = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let stream1 = listener.accept().unwrap().0;
            transports0.push(Box::new(
                crate::model::transport::TcpKvTransport::new(stream0, device).unwrap(),
            ));
            transports1.push(Box::new(
                crate::model::transport::TcpKvTransport::new(stream1, device).unwrap(),
            ));
        }
        backend0.setup_kv_transports(transports0);
        backend1.setup_kv_transports(transports1);

        let prompt = [3_i64, 5, 7, 9];
        let mut reference_caches = reference.create_kv_caches();
        let reference_logits = reference
            .forward(
                &Tensor::from_slice(&prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap();
        let ((logits0, _), (logits1, _)) = std::thread::scope(|scope| {
            let worker0 = scope.spawn(|| {
                backend0.prefill_request_with_reservation(
                    80,
                    &prompt[..1],
                    0,
                    Some(&[0]),
                    Some(&[1, 1]),
                )
            });
            let worker1 = scope.spawn(|| {
                backend1.prefill_request_with_reservation(
                    80,
                    &prompt[1..],
                    1,
                    Some(&[1, 2, 3]),
                    Some(&[3, 3]),
                )
            });
            (
                worker0.join().unwrap().unwrap(),
                worker1.join().unwrap().unwrap(),
            )
        });

        let first_diff = (Tensor::from_slice(&logits0) - reference_logits.select(1, 0).squeeze())
            .abs()
            .max()
            .double_value(&[]);
        let last_diff = (Tensor::from_slice(&logits1) - reference_logits.select(1, 3).squeeze())
            .abs()
            .max()
            .double_value(&[]);
        assert!(first_diff < 1e-3, "first-position max diff: {first_diff}");
        assert!(last_diff < 1e-3, "last-position max diff: {last_diff}");
    }

    #[test]
    fn positioned_continuation_reuses_two_worker_reserved_request_cache() {
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
        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut backend0 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            0,
        );
        let mut backend1 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            1,
        );

        let mut transports0: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        let mut transports1: Vec<Box<dyn KvTransport>> = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let stream0 = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let stream1 = listener.accept().unwrap().0;
            transports0.push(Box::new(
                crate::model::transport::TcpKvTransport::new(stream0, device).unwrap(),
            ));
            transports1.push(Box::new(
                crate::model::transport::TcpKvTransport::new(stream1, device).unwrap(),
            ));
        }
        backend0.setup_kv_transports(transports0);
        backend1.setup_kv_transports(transports1);

        let request_id = 81;
        let initial_prompt = [3_i64, 5, 7, 9];
        let continuation_prompt = [11_i64, 13, 17, 19];
        let schedule = FrozenKvAssigneeSchedule::new(&[1, 3], request_id, config.num_layers)
            .expect("one decode token must have a complete assignee schedule");
        assert_eq!(schedule.counts(), &[6, 18]);
        let assignees = (0..config.num_layers)
            .map(|layer_idx| {
                schedule
                    .assignee_for(0, layer_idx, config.num_layers)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let capacities0 = assignees
            .iter()
            .map(|&assignee| 1 + usize::from(assignee == 0) + 1)
            .collect::<Vec<_>>();
        let capacities1 = assignees
            .iter()
            .map(|&assignee| 3 + usize::from(assignee == 1) + 3)
            .collect::<Vec<_>>();

        let mut reference_caches = reference.create_kv_caches();
        let reference_prefill_logits = reference
            .forward(
                &Tensor::from_slice(&initial_prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, initial_prompt.len() as i64 - 1)
            .squeeze();
        let (_, (distributed_prefill_logits, _)) = std::thread::scope(|scope| {
            let worker0 = scope.spawn(|| {
                backend0.prefill_request_with_reservation(
                    request_id,
                    &initial_prompt[..1],
                    0,
                    Some(&[0]),
                    Some(&capacities0),
                )
            });
            let worker1 = scope.spawn(|| {
                backend1.prefill_request_with_reservation(
                    request_id,
                    &initial_prompt[1..],
                    1,
                    Some(&[1, 2, 3]),
                    Some(&capacities1),
                )
            });
            (
                worker0.join().unwrap().unwrap(),
                worker1.join().unwrap().unwrap(),
            )
        });
        let distributed_prefill_logits = Tensor::from_slice(&distributed_prefill_logits);
        let prefill_diff = (&distributed_prefill_logits - &reference_prefill_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            prefill_diff < 1e-3,
            "initial prefill max diff: {prefill_diff}"
        );

        let decode_token = distributed_prefill_logits
            .argmax(-1, false)
            .int64_value(&[]);
        let mut backends = vec![backend0, backend1];
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let stream0 = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let stream1 = listener.accept().unwrap().0;
        let mut decode_transports = vec![
            crate::model::transport::TcpKvTransport::new(stream0, device).unwrap(),
            crate::model::transport::TcpKvTransport::new(stream1, device).unwrap(),
        ];
        let (distributed_decode_logits, _, hops, _) = run_two_backend_reserved_tcp_decode(
            &mut backends,
            &mut decode_transports,
            request_id,
            decode_token,
            initial_prompt.len() as i64,
            1,
            &assignees,
        );
        assert_eq!(hops, config.num_layers);
        let reference_decode_logits = reference
            .forward(
                &Tensor::from_slice(&[decode_token]).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .squeeze();
        let decode_diff = (&distributed_decode_logits.squeeze() - &reference_decode_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(decode_diff < 1e-3, "decode max diff: {decode_diff}");

        let storage_before = backends
            .iter()
            .map(|backend| {
                let context = backend.request_contexts.get(&request_id).unwrap();
                context
                    .kv_caches
                    .iter()
                    .map(|cache| {
                        let Some(KvCacheImpl::ReservedPositioned(shard)) = cache else {
                            panic!("mixed-history cache must remain reserved positioned KV");
                        };
                        (
                            shard.active_k().data_ptr() as usize,
                            shard.active_v().data_ptr() as usize,
                            shard.reserved_capacity(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let ((_, global_len0), (continuation_logits, global_len1)) = {
            let (left, right) = backends.split_at_mut(1);
            std::thread::scope(|scope| {
                let worker0 = scope.spawn(|| {
                    left[0].prefill_request_with_reservation(
                        request_id,
                        &continuation_prompt[..1],
                        5,
                        Some(&[5]),
                        None,
                    )
                });
                let worker1 = scope.spawn(|| {
                    right[0].prefill_request_with_reservation(
                        request_id,
                        &continuation_prompt[1..],
                        6,
                        Some(&[6, 7, 8]),
                        None,
                    )
                });
                (
                    worker0.join().unwrap().unwrap(),
                    worker1.join().unwrap().unwrap(),
                )
            })
        };
        assert_eq!(global_len0, 6);
        assert_eq!(global_len1, 9);

        for layer_idx in 0..config.num_layers {
            let mut positions = Vec::new();
            for domain in 0..2 {
                let context = backends[domain].request_contexts.get(&request_id).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) = &context.kv_caches[layer_idx]
                else {
                    panic!("worker {domain} layer {layer_idx} rebuilt its request cache");
                };
                assert_eq!(
                    shard.active_k().data_ptr() as usize,
                    storage_before[domain][layer_idx].0
                );
                assert_eq!(
                    shard.active_v().data_ptr() as usize,
                    storage_before[domain][layer_idx].1
                );
                assert_eq!(
                    shard.reserved_capacity(),
                    storage_before[domain][layer_idx].2
                );
                positions.extend_from_slice(shard.positions());
            }
            positions.sort_unstable();
            assert_eq!(positions, (0_i64..=8).collect::<Vec<_>>());
        }

        reference.set_prefill_position_ids(&[5, 6, 7, 8], device);
        let reference_continuation_logits = reference
            .forward(
                &Tensor::from_slice(&continuation_prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, continuation_prompt.len() as i64 - 1)
            .squeeze();
        let continuation_diff = (Tensor::from_slice(&continuation_logits)
            - reference_continuation_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            continuation_diff < 1e-3,
            "continuation max diff: {continuation_diff}"
        );
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

    /// Decode one token through every layer in-process, without any wire transport.
    fn run_two_backend_reserved_local_decode(
        backends: &mut [TchWorkerBackend],
        request_id: u64,
        token: i64,
        position: i64,
        starter: usize,
        assignees: &[usize],
    ) -> (Tensor, usize) {
        let domains = backends.len();
        assert_eq!(domains, 2);
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
                let outcome = {
                    let backend = &mut backends[domain];
                    let context = backend.request_contexts.get_mut(&request_id).unwrap();
                    let Some(KvCacheImpl::ReservedPositioned(shard)) =
                        &mut context.kv_caches[layer_idx]
                    else {
                        panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                    };
                    process_layer_packet_with_reserved_history(
                        &mut backend.model.layers[layer_idx],
                        packet.take().unwrap(),
                        shard,
                    )
                    .unwrap()
                };
                match outcome {
                    LayerStepOutcome::Forward(next_packet) => {
                        packet = Some(next_packet);
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
        for backend in backends.iter_mut() {
            let context = backend.request_contexts.get_mut(&request_id).unwrap();
            context.global_seq_len = position as usize + 1;
        }
        (logits, current_starter)
    }

    #[test]
    #[ignore = "requires the local Qwen2-0.5B model weights"]
    fn real_qwen_two_worker_stationary_continuation_matches_reference() {
        let device = Device::Cpu;
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("Qwen2-0.5B");
        let config = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
        assert_eq!(config.num_layers, 24);
        let weights = ModelWeights::from_dir(&model_dir, device).unwrap();

        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut backend0 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            0,
        );
        let mut backend1 = TchWorkerBackend::from_model(
            LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
            device,
            1,
        );
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
        let mut backends = vec![backend0, backend1];

        let domains = 2_usize;
        let layers = config.num_layers;
        let request_id = 75_u64;
        let capacity_tickets = [1_u64, 3];
        let prompt = [151644_i64, 9707, 0, 16];
        let continuation_prompt = [11_i64, 13, 17, 19];
        let prefix_splits = [1_usize, 3];
        let prefix_len = prompt.len();
        let decode_position = prefix_len as i64;
        let continuation_len = continuation_prompt.len();
        let continuation_positions =
            (decode_position + 1..=decode_position + continuation_len as i64).collect::<Vec<_>>();

        let decode_schedule =
            FrozenKvAssigneeSchedule::new(&capacity_tickets, request_id, layers).unwrap();
        assert_eq!(decode_schedule.counts(), &[6, 18]);
        let decode_assignees = (0..layers)
            .map(|layer_idx| decode_schedule.assignee_for(0, layer_idx, layers).unwrap())
            .collect::<Vec<_>>();

        let continuation_schedule =
            FrozenKvAssigneeSchedule::new(&capacity_tickets, request_id, continuation_len).unwrap();
        assert_eq!(continuation_schedule.counts(), prefix_splits);
        let mut continuation_offsets_by_domain = vec![Vec::new(); domains];
        for offset in 0..continuation_len {
            let domain = continuation_schedule.assignee_for(offset, 0, 1).unwrap();
            continuation_offsets_by_domain[domain].push(offset);
        }
        let mut assigned_offsets = continuation_offsets_by_domain
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assigned_offsets.sort_unstable();
        assert_eq!(assigned_offsets, (0..continuation_len).collect::<Vec<_>>());

        let capacities = (0..domains)
            .map(|domain| {
                (0..layers)
                    .map(|layer_idx| {
                        prefix_splits[domain]
                            + usize::from(decode_assignees[layer_idx] == domain)
                            + continuation_offsets_by_domain[domain].len()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut reference_caches = reference.create_kv_caches();
        let reference_prefill_logits = reference
            .forward(
                &Tensor::from_slice(&prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, prefix_len as i64 - 1)
            .squeeze();

        let (_, global_len0) = backends[0]
            .prefill_request_with_reservation(
                request_id,
                &prompt[..1],
                0,
                Some(&[0]),
                Some(&capacities[0]),
            )
            .unwrap();
        let (distributed_prefill_logits, global_len1) = backends[1]
            .prefill_request_with_reservation(
                request_id,
                &prompt[1..],
                1,
                Some(&[1, 2, 3]),
                Some(&capacities[1]),
            )
            .unwrap();
        assert_eq!((global_len0, global_len1), (1, prefix_len));
        let distributed_prefill_logits = Tensor::from_slice(&distributed_prefill_logits);
        let decode_token = distributed_prefill_logits
            .argmax(-1, false)
            .int64_value(&[]);
        assert_eq!(
            decode_token,
            reference_prefill_logits.argmax(-1, false).int64_value(&[])
        );

        let storage_before = backends
            .iter()
            .map(|backend| {
                let context = backend.request_contexts.get(&request_id).unwrap();
                context
                    .kv_caches
                    .iter()
                    .map(|cache| {
                        let Some(KvCacheImpl::ReservedPositioned(shard)) = cache else {
                            panic!("mixed-history cache must remain reserved positioned KV");
                        };
                        (
                            shard.active_k().data_ptr() as usize,
                            shard.active_v().data_ptr() as usize,
                            shard.reserved_capacity(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let (distributed_decode_logits, decode_finisher) = run_two_backend_reserved_local_decode(
            &mut backends,
            request_id,
            decode_token,
            decode_position,
            1,
            &decode_assignees,
        );
        let reference_decode_logits = reference
            .forward(
                &Tensor::from_slice(&[decode_token]).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .squeeze();
        let distributed_decode_last = distributed_decode_logits.squeeze();
        let decode_max_diff = (&distributed_decode_last - &reference_decode_logits)
            .abs()
            .max()
            .double_value(&[]);
        println!("real pre-continuation decode: max_diff={decode_max_diff:.6}");
        assert_eq!(
            distributed_decode_last.argmax(-1, false).int64_value(&[]),
            reference_decode_logits.argmax(-1, false).int64_value(&[])
        );

        // Route B stationary continuation: historical KV never enters the packet;
        // each worker projects and appends only its own position offsets.
        let mut current_starter = decode_finisher;
        let continuation_input = Tensor::from_slice(&continuation_prompt)
            .unsqueeze(0)
            .to_device(device);
        let mut hidden_states = Tensor::embedding(
            &backends[current_starter].model.embedding,
            &continuation_input,
            -1,
            false,
            false,
        );
        let position_ids = Tensor::from_slice(&continuation_positions)
            .unsqueeze(0)
            .to_device(device);
        let mut handoffs = 0_usize;
        for layer_idx in 0..layers {
            // Position ownership comes from the frozen offsets, not the legacy scalar.
            let mut packet = Some(
                LayerPacket::start(
                    &mut backends[current_starter].model.layers[layer_idx],
                    &hidden_states,
                    &position_ids,
                    current_starter,
                    current_starter,
                    domains,
                )
                .unwrap(),
            );
            let mut next_hidden = None;
            for visit_index in 0..domains {
                let domain = (current_starter + visit_index) % domains;
                let outcome = {
                    let backend = &mut backends[domain];
                    let context = backend.request_contexts.get_mut(&request_id).unwrap();
                    let Some(KvCacheImpl::ReservedPositioned(shard)) =
                        &mut context.kv_caches[layer_idx]
                    else {
                        panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                    };
                    process_layer_packet_with_reserved_history_for_positions(
                        &mut backend.model.layers[layer_idx],
                        packet.take().unwrap(),
                        shard,
                        &continuation_offsets_by_domain[domain],
                    )
                    .unwrap()
                };
                match outcome {
                    LayerStepOutcome::Forward(next_packet) => {
                        handoffs += 1;
                        packet = Some(next_packet);
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
        assert_eq!(handoffs, layers * (domains - 1));

        let continuation_logits =
            project_final_logits(&backends[current_starter].model, &hidden_states);
        let distributed_continuation_last = continuation_logits
            .select(1, continuation_len as i64 - 1)
            .squeeze();

        reference.set_prefill_position_ids(&continuation_positions, device);
        let reference_continuation_logits = reference
            .forward(
                &Tensor::from_slice(&continuation_prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, continuation_len as i64 - 1)
            .squeeze();

        let max_diff = (&distributed_continuation_last - &reference_continuation_logits)
            .abs()
            .max()
            .double_value(&[]);
        let mean_diff = (&distributed_continuation_last - &reference_continuation_logits)
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        let reference_token = reference_continuation_logits
            .argmax(-1, false)
            .int64_value(&[]);
        let distributed_token = distributed_continuation_last
            .argmax(-1, false)
            .int64_value(&[]);
        println!(
            "real stationary continuation: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, tokens={reference_token}/{distributed_token}"
        );
        assert_eq!(reference_token, distributed_token);
        assert!(
            mean_diff < 0.1,
            "continuation mean logits diff: {mean_diff}"
        );
        assert!(max_diff < 0.75, "continuation max logits diff: {max_diff}");

        let domain_totals = (0..domains)
            .map(|domain| {
                (0..layers)
                    .map(|layer_idx| {
                        let context = backends[domain].request_contexts.get(&request_id).unwrap();
                        let Some(KvCacheImpl::ReservedPositioned(shard)) =
                            &context.kv_caches[layer_idx]
                        else {
                            panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                        };
                        assert_eq!(
                            shard.active_k().data_ptr() as usize,
                            storage_before[domain][layer_idx].0
                        );
                        assert_eq!(
                            shard.active_v().data_ptr() as usize,
                            storage_before[domain][layer_idx].1
                        );
                        assert_eq!(
                            shard.reserved_capacity(),
                            storage_before[domain][layer_idx].2
                        );
                        assert_eq!(shard.committed_len(), shard.reserved_capacity());
                        shard.committed_len()
                    })
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        assert_eq!(domain_totals, vec![54, 162]);

        for layer_idx in 0..layers {
            let mut positions = Vec::new();
            for domain in 0..domains {
                let context = backends[domain].request_contexts.get(&request_id).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) = &context.kv_caches[layer_idx]
                else {
                    panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                };
                positions.extend_from_slice(shard.positions());
            }
            positions.sort_unstable();
            assert_eq!(positions, (0_i64..=8).collect::<Vec<_>>());
        }
    }

    /// The production `run_stationary_continuation` driver on a mock ring,
    /// synthetic config: same 97ca355-style scenario, continuation phase
    /// executed by the new method instead of the in-test hand-to-hand loop.
    #[test]
    fn stationary_continuation_driver_matches_reference_on_mock_ring() {
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
        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let layers = config.num_layers;
        let domains = 2_usize;
        let mut backends = (0..domains)
            .map(|domain| {
                TchWorkerBackend::from_model(
                    LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap(),
                    device,
                    domain,
                )
            })
            .collect::<Vec<_>>();
        let mut per_backend: Vec<Vec<Box<dyn KvTransport>>> =
            (0..domains).map(|_| Vec::with_capacity(layers)).collect();
        for _ in 0..layers {
            for (domain, endpoint) in
                crate::model::transport::LinkedMockKvTransport::create_ring(domains)
                    .into_iter()
                    .enumerate()
            {
                per_backend[domain].push(Box::new(endpoint));
            }
        }
        for (domain, transports) in per_backend.into_iter().enumerate() {
            backends[domain].setup_kv_transports(transports);
        }

        let request_id = 81_u64;
        let prompt = [3_i64, 5, 7, 9];
        let continuation_prompt = [11_i64, 13, 17, 19];
        let continuation_positions = [5_i64, 6, 7, 8];
        let capacity_tickets = [1_u64, 3];
        let prefix_splits = [1_usize, 3];

        let decode_schedule =
            FrozenKvAssigneeSchedule::new(&capacity_tickets, request_id, layers).unwrap();
        assert_eq!(decode_schedule.counts(), &[6, 18]);
        let decode_assignees = (0..layers)
            .map(|layer_idx| decode_schedule.assignee_for(0, layer_idx, layers).unwrap())
            .collect::<Vec<_>>();
        let continuation_schedule =
            FrozenKvAssigneeSchedule::new(&capacity_tickets, request_id, continuation_prompt.len())
                .unwrap();
        assert_eq!(continuation_schedule.counts(), prefix_splits);
        let mut continuation_offsets_by_domain = vec![Vec::new(); domains];
        for offset in 0..continuation_prompt.len() {
            let domain = continuation_schedule.assignee_for(offset, 0, 1).unwrap();
            continuation_offsets_by_domain[domain].push(offset);
        }
        let capacities = (0..domains)
            .map(|domain| {
                (0..layers)
                    .map(|layer_idx| {
                        prefix_splits[domain]
                            + usize::from(decode_assignees[layer_idx] == domain)
                            + continuation_offsets_by_domain[domain].len()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut reference_caches = reference.create_kv_caches();
        let reference_prefill_logits = reference
            .forward(
                &Tensor::from_slice(&prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, prompt.len() as i64 - 1)
            .squeeze();
        backends[0]
            .prefill_request_with_reservation(
                request_id,
                &prompt[..1],
                0,
                Some(&[0]),
                Some(&capacities[0]),
            )
            .unwrap();
        let (prefill_logits1, _) = backends[1]
            .prefill_request_with_reservation(
                request_id,
                &prompt[1..],
                1,
                Some(&[1, 2, 3]),
                Some(&capacities[1]),
            )
            .unwrap();
        let decode_token = Tensor::from_slice(&prefill_logits1)
            .argmax(-1, false)
            .int64_value(&[]);
        assert_eq!(
            decode_token,
            reference_prefill_logits.argmax(-1, false).int64_value(&[])
        );

        let (distributed_decode_logits, decode_finisher) = run_two_backend_reserved_local_decode(
            &mut backends,
            request_id,
            decode_token,
            prompt.len() as i64,
            1,
            &decode_assignees,
        );
        assert_eq!(decode_finisher, 1);
        let reference_decode_logits = reference
            .forward(
                &Tensor::from_slice(&[decode_token]).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .squeeze();
        assert_eq!(
            distributed_decode_logits
                .squeeze()
                .argmax(-1, false)
                .int64_value(&[]),
            reference_decode_logits.argmax(-1, false).int64_value(&[])
        );

        // The continuation phase runs through the production driver; the two
        // backends must proceed concurrently because mock receives busy-wait.
        let (left, right) = backends.split_at_mut(1);
        let (result0, result1) = std::thread::scope(|scope| {
            let worker0 = scope.spawn(|| {
                left[0].run_stationary_continuation(
                    request_id,
                    &continuation_prompt,
                    &continuation_positions,
                    &capacity_tickets,
                    decode_finisher,
                )
            });
            let worker1 = scope.spawn(|| {
                right[0].run_stationary_continuation(
                    request_id,
                    &continuation_prompt,
                    &continuation_positions,
                    &capacity_tickets,
                    decode_finisher,
                )
            });
            (worker0.join().unwrap(), worker1.join().unwrap())
        });
        let logits0 = result0.unwrap();
        let logits1 = result1.unwrap();
        assert!(logits0.is_none(), "non-finisher must not return logits");
        let continuation_logits =
            Tensor::from_slice(&logits1.expect("final finisher must return logits"));

        reference.set_prefill_position_ids(&continuation_positions, device);
        let reference_continuation_logits = reference
            .forward(
                &Tensor::from_slice(&continuation_prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, continuation_prompt.len() as i64 - 1)
            .squeeze();
        let max_diff = (&continuation_logits - &reference_continuation_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert_eq!(
            continuation_logits.argmax(-1, false).int64_value(&[]),
            reference_continuation_logits
                .argmax(-1, false)
                .int64_value(&[])
        );
        assert!(max_diff < 1e-3, "driver continuation max diff: {max_diff}");

        // Both backends resynced the request horizon to the segment end.
        for backend in &backends {
            let context = backend.request_contexts.get(&request_id).unwrap();
            assert_eq!(context.global_seq_len, 9);
        }

        let domain_totals = (0..domains)
            .map(|domain| {
                let context = backends[domain].request_contexts.get(&request_id).unwrap();
                (0..layers)
                    .map(|layer_idx| {
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
        assert_eq!(domain_totals, vec![54, 162]);

        for layer_idx in 0..layers {
            let mut positions = Vec::new();
            for backend in &backends {
                let context = backend.request_contexts.get(&request_id).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) = &context.kv_caches[layer_idx]
                else {
                    panic!("layer {layer_idx} did not use reserved KV");
                };
                positions.extend_from_slice(shard.positions());
            }
            positions.sort_unstable();
            assert_eq!(positions, (0_i64..=8).collect::<Vec<_>>());
        }

        // Validation rejects malformed segments before touching state.
        backends[0]
            .run_stationary_continuation(request_id, &[], &[], &capacity_tickets, 1)
            .unwrap_err();
        backends[0]
            .run_stationary_continuation(
                request_id,
                &continuation_prompt,
                &continuation_positions[..2],
                &capacity_tickets,
                1,
            )
            .unwrap_err();
        backends[0]
            .run_stationary_continuation(
                999,
                &continuation_prompt,
                &continuation_positions,
                &capacity_tickets,
                1,
            )
            .unwrap_err();
    }

    /// Real-weight 97ca355 scenario: prefill [1,3] split + one local decode
    /// step + continuation via the production driver, compared against the
    /// contiguous reference with the golden tolerances.
    #[test]
    #[ignore = "requires the local Qwen2-0.5B model weights"]
    fn real_qwen_stationary_continuation_driver_matches_reference() {
        let device = Device::Cpu;
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("Qwen2-0.5B");
        let config = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
        assert_eq!(config.num_layers, 24);
        let layers = config.num_layers;
        let domains = 2_usize;
        let weights = ModelWeights::from_dir(&model_dir, device).unwrap();

        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut backends = (0..domains)
            .map(|domain| {
                TchWorkerBackend::from_model(
                    LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap(),
                    device,
                    domain,
                )
            })
            .collect::<Vec<_>>();
        let mut per_backend: Vec<Vec<Box<dyn KvTransport>>> =
            (0..domains).map(|_| Vec::with_capacity(layers)).collect();
        for _ in 0..layers {
            for (domain, endpoint) in
                crate::model::transport::LinkedMockKvTransport::create_ring(domains)
                    .into_iter()
                    .enumerate()
            {
                per_backend[domain].push(Box::new(endpoint));
            }
        }
        for (domain, transports) in per_backend.into_iter().enumerate() {
            backends[domain].setup_kv_transports(transports);
        }

        let request_id = 75_u64;
        let prompt = [151644_i64, 9707, 0, 16];
        let continuation_prompt = [11_i64, 13, 17, 19];
        let continuation_positions = [5_i64, 6, 7, 8];
        let capacity_tickets = [1_u64, 3];
        let prefix_splits = [1_usize, 3];

        let decode_schedule =
            FrozenKvAssigneeSchedule::new(&capacity_tickets, request_id, layers).unwrap();
        assert_eq!(decode_schedule.counts(), &[6, 18]);
        let decode_assignees = (0..layers)
            .map(|layer_idx| decode_schedule.assignee_for(0, layer_idx, layers).unwrap())
            .collect::<Vec<_>>();
        let continuation_schedule =
            FrozenKvAssigneeSchedule::new(&capacity_tickets, request_id, continuation_prompt.len())
                .unwrap();
        assert_eq!(continuation_schedule.counts(), prefix_splits);
        let mut continuation_offsets_by_domain = vec![Vec::new(); domains];
        for offset in 0..continuation_prompt.len() {
            let domain = continuation_schedule.assignee_for(offset, 0, 1).unwrap();
            continuation_offsets_by_domain[domain].push(offset);
        }
        let capacities = (0..domains)
            .map(|domain| {
                (0..layers)
                    .map(|layer_idx| {
                        prefix_splits[domain]
                            + usize::from(decode_assignees[layer_idx] == domain)
                            + continuation_offsets_by_domain[domain].len()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut reference_caches = reference.create_kv_caches();
        let reference_prefill_logits = reference
            .forward(
                &Tensor::from_slice(&prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, prompt.len() as i64 - 1)
            .squeeze();
        backends[0]
            .prefill_request_with_reservation(
                request_id,
                &prompt[..1],
                0,
                Some(&[0]),
                Some(&capacities[0]),
            )
            .unwrap();
        let (prefill_logits1, _) = backends[1]
            .prefill_request_with_reservation(
                request_id,
                &prompt[1..],
                1,
                Some(&[1, 2, 3]),
                Some(&capacities[1]),
            )
            .unwrap();
        let decode_token = Tensor::from_slice(&prefill_logits1)
            .argmax(-1, false)
            .int64_value(&[]);
        assert_eq!(
            decode_token,
            reference_prefill_logits.argmax(-1, false).int64_value(&[])
        );

        let (distributed_decode_logits, decode_finisher) = run_two_backend_reserved_local_decode(
            &mut backends,
            request_id,
            decode_token,
            prompt.len() as i64,
            1,
            &decode_assignees,
        );
        let reference_decode_logits = reference
            .forward(
                &Tensor::from_slice(&[decode_token]).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .squeeze();
        assert_eq!(
            distributed_decode_logits
                .squeeze()
                .argmax(-1, false)
                .int64_value(&[]),
            reference_decode_logits.argmax(-1, false).int64_value(&[])
        );

        let (left, right) = backends.split_at_mut(1);
        let (result0, result1) = std::thread::scope(|scope| {
            let worker0 = scope.spawn(|| {
                left[0].run_stationary_continuation(
                    request_id,
                    &continuation_prompt,
                    &continuation_positions,
                    &capacity_tickets,
                    decode_finisher,
                )
            });
            let worker1 = scope.spawn(|| {
                right[0].run_stationary_continuation(
                    request_id,
                    &continuation_prompt,
                    &continuation_positions,
                    &capacity_tickets,
                    decode_finisher,
                )
            });
            (worker0.join().unwrap(), worker1.join().unwrap())
        });
        assert!(result0.unwrap().is_none());
        let continuation_logits =
            Tensor::from_slice(&result1.unwrap().expect("final finisher must return logits"));

        reference.set_prefill_position_ids(&continuation_positions, device);
        let reference_continuation_logits = reference
            .forward(
                &Tensor::from_slice(&continuation_prompt).unsqueeze(0),
                &mut reference_caches,
            )
            .unwrap()
            .select(1, continuation_prompt.len() as i64 - 1)
            .squeeze();
        let max_diff = (&continuation_logits - &reference_continuation_logits)
            .abs()
            .max()
            .double_value(&[]);
        let mean_diff = (&continuation_logits - &reference_continuation_logits)
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        let reference_token = reference_continuation_logits
            .argmax(-1, false)
            .int64_value(&[]);
        let driver_token = continuation_logits.argmax(-1, false).int64_value(&[]);
        println!(
            "real stationary continuation driver: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, tokens={driver_token}/{reference_token}"
        );
        assert_eq!(reference_token, driver_token);
        assert!(
            mean_diff < 0.1,
            "driver continuation mean logits diff: {mean_diff}"
        );
        assert!(
            max_diff < 0.75,
            "driver continuation max logits diff: {max_diff}"
        );

        let domain_totals = (0..domains)
            .map(|domain| {
                let context = backends[domain].request_contexts.get(&request_id).unwrap();
                (0..layers)
                    .map(|layer_idx| {
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
        assert_eq!(domain_totals, vec![54, 162]);

        for layer_idx in 0..layers {
            let mut positions = Vec::new();
            for backend in &backends {
                let context = backend.request_contexts.get(&request_id).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) = &context.kv_caches[layer_idx]
                else {
                    panic!("layer {layer_idx} did not use reserved KV");
                };
                positions.extend_from_slice(shard.positions());
            }
            positions.sort_unstable();
            assert_eq!(positions, (0_i64..=8).collect::<Vec<_>>());
        }
    }
}
