//! Experimental single-layer self-driving decode ring.

use crate::model::attention::HcpRingAttentionBackend;
use crate::model::layers::DecoderLayer;
use crate::model::model::LlamaModel;
use crate::model::transport::SelfDrivingPacket;
use crate::model::ModelError;
use tch::{Kind, Tensor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenKvAssigneeSchedule {
    sequence: Vec<usize>,
    counts: Vec<usize>,
    phase: usize,
}

impl FrozenKvAssigneeSchedule {
    pub fn new(
        capacity_tickets: &[u64],
        request_id: u64,
        total_kv_units: usize,
    ) -> Result<Self, String> {
        if capacity_tickets.is_empty() {
            return Err("KV assignee schedule requires at least one worker".to_string());
        }
        if total_kv_units == 0 {
            return Err("KV assignee schedule requires at least one KV unit".to_string());
        }
        let capacity_sum = capacity_tickets
            .iter()
            .try_fold(0_u128, |sum, &tickets| sum.checked_add(tickets as u128))
            .ok_or_else(|| "KV assignee capacity sum overflow".to_string())?;
        if capacity_sum == 0 {
            return Err("KV assignee schedule requires non-zero capacity".to_string());
        }

        let total = total_kv_units as u128;
        let mut counts = Vec::with_capacity(capacity_tickets.len());
        let mut remainders = Vec::with_capacity(capacity_tickets.len());
        for (domain, &tickets) in capacity_tickets.iter().enumerate() {
            let numerator = total
                .checked_mul(tickets as u128)
                .ok_or_else(|| "KV assignee allocation overflow".to_string())?;
            counts.push((numerator / capacity_sum) as usize);
            remainders.push((domain, numerator % capacity_sum));
        }
        remainders.sort_by(|(left_domain, left), (right_domain, right)| {
            right.cmp(left).then_with(|| left_domain.cmp(right_domain))
        });
        let assigned = counts.iter().sum::<usize>();
        let remaining = total_kv_units
            .checked_sub(assigned)
            .ok_or_else(|| "KV assignee allocation overflow".to_string())?;
        for &(domain, _) in remainders.iter().take(remaining) {
            counts[domain] += 1;
        }

        let mut scores = vec![0_i128; counts.len()];
        let total_score = total_kv_units as i128;
        let mut sequence = Vec::with_capacity(total_kv_units);
        for _ in 0..total_kv_units {
            for (score, &count) in scores.iter_mut().zip(&counts) {
                *score += count as i128;
            }
            let selected = scores
                .iter()
                .enumerate()
                .max_by(|(left_domain, left), (right_domain, right)| {
                    left.cmp(right).then_with(|| right_domain.cmp(left_domain))
                })
                .map(|(domain, _)| domain)
                .ok_or_else(|| "KV assignee schedule is empty".to_string())?;
            scores[selected] -= total_score;
            sequence.push(selected);
        }

        Ok(Self {
            sequence,
            counts,
            phase: (request_id as u128 % total) as usize,
        })
    }

    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    pub fn total_units(&self) -> usize {
        self.sequence.len()
    }

    pub fn phase(&self) -> usize {
        self.phase
    }

    pub fn assignee_for(
        &self,
        token_offset: usize,
        layer_idx: usize,
        num_layers: usize,
    ) -> Option<usize> {
        if num_layers == 0 || layer_idx >= num_layers {
            return None;
        }
        let ordinal = token_offset
            .checked_mul(num_layers)?
            .checked_add(layer_idx)?;
        if ordinal >= self.sequence.len() {
            return None;
        }
        let index = self.phase.checked_add(ordinal)? % self.sequence.len();
        self.sequence.get(index).copied()
    }
}

#[derive(Debug)]
pub struct LayerPacket {
    residual: Tensor,
    normalized: Tensor,
    position_ids: Tensor,
    q: Tensor,
    attention_output: Option<Tensor>,
    lse: Option<Tensor>,
    assignee: usize,
    current_domain: usize,
    domains: usize,
    visited_domains: usize,
}

impl LayerPacket {
    pub fn start(
        layer: &mut DecoderLayer,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        starter: usize,
        assignee: usize,
        domains: usize,
    ) -> Result<Self, ModelError> {
        validate_route(hidden_states, starter, assignee, domains)?;
        let residual = hidden_states.shallow_clone();
        let normalized = layer.input_layernorm.forward(hidden_states);
        let q = ring_backend(layer)?.project_decode_q(&normalized, position_ids)?;
        Ok(Self {
            residual,
            normalized,
            position_ids: position_ids.shallow_clone(),
            q,
            attention_output: None,
            lse: None,
            assignee,
            current_domain: starter,
            domains,
            visited_domains: 0,
        })
    }

    pub fn tensor_payload_elements(&self) -> usize {
        let fixed = self.residual.numel()
            + self.normalized.numel()
            + self.position_ids.numel()
            + self.q.numel();
        fixed
            + self.attention_output.as_ref().map_or(0, Tensor::numel)
            + self.lse.as_ref().map_or(0, Tensor::numel)
    }

    pub fn into_self_driving_packet(
        self,
        layer_idx: usize,
    ) -> Result<SelfDrivingPacket, ModelError> {
        let attention_output = self.attention_output.ok_or_else(|| {
            ModelError::Backend(
                "self-driving wire packet requires an attention accumulator".to_string(),
            )
        })?;
        let lse = self.lse.ok_or_else(|| {
            ModelError::Backend("self-driving wire packet requires an LSE accumulator".to_string())
        })?;
        Ok(SelfDrivingPacket {
            layer_idx,
            residual: self.residual,
            normalized: self.normalized,
            position_ids: self.position_ids,
            q: self.q,
            attention_output,
            lse,
            assignee: self.assignee,
            current_domain: self.current_domain,
            domains: self.domains,
            visited_domains: self.visited_domains,
        })
    }

    pub fn from_self_driving_packet(packet: SelfDrivingPacket) -> Result<Self, ModelError> {
        if packet.domains == 0
            || packet.assignee >= packet.domains
            || packet.current_domain >= packet.domains
            || packet.visited_domains >= packet.domains
        {
            return Err(ModelError::Backend(format!(
                "invalid self-driving wire route: domains={}, assignee={}, current_domain={}, visited_domains={}",
                packet.domains, packet.assignee, packet.current_domain, packet.visited_domains
            )));
        }
        Ok(Self {
            residual: packet.residual,
            normalized: packet.normalized,
            position_ids: packet.position_ids,
            q: packet.q,
            attention_output: Some(packet.attention_output),
            lse: Some(packet.lse),
            assignee: packet.assignee,
            current_domain: packet.current_domain,
            domains: packet.domains,
            visited_domains: packet.visited_domains,
        })
    }
}

#[derive(Debug)]
pub enum LayerStepOutcome {
    Forward(LayerPacket),
    Finished {
        attention_output: Tensor,
        hidden_states: Tensor,
    },
}

fn validate_route(
    hidden_states: &Tensor,
    starter: usize,
    assignee: usize,
    domains: usize,
) -> Result<(), ModelError> {
    if domains == 0 {
        return Err(ModelError::Backend(
            "self-driving layer requires at least one domain".to_string(),
        ));
    }
    if starter >= domains || assignee >= domains {
        return Err(ModelError::Backend(format!(
            "self-driving route out of range: domains={domains}, starter={starter}, assignee={assignee}"
        )));
    }
    if hidden_states.size().get(1) != Some(&1) {
        return Err(ModelError::Backend(
            "self-driving layer requires one decode token".to_string(),
        ));
    }
    Ok(())
}

fn ring_backend(layer: &mut DecoderLayer) -> Result<&mut HcpRingAttentionBackend, ModelError> {
    layer
        .attention
        .as_any_mut()
        .downcast_mut::<HcpRingAttentionBackend>()
        .ok_or_else(|| {
            ModelError::Backend("self-driving layer requires HcpRingAttentionBackend".to_string())
        })
}

fn project_final_logits(model: &LlamaModel, hidden_states: &Tensor) -> Tensor {
    let normalized = model.norm.forward(hidden_states);
    let lm_head = model.lm_head.as_ref().unwrap_or(&model.embedding);
    let logits = normalized.matmul(&lm_head.transpose(0, 1));
    if model.dtype != Kind::Float {
        logits.to_kind(Kind::Float)
    } else {
        logits
    }
}

pub fn process_layer_packet(
    layer: &mut DecoderLayer,
    mut packet: LayerPacket,
    local_history: &mut (Tensor, Tensor),
) -> Result<LayerStepOutcome, ModelError> {
    if local_history.0.size() != local_history.1.size() {
        return Err(ModelError::Backend(format!(
            "domain {} has mismatched K/V shapes",
            packet.current_domain
        )));
    }

    let finished = packet.visited_domains + 1 == packet.domains;
    let projected_output = {
        let ring = ring_backend(layer)?;
        if packet.current_domain == packet.assignee {
            let (current_k, current_v) =
                ring.project_decode_current_kv(&packet.normalized, &packet.position_ids)?;
            local_history.0 = Tensor::cat(&[&local_history.0, &current_k], 2);
            local_history.1 = Tensor::cat(&[&local_history.1, &current_v], 2);
        }

        let (next_output, next_lse) = match (packet.attention_output.take(), packet.lse.take()) {
            (None, None) => {
                ring.decode_local_compact_partial(&packet.q, &local_history.0, &local_history.1)
            }
            (Some(output), Some(lse)) => ring.decode_merge_compact_partial(
                &packet.q,
                &output,
                &lse,
                &local_history.0,
                &local_history.1,
            ),
            _ => {
                return Err(ModelError::Backend(
                    "self-driving packet has incomplete attention accumulator".to_string(),
                ));
            }
        };

        if finished {
            Some(ring.project_decode_output(&next_output))
        } else {
            packet.attention_output = Some(next_output);
            packet.lse = Some(next_lse);
            None
        }
    };

    if let Some(attention_output) = projected_output {
        let post_attention = &attention_output + &packet.residual;
        let mlp_output = layer
            .mlp
            .forward(&layer.post_attention_layernorm.forward(&post_attention));
        return Ok(LayerStepOutcome::Finished {
            attention_output,
            hidden_states: post_attention + mlp_output,
        });
    }

    packet.visited_domains += 1;
    packet.current_domain = (packet.current_domain + 1) % packet.domains;
    Ok(LayerStepOutcome::Forward(packet))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleLayerRingStats {
    pub domains: usize,
    pub hops: usize,
    pub visited_domains: Vec<usize>,
    pub local_partials: usize,
    pub q_projections: usize,
    pub q_projection_domain: usize,
    pub current_kv_projections: usize,
    pub current_kv_projection_domain: usize,
    pub current_kv_commits: usize,
    pub layer_finishes: usize,
    pub starter: usize,
    pub assignee: usize,
    pub finisher: usize,
}

#[derive(Debug)]
pub struct SingleLayerRingResult {
    pub attention_output: Tensor,
    pub hidden_states: Tensor,
    pub stats: SingleLayerRingStats,
}

#[derive(Debug)]
pub struct TwoLayerRingResult {
    pub hidden_states: Tensor,
    pub layer_stats: [SingleLayerRingStats; 2],
}

#[derive(Debug)]
pub struct TwoLayerLogitsResult {
    pub hidden_states: Tensor,
    pub logits: Tensor,
    pub layer_stats: [SingleLayerRingStats; 2],
    pub logits_producer_domain: usize,
    pub logits_projections: usize,
}

#[derive(Debug)]
pub struct ModelRingResult {
    pub hidden_states: Tensor,
    pub logits: Tensor,
    pub layer_stats: Vec<SingleLayerRingStats>,
    pub logits_producer_domain: usize,
    pub logits_projections: usize,
}

/// Run one real decoder layer over already-sharded history KV.
///
/// This is deliberately an in-process experiment: it proves the tensor data
/// flow without introducing transport, runtime, admission, or retry behavior.
pub fn run_single_layer_ring(
    layer: &mut DecoderLayer,
    hidden_states: &Tensor,
    position_ids: &Tensor,
    history_shards: &mut [(Tensor, Tensor)],
    starter: usize,
    assignee: usize,
) -> Result<SingleLayerRingResult, ModelError> {
    let domains = history_shards.len();
    let mut packet = LayerPacket::start(
        layer,
        hidden_states,
        position_ids,
        starter,
        assignee,
        domains,
    )?;
    let mut visited_domains = Vec::with_capacity(domains);
    let mut current_kv_projections = 0;
    let mut current_kv_commits = 0;
    let (attention_output, hidden_states) = loop {
        let domain = packet.current_domain;
        visited_domains.push(domain);
        let is_assignee = domain == packet.assignee;
        match process_layer_packet(layer, packet, &mut history_shards[domain])? {
            LayerStepOutcome::Forward(next_packet) => {
                if is_assignee {
                    current_kv_projections += 1;
                    current_kv_commits += 1;
                }
                packet = next_packet;
            }
            LayerStepOutcome::Finished {
                attention_output,
                hidden_states,
            } => {
                if is_assignee {
                    current_kv_projections += 1;
                    current_kv_commits += 1;
                }
                break (attention_output, hidden_states);
            }
        }
    };
    let finisher = (starter + domains - 1) % domains;

    Ok(SingleLayerRingResult {
        attention_output,
        hidden_states,
        stats: SingleLayerRingStats {
            domains,
            hops: domains - 1,
            visited_domains,
            local_partials: domains,
            q_projections: 1,
            q_projection_domain: starter,
            current_kv_projections,
            current_kv_projection_domain: assignee,
            current_kv_commits,
            layer_finishes: 1,
            starter,
            assignee,
            finisher,
        },
    })
}

/// Run one decode token through every model layer using finisher-to-starter handoff.
pub fn run_model_ring(
    model: &mut LlamaModel,
    hidden_states: &Tensor,
    position_ids: &Tensor,
    layer_history_shards: &mut [Vec<(Tensor, Tensor)>],
    starter: usize,
    assignees: &[usize],
) -> Result<ModelRingResult, ModelError> {
    let layers = model.layers.len();
    if layers == 0 || layer_history_shards.len() != layers || assignees.len() != layers {
        return Err(ModelError::Backend(format!(
            "self-driving model requires matching non-empty layers, shard sets, and assignees: layers={layers}, shard_sets={}, assignees={}",
            layer_history_shards.len(),
            assignees.len()
        )));
    }

    let mut current_hidden = hidden_states.shallow_clone();
    let mut current_starter = starter;
    let mut layer_stats = Vec::with_capacity(layers);
    for layer_idx in 0..layers {
        let layer_result = run_single_layer_ring(
            &mut model.layers[layer_idx],
            &current_hidden,
            position_ids,
            &mut layer_history_shards[layer_idx],
            current_starter,
            assignees[layer_idx],
        )?;
        current_hidden = layer_result.hidden_states;
        current_starter = layer_result.stats.finisher;
        layer_stats.push(layer_result.stats);
    }

    let logits = project_final_logits(model, &current_hidden);

    Ok(ModelRingResult {
        hidden_states: current_hidden,
        logits,
        layer_stats,
        logits_producer_domain: current_starter,
        logits_projections: 1,
    })
}

/// Run exactly two decoder layers, continuing layer 1 on layer 0's finisher.
pub fn run_two_layer_ring(
    layers: &mut [DecoderLayer],
    hidden_states: &Tensor,
    position_ids: &Tensor,
    layer_history_shards: &mut [Vec<(Tensor, Tensor)>],
    starter: usize,
    assignees: [usize; 2],
) -> Result<TwoLayerRingResult, ModelError> {
    if layers.len() != 2 || layer_history_shards.len() != 2 {
        return Err(ModelError::Backend(format!(
            "two-layer self-driving experiment requires exactly two layers and shard sets: layers={}, shard_sets={}",
            layers.len(),
            layer_history_shards.len()
        )));
    }

    let first = run_single_layer_ring(
        &mut layers[0],
        hidden_states,
        position_ids,
        &mut layer_history_shards[0],
        starter,
        assignees[0],
    )?;
    let second_starter = first.stats.finisher;
    let second = run_single_layer_ring(
        &mut layers[1],
        &first.hidden_states,
        position_ids,
        &mut layer_history_shards[1],
        second_starter,
        assignees[1],
    )?;

    Ok(TwoLayerRingResult {
        hidden_states: second.hidden_states,
        layer_stats: [first.stats, second.stats],
    })
}

pub fn run_two_layer_ring_with_logits(
    model: &mut LlamaModel,
    hidden_states: &Tensor,
    position_ids: &Tensor,
    layer_history_shards: &mut [Vec<(Tensor, Tensor)>],
    starter: usize,
    assignees: [usize; 2],
) -> Result<TwoLayerLogitsResult, ModelError> {
    let layer_result = run_two_layer_ring(
        &mut model.layers,
        hidden_states,
        position_ids,
        layer_history_shards,
        starter,
        assignees,
    )?;
    let logits_producer_domain = layer_result.layer_stats[1].finisher;
    let logits = project_final_logits(model, &layer_result.hidden_states);

    Ok(TwoLayerLogitsResult {
        hidden_states: layer_result.hidden_states,
        logits,
        layer_stats: layer_result.layer_stats,
        logits_producer_domain,
        logits_projections: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cache::{ContiguousKvCache, KvCache};
    use crate::model::layers::{GqaAttention, Mlp, RmsNorm, RotaryEmbedding};
    use crate::model::model::LlamaModel;
    use crate::model::transport::TcpKvTransport;
    use crate::model::{ModelConfig, ModelWeights, WeightNames};
    use std::collections::HashMap;
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use tch::{Device, Kind, Tensor};

    fn test_config() -> ModelConfig {
        ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 1,
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
        }
    }

    fn deterministic_tensor(shape: &[i64], phase: f64, device: Device) -> Tensor {
        let numel = shape.iter().product::<i64>();
        ((Tensor::arange(numel, (Kind::Float, device)) + phase).sin() * 0.05).reshape(shape)
    }

    fn deterministic_weights(config: &ModelConfig, device: Device) -> ModelWeights {
        let hidden = config.hidden_size as i64;
        let intermediate = config.intermediate_size as i64;
        let q_width = (config.num_heads * config.head_dim()) as i64;
        let kv_width = (config.num_kv_heads() * config.head_dim()) as i64;
        let mut tensors = HashMap::new();
        tensors.insert(
            WeightNames::embedding().to_string(),
            deterministic_tensor(&[config.vocab_size as i64, hidden], 1.0, device),
        );
        tensors.insert(
            WeightNames::layer_norm().to_string(),
            Tensor::ones([hidden], (Kind::Float, device)),
        );
        tensors.insert(
            WeightNames::lm_head().to_string(),
            deterministic_tensor(&[config.vocab_size as i64, hidden], 2.0, device),
        );
        for layer_idx in 0..config.num_layers {
            let phase = layer_idx as f64 * 10.0;
            tensors.insert(
                WeightNames::rms_norm_weight(layer_idx),
                Tensor::ones([hidden], (Kind::Float, device)),
            );
            tensors.insert(
                WeightNames::post_attn_norm_weight(layer_idx),
                Tensor::ones([hidden], (Kind::Float, device)),
            );
            for (name, shape, tensor_phase) in [
                (
                    WeightNames::q_proj_weight(layer_idx),
                    vec![q_width, hidden],
                    3.0 + phase,
                ),
                (
                    WeightNames::k_proj_weight(layer_idx),
                    vec![kv_width, hidden],
                    4.0 + phase,
                ),
                (
                    WeightNames::v_proj_weight(layer_idx),
                    vec![kv_width, hidden],
                    5.0 + phase,
                ),
                (
                    WeightNames::o_proj_weight(layer_idx),
                    vec![hidden, q_width],
                    6.0 + phase,
                ),
                (
                    WeightNames::gate_proj_weight(layer_idx),
                    vec![intermediate, hidden],
                    7.0 + phase,
                ),
                (
                    WeightNames::up_proj_weight(layer_idx),
                    vec![intermediate, hidden],
                    8.0 + phase,
                ),
                (
                    WeightNames::down_proj_weight(layer_idx),
                    vec![hidden, intermediate],
                    9.0 + phase,
                ),
            ] {
                tensors.insert(name, deterministic_tensor(&shape, tensor_phase, device));
            }
        }
        ModelWeights { tensors }
    }

    fn reference_layer_output(
        layer_idx: usize,
        config: &ModelConfig,
        weights: &crate::model::ModelWeights,
        hidden: &Tensor,
        position_ids: &Tensor,
        history_k: &Tensor,
        history_v: &Tensor,
    ) -> (Tensor, Tensor) {
        let device = hidden.device();
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings.unwrap(),
            config.rope_theta,
            device,
        );
        let attention = GqaAttention::from_weights(weights, layer_idx, config, &rope).unwrap();
        let input_norm = RmsNorm::from_weights(
            weights,
            &WeightNames::rms_norm_weight(layer_idx),
            config.rms_norm_eps,
        )
        .unwrap();
        let post_norm = RmsNorm::from_weights(
            weights,
            &WeightNames::post_attn_norm_weight(layer_idx),
            config.rms_norm_eps,
        )
        .unwrap();
        let mlp = Mlp::from_weights(weights, layer_idx).unwrap();
        let mut cache = ContiguousKvCache::new();
        let _ = cache.update(history_k, history_v).unwrap();

        let normed = input_norm.forward(hidden);
        let attention_output = attention
            .forward(&normed, position_ids, Some(&mut cache), None)
            .unwrap();
        let post_attention = &attention_output + hidden;
        let mlp_output = mlp.forward(&post_norm.forward(&post_attention));
        (attention_output, post_attention + mlp_output)
    }

    #[test]
    fn frozen_kv_assignee_schedule_is_capacity_weighted_and_phase_stable() {
        let first = FrozenKvAssigneeSchedule::new(&[1, 3, 2], 41, 24).unwrap();
        assert_eq!(first.counts(), &[4, 12, 8]);
        assert_eq!(first.total_units(), 24);

        let mut observed = vec![0_usize; 3];
        for token_offset in 0..8 {
            for layer_idx in 0..3 {
                let assignee = first.assignee_for(token_offset, layer_idx, 3).unwrap();
                observed[assignee] += 1;
            }
        }
        assert_eq!(observed, first.counts());

        let same_request = FrozenKvAssigneeSchedule::new(&[1, 3, 2], 41, 24).unwrap();
        assert_eq!(first, same_request);
        let other_request = FrozenKvAssigneeSchedule::new(&[1, 3, 2], 42, 24).unwrap();
        assert_ne!(first.phase(), other_request.phase());
        assert_eq!(first.counts(), other_request.counts());

        let mut prefix_counts = [0_i128; 3];
        for ordinal in 0..first.total_units() {
            let assignee = first.sequence[(first.phase() + ordinal) % first.total_units()];
            prefix_counts[assignee] += 1;
            let prefix_units = (ordinal + 1) as i128;
            for (domain, &target_count) in first.counts().iter().enumerate() {
                let scaled_error = (prefix_counts[domain] * first.total_units() as i128
                    - prefix_units * target_count as i128)
                    .abs();
                assert!(scaled_error <= first.total_units() as i128);
            }
        }

        assert!(FrozenKvAssigneeSchedule::new(&[], 41, 24).is_err());
        assert!(FrozenKvAssigneeSchedule::new(&[0, 0, 0], 41, 24).is_err());
        assert!(FrozenKvAssigneeSchedule::new(&[1, 3, 2], 41, 0).is_err());
        assert_eq!(first.assignee_for(0, 3, 3), None);
        assert_eq!(first.assignee_for(8, 0, 3), None);
    }

    #[test]
    fn frozen_kv_assignee_schedule_excludes_zero_capacity_for_arbitrary_n() {
        let with_zero = FrozenKvAssigneeSchedule::new(&[1, 0, 3], 7, 16).unwrap();
        assert_eq!(with_zero.counts(), &[4, 0, 12]);
        assert!(!with_zero.sequence.contains(&1));

        for (tickets, total_units, expected_counts) in [
            (vec![7], 5, vec![5]),
            (vec![1, 1], 6, vec![3, 3]),
            (vec![1, 1, 1, 1], 8, vec![2, 2, 2, 2]),
        ] {
            let schedule = FrozenKvAssigneeSchedule::new(&tickets, 3, total_units).unwrap();
            assert_eq!(schedule.counts(), expected_counts);
            assert!(schedule
                .sequence
                .iter()
                .all(|&assignee| assignee < tickets.len()));
        }
    }

    #[test]
    fn three_domain_uneven_ring_matches_reference_layer() {
        let device = Device::Cpu;
        let config = test_config();
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 3).unwrap();
        let mut layer = model.layers.remove(0);
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 10.0, device);
        let shard_lengths = [2_i64, 5, 3];
        let history_len = shard_lengths.iter().sum::<i64>();
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let mut shards = Vec::new();
        for &len in &shard_lengths {
            let shape = [
                1,
                config.num_kv_heads() as i64,
                len,
                config.head_dim() as i64,
            ];
            let domain = shards.len() as f64;
            shards.push((
                deterministic_tensor(&shape, 20.0 + domain, device),
                deterministic_tensor(&shape, 30.0 + domain, device),
            ));
        }

        let history_k = Tensor::cat(
            &shards
                .iter()
                .map(|(k, _)| k.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let history_v = Tensor::cat(
            &shards
                .iter()
                .map(|(_, v)| v.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let result =
            run_single_layer_ring(&mut layer, &hidden, &position_ids, &mut shards, 1, 2).unwrap();
        assert_eq!(shards[0].0.size()[2], shard_lengths[0]);
        assert_eq!(shards[1].0.size()[2], shard_lengths[1]);
        assert_eq!(shards[2].0.size()[2], shard_lengths[2] + 1);
        let (reference_attention, reference) = reference_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &history_k,
            &history_v,
        );

        let mut single_model =
            LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut single_layer = single_model.layers.remove(0);
        let mut single_shard = vec![(history_k.shallow_clone(), history_v.shallow_clone())];
        let single = run_single_layer_ring(
            &mut single_layer,
            &hidden,
            &position_ids,
            &mut single_shard,
            0,
            0,
        )
        .unwrap();

        let max_diff = (&result.hidden_states - &reference)
            .abs()
            .max()
            .double_value(&[]);
        let single_reference_diff = (&single.hidden_states - &reference)
            .abs()
            .max()
            .double_value(&[]);
        let split_single_diff = (&result.hidden_states - &single.hidden_states)
            .abs()
            .max()
            .double_value(&[]);
        let attention_max_diff = (&result.attention_output - &reference_attention)
            .abs()
            .max()
            .double_value(&[]);
        let attention_mean_diff = (&result.attention_output - &reference_attention)
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        eprintln!(
            "attention-max={attention_max_diff}, attention-mean={attention_mean_diff}, split-reference={max_diff}, single-reference={single_reference_diff}, split-single={split_single_diff}"
        );
        assert!(
            attention_max_diff < 1e-4,
            "max attention output diff: {attention_max_diff}"
        );
        assert!(
            split_single_diff < 1e-4,
            "split ring differs from single-block online softmax: {split_single_diff}"
        );
        // Synthetic weights are intentionally unscaled; O projection and MLP
        // amplify the much smaller attention difference measured above.
        assert!(max_diff < 2e-4, "max layer output diff: {max_diff}");
        assert_eq!(result.stats.domains, 3);
        assert_eq!(result.stats.hops, 2);
        assert_eq!(result.stats.local_partials, 3);
        assert_eq!(result.stats.q_projections, 1);
        assert_eq!(result.stats.q_projection_domain, 1);
        assert_eq!(result.stats.current_kv_projections, 1);
        assert_eq!(result.stats.current_kv_projection_domain, 2);
        assert_eq!(result.stats.current_kv_commits, 1);
        assert_eq!(result.stats.layer_finishes, 1);
        assert_eq!(result.stats.starter, 1);
        assert_eq!(result.stats.assignee, 2);
        assert_eq!(result.stats.finisher, 0);
    }

    #[test]
    fn arbitrary_domain_count_follows_successor_ring() {
        for domains in [1_usize, 2, 4] {
            let device = Device::Cpu;
            let config = test_config();
            let weights = deterministic_weights(&config, device);
            let mut model =
                LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap();
            let mut layer = model.layers.remove(0);
            let hidden = deterministic_tensor(
                &[1, 1, config.hidden_size as i64],
                40.0 + domains as f64,
                device,
            );
            let shard_lengths = (0..domains)
                .map(|domain| domain as i64 + 1)
                .collect::<Vec<_>>();
            let history_len = shard_lengths.iter().sum::<i64>();
            let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
            let mut shards = shard_lengths
                .iter()
                .enumerate()
                .map(|(domain, &len)| {
                    let shape = [
                        1,
                        config.num_kv_heads() as i64,
                        len,
                        config.head_dim() as i64,
                    ];
                    (
                        deterministic_tensor(&shape, 50.0 + domain as f64, device),
                        deterministic_tensor(&shape, 60.0 + domain as f64, device),
                    )
                })
                .collect::<Vec<_>>();
            let starter = domains - 1;
            let assignee = (starter + 1) % domains;
            let history_k = Tensor::cat(
                &shards
                    .iter()
                    .map(|(k, _)| k.shallow_clone())
                    .collect::<Vec<_>>(),
                2,
            );
            let history_v = Tensor::cat(
                &shards
                    .iter()
                    .map(|(_, v)| v.shallow_clone())
                    .collect::<Vec<_>>(),
                2,
            );

            let result = run_single_layer_ring(
                &mut layer,
                &hidden,
                &position_ids,
                &mut shards,
                starter,
                assignee,
            )
            .unwrap();
            let expected_route = (0..domains)
                .map(|step| (starter + step) % domains)
                .collect::<Vec<_>>();

            assert_eq!(result.stats.visited_domains, expected_route);
            assert_eq!(result.stats.domains, domains);
            assert_eq!(result.stats.hops, domains - 1);
            assert_eq!(result.stats.local_partials, domains);
            assert_eq!(result.stats.q_projections, 1);
            assert_eq!(result.stats.q_projection_domain, starter);
            assert_eq!(result.stats.current_kv_projections, 1);
            assert_eq!(result.stats.current_kv_projection_domain, assignee);
            assert_eq!(result.stats.current_kv_commits, 1);
            assert_eq!(result.stats.layer_finishes, 1);
            assert_eq!(result.stats.finisher, (starter + domains - 1) % domains);

            let (reference_attention, reference) = reference_layer_output(
                0,
                &config,
                &weights,
                &hidden,
                &position_ids,
                &history_k,
                &history_v,
            );
            let attention_diff = (&result.attention_output - reference_attention)
                .abs()
                .max()
                .double_value(&[]);
            let layer_diff = (&result.hidden_states - reference)
                .abs()
                .max()
                .double_value(&[]);
            assert!(
                attention_diff < 1e-4,
                "N={domains} attention diff: {attention_diff}"
            );
            assert!(layer_diff < 2e-4, "N={domains} layer diff: {layer_diff}");
        }
    }

    #[test]
    fn layer_packet_is_sufficient_for_each_domain_step() {
        let device = Device::Cpu;
        let config = test_config();
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let mut layer = model.layers.remove(0);
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 70.0, device);
        let shard_lengths = [3_i64, 5];
        let history_len = shard_lengths.iter().sum::<i64>();
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let mut shards = shard_lengths
            .iter()
            .enumerate()
            .map(|(domain, &len)| {
                let shape = [
                    1,
                    config.num_kv_heads() as i64,
                    len,
                    config.head_dim() as i64,
                ];
                (
                    deterministic_tensor(&shape, 80.0 + domain as f64, device),
                    deterministic_tensor(&shape, 90.0 + domain as f64, device),
                )
            })
            .collect::<Vec<_>>();
        let history_k = Tensor::cat(
            &shards
                .iter()
                .map(|(k, _)| k.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let history_v = Tensor::cat(
            &shards
                .iter()
                .map(|(_, v)| v.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let (_, reference) = reference_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &history_k,
            &history_v,
        );

        let packet = LayerPacket::start(&mut layer, &hidden, &position_ids, 0, 1, 2).unwrap();
        let packet = match process_layer_packet(&mut layer, packet, &mut shards[0]).unwrap() {
            LayerStepOutcome::Forward(packet) => packet,
            LayerStepOutcome::Finished { .. } => panic!("N=2 packet finished before successor"),
        };
        let hidden_states = match process_layer_packet(&mut layer, packet, &mut shards[1]).unwrap()
        {
            LayerStepOutcome::Finished { hidden_states, .. } => hidden_states,
            LayerStepOutcome::Forward(_) => panic!("N=2 packet did not finish at successor"),
        };

        let max_diff = (&hidden_states - reference).abs().max().double_value(&[]);
        assert!(max_diff < 2e-4, "explicit packet layer diff: {max_diff}");
        assert_eq!(shards[0].0.size()[2], shard_lengths[0]);
        assert_eq!(shards[1].0.size()[2], shard_lengths[1] + 1);
    }

    #[test]
    fn layer_packet_payload_does_not_grow_with_history_context() {
        fn payload_after_first_hop(history_len: i64) -> usize {
            let device = Device::Cpu;
            let config = test_config();
            let weights = deterministic_weights(&config, device);
            let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
            let mut layer = model.layers.remove(0);
            let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 100.0, device);
            let position_ids = Tensor::from_slice(&[history_len * 2]).unsqueeze(0);
            let shape = [
                1,
                config.num_kv_heads() as i64,
                history_len,
                config.head_dim() as i64,
            ];
            let mut local_history = (
                deterministic_tensor(&shape, 110.0, device),
                deterministic_tensor(&shape, 120.0, device),
            );
            let packet = LayerPacket::start(&mut layer, &hidden, &position_ids, 0, 1, 2).unwrap();
            match process_layer_packet(&mut layer, packet, &mut local_history).unwrap() {
                LayerStepOutcome::Forward(packet) => packet.tensor_payload_elements(),
                LayerStepOutcome::Finished { .. } => panic!("N=2 packet finished before successor"),
            }
        }

        let short_context_payload = payload_after_first_hop(2);
        let long_context_payload = payload_after_first_hop(47);
        assert_eq!(short_context_payload, long_context_payload);
        assert!(short_context_payload > 0);
    }

    fn assert_tcp_ring_case(domains: usize, starter: usize, assignee: usize) -> Vec<usize> {
        let device = Device::Cpu;
        let config = test_config();
        let weights = deterministic_weights(&config, device);
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 125.0, device);
        let shard_lengths = [2_i64, 5, 3, 4][..domains].to_vec();
        let history_len = shard_lengths.iter().sum::<i64>();
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let shards = shard_lengths
            .iter()
            .enumerate()
            .map(|(domain, &len)| {
                let shape = [
                    1,
                    config.num_kv_heads() as i64,
                    len,
                    config.head_dim() as i64,
                ];
                (
                    deterministic_tensor(&shape, 126.0 + domain as f64, device),
                    deterministic_tensor(&shape, 136.0 + domain as f64, device),
                )
            })
            .collect::<Vec<_>>();
        let history_k = Tensor::cat(
            &shards
                .iter()
                .map(|(k, _)| k.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let history_v = Tensor::cat(
            &shards
                .iter()
                .map(|(_, v)| v.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let (reference_attention, reference_hidden) = reference_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &history_k,
            &history_v,
        );

        let layers = (0..domains)
            .map(|_| {
                let mut model =
                    LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap();
                model.layers.remove(0)
            })
            .collect::<Vec<_>>();
        let listeners = (0..domains)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect::<Vec<_>>();
        let addresses = listeners
            .iter()
            .map(|listener| listener.local_addr().unwrap())
            .collect::<Vec<_>>();
        let outgoing = (0..domains)
            .map(|domain| TcpStream::connect(addresses[(domain + 1) % domains]).unwrap())
            .collect::<Vec<_>>();
        let incoming = listeners
            .into_iter()
            .map(|listener| listener.accept().unwrap().0)
            .collect::<Vec<_>>();

        let workers = layers
            .into_iter()
            .zip(shards)
            .zip(incoming)
            .zip(outgoing)
            .enumerate()
            .map(|(domain, (((mut layer, mut shard), incoming), outgoing))| {
                let worker_hidden = hidden.shallow_clone();
                let worker_position_ids = position_ids.shallow_clone();
                thread::spawn(move || {
                    let mut predecessor = TcpKvTransport::new(incoming, device).unwrap();
                    let mut successor = TcpKvTransport::new(outgoing, device).unwrap();
                    let started = domain == starter;
                    let packet = if started {
                        LayerPacket::start(
                            &mut layer,
                            &worker_hidden,
                            &worker_position_ids,
                            starter,
                            assignee,
                            domains,
                        )
                        .unwrap()
                    } else {
                        let wire = predecessor
                            .recv_self_driving_packet()
                            .unwrap()
                            .expect("predecessor closed before sending a packet");
                        LayerPacket::from_self_driving_packet(wire).unwrap()
                    };
                    assert_eq!(packet.current_domain, domain);
                    let visit_index = packet.visited_domains;
                    let before = shard.0.size()[2];
                    match process_layer_packet(&mut layer, packet, &mut shard).unwrap() {
                        LayerStepOutcome::Forward(packet) => {
                            let wire = packet.into_self_driving_packet(0).unwrap();
                            let sent_bytes = successor.send_self_driving_packet(&wire).unwrap();
                            (
                                visit_index,
                                domain,
                                started,
                                before,
                                shard.0.size()[2],
                                sent_bytes,
                                None,
                            )
                        }
                        LayerStepOutcome::Finished {
                            attention_output,
                            hidden_states,
                        } => (
                            visit_index,
                            domain,
                            started,
                            before,
                            shard.0.size()[2],
                            0,
                            Some((attention_output, hidden_states)),
                        ),
                    }
                })
            })
            .collect::<Vec<_>>();

        let mut started = 0_usize;
        let mut sends = 0_usize;
        let mut finished = 0_usize;
        let mut finisher_domain = None;
        let mut route = Vec::with_capacity(domains);
        let mut actual_attention = None;
        let mut actual_hidden = None;
        for worker in workers {
            let (visit_index, domain, worker_started, before, after, sent_bytes, output) =
                worker.join().unwrap();
            route.push((visit_index, domain));
            started += usize::from(worker_started);
            sends += usize::from(sent_bytes > 0);
            let expected_growth = i64::from(domain == assignee);
            assert_eq!(after, before + expected_growth);
            if let Some((attention, hidden)) = output {
                finished += 1;
                finisher_domain = Some(domain);
                actual_attention = Some(attention);
                actual_hidden = Some(hidden);
            }
        }

        assert_eq!(started, 1);
        assert_eq!(sends, domains - 1);
        assert_eq!(finished, 1);
        assert_eq!(finisher_domain, Some((starter + domains - 1) % domains));
        route.sort_by_key(|(visit_index, _)| *visit_index);
        let route = route
            .into_iter()
            .map(|(_, domain)| domain)
            .collect::<Vec<_>>();
        assert_eq!(
            route,
            (0..domains)
                .map(|step| (starter + step) % domains)
                .collect::<Vec<_>>()
        );
        let attention_diff = (actual_attention.unwrap() - reference_attention)
            .abs()
            .max()
            .double_value(&[]);
        let hidden_diff = (actual_hidden.unwrap() - reference_hidden)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            attention_diff < 1e-4,
            "TCP ring attention diff: {attention_diff}"
        );
        assert!(hidden_diff < 2e-4, "TCP ring layer diff: {hidden_diff}");
        route
    }

    #[test]
    fn three_domain_tcp_ring_runs_one_real_layer_in_two_hops() {
        assert_eq!(assert_tcp_ring_case(3, 0, 1), vec![0, 1, 2]);
    }

    #[test]
    fn tcp_ring_handles_arbitrary_domain_counts_and_wraparound() {
        for domains in [2_usize, 3, 4] {
            let starter = domains - 1;
            let assignee = (starter + 1) % domains;
            let route = assert_tcp_ring_case(domains, starter, assignee);
            assert_eq!(route[0], starter);
            assert_eq!(
                route[1], 0,
                "N={domains} did not cross the wrap-around edge"
            );
        }
    }

    #[derive(Debug)]
    struct TcpLayerEvent {
        layer_idx: usize,
        visit_index: usize,
        domain: usize,
        started: bool,
        finished: bool,
        sent_bytes: usize,
        kv_before: i64,
        kv_after: i64,
    }

    #[test]
    fn two_layer_tcp_ring_uses_scheduled_assignees_and_produces_final_logits() {
        let domains = 3_usize;
        let layers_count = 2_usize;
        let initial_starter = 1_usize;
        let schedule = FrozenKvAssigneeSchedule::new(&[1, 3, 2], 1, layers_count).unwrap();
        let assignees: [usize; 2] = std::array::from_fn(|layer_idx| {
            schedule.assignee_for(0, layer_idx, layers_count).unwrap()
        });
        assert_eq!(schedule.counts(), &[0, 1, 1]);
        assert_eq!(assignees, [2, 1]);
        let device = Device::Cpu;
        let mut config = test_config();
        config.num_layers = layers_count;
        let weights = deterministic_weights(&config, device);
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 129.0, device);
        let shard_lengths = [2_i64, 4, 3];
        let history_len = shard_lengths.iter().sum::<i64>();
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let layer_shards = (0..layers_count)
            .map(|layer_idx| {
                shard_lengths
                    .iter()
                    .enumerate()
                    .map(|(domain, &len)| {
                        let shape = [
                            1,
                            config.num_kv_heads() as i64,
                            len,
                            config.head_dim() as i64,
                        ];
                        (
                            deterministic_tensor(
                                &shape,
                                130.0 + layer_idx as f64 * 20.0 + domain as f64,
                                device,
                            ),
                            deterministic_tensor(
                                &shape,
                                140.0 + layer_idx as f64 * 20.0 + domain as f64,
                                device,
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let reference_histories = layer_shards
            .iter()
            .map(|shards| {
                (
                    Tensor::cat(
                        &shards
                            .iter()
                            .map(|(k, _)| k.shallow_clone())
                            .collect::<Vec<_>>(),
                        2,
                    ),
                    Tensor::cat(
                        &shards
                            .iter()
                            .map(|(_, v)| v.shallow_clone())
                            .collect::<Vec<_>>(),
                        2,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let (_, reference_layer_0) = reference_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &reference_histories[0].0,
            &reference_histories[0].1,
        );
        let (_, reference_layer_1) = reference_layer_output(
            1,
            &config,
            &weights,
            &reference_layer_0,
            &position_ids,
            &reference_histories[1].0,
            &reference_histories[1].1,
        );
        let reference_final_norm =
            RmsNorm::from_weights(&weights, WeightNames::layer_norm(), config.rms_norm_eps)
                .unwrap();
        let reference_logits = reference_final_norm
            .forward(&reference_layer_1)
            .matmul(&weights.get(WeightNames::lm_head()).unwrap().transpose(0, 1));

        let worker_models = (0..domains)
            .map(|_| LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap())
            .collect::<Vec<_>>();
        let mut worker_shards = (0..domains)
            .map(|_| Vec::with_capacity(layers_count))
            .collect::<Vec<_>>();
        for shards in layer_shards {
            for (domain, shard) in shards.into_iter().enumerate() {
                worker_shards[domain].push(shard);
            }
        }

        let listeners = (0..domains)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect::<Vec<_>>();
        let addresses = listeners
            .iter()
            .map(|listener| listener.local_addr().unwrap())
            .collect::<Vec<_>>();
        let outgoing = (0..domains)
            .map(|domain| TcpStream::connect(addresses[(domain + 1) % domains]).unwrap())
            .collect::<Vec<_>>();
        let incoming = listeners
            .into_iter()
            .map(|listener| listener.accept().unwrap().0)
            .collect::<Vec<_>>();

        let workers = worker_models
            .into_iter()
            .zip(worker_shards)
            .zip(incoming)
            .zip(outgoing)
            .enumerate()
            .map(
                |(domain, (((mut model, mut shards), incoming), outgoing))| {
                    let initial_hidden =
                        (domain == initial_starter).then(|| hidden.shallow_clone());
                    let worker_position_ids = position_ids.shallow_clone();
                    thread::spawn(move || {
                        let mut predecessor = TcpKvTransport::new(incoming, device).unwrap();
                        let mut successor = TcpKvTransport::new(outgoing, device).unwrap();
                        let mut next_layer_hidden = initial_hidden;
                        let mut events = Vec::with_capacity(layers_count);
                        let mut final_hidden = None;
                        let mut final_logits = None;

                        for layer_idx in 0..layers_count {
                            let started = next_layer_hidden.is_some();
                            let packet = if let Some(layer_hidden) = next_layer_hidden.take() {
                                LayerPacket::start(
                                    &mut model.layers[layer_idx],
                                    &layer_hidden,
                                    &worker_position_ids,
                                    domain,
                                    assignees[layer_idx],
                                    domains,
                                )
                                .unwrap()
                            } else {
                                let wire = predecessor
                                    .recv_self_driving_packet()
                                    .unwrap()
                                    .expect("predecessor closed before the next layer packet");
                                assert_eq!(wire.layer_idx, layer_idx);
                                LayerPacket::from_self_driving_packet(wire).unwrap()
                            };
                            assert_eq!(packet.current_domain, domain);
                            let visit_index = packet.visited_domains;
                            let kv_before = shards[layer_idx].0.size()[2];
                            let (finished, sent_bytes) = match process_layer_packet(
                                &mut model.layers[layer_idx],
                                packet,
                                &mut shards[layer_idx],
                            )
                            .unwrap()
                            {
                                LayerStepOutcome::Forward(packet) => {
                                    let wire = packet.into_self_driving_packet(layer_idx).unwrap();
                                    let sent_bytes =
                                        successor.send_self_driving_packet(&wire).unwrap();
                                    (false, sent_bytes)
                                }
                                LayerStepOutcome::Finished { hidden_states, .. } => {
                                    if layer_idx + 1 == layers_count {
                                        final_logits =
                                            Some(project_final_logits(&model, &hidden_states));
                                        final_hidden = Some(hidden_states);
                                    } else {
                                        next_layer_hidden = Some(hidden_states);
                                    }
                                    (true, 0)
                                }
                            };
                            events.push(TcpLayerEvent {
                                layer_idx,
                                visit_index,
                                domain,
                                started,
                                finished,
                                sent_bytes,
                                kv_before,
                                kv_after: shards[layer_idx].0.size()[2],
                            });
                        }

                        (events, final_hidden, final_logits)
                    })
                },
            )
            .collect::<Vec<_>>();

        let mut events = Vec::with_capacity(domains * layers_count);
        let mut final_outputs = Vec::new();
        let mut logits_outputs = Vec::new();
        for (domain, worker) in workers.into_iter().enumerate() {
            let (worker_events, final_hidden, final_logits) = worker.join().unwrap();
            assert_eq!(worker_events.len(), layers_count);
            events.extend(worker_events);
            if let Some(hidden_states) = final_hidden {
                final_outputs.push((domain, hidden_states));
            }
            if let Some(logits) = final_logits {
                logits_outputs.push((domain, logits));
            }
        }

        assert_eq!(events.len(), domains * layers_count);
        assert_eq!(
            events.iter().filter(|event| event.sent_bytes > 0).count(),
            layers_count * (domains - 1)
        );
        let expected_routes = [vec![1_usize, 2, 0], vec![0_usize, 1, 2]];
        for layer_idx in 0..layers_count {
            let mut layer_events = events
                .iter()
                .filter(|event| event.layer_idx == layer_idx)
                .collect::<Vec<_>>();
            layer_events.sort_by_key(|event| event.visit_index);
            assert_eq!(
                layer_events
                    .iter()
                    .map(|event| event.domain)
                    .collect::<Vec<_>>(),
                expected_routes[layer_idx]
            );
            assert_eq!(
                layer_events
                    .iter()
                    .filter(|event| event.started)
                    .map(|event| event.domain)
                    .collect::<Vec<_>>(),
                vec![expected_routes[layer_idx][0]]
            );
            assert_eq!(
                layer_events
                    .iter()
                    .filter(|event| event.finished)
                    .map(|event| event.domain)
                    .collect::<Vec<_>>(),
                vec![expected_routes[layer_idx][domains - 1]]
            );
            for event in layer_events {
                assert_eq!(
                    event.kv_after,
                    event.kv_before + i64::from(event.domain == assignees[layer_idx])
                );
            }
        }
        assert_eq!(expected_routes[1][0], expected_routes[0][domains - 1]);
        assert_eq!(final_outputs.len(), 1);
        let (final_domain, actual_hidden) = final_outputs.pop().unwrap();
        assert_eq!(final_domain, 2);
        let hidden_diff = (actual_hidden - reference_layer_1)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            hidden_diff < 4e-4,
            "two-layer TCP hidden diff: {hidden_diff}"
        );
        assert_eq!(logits_outputs.len(), 1);
        let (logits_domain, actual_logits) = logits_outputs.pop().unwrap();
        assert_eq!(logits_domain, final_domain);
        let logits_diff = (actual_logits - reference_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            logits_diff < 4e-4,
            "two-layer TCP final logits diff: {logits_diff}"
        );
    }

    #[test]
    fn self_driving_tcp_wire_bytes_do_not_grow_with_history_context() {
        fn first_hop_wire_bytes(history_len: i64) -> usize {
            let device = Device::Cpu;
            let config = test_config();
            let weights = deterministic_weights(&config, device);
            let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 3).unwrap();
            let mut layer = model.layers.remove(0);
            let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 127.0, device);
            let position_ids = Tensor::from_slice(&[history_len * 2]).unsqueeze(0);
            let shape = [
                1,
                config.num_kv_heads() as i64,
                history_len,
                config.head_dim() as i64,
            ];
            let mut local_history = (
                deterministic_tensor(&shape, 128.0, device),
                deterministic_tensor(&shape, 138.0, device),
            );
            let packet = LayerPacket::start(&mut layer, &hidden, &position_ids, 0, 1, 3).unwrap();
            let packet = match process_layer_packet(&mut layer, packet, &mut local_history).unwrap()
            {
                LayerStepOutcome::Forward(packet) => packet.into_self_driving_packet(0).unwrap(),
                LayerStepOutcome::Finished { .. } => panic!("N=3 starter unexpectedly finished"),
            };

            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let receiver = thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                TcpKvTransport::new(stream, device)
                    .unwrap()
                    .recv_self_driving_packet()
                    .unwrap()
                    .unwrap()
            });
            let stream = TcpStream::connect(address).unwrap();
            let sent_bytes = TcpKvTransport::new(stream, device)
                .unwrap()
                .send_self_driving_packet(&packet)
                .unwrap();
            let _ = receiver.join().unwrap();
            sent_bytes
        }

        let short_context_bytes = first_hop_wire_bytes(2);
        let long_context_bytes = first_hop_wire_bytes(47);
        assert_eq!(short_context_bytes, long_context_bytes);
        assert!(short_context_bytes > 0);
    }

    #[test]
    fn two_layer_handoff_continues_from_each_finisher() {
        let device = Device::Cpu;
        let mut config = test_config();
        config.num_layers = 2;
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 3).unwrap();
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 130.0, device);
        let shard_lengths = [2_i64, 4, 3];
        let history_len = shard_lengths.iter().sum::<i64>();
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let mut layer_shards = (0..2)
            .map(|layer_idx| {
                shard_lengths
                    .iter()
                    .enumerate()
                    .map(|(domain, &len)| {
                        let shape = [
                            1,
                            config.num_kv_heads() as i64,
                            len,
                            config.head_dim() as i64,
                        ];
                        (
                            deterministic_tensor(
                                &shape,
                                140.0 + layer_idx as f64 * 20.0 + domain as f64,
                                device,
                            ),
                            deterministic_tensor(
                                &shape,
                                150.0 + layer_idx as f64 * 20.0 + domain as f64,
                                device,
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let reference_histories = layer_shards
            .iter()
            .map(|shards| {
                (
                    Tensor::cat(
                        &shards
                            .iter()
                            .map(|(k, _)| k.shallow_clone())
                            .collect::<Vec<_>>(),
                        2,
                    ),
                    Tensor::cat(
                        &shards
                            .iter()
                            .map(|(_, v)| v.shallow_clone())
                            .collect::<Vec<_>>(),
                        2,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let (_, reference_layer_0) = reference_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &reference_histories[0].0,
            &reference_histories[0].1,
        );
        let (_, reference_layer_1) = reference_layer_output(
            1,
            &config,
            &weights,
            &reference_layer_0,
            &position_ids,
            &reference_histories[1].0,
            &reference_histories[1].1,
        );
        let reference_final_norm =
            RmsNorm::from_weights(&weights, WeightNames::layer_norm(), config.rms_norm_eps)
                .unwrap();
        let reference_logits = reference_final_norm
            .forward(&reference_layer_1)
            .matmul(&weights.get(WeightNames::lm_head()).unwrap().transpose(0, 1));

        let result = run_two_layer_ring_with_logits(
            &mut model,
            &hidden,
            &position_ids,
            &mut layer_shards,
            1,
            [2, 1],
        )
        .unwrap();

        let max_diff = (&result.hidden_states - reference_layer_1)
            .abs()
            .max()
            .double_value(&[]);
        assert!(max_diff < 4e-4, "two-layer output diff: {max_diff}");
        let logits_diff = (&result.logits - reference_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(logits_diff < 4e-4, "final logits diff: {logits_diff}");
        assert_eq!(result.logits_producer_domain, 2);
        assert_eq!(result.logits_projections, 1);
        assert_eq!(result.layer_stats[0].starter, 1);
        assert_eq!(result.layer_stats[0].finisher, 0);
        assert_eq!(result.layer_stats[1].starter, 0);
        assert_eq!(result.layer_stats[1].finisher, 2);
        assert_eq!(
            result
                .layer_stats
                .iter()
                .map(|stats| stats.hops)
                .sum::<usize>(),
            4
        );
        for (layer_idx, &assignee) in [2_usize, 1].iter().enumerate() {
            assert_eq!(result.layer_stats[layer_idx].q_projections, 1);
            assert_eq!(result.layer_stats[layer_idx].local_partials, 3);
            assert_eq!(result.layer_stats[layer_idx].current_kv_projections, 1);
            assert_eq!(result.layer_stats[layer_idx].current_kv_commits, 1);
            assert_eq!(result.layer_stats[layer_idx].layer_finishes, 1);
            for domain in 0..3 {
                let expected = shard_lengths[domain] + i64::from(domain == assignee);
                assert_eq!(layer_shards[layer_idx][domain].0.size()[2], expected);
            }
        }
    }

    fn assert_full_model_ring_case(num_layers: usize, starter: usize) {
        let domains = 3_usize;
        let device = Device::Cpu;
        let mut config = test_config();
        config.num_layers = num_layers;
        let weights = deterministic_weights(&config, device);
        let mut model =
            LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap();
        let hidden = deterministic_tensor(
            &[1, 1, config.hidden_size as i64],
            210.0 + num_layers as f64,
            device,
        );
        let shard_lengths = [2_i64, 4, 3];
        let history_len = shard_lengths.iter().sum::<i64>();
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let mut layer_shards = (0..config.num_layers)
            .map(|layer_idx| {
                shard_lengths
                    .iter()
                    .enumerate()
                    .map(|(domain, &len)| {
                        let shape = [
                            1,
                            config.num_kv_heads() as i64,
                            len,
                            config.head_dim() as i64,
                        ];
                        (
                            deterministic_tensor(
                                &shape,
                                220.0 + layer_idx as f64 * 20.0 + domain as f64,
                                device,
                            ),
                            deterministic_tensor(
                                &shape,
                                230.0 + layer_idx as f64 * 20.0 + domain as f64,
                                device,
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let reference_histories = layer_shards
            .iter()
            .map(|shards| {
                (
                    Tensor::cat(
                        &shards
                            .iter()
                            .map(|(k, _)| k.shallow_clone())
                            .collect::<Vec<_>>(),
                        2,
                    ),
                    Tensor::cat(
                        &shards
                            .iter()
                            .map(|(_, v)| v.shallow_clone())
                            .collect::<Vec<_>>(),
                        2,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let mut reference_hidden = hidden.shallow_clone();
        for (layer_idx, (history_k, history_v)) in reference_histories.iter().enumerate() {
            let (_, next_hidden) = reference_layer_output(
                layer_idx,
                &config,
                &weights,
                &reference_hidden,
                &position_ids,
                history_k,
                history_v,
            );
            reference_hidden = next_hidden;
        }
        let reference_final_norm =
            RmsNorm::from_weights(&weights, WeightNames::layer_norm(), config.rms_norm_eps)
                .unwrap();
        let reference_logits = reference_final_norm
            .forward(&reference_hidden)
            .matmul(&weights.get(WeightNames::lm_head()).unwrap().transpose(0, 1));
        let assignees = (0..num_layers)
            .map(|layer_idx| (2 + domains - layer_idx % domains) % domains)
            .collect::<Vec<_>>();

        let result = run_model_ring(
            &mut model,
            &hidden,
            &position_ids,
            &mut layer_shards,
            starter,
            &assignees,
        )
        .unwrap();

        let logits_diff = (&result.logits - reference_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            logits_diff < 1e-3,
            "L={num_layers} full-model logits diff: {logits_diff}"
        );
        let expected_producer = (starter + domains - num_layers % domains) % domains;
        assert_eq!(result.logits_producer_domain, expected_producer);
        assert_eq!(result.logits_projections, 1);
        assert_eq!(result.layer_stats.len(), num_layers);
        assert_eq!(
            result
                .layer_stats
                .iter()
                .map(|stats| stats.hops)
                .sum::<usize>(),
            num_layers * (domains - 1)
        );
        let expected_starters = (0..num_layers)
            .scan(starter, |current, _| {
                let layer_starter = *current;
                *current = (layer_starter + domains - 1) % domains;
                Some(layer_starter)
            })
            .collect::<Vec<_>>();
        let expected_finishers = expected_starters
            .iter()
            .map(|layer_starter| (layer_starter + domains - 1) % domains)
            .collect::<Vec<_>>();
        assert_eq!(
            result
                .layer_stats
                .iter()
                .map(|stats| stats.starter)
                .collect::<Vec<_>>(),
            expected_starters
        );
        assert_eq!(
            result
                .layer_stats
                .iter()
                .map(|stats| stats.finisher)
                .collect::<Vec<_>>(),
            expected_finishers
        );
        for (layer_idx, &assignee) in assignees.iter().enumerate() {
            let stats = &result.layer_stats[layer_idx];
            assert_eq!(stats.q_projections, 1);
            assert_eq!(stats.local_partials, 3);
            assert_eq!(stats.current_kv_projections, 1);
            assert_eq!(stats.current_kv_commits, 1);
            assert_eq!(stats.layer_finishes, 1);
            for domain in 0..3 {
                let expected = shard_lengths[domain] + i64::from(domain == assignee);
                assert_eq!(layer_shards[layer_idx][domain].0.size()[2], expected);
            }
        }
    }

    #[test]
    fn full_model_ring_returns_to_starter_when_layers_divide_domains() {
        assert_full_model_ring_case(3, 1);
    }

    #[test]
    fn full_model_ring_rotates_producer_when_layers_do_not_divide_domains() {
        assert_full_model_ring_case(4, 1);
    }

    #[test]
    fn two_layer_final_logits_support_tied_embeddings() {
        let device = Device::Cpu;
        let mut config = test_config();
        config.num_layers = 2;
        config.tie_word_embeddings = true;
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        assert!(model.lm_head.is_none());
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 180.0, device);
        let history_len = 3_i64;
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let mut layer_shards = (0..2)
            .map(|layer_idx| {
                let shape = [
                    1,
                    config.num_kv_heads() as i64,
                    history_len,
                    config.head_dim() as i64,
                ];
                vec![(
                    deterministic_tensor(&shape, 190.0 + layer_idx as f64 * 10.0, device),
                    deterministic_tensor(&shape, 200.0 + layer_idx as f64 * 10.0, device),
                )]
            })
            .collect::<Vec<_>>();
        let reference_histories = layer_shards
            .iter()
            .map(|shards| (shards[0].0.shallow_clone(), shards[0].1.shallow_clone()))
            .collect::<Vec<_>>();
        let (_, reference_layer_0) = reference_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &reference_histories[0].0,
            &reference_histories[0].1,
        );
        let (_, reference_layer_1) = reference_layer_output(
            1,
            &config,
            &weights,
            &reference_layer_0,
            &position_ids,
            &reference_histories[1].0,
            &reference_histories[1].1,
        );
        let reference_final_norm =
            RmsNorm::from_weights(&weights, WeightNames::layer_norm(), config.rms_norm_eps)
                .unwrap();
        let reference_logits = reference_final_norm.forward(&reference_layer_1).matmul(
            &weights
                .get(WeightNames::embedding())
                .unwrap()
                .transpose(0, 1),
        );

        let result = run_two_layer_ring_with_logits(
            &mut model,
            &hidden,
            &position_ids,
            &mut layer_shards,
            0,
            [0, 0],
        )
        .unwrap();

        let logits_diff = (&result.logits - reference_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(logits_diff < 4e-4, "tied final logits diff: {logits_diff}");
        assert_eq!(result.logits_producer_domain, 0);
        assert_eq!(result.logits_projections, 1);
        assert_eq!(
            result
                .layer_stats
                .iter()
                .map(|stats| stats.hops)
                .sum::<usize>(),
            0
        );
    }
}
