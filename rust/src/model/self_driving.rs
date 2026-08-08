//! Experimental self-driving inference ring.

use crate::model::attention::HcpRingAttentionBackend;
use crate::model::cache::KvCache;
use crate::model::layers::DecoderLayer;
use crate::model::model::LlamaModel;
use crate::model::transport::SelfDrivingPacket;
use crate::model::{ModelConfig, ModelError};
use tch::{Device, Kind, Tensor};

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

/// Experimental finite-horizon KV shard with stable, position-indexed storage.
#[derive(Debug)]
pub struct ReservedPositionedKvShard {
    k_storage: Tensor,
    v_storage: Tensor,
    positions: Vec<i64>,
    committed_len: usize,
    pending_positions: Option<Vec<i64>>,
}

impl ReservedPositionedKvShard {
    pub(crate) fn new(config: &ModelConfig, capacity: usize, device: Device) -> Self {
        Self::new_with_kind(config, capacity, device, Kind::Float)
    }

    pub(crate) fn new_with_kind(
        config: &ModelConfig,
        capacity: usize,
        device: Device,
        kind: Kind,
    ) -> Self {
        let shape = [
            1,
            config.num_kv_heads() as i64,
            capacity as i64,
            config.head_dim() as i64,
        ];
        Self {
            k_storage: Tensor::zeros(shape, (kind, device)),
            v_storage: Tensor::zeros(shape, (kind, device)),
            positions: Vec::with_capacity(capacity),
            committed_len: 0,
            pending_positions: None,
        }
    }

    pub(crate) fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        positions: &[i64],
    ) -> Result<(), String> {
        if k.size() != v.size() {
            return Err("positioned KV slab requires matching K/V shapes".to_string());
        }
        let shape = k.size();
        if shape.len() != 4
            || shape[0] != self.k_storage.size()[0]
            || shape[1] != self.k_storage.size()[1]
            || shape[3] != self.k_storage.size()[3]
            || shape[2] as usize != positions.len()
            || k.kind() != self.k_storage.kind()
            || v.kind() != self.v_storage.kind()
            || k.device() != self.k_storage.device()
            || v.device() != self.v_storage.device()
        {
            return Err("positioned KV slab append shape, dtype, or device mismatch".to_string());
        }
        let append_len = positions.len();
        let next_len = self
            .committed_len
            .checked_add(append_len)
            .ok_or_else(|| "positioned KV slab capacity overflow".to_string())?;
        if next_len > self.reserved_capacity() {
            return Err(format!(
                "positioned KV slab capacity exceeded: committed={}, append={}, capacity={}",
                self.committed_len,
                append_len,
                self.reserved_capacity()
            ));
        }

        let mut k_slot = self
            .k_storage
            .narrow(2, self.committed_len as i64, append_len as i64);
        let mut v_slot = self
            .v_storage
            .narrow(2, self.committed_len as i64, append_len as i64);
        k_slot.copy_(k);
        v_slot.copy_(v);
        self.positions.extend_from_slice(positions);
        self.committed_len = next_len;
        Ok(())
    }

    pub(crate) fn reserved_capacity(&self) -> usize {
        self.k_storage.size()[2] as usize
    }

    pub(crate) fn committed_len(&self) -> usize {
        self.committed_len
    }

    pub(crate) fn positions(&self) -> &[i64] {
        &self.positions
    }

    pub(crate) fn active_k(&self) -> Tensor {
        self.k_storage.narrow(2, 0, self.committed_len as i64)
    }

    pub(crate) fn active_v(&self) -> Tensor {
        self.v_storage.narrow(2, 0, self.committed_len as i64)
    }

    pub(crate) fn position_tensor(&self) -> Tensor {
        Tensor::from_slice(&self.positions).to_kind(Kind::Int64)
    }

    #[cfg(test)]
    fn storage_ptrs(&self) -> (usize, usize) {
        (
            self.k_storage.data_ptr() as usize,
            self.v_storage.data_ptr() as usize,
        )
    }
}

impl KvCache for ReservedPositionedKvShard {
    fn prepare_positions(&mut self, position_ids: &Tensor) -> Result<(), ModelError> {
        let shape = position_ids.size();
        if shape.len() != 2 || shape[0] != 1 {
            return Err(ModelError::Backend(
                "reserved positioned KV requires position_ids shaped [1, seq_len]".to_string(),
            ));
        }
        self.pending_positions = Some(
            (0..shape[1])
                .map(|offset| position_ids.int64_value(&[0, offset]))
                .collect(),
        );
        Ok(())
    }

    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError> {
        let positions = self.pending_positions.take().ok_or_else(|| {
            ModelError::Backend(
                "reserved positioned KV update requires prepared positions".to_string(),
            )
        })?;
        self.append(new_k, new_v, &positions)
            .map_err(ModelError::Backend)?;
        Ok((self.active_k(), self.active_v()))
    }

    fn committed_position_ids(&self) -> Option<Tensor> {
        Some(self.position_tensor())
    }

    fn update_sharded(
        &mut self,
        new_k: &Tensor,
        new_v: &Tensor,
        keep: bool,
    ) -> Result<(Tensor, Tensor), ModelError> {
        if keep {
            return self.update(new_k, new_v);
        }
        Err(ModelError::Backend(
            "reserved positioned KV decode must use the self-driving ring".to_string(),
        ))
    }

    fn seq_len(&self) -> usize {
        self.committed_len
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.committed_len = 0;
        self.pending_positions = None;
    }

    fn is_empty(&self) -> bool {
        self.committed_len == 0
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
        validate_route(hidden_states, position_ids, starter, assignee, domains)?;
        let residual = hidden_states.shallow_clone();
        let normalized = layer.input_layernorm.forward(hidden_states);
        let q = ring_backend(layer)?.project_packet_q(&normalized, position_ids)?;
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
    position_ids: &Tensor,
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
    let hidden_shape = hidden_states.size();
    if hidden_shape.len() != 3 || hidden_shape[0] != 1 || hidden_shape[1] < 1 {
        return Err(ModelError::Backend(
            "self-driving layer requires hidden_states shaped [1, seq_len>=1, hidden]".to_string(),
        ));
    }
    if position_ids.size() != [1, hidden_shape[1]] {
        return Err(ModelError::Backend(format!(
            "self-driving layer requires position_ids [1, {}], got {:?}",
            hidden_shape[1],
            position_ids.size()
        )));
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

pub(crate) fn project_final_logits(model: &LlamaModel, hidden_states: &Tensor) -> Tensor {
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
    packet: LayerPacket,
    local_history: &mut (Tensor, Tensor),
) -> Result<LayerStepOutcome, ModelError> {
    if packet.position_ids.size()[1] != 1 {
        return Err(ModelError::Backend(
            "multi-token self-driving packets require positioned reserved KV history".to_string(),
        ));
    }
    if local_history.0.size() != local_history.1.size() {
        return Err(ModelError::Backend(format!(
            "domain {} has mismatched K/V shapes",
            packet.current_domain
        )));
    }

    if packet.current_domain == packet.assignee {
        let ring = ring_backend(layer)?;
        let (current_k, current_v) =
            ring.project_packet_current_kv(&packet.normalized, &packet.position_ids)?;
        local_history.0 = Tensor::cat(&[&local_history.0, &current_k], 2);
        local_history.1 = Tensor::cat(&[&local_history.1, &current_v], 2);
    }

    continue_layer_packet(layer, packet, &local_history.0, &local_history.1, None)
}

pub(crate) fn process_layer_packet_with_reserved_history(
    layer: &mut DecoderLayer,
    packet: LayerPacket,
    local_history: &mut ReservedPositionedKvShard,
) -> Result<LayerStepOutcome, ModelError> {
    let query_len = packet.position_ids.size()[1] as usize;
    let new_position_offsets = if packet.current_domain == packet.assignee {
        (0..query_len).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    process_layer_packet_with_reserved_history_for_positions(
        layer,
        packet,
        local_history,
        &new_position_offsets,
    )
}

pub(crate) fn process_layer_packet_with_reserved_history_for_positions(
    layer: &mut DecoderLayer,
    packet: LayerPacket,
    local_history: &mut ReservedPositionedKvShard,
    new_position_offsets: &[usize],
) -> Result<LayerStepOutcome, ModelError> {
    let active_k = local_history.active_k();
    let active_v = local_history.active_v();
    if active_k.size() != active_v.size() {
        return Err(ModelError::Backend(format!(
            "domain {} has mismatched K/V shapes",
            packet.current_domain
        )));
    }

    let query_len = packet.position_ids.size()[1] as usize;
    let mut seen_offsets = vec![false; query_len];
    for &offset in new_position_offsets {
        if offset >= query_len || seen_offsets[offset] {
            return Err(ModelError::Backend(format!(
                "domain {} received invalid or duplicate new position offset {offset} for query_len={query_len}",
                packet.current_domain
            )));
        }
        seen_offsets[offset] = true;
    }

    if !new_position_offsets.is_empty() {
        let index_values = new_position_offsets
            .iter()
            .map(|&offset| offset as i64)
            .collect::<Vec<_>>();
        let normalized_indices =
            Tensor::from_slice(&index_values).to_device(packet.normalized.device());
        let position_indices =
            Tensor::from_slice(&index_values).to_device(packet.position_ids.device());
        let local_normalized = packet.normalized.index_select(1, &normalized_indices);
        let local_position_ids = packet.position_ids.index_select(1, &position_indices);
        let ring = ring_backend(layer)?;
        let (current_k, current_v) =
            ring.project_packet_current_kv(&local_normalized, &local_position_ids)?;
        let positions = (0..local_position_ids.size()[1])
            .map(|offset| local_position_ids.int64_value(&[0, offset]))
            .collect::<Vec<_>>();
        local_history
            .append(&current_k, &current_v, &positions)
            .map_err(ModelError::Backend)?;
    }

    let local_positions = local_history.position_tensor();
    continue_layer_packet(
        layer,
        packet,
        &local_history.active_k(),
        &local_history.active_v(),
        Some(&local_positions),
    )
}

fn continue_layer_packet(
    layer: &mut DecoderLayer,
    mut packet: LayerPacket,
    local_k: &Tensor,
    local_v: &Tensor,
    local_positions: Option<&Tensor>,
) -> Result<LayerStepOutcome, ModelError> {
    let finished = packet.visited_domains + 1 == packet.domains;
    let projected_output = {
        let ring = ring_backend(layer)?;

        let (next_output, next_lse) = match (packet.attention_output.take(), packet.lse.take()) {
            (None, None) => match local_positions {
                Some(k_positions) => ring.positioned_local_compact_partial(
                    &packet.q,
                    &packet.position_ids,
                    local_k,
                    local_v,
                    k_positions,
                ),
                None => ring.decode_local_compact_partial(&packet.q, local_k, local_v),
            },
            (Some(output), Some(lse)) => match local_positions {
                Some(k_positions) => ring.positioned_merge_compact_partial(
                    &packet.q,
                    &packet.position_ids,
                    &output,
                    &lse,
                    local_k,
                    local_v,
                    k_positions,
                ),
                None => {
                    ring.decode_merge_compact_partial(&packet.q, &output, &lse, local_k, local_v)
                }
            },
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PositionedLayerRingStats {
    pub domains: usize,
    pub hops: usize,
    pub visited_domains: Vec<usize>,
    pub new_kv_positions_by_domain: Vec<usize>,
    pub starter: usize,
    pub finisher: usize,
}

#[derive(Debug)]
pub(crate) struct PositionedModelRingResult {
    pub hidden_states: Tensor,
    pub logits: Tensor,
    pub layer_stats: Vec<PositionedLayerRingStats>,
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

pub(crate) fn run_model_ring_with_reserved_history_for_positions(
    model: &mut LlamaModel,
    hidden_states: &Tensor,
    position_ids: &Tensor,
    layer_history_shards: &mut [Vec<ReservedPositionedKvShard>],
    starter: usize,
    new_position_offsets_by_domain: &[Vec<usize>],
) -> Result<PositionedModelRingResult, ModelError> {
    let layers = model.layers.len();
    if layers == 0 || layer_history_shards.len() != layers {
        return Err(ModelError::Backend(format!(
            "reserved positioned model ring requires matching non-empty layers and shard sets: layers={layers}, shard_sets={}",
            layer_history_shards.len()
        )));
    }
    let domains = layer_history_shards[0].len();
    validate_route(hidden_states, position_ids, starter, starter, domains)?;
    if new_position_offsets_by_domain.len() != domains
        || layer_history_shards
            .iter()
            .any(|shards| shards.len() != domains)
    {
        return Err(ModelError::Backend(format!(
            "reserved positioned model ring requires {domains} domain offset and shard sets"
        )));
    }

    let query_len = position_ids.size()[1] as usize;
    let mut seen_offsets = vec![false; query_len];
    for offsets in new_position_offsets_by_domain {
        for &offset in offsets {
            if offset >= query_len || seen_offsets[offset] {
                return Err(ModelError::Backend(format!(
                    "reserved positioned model ring received invalid or duplicate new position offset {offset} for query_len={query_len}"
                )));
            }
            seen_offsets[offset] = true;
        }
    }
    if let Some(missing) = seen_offsets.iter().position(|seen| !seen) {
        return Err(ModelError::Backend(format!(
            "reserved positioned model ring is missing new position offset {missing} for query_len={query_len}"
        )));
    }

    let new_kv_positions_by_domain = new_position_offsets_by_domain
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let mut current_hidden = hidden_states.shallow_clone();
    let mut current_starter = starter;
    let mut layer_stats = Vec::with_capacity(layers);

    for (layer_idx, shards) in layer_history_shards.iter_mut().enumerate() {
        // Position ownership comes from the frozen offsets, not this legacy scalar field.
        let mut packet = LayerPacket::start(
            &mut model.layers[layer_idx],
            &current_hidden,
            position_ids,
            current_starter,
            current_starter,
            domains,
        )?;
        let mut visited_domains = Vec::with_capacity(domains);
        current_hidden = loop {
            let domain = packet.current_domain;
            visited_domains.push(domain);
            match process_layer_packet_with_reserved_history_for_positions(
                &mut model.layers[layer_idx],
                packet,
                &mut shards[domain],
                &new_position_offsets_by_domain[domain],
            )? {
                LayerStepOutcome::Forward(next_packet) => packet = next_packet,
                LayerStepOutcome::Finished { hidden_states, .. } => break hidden_states,
            }
        };

        let finisher = (current_starter + domains - 1) % domains;
        layer_stats.push(PositionedLayerRingStats {
            domains,
            hops: domains - 1,
            visited_domains,
            new_kv_positions_by_domain: new_kv_positions_by_domain.clone(),
            starter: current_starter,
            finisher,
        });
        current_starter = finisher;
    }

    let logits = project_final_logits(model, &current_hidden);
    Ok(PositionedModelRingResult {
        hidden_states: current_hidden,
        logits,
        layer_stats,
        logits_producer_domain: current_starter,
        logits_projections: 1,
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
    use crate::model::cache::{ContiguousKvCache, KvCache, KvCacheImpl};
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

    fn reference_positioned_layer_output(
        layer_idx: usize,
        config: &ModelConfig,
        weights: &ModelWeights,
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

        let query_len = hidden.size()[1];
        let history_len = history_k.size()[2];
        let key_len = history_len + query_len;
        let key_positions = Tensor::arange(key_len, (Kind::Int64, device));
        let causal = position_ids
            .view([query_len, 1])
            .ge_tensor(&key_positions.view([1, key_len]));
        let mask = Tensor::zeros([query_len, key_len], (Kind::Float, device))
            .masked_fill(&causal.logical_not(), f64::NEG_INFINITY)
            .unsqueeze(0)
            .unsqueeze(0);

        let normed = input_norm.forward(hidden);
        let attention_output = attention
            .forward(&normed, position_ids, Some(&mut cache), Some(&mask))
            .unwrap();
        let post_attention = &attention_output + hidden;
        let mlp_output = mlp.forward(&post_norm.forward(&post_attention));
        (attention_output, post_attention + mlp_output)
    }

    fn reference_current_kv(
        layer_idx: usize,
        config: &ModelConfig,
        weights: &ModelWeights,
        hidden: &Tensor,
        position_ids: &Tensor,
    ) -> (Tensor, Tensor) {
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings.unwrap(),
            config.rope_theta,
            hidden.device(),
        );
        let backend =
            HcpRingAttentionBackend::from_weights(weights, layer_idx, config, &rope, 1).unwrap();
        let input_norm = RmsNorm::from_weights(
            weights,
            &WeightNames::rms_norm_weight(layer_idx),
            config.rms_norm_eps,
        )
        .unwrap();
        backend
            .project_packet_current_kv(&input_norm.forward(hidden), position_ids)
            .unwrap()
    }

    #[test]
    fn reserved_positioned_kv_core_api_preserves_committed_prefix() {
        let config = test_config();
        let device = Device::Cpu;
        let mut slab = super::ReservedPositionedKvShard::new(&config, 2, device);
        let shape = [1, config.num_kv_heads() as i64, 2, config.head_dim() as i64];
        let k = deterministic_tensor(&shape, 299.0, device);
        let v = deterministic_tensor(&shape, 300.0, device);

        slab.append(&k, &v, &[4, 5]).unwrap();

        assert_eq!(slab.reserved_capacity(), 2);
        assert_eq!(slab.committed_len(), 2);
        assert_eq!(slab.positions(), &[4, 5]);
        assert_eq!(slab.active_k().size(), shape);
        assert_eq!(slab.active_v().size(), shape);
        let _processor: fn(
            &mut DecoderLayer,
            LayerPacket,
            &mut super::ReservedPositionedKvShard,
        ) -> Result<LayerStepOutcome, ModelError> =
            super::process_layer_packet_with_reserved_history;
    }

    #[test]
    fn reserved_positioned_kv_accepts_explicit_runtime_dtype() {
        let config = test_config();
        let device = Device::Cpu;
        let mut slab =
            super::ReservedPositionedKvShard::new_with_kind(&config, 2, device, Kind::BFloat16);
        let shape = [1, config.num_kv_heads() as i64, 2, config.head_dim() as i64];
        let k = deterministic_tensor(&shape, 301.0, device).to_kind(Kind::BFloat16);
        let v = deterministic_tensor(&shape, 302.0, device).to_kind(Kind::BFloat16);

        slab.append(&k, &v, &[6, 7]).unwrap();

        assert_eq!(slab.active_k().kind(), Kind::BFloat16);
        assert_eq!(slab.active_v().kind(), Kind::BFloat16);
        assert_eq!(slab.committed_len(), 2);
        assert_eq!(slab.positions(), &[6, 7]);
    }

    #[test]
    fn llama_prefill_writes_reserved_positioned_cache() {
        let device = Device::Cpu;
        let mut config = test_config();
        config.num_layers = 2;
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut caches = (0..config.num_layers)
            .map(|_| {
                Some(KvCacheImpl::ReservedPositioned(
                    ReservedPositionedKvShard::new_with_kind(&config, 4, device, model.dtype),
                ))
            })
            .collect::<Vec<_>>();
        let input_ids = Tensor::from_slice(&[3_i64, 5]).unsqueeze(0);

        let logits = model.forward(&input_ids, &mut caches).unwrap();

        assert_eq!(logits.size(), [1, 2, config.vocab_size as i64]);
        for cache in &caches {
            let Some(KvCacheImpl::ReservedPositioned(shard)) = cache else {
                panic!("expected reserved positioned cache");
            };
            assert_eq!(shard.committed_len(), 2);
            assert_eq!(shard.positions(), &[0, 1]);
        }
    }

    #[test]
    #[ignore = "requires the local Qwen2-0.5B model weights"]
    fn real_qwen_prefill_matches_reserved_positioned_cache() {
        let device = Device::Cpu;
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("Qwen2-0.5B");
        let config = ModelConfig::from_file(model_dir.join("config.json")).unwrap();
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));

        let weights = ModelWeights::from_dir(&model_dir, device).unwrap();
        let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut reserved = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        assert_eq!(reserved.dtype, Kind::BFloat16);

        let prompt = [151644_i64, 9707, 0, 16];
        let input_ids = Tensor::from_slice(&prompt).unsqueeze(0);
        let mut reference_caches = reference.create_kv_caches();
        let mut reserved_caches = (0..config.num_layers)
            .map(|_| {
                Some(KvCacheImpl::ReservedPositioned(
                    ReservedPositionedKvShard::new_with_kind(
                        &config,
                        prompt.len() + 2,
                        device,
                        reserved.dtype,
                    ),
                ))
            })
            .collect::<Vec<_>>();

        let reference_logits = reference
            .forward(&input_ids, &mut reference_caches)
            .unwrap();
        let reserved_logits = reserved.forward(&input_ids, &mut reserved_caches).unwrap();

        let logits_diff = (&reference_logits - &reserved_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            logits_diff < 1e-3,
            "real Qwen reserved prefill logits differ from contiguous reference: {logits_diff}"
        );
        assert_eq!(
            reference_logits
                .narrow(1, prompt.len() as i64 - 1, 1)
                .argmax(-1, false)
                .int64_value(&[0, 0]),
            reserved_logits
                .narrow(1, prompt.len() as i64 - 1, 1)
                .argmax(-1, false)
                .int64_value(&[0, 0])
        );

        for (layer_idx, (reference_cache, reserved_cache)) in
            reference_caches.iter().zip(&reserved_caches).enumerate()
        {
            let (reference_k, reference_v) = reference_cache
                .as_ref()
                .and_then(KvCacheImpl::get_kv)
                .unwrap();
            let Some(KvCacheImpl::ReservedPositioned(shard)) = reserved_cache else {
                panic!("layer {layer_idx} did not use reserved positioned cache");
            };
            let reserved_k = shard.active_k();
            let reserved_v = shard.active_v();
            assert_eq!(shard.reserved_capacity(), prompt.len() + 2);
            assert_eq!(shard.committed_len(), prompt.len());
            assert_eq!(shard.positions(), &[0, 1, 2, 3]);
            assert_eq!(reserved_k.kind(), Kind::BFloat16);
            assert_eq!(reserved_v.kind(), Kind::BFloat16);
            assert_eq!(reference_k.size(), reserved_k.size());
            assert_eq!(reference_v.size(), reserved_v.size());
            assert_eq!(
                (&reference_k - &reserved_k).abs().max().double_value(&[]),
                0.0,
                "layer {layer_idx} K cache differs"
            );
            assert_eq!(
                (&reference_v - &reserved_v).abs().max().double_value(&[]),
                0.0,
                "layer {layer_idx} V cache differs"
            );
        }
    }

    fn reserved_positioned_layer_shards(
        config: &ModelConfig,
        reservation_plan: &[Vec<usize>],
        device: Device,
    ) -> Vec<Vec<ReservedPositionedKvShard>> {
        assert_eq!(reservation_plan.len(), config.num_layers);
        reservation_plan
            .iter()
            .map(|capacities| {
                capacities
                    .iter()
                    .map(|&capacity| ReservedPositionedKvShard::new(config, capacity, device))
                    .collect()
            })
            .collect()
    }

    fn local_reference_model(
        config: &ModelConfig,
        weights: &ModelWeights,
        device: Device,
    ) -> LlamaModel {
        let mut model = LlamaModel::from_weights(config.clone(), weights, device, 1).unwrap();
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings.unwrap(),
            config.rope_theta,
            device,
        );
        for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
            layer.attention = Box::new(crate::model::attention::backend::LocalAttentionBackend {
                attention: GqaAttention::from_weights(weights, layer_idx, config, &rope).unwrap(),
            });
        }
        model
    }

    fn run_contiguous_reference_block(
        model: &mut LlamaModel,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        caches: &mut [ContiguousKvCache],
    ) -> (Tensor, Tensor) {
        let query_len = hidden_states.size()[1];
        assert_eq!(position_ids.size(), vec![1, query_len]);
        assert_eq!(caches.len(), model.layers.len());
        let mut current_hidden = hidden_states.shallow_clone();

        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            let history_len = cache.seq_len() as i64;
            let key_len = history_len + query_len;
            let query_positions =
                Tensor::arange(query_len, (Kind::Int64, hidden_states.device())) + history_len;
            let key_positions = Tensor::arange(key_len, (Kind::Int64, hidden_states.device()));
            let causal = query_positions
                .unsqueeze(1)
                .ge_tensor(&key_positions.unsqueeze(0));
            let mask = Tensor::zeros([query_len, key_len], (Kind::Float, hidden_states.device()))
                .masked_fill(&causal.logical_not(), f64::NEG_INFINITY)
                .unsqueeze(0)
                .unsqueeze(0);
            current_hidden = model.layers[layer_idx]
                .forward(&current_hidden, position_ids, Some(cache), Some(&mask))
                .unwrap();
        }

        let logits = project_final_logits(model, &current_hidden);
        (current_hidden, logits)
    }

    fn run_reserved_positioned_prefill_block(
        model: &mut LlamaModel,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        layer_shards: &mut [Vec<ReservedPositionedKvShard>],
        token_splits: &[usize],
    ) -> (Tensor, Tensor, usize) {
        let seq_len = hidden_states.size()[1] as usize;
        assert_eq!(position_ids.size(), vec![1, seq_len as i64]);
        assert_eq!(layer_shards.len(), model.layers.len());
        assert_eq!(token_splits.iter().sum::<usize>(), seq_len);
        assert!(layer_shards
            .iter()
            .all(|shards| shards.len() == token_splits.len()));

        let new_positions = (0..seq_len)
            .map(|offset| position_ids.int64_value(&[0, offset as i64]))
            .collect::<Vec<_>>();
        let mut current_hidden = hidden_states.shallow_clone();
        let mut projected_positions = 0_usize;

        for (layer_idx, shards) in layer_shards.iter_mut().enumerate() {
            let residual = current_hidden.shallow_clone();
            let normalized = model.layers[layer_idx]
                .input_layernorm
                .forward(&current_hidden);
            let projected_attention = {
                let ring = ring_backend(&mut model.layers[layer_idx]).unwrap();
                let (q, current_k, current_v) = ring
                    .project_positioned_qkv(&normalized, position_ids)
                    .unwrap();
                projected_positions += seq_len;

                let mut offset = 0_i64;
                for (domain, &count) in token_splits.iter().enumerate() {
                    let count = count as i64;
                    shards[domain]
                        .append(
                            &current_k.narrow(2, offset, count),
                            &current_v.narrow(2, offset, count),
                            &new_positions[offset as usize..(offset + count) as usize],
                        )
                        .unwrap();
                    offset += count;
                }

                let mut partial = None;
                for shard in shards.iter() {
                    let k_positions = shard.position_tensor();
                    let active_k = shard.active_k();
                    let active_v = shard.active_v();
                    partial = Some(match partial {
                        None => ring.positioned_local_compact_partial(
                            &q,
                            position_ids,
                            &active_k,
                            &active_v,
                            &k_positions,
                        ),
                        Some((output, lse)) => ring.positioned_merge_compact_partial(
                            &q,
                            position_ids,
                            &output,
                            &lse,
                            &active_k,
                            &active_v,
                            &k_positions,
                        ),
                    });
                }
                let (attention_output, _) = partial.expect("prefill requires at least one domain");
                ring.project_decode_output(&attention_output)
            };

            let post_attention = projected_attention + residual;
            let mlp_output = model.layers[layer_idx].mlp.forward(
                &model.layers[layer_idx]
                    .post_attention_layernorm
                    .forward(&post_attention),
            );
            current_hidden = post_attention + mlp_output;
        }

        let logits = project_final_logits(model, &current_hidden);
        (current_hidden, logits, projected_positions)
    }

    fn run_reserved_positioned_decode(
        model: &mut LlamaModel,
        hidden_states: &Tensor,
        position: i64,
        layer_shards: &mut [Vec<ReservedPositionedKvShard>],
        starter: usize,
        assignees: &[usize],
    ) -> ModelRingResult {
        assert_eq!(layer_shards.len(), model.layers.len());
        assert_eq!(assignees.len(), model.layers.len());
        let domains = layer_shards[0].len();
        assert!(domains > 0);
        assert!(layer_shards.iter().all(|shards| shards.len() == domains));

        let position_ids = Tensor::from_slice(&[position]).unsqueeze(0);
        let mut current_hidden = hidden_states.shallow_clone();
        let mut current_starter = starter;
        let mut layer_stats = Vec::with_capacity(model.layers.len());

        for (layer_idx, shards) in layer_shards.iter_mut().enumerate() {
            let residual = current_hidden.shallow_clone();
            let normalized = model.layers[layer_idx]
                .input_layernorm
                .forward(&current_hidden);
            let assignee = assignees[layer_idx];
            let finisher = (current_starter + domains - 1) % domains;
            let (projected_attention, visited_domains) = {
                let ring = ring_backend(&mut model.layers[layer_idx]).unwrap();
                let q = ring.project_packet_q(&normalized, &position_ids).unwrap();
                let mut partial = None;
                let mut visited_domains = Vec::with_capacity(domains);

                for visit_index in 0..domains {
                    let domain = (current_starter + visit_index) % domains;
                    visited_domains.push(domain);
                    if domain == assignee {
                        let (current_k, current_v) = ring
                            .project_packet_current_kv(&normalized, &position_ids)
                            .unwrap();
                        shards[domain]
                            .append(&current_k, &current_v, &[position])
                            .unwrap();
                    }
                    let active_k = shards[domain].active_k();
                    let active_v = shards[domain].active_v();
                    partial = Some(match partial {
                        None => ring.decode_local_compact_partial(&q, &active_k, &active_v),
                        Some((output, lse)) => ring
                            .decode_merge_compact_partial(&q, &output, &lse, &active_k, &active_v),
                    });
                }

                let (attention_output, _) =
                    partial.expect("decode requires at least one reserved KV shard");
                (
                    ring.project_decode_output(&attention_output),
                    visited_domains,
                )
            };

            let post_attention = projected_attention + residual;
            let mlp_output = model.layers[layer_idx].mlp.forward(
                &model.layers[layer_idx]
                    .post_attention_layernorm
                    .forward(&post_attention),
            );
            current_hidden = post_attention + mlp_output;
            layer_stats.push(SingleLayerRingStats {
                domains,
                hops: domains - 1,
                visited_domains,
                local_partials: domains,
                q_projections: 1,
                q_projection_domain: current_starter,
                current_kv_projections: 1,
                current_kv_projection_domain: assignee,
                current_kv_commits: 1,
                layer_finishes: 1,
                starter: current_starter,
                assignee,
                finisher,
            });
            current_starter = finisher;
        }

        let logits = project_final_logits(model, &current_hidden);
        ModelRingResult {
            hidden_states: current_hidden,
            logits,
            layer_stats,
            logits_producer_domain: current_starter,
            logits_projections: 1,
        }
    }

    fn assert_phase_matches(
        phase: &str,
        distributed_hidden: &Tensor,
        distributed_logits: &Tensor,
        reference_hidden: &Tensor,
        reference_logits: &Tensor,
    ) {
        let hidden_diff = (distributed_hidden - reference_hidden)
            .abs()
            .max()
            .double_value(&[]);
        assert!(hidden_diff < 1e-3, "{phase} hidden diff: {hidden_diff}");
        let logits_diff = (distributed_logits - reference_logits)
            .abs()
            .max()
            .double_value(&[]);
        assert!(logits_diff < 1e-3, "{phase} logits diff: {logits_diff}");
    }

    fn sample_last_token(logits: &Tensor) -> i64 {
        let last = logits.select(1, logits.size()[1] - 1);
        last.argmax(-1, false).int64_value(&[0])
    }

    fn assert_reserved_positioned_history(
        layer_shards: &[Vec<ReservedPositionedKvShard>],
        expected_positions: std::ops::Range<i64>,
    ) {
        let expected = expected_positions.collect::<Vec<_>>();
        for (layer_idx, shards) in layer_shards.iter().enumerate() {
            let mut actual = Vec::new();
            for (domain, shard) in shards.iter().enumerate() {
                let active_k = shard.active_k();
                let active_v = shard.active_v();
                assert_eq!(
                    active_k.size(),
                    active_v.size(),
                    "layer {layer_idx} domain {domain} K/V shape"
                );
                assert_eq!(
                    shard.committed_len(),
                    shard.positions.len(),
                    "layer {layer_idx} domain {domain} position length"
                );
                assert_eq!(
                    active_k.size()[2] as usize,
                    shard.committed_len(),
                    "layer {layer_idx} domain {domain} committed prefix length"
                );
                actual.extend_from_slice(&shard.positions);
            }
            actual.sort_unstable();
            assert_eq!(actual, expected, "layer {layer_idx} position union");
        }
    }

    fn reserved_positioned_domain_totals(
        layer_shards: &[Vec<ReservedPositionedKvShard>],
    ) -> Vec<usize> {
        let domains = layer_shards[0].len();
        let mut totals = vec![0_usize; domains];
        for shards in layer_shards {
            for (domain, shard) in shards.iter().enumerate() {
                totals[domain] += shard.positions.len();
            }
        }
        totals
    }

    #[test]
    fn positioned_kv_slab_appends_in_place_and_rejects_overflow() {
        let device = Device::Cpu;
        let config = test_config();
        let mut slab = ReservedPositionedKvShard::new(&config, 3, device);
        let shape = [1, config.num_kv_heads() as i64, 3, config.head_dim() as i64];
        let expected_k = deterministic_tensor(&shape, 301.0, device);
        let expected_v = deterministic_tensor(&shape, 302.0, device);
        let storage_ptrs = slab.storage_ptrs();

        slab.append(
            &expected_k.narrow(2, 0, 1),
            &expected_v.narrow(2, 0, 1),
            &[7],
        )
        .unwrap();
        slab.append(
            &expected_k.narrow(2, 1, 2),
            &expected_v.narrow(2, 1, 2),
            &[9, 12],
        )
        .unwrap();

        assert_eq!(slab.reserved_capacity(), 3);
        assert_eq!(slab.committed_len(), 3);
        assert_eq!(slab.positions, [7, 9, 12]);
        assert_eq!(slab.storage_ptrs(), storage_ptrs);
        assert_eq!(
            (&slab.active_k() - &expected_k)
                .abs()
                .max()
                .double_value(&[]),
            0.0
        );
        assert_eq!(
            (&slab.active_v() - &expected_v)
                .abs()
                .max()
                .double_value(&[]),
            0.0
        );

        let committed_k = slab.active_k().copy();
        let committed_v = slab.active_v().copy();
        let error = slab
            .append(
                &expected_k.narrow(2, 0, 1),
                &expected_v.narrow(2, 0, 1),
                &[13],
            )
            .unwrap_err();
        assert!(error.contains("capacity"));
        assert_eq!(slab.committed_len(), 3);
        assert_eq!(slab.positions, [7, 9, 12]);
        assert_eq!(slab.storage_ptrs(), storage_ptrs);
        assert_eq!(
            (&slab.active_k() - committed_k)
                .abs()
                .max()
                .double_value(&[]),
            0.0
        );
        assert_eq!(
            (&slab.active_v() - committed_v)
                .abs()
                .max()
                .double_value(&[]),
            0.0
        );
    }

    #[test]
    fn frozen_kv_assignee_schedule_is_capacity_weighted_and_request_stable() {
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

        assert!(FrozenKvAssigneeSchedule::new(&[], 41, 24).is_err());
        assert!(FrozenKvAssigneeSchedule::new(&[0, 0, 0], 41, 24).is_err());
        assert!(FrozenKvAssigneeSchedule::new(&[1, 3, 2], 41, 0).is_err());
        assert_eq!(first.assignee_for(0, 3, 3), None);
        assert_eq!(first.assignee_for(8, 0, 3), None);
    }

    #[test]
    fn frozen_kv_assignee_schedule_full_horizon_reservation_bounds_every_phase_prefix() {
        for (tickets, total_units) in [
            (vec![7], 5_usize),
            (vec![1, 1], 6),
            (vec![1, 1, 1], 7),
            (vec![1, 3, 2], 24),
            (vec![1, 0, 3], 16),
            (vec![2, 5, 1, 4], 19),
        ] {
            for request_id in 0..total_units as u64 {
                let schedule =
                    FrozenKvAssigneeSchedule::new(&tickets, request_id, total_units).unwrap();
                let reservation = schedule.counts().to_vec();
                let mut consumed = vec![0_usize; tickets.len()];

                for ordinal in 0..total_units {
                    let assignee = schedule.assignee_for(ordinal, 0, 1).unwrap();
                    consumed[assignee] += 1;
                    for domain in 0..tickets.len() {
                        assert!(
                            consumed[domain] <= reservation[domain],
                            "tickets={tickets:?}, phase={request_id}, prefix={}, domain={domain}, consumed={}, reservation={}",
                            ordinal + 1,
                            consumed[domain],
                            reservation[domain]
                        );
                    }
                }

                assert_eq!(consumed, reservation);
                assert_eq!(schedule.assignee_for(total_units, 0, 1), None);
            }
        }
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
    fn multi_token_layer_packet_completes_positioned_causal_layer_without_history_payload() {
        let device = Device::Cpu;
        let config = test_config();
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let mut layer = model.layers.remove(0);
        let query_len = 3_i64;
        let history_len = 6_i64;
        let hidden =
            deterministic_tensor(&[1, query_len, config.hidden_size as i64], 301.0, device);
        let position_ids =
            Tensor::arange_start(history_len, history_len + query_len, (Kind::Int64, device))
                .unsqueeze(0);
        let history_shape = [
            1,
            config.num_kv_heads() as i64,
            history_len,
            config.head_dim() as i64,
        ];
        let history_k = deterministic_tensor(&history_shape, 302.0, device);
        let history_v = deterministic_tensor(&history_shape, 303.0, device);
        let domain_positions = [vec![0_i64, 3, 4], vec![1_i64, 2, 5]];
        let mut shards = domain_positions
            .iter()
            .enumerate()
            .map(|(domain, positions)| {
                let capacity = positions.len() + if domain == 1 { query_len as usize } else { 0 };
                let mut shard = ReservedPositionedKvShard::new(&config, capacity, device);
                let indices = Tensor::from_slice(positions);
                shard
                    .append(
                        &history_k.index_select(2, &indices),
                        &history_v.index_select(2, &indices),
                        positions,
                    )
                    .unwrap();
                shard
            })
            .collect::<Vec<_>>();
        let storage_ptrs = shards
            .iter()
            .map(ReservedPositionedKvShard::storage_ptrs)
            .collect::<Vec<_>>();
        let (reference_attention, reference_hidden) = reference_positioned_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &history_k,
            &history_v,
        );

        let packet = LayerPacket::start(&mut layer, &hidden, &position_ids, 0, 1, 2).unwrap();
        let packet =
            match process_layer_packet_with_reserved_history(&mut layer, packet, &mut shards[0])
                .unwrap()
            {
                LayerStepOutcome::Forward(packet) => packet,
                LayerStepOutcome::Finished { .. } => panic!("N=2 packet finished before successor"),
            };
        let expected_payload = query_len as usize * (4 * config.hidden_size + config.num_heads + 1);
        assert_eq!(packet.tensor_payload_elements(), expected_payload);
        let (attention_output, hidden_states) =
            match process_layer_packet_with_reserved_history(&mut layer, packet, &mut shards[1])
                .unwrap()
            {
                LayerStepOutcome::Finished {
                    attention_output,
                    hidden_states,
                } => (attention_output, hidden_states),
                LayerStepOutcome::Forward(_) => panic!("N=2 packet did not finish at successor"),
            };

        let attention_diff = (&attention_output - reference_attention)
            .abs()
            .max()
            .double_value(&[]);
        let hidden_diff = (&hidden_states - reference_hidden)
            .abs()
            .max()
            .double_value(&[]);
        assert!(attention_diff < 1e-4, "attention diff: {attention_diff}");
        assert!(hidden_diff < 2e-4, "hidden diff: {hidden_diff}");
        assert_eq!(shards[0].positions(), &[0, 3, 4]);
        assert_eq!(shards[1].positions(), &[1, 2, 5, 6, 7, 8]);
        assert_eq!(
            shards
                .iter()
                .map(ReservedPositionedKvShard::storage_ptrs)
                .collect::<Vec<_>>(),
            storage_ptrs
        );
    }

    #[test]
    fn multi_token_packet_generates_new_kv_by_capacity_weighted_position_owner() {
        let device = Device::Cpu;
        let config = test_config();
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 3).unwrap();
        let mut layer = model.layers.remove(0);
        let domains = 3_usize;
        let query_len = 6_i64;
        let history_len = 9_i64;
        let hidden =
            deterministic_tensor(&[1, query_len, config.hidden_size as i64], 311.0, device);
        let position_ids =
            Tensor::arange_start(history_len, history_len + query_len, (Kind::Int64, device))
                .unsqueeze(0);
        let history_shape = [
            1,
            config.num_kv_heads() as i64,
            history_len,
            config.head_dim() as i64,
        ];
        let history_k = deterministic_tensor(&history_shape, 312.0, device);
        let history_v = deterministic_tensor(&history_shape, 313.0, device);
        let history_positions = [vec![0_i64, 3, 6], vec![1_i64, 4, 7], vec![2_i64, 5, 8]];

        let schedule = FrozenKvAssigneeSchedule::new(&[1, 3, 2], 0, query_len as usize).unwrap();
        assert_eq!(schedule.counts(), &[1, 3, 2]);
        let mut owner_offsets = vec![Vec::new(); domains];
        for offset in 0..query_len as usize {
            let owner = schedule.assignee_for(offset, 0, 1).unwrap();
            owner_offsets[owner].push(offset);
        }
        let mut assigned_offsets = owner_offsets.iter().flatten().copied().collect::<Vec<_>>();
        assigned_offsets.sort_unstable();
        assert_eq!(
            assigned_offsets,
            (0..query_len as usize).collect::<Vec<_>>()
        );
        assert_eq!(
            owner_offsets.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );

        let mut shards = history_positions
            .iter()
            .enumerate()
            .map(|(domain, positions)| {
                let mut shard = ReservedPositionedKvShard::new(
                    &config,
                    positions.len() + owner_offsets[domain].len(),
                    device,
                );
                let indices = Tensor::from_slice(positions);
                shard
                    .append(
                        &history_k.index_select(2, &indices),
                        &history_v.index_select(2, &indices),
                        positions,
                    )
                    .unwrap();
                shard
            })
            .collect::<Vec<_>>();
        let storage_ptrs = shards
            .iter()
            .map(ReservedPositionedKvShard::storage_ptrs)
            .collect::<Vec<_>>();
        let (reference_attention, reference_hidden) = reference_positioned_layer_output(
            0,
            &config,
            &weights,
            &hidden,
            &position_ids,
            &history_k,
            &history_v,
        );

        let mut packet =
            LayerPacket::start(&mut layer, &hidden, &position_ids, 0, 0, domains).unwrap();
        let mut packet_payload = None;
        let (attention_output, hidden_states) = loop {
            let domain = packet.current_domain;
            match process_layer_packet_with_reserved_history_for_positions(
                &mut layer,
                packet,
                &mut shards[domain],
                &owner_offsets[domain],
            )
            .unwrap()
            {
                LayerStepOutcome::Forward(next_packet) => {
                    packet_payload.get_or_insert_with(|| next_packet.tensor_payload_elements());
                    packet = next_packet;
                }
                LayerStepOutcome::Finished {
                    attention_output,
                    hidden_states,
                } => break (attention_output, hidden_states),
            }
        };

        let expected_payload = query_len as usize * (4 * config.hidden_size + config.num_heads + 1);
        assert_eq!(packet_payload, Some(expected_payload));
        let attention_diff = (&attention_output - reference_attention)
            .abs()
            .max()
            .double_value(&[]);
        let hidden_diff = (&hidden_states - reference_hidden)
            .abs()
            .max()
            .double_value(&[]);
        assert!(attention_diff < 1e-4, "attention diff: {attention_diff}");
        assert!(hidden_diff < 2e-4, "hidden diff: {hidden_diff}");

        for domain in 0..domains {
            let mut expected_positions = history_positions[domain].clone();
            expected_positions.extend(
                owner_offsets[domain]
                    .iter()
                    .map(|&offset| history_len + offset as i64),
            );
            assert_eq!(shards[domain].positions(), expected_positions);
            assert_eq!(
                shards[domain].committed_len(),
                shards[domain].reserved_capacity()
            );
        }
        assert_eq!(
            shards
                .iter()
                .map(ReservedPositionedKvShard::storage_ptrs)
                .collect::<Vec<_>>(),
            storage_ptrs
        );
    }

    #[test]
    fn position_owner_offsets_reject_duplicates_and_out_of_range_values() {
        fn assert_invalid_offsets(offsets: &[usize], expected: &str) {
            let device = Device::Cpu;
            let config = test_config();
            let weights = deterministic_weights(&config, device);
            let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
            let mut layer = model.layers.remove(0);
            let hidden = deterministic_tensor(&[1, 2, config.hidden_size as i64], 321.0, device);
            let position_ids = Tensor::from_slice(&[4_i64, 5]).unsqueeze(0);
            let packet = LayerPacket::start(&mut layer, &hidden, &position_ids, 0, 0, 1).unwrap();
            let mut shard = ReservedPositionedKvShard::new(&config, 2, device);

            let error = process_layer_packet_with_reserved_history_for_positions(
                &mut layer, packet, &mut shard, offsets,
            )
            .unwrap_err()
            .to_string();

            assert!(error.contains(expected), "unexpected error: {error}");
            assert_eq!(shard.committed_len(), 0);
        }

        assert_invalid_offsets(&[0, 0], "duplicate");
        assert_invalid_offsets(&[2], "query_len=2");
    }

    #[test]
    fn layer_packet_payload_does_not_grow_with_history_context() {
        fn payload_after_first_hop(history_len: i64) -> usize {
            let device = Device::Cpu;
            let config = test_config();
            let weights = deterministic_weights(&config, device);
            let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
            let mut layer = model.layers.remove(0);
            let query_len = 3_i64;
            let hidden =
                deterministic_tensor(&[1, query_len, config.hidden_size as i64], 100.0, device);
            let position_ids = Tensor::arange_start(
                history_len * 2,
                history_len * 2 + query_len,
                (Kind::Int64, device),
            )
            .unsqueeze(0);
            let shape = [
                1,
                config.num_kv_heads() as i64,
                history_len,
                config.head_dim() as i64,
            ];
            let mut local_history =
                ReservedPositionedKvShard::new(&config, history_len as usize, device);
            let positions = (0..history_len).collect::<Vec<_>>();
            local_history
                .append(
                    &deterministic_tensor(&shape, 110.0, device),
                    &deterministic_tensor(&shape, 120.0, device),
                    &positions,
                )
                .unwrap();
            let packet = LayerPacket::start(&mut layer, &hidden, &position_ids, 0, 1, 2).unwrap();
            match process_layer_packet_with_reserved_history(&mut layer, packet, &mut local_history)
                .unwrap()
            {
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
        token_offset: usize,
        layer_idx: usize,
        visit_index: usize,
        domain: usize,
        started: bool,
        finished: bool,
        sent_bytes: usize,
        kv_before: i64,
        kv_after: i64,
        reserved_capacity: usize,
        storage_ptrs_before: (usize, usize),
        storage_ptrs_after: (usize, usize),
    }

    #[derive(Debug)]
    struct TcpTokenOutput {
        token_offset: usize,
        hidden_states: Tensor,
        logits: Tensor,
        sampled_token: i64,
    }

    #[test]
    fn two_token_tcp_ring_continues_from_finisher_with_scheduled_assignees() {
        let domains = 3_usize;
        let layers_count = 2_usize;
        let token_steps = 2_usize;
        let initial_starter = 1_usize;
        let schedule =
            FrozenKvAssigneeSchedule::new(&[1, 3, 2], 2, token_steps * layers_count).unwrap();
        let assignees = (0..token_steps)
            .map(|token_offset| {
                (0..layers_count)
                    .map(|layer_idx| {
                        schedule
                            .assignee_for(token_offset, layer_idx, layers_count)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(schedule.counts(), &[1, 2, 1]);
        assert_eq!(assignees, [vec![2, 1], vec![1, 0]]);
        let mut growth_reservations = vec![vec![0_usize; domains]; layers_count];
        for token_assignees in &assignees {
            for (layer_idx, &assignee) in token_assignees.iter().enumerate() {
                growth_reservations[layer_idx][assignee] += 1;
            }
        }
        let mut domain_growth = vec![0_usize; domains];
        for layer_reservations in &growth_reservations {
            for (domain, &reserved) in layer_reservations.iter().enumerate() {
                domain_growth[domain] += reserved;
            }
        }
        assert_eq!(domain_growth, schedule.counts());
        let device = Device::Cpu;
        let mut config = test_config();
        config.num_layers = layers_count;
        let weights = deterministic_weights(&config, device);
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 129.0, device);
        let shard_lengths = [2_i64, 4, 3];
        let history_len = shard_lengths.iter().sum::<i64>();
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
        let mut reference_histories = layer_shards
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
        let reference_final_norm =
            RmsNorm::from_weights(&weights, WeightNames::layer_norm(), config.rms_norm_eps)
                .unwrap();
        let reference_lm_head = weights.get(WeightNames::lm_head()).unwrap();
        let mut reference_hidden = hidden.shallow_clone();
        let mut reference_outputs = Vec::with_capacity(token_steps);
        for token_offset in 0..token_steps {
            let position_ids =
                Tensor::from_slice(&[history_len + token_offset as i64]).unsqueeze(0);
            for layer_idx in 0..layers_count {
                let (current_k, current_v) = reference_current_kv(
                    layer_idx,
                    &config,
                    &weights,
                    &reference_hidden,
                    &position_ids,
                );
                let (_, next_hidden) = reference_layer_output(
                    layer_idx,
                    &config,
                    &weights,
                    &reference_hidden,
                    &position_ids,
                    &reference_histories[layer_idx].0,
                    &reference_histories[layer_idx].1,
                );
                reference_histories[layer_idx].0 =
                    Tensor::cat(&[&reference_histories[layer_idx].0, &current_k], 2);
                reference_histories[layer_idx].1 =
                    Tensor::cat(&[&reference_histories[layer_idx].1, &current_v], 2);
                reference_hidden = next_hidden;
            }
            let logits = reference_final_norm
                .forward(&reference_hidden)
                .matmul(&reference_lm_head.transpose(0, 1));
            let sampled_token = logits.squeeze().argmax(-1, false).int64_value(&[]);
            reference_outputs.push(TcpTokenOutput {
                token_offset,
                hidden_states: reference_hidden.shallow_clone(),
                logits,
                sampled_token,
            });
            if token_offset + 1 < token_steps {
                let token_ids = Tensor::from_slice(&[sampled_token]).unsqueeze(0);
                reference_hidden = Tensor::embedding(
                    weights.get(WeightNames::embedding()).unwrap(),
                    &token_ids,
                    -1,
                    false,
                    false,
                );
            }
        }

        let worker_models = (0..domains)
            .map(|_| LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap())
            .collect::<Vec<_>>();
        let mut worker_shards = (0..domains)
            .map(|_| Vec::with_capacity(layers_count))
            .collect::<Vec<_>>();
        for (layer_idx, shards) in layer_shards.into_iter().enumerate() {
            let mut position_offset = 0_i64;
            for (domain, (k, v)) in shards.into_iter().enumerate() {
                let history_positions =
                    (position_offset..position_offset + k.size()[2]).collect::<Vec<_>>();
                let capacity = k.size()[2] as usize + growth_reservations[layer_idx][domain];
                let mut slab = ReservedPositionedKvShard::new(&config, capacity, device);
                slab.append(&k, &v, &history_positions).unwrap();
                position_offset += k.size()[2];
                worker_shards[domain].push(slab);
            }
            assert_eq!(position_offset, history_len);
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
                    let worker_assignees = assignees.clone();
                    thread::spawn(move || {
                        let mut predecessor = TcpKvTransport::new(incoming, device).unwrap();
                        let mut successor = TcpKvTransport::new(outgoing, device).unwrap();
                        let mut next_token_hidden = initial_hidden;
                        let mut events = Vec::with_capacity(token_steps * layers_count);
                        let mut outputs = Vec::with_capacity(token_steps);

                        for token_offset in 0..token_steps {
                            let position_ids =
                                Tensor::from_slice(&[history_len + token_offset as i64])
                                    .unsqueeze(0);
                            let mut next_layer_hidden = next_token_hidden.take();
                            for layer_idx in 0..layers_count {
                                let started = next_layer_hidden.is_some();
                                let packet = if let Some(layer_hidden) = next_layer_hidden.take() {
                                    LayerPacket::start(
                                        &mut model.layers[layer_idx],
                                        &layer_hidden,
                                        &position_ids,
                                        domain,
                                        worker_assignees[token_offset][layer_idx],
                                        domains,
                                    )
                                    .unwrap()
                                } else {
                                    let wire = predecessor
                                        .recv_self_driving_packet()
                                        .unwrap()
                                        .expect("predecessor closed before the next packet");
                                    assert_eq!(wire.layer_idx, layer_idx);
                                    LayerPacket::from_self_driving_packet(wire).unwrap()
                                };
                                assert_eq!(packet.current_domain, domain);
                                let visit_index = packet.visited_domains;
                                let kv_before = shards[layer_idx].committed_len() as i64;
                                let reserved_capacity = shards[layer_idx].reserved_capacity();
                                let storage_ptrs_before = shards[layer_idx].storage_ptrs();
                                let (finished, sent_bytes) =
                                    match process_layer_packet_with_reserved_history(
                                        &mut model.layers[layer_idx],
                                        packet,
                                        &mut shards[layer_idx],
                                    )
                                    .unwrap()
                                    {
                                        LayerStepOutcome::Forward(packet) => {
                                            let wire =
                                                packet.into_self_driving_packet(layer_idx).unwrap();
                                            let sent_bytes =
                                                successor.send_self_driving_packet(&wire).unwrap();
                                            (false, sent_bytes)
                                        }
                                        LayerStepOutcome::Finished { hidden_states, .. } => {
                                            if layer_idx + 1 == layers_count {
                                                let logits =
                                                    project_final_logits(&model, &hidden_states);
                                                let sampled_token = logits
                                                    .squeeze()
                                                    .argmax(-1, false)
                                                    .int64_value(&[]);
                                                if token_offset + 1 < token_steps {
                                                    let token_ids =
                                                        Tensor::from_slice(&[sampled_token])
                                                            .unsqueeze(0);
                                                    next_token_hidden = Some(Tensor::embedding(
                                                        &model.embedding,
                                                        &token_ids,
                                                        -1,
                                                        false,
                                                        false,
                                                    ));
                                                }
                                                outputs.push(TcpTokenOutput {
                                                    token_offset,
                                                    hidden_states,
                                                    logits,
                                                    sampled_token,
                                                });
                                            } else {
                                                next_layer_hidden = Some(hidden_states);
                                            }
                                            (true, 0)
                                        }
                                    };
                                let storage_ptrs_after = shards[layer_idx].storage_ptrs();
                                events.push(TcpLayerEvent {
                                    token_offset,
                                    layer_idx,
                                    visit_index,
                                    domain,
                                    started,
                                    finished,
                                    sent_bytes,
                                    kv_before,
                                    kv_after: shards[layer_idx].committed_len() as i64,
                                    reserved_capacity,
                                    storage_ptrs_before,
                                    storage_ptrs_after,
                                });
                            }
                        }

                        (events, outputs)
                    })
                },
            )
            .collect::<Vec<_>>();

        let mut events = Vec::with_capacity(domains * token_steps * layers_count);
        let mut outputs = Vec::with_capacity(token_steps);
        for (domain, worker) in workers.into_iter().enumerate() {
            let (worker_events, worker_outputs) = worker.join().unwrap();
            assert_eq!(worker_events.len(), token_steps * layers_count);
            events.extend(worker_events);
            for output in worker_outputs {
                outputs.push((domain, output));
            }
        }

        assert_eq!(events.len(), domains * token_steps * layers_count);
        assert_eq!(
            events.iter().filter(|event| event.sent_bytes > 0).count(),
            token_steps * layers_count * (domains - 1)
        );
        let expected_routes = (0..token_steps)
            .map(|token_offset| {
                (0..layers_count)
                    .map(|layer_idx| {
                        let layer_ordinal = token_offset * layers_count + layer_idx;
                        let starter =
                            (initial_starter + domains - layer_ordinal % domains) % domains;
                        (0..domains)
                            .map(|visit_index| (starter + visit_index) % domains)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            expected_routes,
            [
                vec![vec![1, 2, 0], vec![0, 1, 2]],
                vec![vec![2, 0, 1], vec![1, 2, 0]],
            ]
        );
        for token_offset in 0..token_steps {
            for layer_idx in 0..layers_count {
                let mut layer_events = events
                    .iter()
                    .filter(|event| {
                        event.token_offset == token_offset && event.layer_idx == layer_idx
                    })
                    .collect::<Vec<_>>();
                layer_events.sort_by_key(|event| event.visit_index);
                assert_eq!(
                    layer_events
                        .iter()
                        .map(|event| event.domain)
                        .collect::<Vec<_>>(),
                    expected_routes[token_offset][layer_idx]
                );
                assert_eq!(
                    layer_events
                        .iter()
                        .filter(|event| event.started)
                        .map(|event| event.domain)
                        .collect::<Vec<_>>(),
                    vec![expected_routes[token_offset][layer_idx][0]]
                );
                assert_eq!(
                    layer_events
                        .iter()
                        .filter(|event| event.finished)
                        .map(|event| event.domain)
                        .collect::<Vec<_>>(),
                    vec![expected_routes[token_offset][layer_idx][domains - 1]]
                );
                for event in layer_events {
                    assert_eq!(event.storage_ptrs_after, event.storage_ptrs_before);
                    assert!(event.kv_after as usize <= event.reserved_capacity);
                    assert_eq!(
                        event.kv_after,
                        event.kv_before
                            + i64::from(event.domain == assignees[token_offset][layer_idx])
                    );
                    if token_offset + 1 == token_steps {
                        assert_eq!(event.kv_after as usize, event.reserved_capacity);
                    }
                }
            }
        }
        assert_eq!(
            expected_routes[1][0][0],
            expected_routes[0][layers_count - 1][domains - 1]
        );

        outputs.sort_by_key(|(_, output)| output.token_offset);
        assert_eq!(outputs.len(), token_steps);
        for (token_offset, (domain, actual)) in outputs.iter().enumerate() {
            let reference = &reference_outputs[token_offset];
            assert_eq!(actual.token_offset, token_offset);
            assert_eq!(reference.token_offset, token_offset);
            assert_eq!(
                *domain,
                expected_routes[token_offset][layers_count - 1][domains - 1]
            );
            let hidden_diff = (&actual.hidden_states - &reference.hidden_states)
                .abs()
                .max()
                .double_value(&[]);
            assert!(
                hidden_diff < 4e-4,
                "token {token_offset} TCP hidden diff: {hidden_diff}"
            );
            let logits_diff = (&actual.logits - &reference.logits)
                .abs()
                .max()
                .double_value(&[]);
            assert!(
                logits_diff < 4e-4,
                "token {token_offset} TCP logits diff: {logits_diff}"
            );
            assert_eq!(actual.sampled_token, reference.sampled_token);
        }
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
    fn twenty_four_layer_stationary_continuation_returns_to_decode() {
        let device = Device::Cpu;
        let domains = 3_usize;
        let layers = 24_usize;
        let continuation_len = 6_usize;
        let prefix_splits = [1_usize, 3, 2];
        let mut config = test_config();
        config.num_layers = layers;
        let weights = deterministic_weights(&config, device);
        let mut distributed_model =
            LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap();
        let mut reference_model = local_reference_model(&config, &weights, device);
        let mut reference_caches = (0..layers)
            .map(|_| ContiguousKvCache::new())
            .collect::<Vec<_>>();

        let decode_schedule = FrozenKvAssigneeSchedule::new(&[1, 3, 2], 41, 2 * layers).unwrap();
        assert_eq!(decode_schedule.counts(), &[8, 24, 16]);
        let decode_assignees = (0..2)
            .map(|token_offset| {
                (0..layers)
                    .map(|layer_idx| {
                        decode_schedule
                            .assignee_for(token_offset, layer_idx, layers)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for assignees in &decode_assignees {
            let mut counts = vec![0_usize; domains];
            for &domain in assignees {
                counts[domain] += 1;
            }
            assert_eq!(counts, [4, 12, 8]);
        }

        let continuation_schedule =
            FrozenKvAssigneeSchedule::new(&[1, 3, 2], 0, continuation_len).unwrap();
        assert_eq!(continuation_schedule.counts(), &[1, 3, 2]);
        let mut continuation_offsets_by_domain = vec![Vec::new(); domains];
        for offset in 0..continuation_len {
            let domain = continuation_schedule.assignee_for(offset, 0, 1).unwrap();
            continuation_offsets_by_domain[domain].push(offset);
        }
        assert_eq!(
            continuation_offsets_by_domain
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [1, 3, 2]
        );

        let reservation_plan = (0..layers)
            .map(|layer_idx| {
                let mut capacities = prefix_splits
                    .iter()
                    .zip(continuation_schedule.counts())
                    .map(|(&prefix, &continuation)| prefix + continuation)
                    .collect::<Vec<_>>();
                for assignees in &decode_assignees {
                    capacities[assignees[layer_idx]] += 1;
                }
                assert_eq!(capacities.iter().sum::<usize>(), 14);
                capacities
            })
            .collect::<Vec<_>>();
        let mut distributed_shards =
            reserved_positioned_layer_shards(&config, &reservation_plan, device);
        let initial_storage_ptrs = distributed_shards
            .iter()
            .map(|shards| {
                shards
                    .iter()
                    .map(ReservedPositionedKvShard::storage_ptrs)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let prefix_ids = Tensor::from_slice(&[3_i64, 5, 7, 9, 11, 13]).unsqueeze(0);
        let prefix_positions = Tensor::arange(6, (Kind::Int64, device)).unsqueeze(0);
        let distributed_prefix_hidden =
            Tensor::embedding(&distributed_model.embedding, &prefix_ids, -1, false, false);
        let reference_prefix_hidden =
            Tensor::embedding(&reference_model.embedding, &prefix_ids, -1, false, false);
        let (distributed_prefix_hidden, distributed_prefix_logits, _) =
            run_reserved_positioned_prefill_block(
                &mut distributed_model,
                &distributed_prefix_hidden,
                &prefix_positions,
                &mut distributed_shards,
                &prefix_splits,
            );
        let (reference_prefix_hidden, reference_prefix_logits) = run_contiguous_reference_block(
            &mut reference_model,
            &reference_prefix_hidden,
            &prefix_positions,
            &mut reference_caches,
        );
        assert_phase_matches(
            "stationary_prefix",
            &distributed_prefix_hidden,
            &distributed_prefix_logits,
            &reference_prefix_hidden,
            &reference_prefix_logits,
        );

        let decode_token = sample_last_token(&distributed_prefix_logits);
        assert_eq!(decode_token, sample_last_token(&reference_prefix_logits));
        let decode_ids = Tensor::from_slice(&[decode_token]).unsqueeze(0);
        let distributed_decode_hidden =
            Tensor::embedding(&distributed_model.embedding, &decode_ids, -1, false, false);
        let reference_decode_hidden =
            Tensor::embedding(&reference_model.embedding, &decode_ids, -1, false, false);
        let distributed_decode = run_reserved_positioned_decode(
            &mut distributed_model,
            &distributed_decode_hidden,
            6,
            &mut distributed_shards,
            1,
            &decode_assignees[0],
        );
        let decode_positions = Tensor::from_slice(&[6_i64]).unsqueeze(0);
        let (reference_decode_hidden, reference_decode_logits) = run_contiguous_reference_block(
            &mut reference_model,
            &reference_decode_hidden,
            &decode_positions,
            &mut reference_caches,
        );
        assert_phase_matches(
            "stationary_decode_history",
            &distributed_decode.hidden_states,
            &distributed_decode.logits,
            &reference_decode_hidden,
            &reference_decode_logits,
        );

        let committed_before_continuation = distributed_shards
            .iter()
            .map(|shards| {
                shards
                    .iter()
                    .map(ReservedPositionedKvShard::committed_len)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let continuation_ids = Tensor::from_slice(&[17_i64, 19, 23, 29, 31, 37]).unsqueeze(0);
        let continuation_positions =
            (Tensor::arange(continuation_len as i64, (Kind::Int64, device)) + 7).unsqueeze(0);
        let distributed_continuation_hidden = Tensor::embedding(
            &distributed_model.embedding,
            &continuation_ids,
            -1,
            false,
            false,
        );
        let reference_continuation_hidden = Tensor::embedding(
            &reference_model.embedding,
            &continuation_ids,
            -1,
            false,
            false,
        );

        let continuation = run_model_ring_with_reserved_history_for_positions(
            &mut distributed_model,
            &distributed_continuation_hidden,
            &continuation_positions,
            &mut distributed_shards,
            distributed_decode.logits_producer_domain,
            &continuation_offsets_by_domain,
        )
        .unwrap();
        let (reference_continuation_hidden, reference_continuation_logits) =
            run_contiguous_reference_block(
                &mut reference_model,
                &reference_continuation_hidden,
                &continuation_positions,
                &mut reference_caches,
            );

        assert_phase_matches(
            "stationary_continuation",
            &continuation.hidden_states,
            &continuation.logits,
            &reference_continuation_hidden,
            &reference_continuation_logits,
        );
        assert!(reference_caches.iter().all(|cache| cache.seq_len() == 13));
        assert_eq!(continuation.layer_stats.len(), layers);
        assert_eq!(
            continuation
                .layer_stats
                .iter()
                .map(|stats| stats.hops)
                .sum::<usize>(),
            layers * (domains - 1)
        );
        assert_eq!(
            continuation
                .layer_stats
                .iter()
                .map(|stats| stats.starter)
                .collect::<Vec<_>>(),
            (0..layers)
                .map(|layer_idx| (1 + domains - (layer_idx % domains)) % domains)
                .collect::<Vec<_>>()
        );
        for stats in &continuation.layer_stats {
            assert_eq!(stats.domains, domains);
            assert_eq!(stats.finisher, (stats.starter + domains - 1) % domains);
            assert_eq!(
                stats.visited_domains,
                (0..domains)
                    .map(|step| (stats.starter + step) % domains)
                    .collect::<Vec<_>>()
            );
            assert_eq!(stats.new_kv_positions_by_domain, [1, 3, 2]);
        }
        assert_eq!(continuation.logits_producer_domain, 1);
        assert_eq!(continuation.logits_projections, 1);

        assert_reserved_positioned_history(&distributed_shards, 0..13);
        assert_eq!(
            reserved_positioned_domain_totals(&distributed_shards),
            [52, 156, 104]
        );
        for (layer_idx, shards) in distributed_shards.iter().enumerate() {
            for (domain, shard) in shards.iter().enumerate() {
                assert_eq!(
                    shard.committed_len() - committed_before_continuation[layer_idx][domain],
                    continuation_schedule.counts()[domain]
                );
                assert_eq!(
                    shard.committed_len() + usize::from(domain == decode_assignees[1][layer_idx]),
                    reservation_plan[layer_idx][domain]
                );
                assert_eq!(
                    shard.storage_ptrs(),
                    initial_storage_ptrs[layer_idx][domain]
                );
            }
        }

        let post_continuation_token = sample_last_token(&continuation.logits);
        assert_eq!(
            post_continuation_token,
            sample_last_token(&reference_continuation_logits)
        );
        let post_continuation_ids = Tensor::from_slice(&[post_continuation_token]).unsqueeze(0);
        let distributed_post_continuation_hidden = Tensor::embedding(
            &distributed_model.embedding,
            &post_continuation_ids,
            -1,
            false,
            false,
        );
        let reference_post_continuation_hidden = Tensor::embedding(
            &reference_model.embedding,
            &post_continuation_ids,
            -1,
            false,
            false,
        );
        let committed_before_post_continuation_decode = distributed_shards
            .iter()
            .map(|shards| {
                shards
                    .iter()
                    .map(ReservedPositionedKvShard::committed_len)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let post_continuation_decode = run_reserved_positioned_decode(
            &mut distributed_model,
            &distributed_post_continuation_hidden,
            13,
            &mut distributed_shards,
            continuation.logits_producer_domain,
            &decode_assignees[1],
        );
        let post_continuation_positions = Tensor::from_slice(&[13_i64]).unsqueeze(0);
        let (reference_post_continuation_hidden, reference_post_continuation_logits) =
            run_contiguous_reference_block(
                &mut reference_model,
                &reference_post_continuation_hidden,
                &post_continuation_positions,
                &mut reference_caches,
            );
        assert_phase_matches(
            "post_stationary_continuation_decode",
            &post_continuation_decode.hidden_states,
            &post_continuation_decode.logits,
            &reference_post_continuation_hidden,
            &reference_post_continuation_logits,
        );
        assert_eq!(
            sample_last_token(&post_continuation_decode.logits),
            sample_last_token(&reference_post_continuation_logits)
        );
        assert!(reference_caches.iter().all(|cache| cache.seq_len() == 14));
        assert_eq!(
            post_continuation_decode
                .layer_stats
                .iter()
                .map(|stats| stats.hops)
                .sum::<usize>(),
            layers * (domains - 1)
        );
        assert_eq!(
            post_continuation_decode.logits_producer_domain,
            continuation.logits_producer_domain
        );

        assert_reserved_positioned_history(&distributed_shards, 0..14);
        assert_eq!(
            reserved_positioned_domain_totals(&distributed_shards),
            [56, 168, 112]
        );
        for (layer_idx, shards) in distributed_shards.iter().enumerate() {
            for (domain, shard) in shards.iter().enumerate() {
                assert_eq!(
                    shard.committed_len()
                        - committed_before_post_continuation_decode[layer_idx][domain],
                    usize::from(domain == decode_assignees[1][layer_idx])
                );
                assert_eq!(shard.committed_len(), reservation_plan[layer_idx][domain]);
                assert_eq!(
                    shard.storage_ptrs(),
                    initial_storage_ptrs[layer_idx][domain]
                );
            }
        }
    }

    #[test]
    fn twenty_four_layers_reuse_positioned_kv_across_prefill_decode_cycles() {
        let device = Device::Cpu;
        let domains = 3_usize;
        let layers = 24_usize;
        let mut config = test_config();
        config.num_layers = layers;
        let weights = deterministic_weights(&config, device);
        let mut distributed_model =
            LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap();
        let mut reference_model = local_reference_model(&config, &weights, device);
        let mut reference_caches = (0..layers)
            .map(|_| ContiguousKvCache::new())
            .collect::<Vec<_>>();

        let decode_schedule = FrozenKvAssigneeSchedule::new(&[1, 3, 2], 41, 2 * layers).unwrap();
        assert_eq!(decode_schedule.counts(), &[8, 24, 16]);
        let decode_assignees = (0..2)
            .map(|token_offset| {
                (0..layers)
                    .map(|layer_idx| {
                        decode_schedule
                            .assignee_for(token_offset, layer_idx, layers)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for assignees in &decode_assignees {
            let mut counts = vec![0_usize; domains];
            for &domain in assignees {
                counts[domain] += 1;
            }
            assert_eq!(counts, [4, 12, 8]);
        }
        let reservation_plan = (0..layers)
            .map(|layer_idx| {
                let mut capacities = vec![2_usize, 6, 4];
                for token_assignees in &decode_assignees {
                    capacities[token_assignees[layer_idx]] += 1;
                }
                assert_eq!(capacities.iter().sum::<usize>(), 14);
                capacities
            })
            .collect::<Vec<_>>();
        let mut distributed_shards =
            reserved_positioned_layer_shards(&config, &reservation_plan, device);
        let initial_storage_ptrs = distributed_shards
            .iter()
            .map(|shards| {
                shards
                    .iter()
                    .map(ReservedPositionedKvShard::storage_ptrs)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let prefill_1_ids = Tensor::from_slice(&[3_i64, 5, 7, 9, 11, 13]).unsqueeze(0);
        let prefill_1_positions = Tensor::arange(6, (Kind::Int64, device)).unsqueeze(0);
        let distributed_prefill_1_hidden = Tensor::embedding(
            &distributed_model.embedding,
            &prefill_1_ids,
            -1,
            false,
            false,
        );
        let reference_prefill_1_hidden =
            Tensor::embedding(&reference_model.embedding, &prefill_1_ids, -1, false, false);
        let (distributed_hidden, distributed_logits, prefill_1_projections) =
            run_reserved_positioned_prefill_block(
                &mut distributed_model,
                &distributed_prefill_1_hidden,
                &prefill_1_positions,
                &mut distributed_shards,
                &[1, 3, 2],
            );
        let (reference_hidden, reference_logits) = run_contiguous_reference_block(
            &mut reference_model,
            &reference_prefill_1_hidden,
            &prefill_1_positions,
            &mut reference_caches,
        );
        assert_eq!(prefill_1_projections, 6 * layers);
        assert!(reference_caches.iter().all(|cache| cache.seq_len() == 6));
        assert_phase_matches(
            "prefill_1",
            &distributed_hidden,
            &distributed_logits,
            &reference_hidden,
            &reference_logits,
        );
        assert_reserved_positioned_history(&distributed_shards, 0..6);

        let decode_1_token = sample_last_token(&distributed_logits);
        assert_eq!(decode_1_token, sample_last_token(&reference_logits));
        let decode_1_ids = Tensor::from_slice(&[decode_1_token]).unsqueeze(0);
        let distributed_decode_1_hidden = Tensor::embedding(
            &distributed_model.embedding,
            &decode_1_ids,
            -1,
            false,
            false,
        );
        let reference_decode_1_hidden =
            Tensor::embedding(&reference_model.embedding, &decode_1_ids, -1, false, false);
        let distributed_decode_1 = run_reserved_positioned_decode(
            &mut distributed_model,
            &distributed_decode_1_hidden,
            6,
            &mut distributed_shards,
            1,
            &decode_assignees[0],
        );
        let decode_1_positions = Tensor::from_slice(&[6_i64]).unsqueeze(0);
        let (reference_decode_1_hidden, reference_decode_1_logits) = run_contiguous_reference_block(
            &mut reference_model,
            &reference_decode_1_hidden,
            &decode_1_positions,
            &mut reference_caches,
        );
        assert_phase_matches(
            "decode_1",
            &distributed_decode_1.hidden_states,
            &distributed_decode_1.logits,
            &reference_decode_1_hidden,
            &reference_decode_1_logits,
        );
        assert_eq!(
            sample_last_token(&distributed_decode_1.logits),
            sample_last_token(&reference_decode_1_logits)
        );
        assert!(reference_caches.iter().all(|cache| cache.seq_len() == 7));
        assert_reserved_positioned_history(&distributed_shards, 0..7);

        let prefill_2_ids = Tensor::from_slice(&[17_i64, 19, 23, 29, 31, 37]).unsqueeze(0);
        let prefill_2_positions = (Tensor::arange(6, (Kind::Int64, device)) + 7).unsqueeze(0);
        let distributed_prefill_2_hidden = Tensor::embedding(
            &distributed_model.embedding,
            &prefill_2_ids,
            -1,
            false,
            false,
        );
        let reference_prefill_2_hidden =
            Tensor::embedding(&reference_model.embedding, &prefill_2_ids, -1, false, false);
        let (distributed_hidden, distributed_logits, prefill_2_projections) =
            run_reserved_positioned_prefill_block(
                &mut distributed_model,
                &distributed_prefill_2_hidden,
                &prefill_2_positions,
                &mut distributed_shards,
                &[1, 3, 2],
            );
        let (reference_hidden, reference_logits) = run_contiguous_reference_block(
            &mut reference_model,
            &reference_prefill_2_hidden,
            &prefill_2_positions,
            &mut reference_caches,
        );
        assert_eq!(prefill_2_projections, 6 * layers);
        assert!(reference_caches.iter().all(|cache| cache.seq_len() == 13));
        assert_phase_matches(
            "prefill_2",
            &distributed_hidden,
            &distributed_logits,
            &reference_hidden,
            &reference_logits,
        );
        assert_reserved_positioned_history(&distributed_shards, 0..13);

        let decode_2_token = sample_last_token(&distributed_logits);
        assert_eq!(decode_2_token, sample_last_token(&reference_logits));
        let decode_2_ids = Tensor::from_slice(&[decode_2_token]).unsqueeze(0);
        let distributed_decode_2_hidden = Tensor::embedding(
            &distributed_model.embedding,
            &decode_2_ids,
            -1,
            false,
            false,
        );
        let reference_decode_2_hidden =
            Tensor::embedding(&reference_model.embedding, &decode_2_ids, -1, false, false);
        let distributed_decode_2 = run_reserved_positioned_decode(
            &mut distributed_model,
            &distributed_decode_2_hidden,
            13,
            &mut distributed_shards,
            distributed_decode_1.logits_producer_domain,
            &decode_assignees[1],
        );
        let decode_2_positions = Tensor::from_slice(&[13_i64]).unsqueeze(0);
        let (reference_decode_2_hidden, reference_decode_2_logits) = run_contiguous_reference_block(
            &mut reference_model,
            &reference_decode_2_hidden,
            &decode_2_positions,
            &mut reference_caches,
        );
        assert_phase_matches(
            "decode_2",
            &distributed_decode_2.hidden_states,
            &distributed_decode_2.logits,
            &reference_decode_2_hidden,
            &reference_decode_2_logits,
        );
        assert_eq!(
            sample_last_token(&distributed_decode_2.logits),
            sample_last_token(&reference_decode_2_logits)
        );
        assert!(reference_caches.iter().all(|cache| cache.seq_len() == 14));
        assert_reserved_positioned_history(&distributed_shards, 0..14);
        assert_eq!(
            reserved_positioned_domain_totals(&distributed_shards),
            [56, 168, 112]
        );
        for (layer_idx, shards) in distributed_shards.iter().enumerate() {
            for (domain, shard) in shards.iter().enumerate() {
                assert_eq!(
                    shard.reserved_capacity(),
                    reservation_plan[layer_idx][domain]
                );
                assert_eq!(shard.committed_len(), reservation_plan[layer_idx][domain]);
                assert_eq!(
                    shard.storage_ptrs(),
                    initial_storage_ptrs[layer_idx][domain]
                );
            }
        }
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
