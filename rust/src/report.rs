use serde::Serialize;
use crate::error::{Tolerance, ToleranceTier};
use crate::protocol::{CpRingNodeSmokeReport, ProtocolSmokeReport, RemoteCpNodeReport};

#[derive(Clone, Debug)]
pub struct DomainSpec {
    pub domain_id: String,
    pub seq_offset: usize,
    pub seq_chunk_len: usize,
    pub block_size: usize,
}

#[derive(Serialize)]
pub struct Report {
    pub status: &'static str,
    pub summary: Summary,
    pub protocol_smoke: ProtocolSmokeReport,
    pub cp_ring_smoke: CpRingNodeSmokeReport,
    pub cxx_bridge: CxxBridgeReport,
    pub torch_bridge: TorchBridgeReport,
    pub torch_attention_bridge: TorchBridgeReport,
    pub tch_attention_bridge: TorchBridgeReport,
    pub torch_block_update_bridge: TorchBlockUpdateReport,
    pub torch_payload_block_bridge: TorchPayloadBlockReport,
    pub torch_payload_online_bridge: TorchPayloadBlockReport,
    pub torch_payload_chunk_bridge: TorchPayloadBlockReport,
    pub torch_query_chunk_bridge: TorchPayloadBlockReport,
    pub torch_query_output_bridge: TorchQueryOutputReport,
    pub tch_payload_block_bridge: TorchPayloadBlockReport,
    pub tch_payload_online_bridge: TorchPayloadBlockReport,
    pub tch_payload_chunk_bridge: TorchPayloadBlockReport,
    pub tch_query_chunk_bridge: TorchPayloadBlockReport,
    pub tch_query_output_bridge: TorchQueryOutputReport,
    pub tch_compute_output_checksum: f64,
    pub cases: Vec<CaseReport>,
}

#[derive(Serialize)]
pub struct RemoteCpNodeRunReport {
    pub status: &'static str,
    pub cp_node: RemoteCpNodeReport,
    pub torch_payload_block_bridge: TorchPayloadBlockReport,
    pub torch_payload_online_bridge: TorchPayloadBlockReport,
    pub torch_payload_chunk_bridge: TorchPayloadBlockReport,
    pub torch_query_chunk_bridge: TorchPayloadBlockReport,
    pub torch_query_output_bridge: TorchQueryOutputReport,
    pub tch_payload_block_bridge: TorchPayloadBlockReport,
    pub tch_payload_online_bridge: TorchPayloadBlockReport,
    pub tch_payload_chunk_bridge: TorchPayloadBlockReport,
    pub tch_query_chunk_bridge: TorchPayloadBlockReport,
    pub tch_query_output_bridge: TorchQueryOutputReport,
    pub tch_compute_output_checksum: f64,
}

#[derive(Serialize)]
pub struct Summary {
    pub cases: usize,
    pub passed: usize,
    pub failed: usize,
}

#[derive(Serialize)]
pub struct CxxBridgeReport {
    pub status: &'static str,
    pub smoke_domains: i32,
}

#[derive(Serialize)]
pub struct TorchBridgeReport {
    pub status: &'static str,
    pub compiled: bool,
    pub requested_device: String,
    pub status_code: i32,
    pub note: String,
    pub message: String,
}

#[derive(Serialize)]
pub struct TorchBlockUpdateReport {
    pub status: &'static str,
    pub compiled: bool,
    pub requested_device: String,
    pub requested_updates: usize,
    pub status_code: i32,
    pub note: String,
    pub message: String,
}

#[derive(Serialize)]
pub struct TorchPayloadBlockReport {
    pub status: &'static str,
    pub compiled: bool,
    pub requested_device: String,
    pub requested_blocks: usize,
    pub processed_blocks: usize,
    pub status_code: i32,
    pub note: String,
    pub message: String,
}

#[derive(Serialize)]
pub struct TorchQueryOutputReport {
    pub status: &'static str,
    pub compiled: bool,
    pub requested_device: String,
    pub requested_blocks: usize,
    pub processed_blocks: usize,
    pub status_code: i32,
    pub note: String,
    pub message: String,
    pub output_groups: Vec<TorchQueryOutputGroup>,
}

#[derive(Serialize)]
pub struct TorchQueryOutputGroup {
    pub compute_domain: String,
    pub layer_index: i32,
    pub output_seq_offset: usize,
    pub query_len: usize,
    pub output_slot_values: usize,
    pub blocks: usize,
    pub output_values: usize,
    pub output_checksum: f64,
    pub max_abs_err: f64,
}

#[derive(Serialize)]
pub struct SeedResult {
    pub seed: u64,
    pub status: &'static str,
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub max_rel_err: f64,
}

#[derive(Serialize)]
pub struct CaseReport {
    pub name: &'static str,
    pub status: &'static str,
    pub seed: u64,
    pub tolerance_tier: ToleranceTier,
    pub tolerance: Tolerance,
    pub metrics: Metrics,
    pub config: CaseConfig,
    pub ring_trace_summary: Vec<DomainTrace>,
    pub seed_results: Vec<SeedResult>,
}

#[derive(Clone, Copy, Serialize)]
pub struct Metrics {
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub max_rel_err: f64,
}

#[derive(Clone, Serialize)]
pub struct CaseConfig {
    pub global_seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub domains: Vec<DomainConfig>,
}

#[derive(Clone, Serialize)]
pub struct DomainConfig {
    pub domain_id: String,
    pub seq_chunk_len: usize,
    pub block_size: usize,
}

#[derive(Clone, Serialize)]
pub struct DomainTrace {
    pub domain_id: String,
    pub seq_offset: usize,
    pub seq_chunk_len: usize,
    pub block_visits: usize,
    pub first_blocks: Vec<BlockTrace>,
}

#[derive(Clone, Serialize)]
pub struct BlockTrace {
    pub source_domain: String,
    pub block_start: usize,
    pub block_stop: usize,
    pub block_len: usize,
}
