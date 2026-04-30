mod protocol;
mod tch_backend;

use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use thiserror::Error;

const PAYLOAD_CHUNK_QUERY_LEN: i32 = 4;

#[derive(Debug, Error)]
enum RingError {
    #[error("sum(seq_chunk_len)={actual} does not match global_seq_len={expected}")]
    InvalidChunkSum { actual: usize, expected: usize },
    #[error("domain {domain_id} has invalid seq_chunk_len or block_size")]
    InvalidDomain { domain_id: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("protocol error: {0}")]
    Protocol(#[from] protocol::ProtocolError),
    #[error("invalid cli args: {0}")]
    InvalidCli(String),
}

#[derive(Clone, Debug)]
struct Tensor3 {
    seq: usize,
    heads: usize,
    dim: usize,
    data: Vec<f64>,
}

impl Tensor3 {
    fn zeros(seq: usize, heads: usize, dim: usize) -> Self {
        Self {
            seq,
            heads,
            dim,
            data: vec![0.0; seq * heads * dim],
        }
    }

    fn random(seq: usize, heads: usize, dim: usize, rng: &mut Lcg) -> Self {
        let mut data = Vec::with_capacity(seq * heads * dim);
        for _ in 0..data.capacity() {
            data.push(rng.standard_normal());
        }
        Self {
            seq,
            heads,
            dim,
            data,
        }
    }

    fn get(&self, q: usize, h: usize, d: usize) -> f64 {
        self.data[((q * self.heads + h) * self.dim) + d]
    }

    fn set(&mut self, q: usize, h: usize, d: usize, value: f64) {
        self.data[((q * self.heads + h) * self.dim) + d] = value;
    }

    fn copy_seq_from(&mut self, dst_seq_offset: usize, src: &Tensor3) {
        for q in 0..src.seq {
            for h in 0..src.heads {
                for d in 0..src.dim {
                    self.set(dst_seq_offset + q, h, d, src.get(q, h, d));
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct DomainSpec {
    domain_id: String,
    seq_offset: usize,
    seq_chunk_len: usize,
    block_size: usize,
}

#[derive(Serialize)]
struct Report {
    status: &'static str,
    summary: Summary,
    protocol_smoke: protocol::ProtocolSmokeReport,
    cp_ring_smoke: protocol::CpRingNodeSmokeReport,
    cxx_bridge: CxxBridgeReport,
    torch_bridge: TorchBridgeReport,
    torch_attention_bridge: TorchBridgeReport,
    tch_attention_bridge: TorchBridgeReport,
    torch_block_update_bridge: TorchBlockUpdateReport,
    torch_payload_block_bridge: TorchPayloadBlockReport,
    torch_payload_online_bridge: TorchPayloadBlockReport,
    torch_payload_chunk_bridge: TorchPayloadBlockReport,
    torch_query_chunk_bridge: TorchPayloadBlockReport,
    torch_query_output_bridge: TorchQueryOutputReport,
    tch_payload_block_bridge: TorchPayloadBlockReport,
    tch_payload_online_bridge: TorchPayloadBlockReport,
    tch_payload_chunk_bridge: TorchPayloadBlockReport,
    tch_query_chunk_bridge: TorchPayloadBlockReport,
    tch_query_output_bridge: TorchQueryOutputReport,
    tch_compute_output_checksum: f64,
    cases: Vec<CaseReport>,
}

#[derive(Serialize)]
struct RemoteCpNodeRunReport {
    status: &'static str,
    cp_node: protocol::RemoteCpNodeReport,
    torch_payload_block_bridge: TorchPayloadBlockReport,
    torch_payload_online_bridge: TorchPayloadBlockReport,
    torch_payload_chunk_bridge: TorchPayloadBlockReport,
    torch_query_chunk_bridge: TorchPayloadBlockReport,
    torch_query_output_bridge: TorchQueryOutputReport,
    tch_payload_block_bridge: TorchPayloadBlockReport,
    tch_payload_online_bridge: TorchPayloadBlockReport,
    tch_payload_chunk_bridge: TorchPayloadBlockReport,
    tch_query_chunk_bridge: TorchPayloadBlockReport,
    tch_query_output_bridge: TorchQueryOutputReport,
    tch_compute_output_checksum: f64,
}

#[derive(Serialize)]
struct Summary {
    cases: usize,
    passed: usize,
    failed: usize,
}

#[derive(Serialize)]
struct CxxBridgeReport {
    status: &'static str,
    smoke_domains: i32,
}

#[derive(Serialize)]
struct TorchBridgeReport {
    status: &'static str,
    compiled: bool,
    requested_device: String,
    status_code: i32,
    note: String,
    message: String,
}

#[derive(Serialize)]
struct TorchBlockUpdateReport {
    status: &'static str,
    compiled: bool,
    requested_device: String,
    requested_updates: usize,
    status_code: i32,
    note: String,
    message: String,
}

#[derive(Serialize)]
struct TorchPayloadBlockReport {
    status: &'static str,
    compiled: bool,
    requested_device: String,
    requested_blocks: usize,
    processed_blocks: usize,
    status_code: i32,
    note: String,
    message: String,
}

#[derive(Serialize)]
struct TorchQueryOutputReport {
    status: &'static str,
    compiled: bool,
    requested_device: String,
    requested_blocks: usize,
    processed_blocks: usize,
    status_code: i32,
    note: String,
    message: String,
    output_groups: Vec<TorchQueryOutputGroup>,
}

#[derive(Serialize)]
struct TorchQueryOutputGroup {
    compute_domain: String,
    layer_index: i32,
    output_seq_offset: usize,
    query_len: usize,
    output_slot_values: usize,
    blocks: usize,
    output_values: usize,
    output_checksum: f64,
    max_abs_err: f64,
}

#[derive(Serialize)]
struct SeedResult {
    seed: u64,
    status: &'static str,
    max_abs_err: f64,
    mean_abs_err: f64,
    max_rel_err: f64,
}

#[derive(Serialize)]
struct CaseReport {
    name: &'static str,
    status: &'static str,
    seed: u64,
    tolerance: Tolerance,
    metrics: Metrics,
    config: CaseConfig,
    ring_trace_summary: Vec<DomainTrace>,
    seed_results: Vec<SeedResult>,
}

#[derive(Clone, Copy, Serialize)]
struct Tolerance {
    max_abs_err: f64,
    mean_abs_err: f64,
    max_rel_err: f64,
}

#[derive(Clone, Copy, Serialize)]
struct Metrics {
    max_abs_err: f64,
    mean_abs_err: f64,
    max_rel_err: f64,
}

#[derive(Clone, Serialize)]
struct CaseConfig {
    global_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    domains: Vec<DomainConfig>,
}

#[derive(Clone, Serialize)]
struct DomainConfig {
    domain_id: String,
    seq_chunk_len: usize,
    block_size: usize,
}

#[derive(Clone, Serialize)]
struct DomainTrace {
    domain_id: String,
    seq_offset: usize,
    seq_chunk_len: usize,
    block_visits: usize,
    first_blocks: Vec<BlockTrace>,
}

#[derive(Clone, Serialize)]
struct BlockTrace {
    source_domain: String,
    block_start: usize,
    block_stop: usize,
    block_len: usize,
}

struct Lcg {
    state: u64,
    spare: Option<f64>,
}

#[derive(Debug)]
struct CliArgs {
    report_path: String,
    remote_p2p_role: Option<String>,
    node_index: Option<usize>,
    bind_addr: Option<String>,
    connect_addr: Option<String>,
    stress_test: bool,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_f64_open01(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = self.state >> 11;
        ((value as f64) + 0.5) / ((1_u64 << 53) as f64)
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

extern "C" {
    fn hcp_ringattn_cxx_smoke_domain_count() -> i32;
    fn hcp_ringattn_torch_smoke() -> i32;
    fn hcp_ringattn_torch_smoke_message() -> *const std::os::raw::c_char;
    fn hcp_ringattn_torch_attention_smoke() -> i32;
    fn hcp_ringattn_torch_attention_smoke_message() -> *const std::os::raw::c_char;
    fn hcp_ringattn_torch_block_update_smoke(block_updates: i32) -> i32;
    fn hcp_ringattn_torch_block_update_smoke_message() -> *const std::os::raw::c_char;
    fn hcp_ringattn_torch_payload_block_smoke(
        payload: *const std::os::raw::c_uchar,
        payload_len: usize,
        block_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    fn hcp_ringattn_torch_payload_block_smoke_message() -> *const std::os::raw::c_char;
    fn hcp_ringattn_torch_payload_online_smoke(
        payload: *const std::os::raw::c_uchar,
        payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    fn hcp_ringattn_torch_payload_online_smoke_message() -> *const std::os::raw::c_char;
    fn hcp_ringattn_torch_payload_chunk_smoke(
        payload: *const std::os::raw::c_uchar,
        payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    fn hcp_ringattn_torch_payload_chunk_smoke_message() -> *const std::os::raw::c_char;
    fn hcp_ringattn_torch_query_chunk_smoke(
        q_payload: *const std::os::raw::c_uchar,
        q_payload_len: usize,
        kv_payload: *const std::os::raw::c_uchar,
        kv_payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    fn hcp_ringattn_torch_query_chunk_smoke_message() -> *const std::os::raw::c_char;
    fn hcp_ringattn_torch_query_chunk_output_smoke(
        q_payload: *const std::os::raw::c_uchar,
        q_payload_len: usize,
        kv_payload: *const std::os::raw::c_uchar,
        kv_payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
        output_checksum: *mut f64,
        max_abs_err: *mut f64,
        output_values: *mut usize,
    ) -> i32;
    fn hcp_ringattn_torch_query_chunk_output_smoke_message() -> *const std::os::raw::c_char;
}

fn build_specs(config: &CaseConfig) -> Result<Vec<DomainSpec>, RingError> {
    let mut specs = Vec::with_capacity(config.domains.len());
    let mut offset = 0;
    for domain in &config.domains {
        if domain.seq_chunk_len == 0 || domain.block_size == 0 {
            return Err(RingError::InvalidDomain {
                domain_id: domain.domain_id.clone(),
            });
        }
        specs.push(DomainSpec {
            domain_id: domain.domain_id.clone(),
            seq_offset: offset,
            seq_chunk_len: domain.seq_chunk_len,
            block_size: domain.block_size,
        });
        offset += domain.seq_chunk_len;
    }
    if offset != config.global_seq_len {
        return Err(RingError::InvalidChunkSum {
            actual: offset,
            expected: config.global_seq_len,
        });
    }
    Ok(specs)
}

fn ring_source_order(target_index: usize, domain_count: usize) -> impl Iterator<Item = usize> {
    (0..domain_count).map(move |step| (target_index + step) % domain_count)
}

fn block_ranges(spec: &DomainSpec) -> impl Iterator<Item = (usize, usize)> + '_ {
    let start = spec.seq_offset;
    let stop = spec.seq_offset + spec.seq_chunk_len;
    (start..stop)
        .step_by(spec.block_size)
        .map(move |block_start| (block_start, usize::min(block_start + spec.block_size, stop)))
}

fn attention_scale(head_dim: usize) -> f64 {
    1.0 / (head_dim as f64).sqrt()
}

fn full_attention_reference(q: &Tensor3, k: &Tensor3, v: &Tensor3) -> Tensor3 {
    let mut out = Tensor3::zeros(q.seq, q.heads, q.dim);
    let scale = attention_scale(q.dim);
    for qi in 0..q.seq {
        for h in 0..q.heads {
            let mut max_score = f64::NEG_INFINITY;
            let mut scores = vec![0.0; k.seq];
            for (ki, score_slot) in scores.iter_mut().enumerate() {
                let mut score = 0.0;
                for d in 0..q.dim {
                    score += q.get(qi, h, d) * k.get(ki, h, d);
                }
                score *= scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }

            let mut denom = 0.0;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }

            for d in 0..q.dim {
                let mut value = 0.0;
                for (ki, weight) in scores.iter().enumerate() {
                    value += (weight / denom) * v.get(ki, h, d);
                }
                out.set(qi, h, d, value);
            }
        }
    }
    out
}

fn ring_domain_output(
    q: &Tensor3,
    k: &Tensor3,
    v: &Tensor3,
    specs: &[DomainSpec],
    target_index: usize,
) -> (Tensor3, Vec<BlockTrace>) {
    let target = &specs[target_index];
    let mut out = Tensor3::zeros(target.seq_chunk_len, q.heads, q.dim);
    let mut running_sum = vec![0.0; target.seq_chunk_len * q.heads];
    let mut running_max = vec![f64::NEG_INFINITY; target.seq_chunk_len * q.heads];
    let scale = attention_scale(q.dim);
    let mut trace = Vec::new();

    for source_index in ring_source_order(target_index, specs.len()) {
        let source = &specs[source_index];
        for (block_start, block_stop) in block_ranges(source) {
            online_update_block(
                q,
                k,
                v,
                target,
                block_start,
                block_stop,
                scale,
                &mut out,
                &mut running_sum,
                &mut running_max,
            );
            trace.push(BlockTrace {
                source_domain: source.domain_id.clone(),
                block_start,
                block_stop,
                block_len: block_stop - block_start,
            });
        }
    }

    (out, trace)
}

#[allow(clippy::too_many_arguments)]
fn online_update_block(
    q: &Tensor3,
    k: &Tensor3,
    v: &Tensor3,
    target: &DomainSpec,
    block_start: usize,
    block_stop: usize,
    scale: f64,
    out: &mut Tensor3,
    running_sum: &mut [f64],
    running_max: &mut [f64],
) {
    for local_q in 0..target.seq_chunk_len {
        let global_q = target.seq_offset + local_q;
        for h in 0..q.heads {
            let state_index = local_q * q.heads + h;
            let mut scores = Vec::with_capacity(block_stop - block_start);
            let mut local_max = f64::NEG_INFINITY;
            for ki in block_start..block_stop {
                let mut score = 0.0;
                for d in 0..q.dim {
                    score += q.get(global_q, h, d) * k.get(ki, h, d);
                }
                score *= scale;
                local_max = local_max.max(score);
                scores.push(score);
            }

            let mut local_sum = 0.0;
            let mut local_pv = vec![0.0; q.dim];
            for (offset, score) in scores.iter().enumerate() {
                let weight = (*score - local_max).exp();
                local_sum += weight;
                let ki = block_start + offset;
                for (d, pv) in local_pv.iter_mut().enumerate() {
                    *pv += weight * v.get(ki, h, d);
                }
            }

            let old_max = running_max[state_index];
            let old_sum = running_sum[state_index];
            let new_max = old_max.max(local_max);
            let exp_prev = (old_max - new_max).exp();
            let exp_local = (local_max - new_max).exp();
            let new_sum = exp_prev * old_sum + exp_local * local_sum;

            for (d, local_value) in local_pv.iter().enumerate() {
                let numerator =
                    exp_prev * old_sum * out.get(local_q, h, d) + exp_local * local_value;
                out.set(local_q, h, d, numerator / new_sum);
            }
            running_sum[state_index] = new_sum;
            running_max[state_index] = new_max;
        }
    }
}

fn ring_attention_model(
    q: &Tensor3,
    k: &Tensor3,
    v: &Tensor3,
    config: &CaseConfig,
) -> Result<(Tensor3, Vec<DomainTrace>), RingError> {
    let specs = build_specs(config)?;
    let mut output = Tensor3::zeros(config.global_seq_len, config.num_heads, config.head_dim);
    let mut traces = Vec::with_capacity(specs.len());

    for (target_index, spec) in specs.iter().enumerate() {
        let (domain_out, trace) = ring_domain_output(q, k, v, &specs, target_index);
        output.copy_seq_from(spec.seq_offset, &domain_out);
        traces.push(DomainTrace {
            domain_id: spec.domain_id.clone(),
            seq_offset: spec.seq_offset,
            seq_chunk_len: spec.seq_chunk_len,
            block_visits: trace.len(),
            first_blocks: trace.into_iter().take(4).collect(),
        });
    }

    Ok((output, traces))
}

fn run_case_single_seed(
    _name: &'static str,
    config: &CaseConfig,
    seed: u64,
    tol: Tolerance,
) -> Result<(Metrics, &'static str, Vec<DomainTrace>), RingError> {
    let mut rng = Lcg::new(seed);
    let q = Tensor3::random(
        config.global_seq_len,
        config.num_heads,
        config.head_dim,
        &mut rng,
    );
    let k = Tensor3::random(
        config.global_seq_len,
        config.num_heads,
        config.head_dim,
        &mut rng,
    );
    let v = Tensor3::random(
        config.global_seq_len,
        config.num_heads,
        config.head_dim,
        &mut rng,
    );

    let reference = full_attention_reference(&q, &k, &v);
    let (modeled, traces) = ring_attention_model(&q, &k, &v, &config)?;
    let mut max_abs_err = 0.0;
    let mut total_abs_err = 0.0;
    let mut max_rel_err = 0.0;
    for (lhs, rhs) in reference.data.iter().zip(modeled.data.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > max_abs_err {
            max_abs_err = diff;
        }
        total_abs_err += diff;
        let ref_mag = lhs.abs();
        if ref_mag > 1e-12 {
            let rel = diff / ref_mag;
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }
    let mean_abs_err = total_abs_err / reference.data.len() as f64;
    let status = if max_abs_err <= tol.max_abs_err
        && mean_abs_err <= tol.mean_abs_err
        && max_rel_err <= tol.max_rel_err
    {
        "pass"
    } else {
        "fail"
    };

    Ok((
        Metrics {
            max_abs_err,
            mean_abs_err,
            max_rel_err,
        },
        status,
        traces,
    ))
}

fn run_case(
    name: &'static str,
    config: CaseConfig,
    seeds: &[u64],
    tol: Tolerance,
) -> Result<CaseReport, RingError> {
    let mut seed_results = Vec::with_capacity(seeds.len());
    let mut worst_status = "pass";
    let mut worst_metrics = None;
    let mut worst_traces = Vec::new();
    let mut first_seed = 0;

    for (i, &seed) in seeds.iter().enumerate() {
        let (metrics, status, traces) = run_case_single_seed(name, &config, seed, tol)?;
        seed_results.push(SeedResult {
            seed,
            status,
            max_abs_err: metrics.max_abs_err,
            mean_abs_err: metrics.mean_abs_err,
            max_rel_err: metrics.max_rel_err,
        });
        if status == "fail" || worst_metrics.is_none() {
            worst_status = status;
            worst_metrics = Some(metrics);
            worst_traces = traces.clone();
            first_seed = seed;
        }
        if i == 0 {
            first_seed = seed;
            worst_metrics = Some(metrics);
            worst_traces = traces.clone();
        }
    }

    let metrics = worst_metrics.unwrap_or(Metrics {
        max_abs_err: 0.0,
        mean_abs_err: 0.0,
        max_rel_err: 0.0,
    });

    Ok(CaseReport {
        name,
        status: worst_status,
        seed: first_seed,
        tolerance: tol,
        metrics,
        config,
        ring_trace_summary: worst_traces,
        seed_results,
    })
}

fn case_config(chunks: &[usize], block_sizes: &[usize], heads: usize, dim: usize) -> CaseConfig {
    let global_seq_len = chunks.iter().sum();
    let domains = chunks
        .iter()
        .zip(block_sizes.iter())
        .enumerate()
        .map(|(index, (&seq_chunk_len, &block_size))| DomainConfig {
            domain_id: format!("domain-{index}"),
            seq_chunk_len,
            block_size,
        })
        .collect();
    CaseConfig {
        global_seq_len,
        num_heads: heads,
        head_dim: dim,
        domains,
    }
}

fn default_cases() -> Vec<(&'static str, CaseConfig)> {
    vec![
        (
            "2domain_uneven_chunks",
            case_config(&[80, 48], &[16, 12], 4, 16),
        ),
        (
            "3domain_uneven_blocks",
            case_config(&[64, 40, 56], &[32, 10, 14], 3, 24),
        ),
        (
            "4domain_small_tail_blocks",
            case_config(&[32, 64, 48, 48], &[7, 16, 11, 13], 2, 32),
        ),
        (
            "3domain_large_seq",
            case_config(&[512, 256, 256], &[128, 64, 64], 8, 64),
        ),
        (
            "1domain_single_block",
            case_config(&[64], &[64], 4, 16),
        ),
        (
            "2domain_unit_blocks",
            case_config(&[16, 16], &[1, 1], 2, 8),
        ),
        (
            "1domain_medium",
            case_config(&[128], &[32], 4, 16),
        ),
    ]
}

fn parse_cli_args() -> Result<CliArgs, RingError> {
    let mut args = env::args().skip(1);
    let mut report_path = String::from("reports/rust_ringattn_correctness.json");
    let mut remote_p2p_role = None;
    let mut node_index = None;
    let mut bind_addr = None;
    let mut connect_addr = None;
    let mut stress_test = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--report-path" => {
                report_path = next_cli_value(&mut args, "--report-path")?;
            }
            "--remote-p2p-role" => {
                remote_p2p_role = Some(next_cli_value(&mut args, "--remote-p2p-role")?);
            }
            "--node-index" => {
                let value = next_cli_value(&mut args, "--node-index")?;
                node_index = Some(value.parse::<usize>().map_err(|error| {
                    RingError::InvalidCli(format!("invalid --node-index: {error}"))
                })?);
            }
            "--bind" => {
                bind_addr = Some(next_cli_value(&mut args, "--bind")?);
            }
            "--connect" => {
                connect_addr = Some(next_cli_value(&mut args, "--connect")?);
            }
            "--stress-test" => {
                stress_test = true;
            }
            _ => {
                return Err(RingError::InvalidCli(format!("unknown argument {arg}")));
            }
        }
    }
    Ok(CliArgs {
        report_path,
        remote_p2p_role,
        node_index,
        bind_addr,
        connect_addr,
        stress_test,
    })
}

fn next_cli_value(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, RingError> {
    args.next()
        .filter(|value| !value.starts_with("--"))
        .ok_or_else(|| RingError::InvalidCli(format!("missing value for {flag}")))
}

fn torch_device_success_code(requested_device: &str) -> Option<i32> {
    match requested_device {
        "cpu" => Some(1),
        "mps" => Some(2),
        "cuda" => Some(3),
        _ => requested_device
            .strip_prefix("cuda:")
            .filter(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
            .map(|_| 3),
    }
}

fn torch_bridge_enabled_by_env() -> bool {
    env::var("HCP_ENABLE_TORCH").ok().as_deref() == Some("1")
}

fn compact_message(message: &str, max_chars: usize) -> String {
    let one_line = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max_chars {
        return one_line;
    }
    let mut compact = one_line.chars().take(max_chars).collect::<String>();
    compact.push_str("...");
    compact
}

fn c_string_from_ptr(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

fn torch_bridge_report() -> TorchBridgeReport {
    let code = unsafe { hcp_ringattn_torch_smoke() };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_smoke_message() });
    torch_report_from_code(
        code,
        message,
        "C++ libtorch bridge executed on CPU",
        "C++ libtorch bridge executed on MPS",
        "C++ libtorch bridge executed on CUDA",
        "C++ libtorch bridge is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch bridge failed or returned an unexpected status",
    )
}

fn torch_attention_bridge_report() -> TorchBridgeReport {
    let code = unsafe { hcp_ringattn_torch_attention_smoke() };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_attention_smoke_message() });
    torch_report_from_code(
        code,
        message,
        "C++ libtorch attention smoke executed on CPU",
        "C++ libtorch attention smoke executed on MPS",
        "C++ libtorch attention smoke executed on CUDA",
        "C++ libtorch attention smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch attention smoke failed or returned an unexpected status",
    )
}

fn tch_attention_bridge_report() -> TorchBridgeReport {
    if !cfg!(feature = "tch-backend") {
        return TorchBridgeReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            status_code: 0,
            note: "tch-rs attention smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    match tch_backend::backend::run_attention_block_updates(1) {
        Ok((code, message, _max_rel_err)) => {
            let device_name = env::var("HCP_TCH_DEVICE")
                .or_else(|_| env::var("HCP_TORCH_DEVICE"))
                .unwrap_or_else(|_| "cpu".to_string());
            let (note, status) = match code {
                1 => ("tch-rs attention smoke executed on CPU".to_string(), "pass"),
                2 => ("tch-rs attention smoke executed on MPS".to_string(), "pass"),
                3 => ("tch-rs attention smoke executed on CUDA".to_string(), "pass"),
                _ => ("tch-rs attention smoke returned unexpected status".to_string(), "fail"),
            };
            TorchBridgeReport {
                status,
                compiled: true,
                requested_device: device_name,
                status_code: code,
                note,
                message,
            }
        }
        Err(e) => {
            let device_name = env::var("HCP_TCH_DEVICE")
                .or_else(|_| env::var("HCP_TORCH_DEVICE"))
                .unwrap_or_else(|_| "cpu".to_string());
            TorchBridgeReport {
                status: "fail",
                compiled: true,
                requested_device: device_name,
                status_code: -1,
                note: "tch-rs attention smoke failed".to_string(),
                message: e,
            }
        }
    }
}

fn tch_device_name() -> String {
    env::var("HCP_TCH_DEVICE")
        .or_else(|_| env::var("HCP_TORCH_DEVICE"))
        .unwrap_or_else(|_| "cpu".to_string())
}

fn tch_status_note_from_code(code: i32, base_note: &str) -> (&'static str, String) {
    match code {
        1 => ("pass", format!("{base_note} executed on CPU")),
        2 => ("pass", format!("{base_note} executed on MPS")),
        3 => ("pass", format!("{base_note} executed on CUDA")),
        _ => ("fail", format!("{base_note} returned unexpected status")),
    }
}

fn tch_payload_block_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs payload block smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = if blocks.is_empty() {
        "no CP payload blocks captured".to_string()
    } else {
        String::new()
    };
    for block in blocks {
        let block_len = i32::try_from(block.block_len()).unwrap_or(i32::MAX);
        let num_heads = i32::try_from(block.num_heads()).unwrap_or(i32::MAX);
        let head_dim = i32::try_from(block.head_dim()).unwrap_or(i32::MAX);
        match tch_backend::backend::run_payload_block_smoke(block.payload(), block_len, num_heads, head_dim) {
            Ok((c, msg, _max_rel_err)) => {
                code = c;
                message = msg;
            }
            Err(e) => {
                code = -1;
                message = e;
            }
        }
        let (status, _) = tch_status_note_from_code(code, "tch-rs payload block smoke");
        if status == "fail" {
            message = format!("sequence_id={} {message}", block.sequence_id());
            break;
        }
        processed_blocks += 1;
    }
    let (status, note) = tch_status_note_from_code(code, "tch-rs payload block smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks,
        status_code: code,
        note,
        message,
    }
}

fn tch_payload_online_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs payload online smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload online smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload online smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
    let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
    let (code, message) = match tch_backend::backend::run_payload_online_smoke(&payload, &block_lens, num_heads_i32, head_dim_i32) {
        Ok((c, msg, _max_rel_err)) => (c, msg),
        Err(e) => (-1, e),
    };
    let (status, note) = tch_status_note_from_code(code, "tch-rs payload online smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks: if status == "pass" { blocks.len() } else { 0 },
        status_code: code,
        note,
        message,
    }
}

fn tch_payload_chunk_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs payload chunk smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload chunk smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
    let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
    let query_len = i32::try_from(first_block.query_len()).unwrap_or(i32::MAX);
    let (code, message) = match tch_backend::backend::run_payload_chunk_smoke(&payload, &block_lens, query_len, num_heads_i32, head_dim_i32) {
        Ok((c, msg, _max_rel_err)) => (c, msg),
        Err(e) => (-1, e),
    };
    let (status, note) = tch_status_note_from_code(code, "tch-rs payload chunk smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks: if status == "pass" { blocks.len() } else { 0 },
        status_code: code,
        note,
        message,
    }
}

fn tch_query_chunk_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs query chunk smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim || block.query_len() != query_len) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query chunk smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent query_len/num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut groups: std::collections::BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = std::collections::BTreeMap::new();
    for block in blocks {
        groups.entry(block.compute_domain()).or_default().push(block);
    }
    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else { continue; };
        if group_blocks.iter().any(|block| block.query_payload() != group_first.query_payload()) {
            code = -6;
            message = format!("inconsistent query payload for compute_domain={compute_domain}");
            break;
        }
        let mut kv_payload = Vec::new();
        let mut block_lens = Vec::with_capacity(group_blocks.len());
        for block in &group_blocks {
            kv_payload.extend_from_slice(block.payload());
            block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
        }
        let query_len_i32 = i32::try_from(query_len).unwrap_or(i32::MAX);
        let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
        let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
        match tch_backend::backend::run_query_chunk_smoke(
            group_first.query_payload(),
            &kv_payload,
            &block_lens,
            query_len_i32,
            num_heads_i32,
            head_dim_i32,
        ) {
            Ok((c, msg, ..)) => {
                code = c;
                message = msg;
            }
            Err(e) => {
                code = -1;
                message = e;
            }
        }
        let (status, _) = tch_status_note_from_code(code, "tch-rs query chunk smoke");
        if status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
        processed_blocks += group_blocks.len();
    }
    let (status, note) = tch_status_note_from_code(code, "tch-rs query chunk smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks,
        status_code: code,
        note,
        message,
    }
}

fn tch_query_output_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchQueryOutputReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchQueryOutputReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs query output smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
            output_groups: Vec::new(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query output smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
            output_groups: Vec::new(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim || block.query_len() != query_len) {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query output smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent query_len/num_heads/head_dim across CP payload blocks".to_string(),
            output_groups: Vec::new(),
        };
    }
    let mut groups: std::collections::BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = std::collections::BTreeMap::new();
    for block in blocks {
        groups.entry(block.compute_domain()).or_default().push(block);
    }
    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    let mut output_groups = Vec::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else { continue; };
        if group_blocks.iter().any(|block| block.query_payload() != group_first.query_payload()) {
            code = -6;
            message = format!("inconsistent query payload for compute_domain={compute_domain}");
            break;
        }
        let mut kv_payload = Vec::new();
        let mut block_lens = Vec::with_capacity(group_blocks.len());
        for block in &group_blocks {
            kv_payload.extend_from_slice(block.payload());
            block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
        }
        let query_len_i32 = i32::try_from(query_len).unwrap_or(i32::MAX);
        let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
        let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
        match tch_backend::backend::run_query_chunk_smoke(
            group_first.query_payload(),
            &kv_payload,
            &block_lens,
            query_len_i32,
            num_heads_i32,
            head_dim_i32,
        ) {
            Ok((c, msg, output_checksum, max_abs_err, _max_rel_err, output_values)) => {
                code = c;
                message = msg;
                let expected_output_values = query_len * num_heads * head_dim;
                if output_values != expected_output_values {
                    code = -6;
                    message = format!(
                        "compute_domain={compute_domain} output values mismatch expected={expected_output_values} actual={output_values}"
                    );
                    break;
                }
                output_groups.push(TorchQueryOutputGroup {
                    compute_domain: compute_domain.to_string(),
                    layer_index: group_first.layer_index(),
                    output_seq_offset: group_first.output_seq_offset(),
                    query_len: group_first.query_len(),
                    output_slot_values: group_first.output_slot_values(),
                    blocks: group_blocks.len(),
                    output_values,
                    output_checksum,
                    max_abs_err,
                });
            }
            Err(e) => {
                code = -1;
                message = e;
            }
        }
        let (status, _) = tch_status_note_from_code(code, "tch-rs query output smoke");
        if status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
        processed_blocks += group_blocks.len();
    }
    let (status, note) = tch_status_note_from_code(code, "tch-rs query output smoke");
    TorchQueryOutputReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks,
        status_code: code,
        note,
        message,
        output_groups,
    }
}

fn torch_block_update_bridge_report(requested_updates: usize) -> TorchBlockUpdateReport {
    let block_updates = i32::try_from(requested_updates).unwrap_or(i32::MAX);
    let code = unsafe { hcp_ringattn_torch_block_update_smoke(block_updates) };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_block_update_smoke_message() });
    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch CP block update smoke executed on CPU",
        "C++ libtorch CP block update smoke executed on MPS",
        "C++ libtorch CP block update smoke executed on CUDA",
        "C++ libtorch CP block update smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch CP block update smoke failed or returned an unexpected status",
    );
    TorchBlockUpdateReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_updates,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}

fn torch_payload_block_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch payload block smoke executed on CPU",
            "C++ libtorch payload block smoke executed on MPS",
            "C++ libtorch payload block smoke executed on CUDA",
            "C++ libtorch payload block smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload block smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
        };
    }

    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = if blocks.is_empty() {
        "no CP payload blocks captured".to_string()
    } else {
        String::new()
    };
    for block in blocks {
        let block_len = i32::try_from(block.block_len()).unwrap_or(i32::MAX);
        let num_heads = i32::try_from(block.num_heads()).unwrap_or(i32::MAX);
        let head_dim = i32::try_from(block.head_dim()).unwrap_or(i32::MAX);
        code = unsafe {
            hcp_ringattn_torch_payload_block_smoke(
                block.payload().as_ptr(),
                block.payload().len(),
                block_len,
                num_heads,
                head_dim,
            )
        };
        message = c_string_from_ptr(unsafe { hcp_ringattn_torch_payload_block_smoke_message() });
        let block_report = torch_report_from_code(
            code,
            message.clone(),
            "C++ libtorch payload block smoke executed on CPU",
            "C++ libtorch payload block smoke executed on MPS",
            "C++ libtorch payload block smoke executed on CUDA",
            "C++ libtorch payload block smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload block smoke failed or returned an unexpected status",
        );
        if block_report.status == "fail" {
            message = format!("sequence_id={} {}", block.sequence_id(), message);
            break;
        }
        processed_blocks += 1;
    }

    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch payload block smoke executed on CPU",
        "C++ libtorch payload block smoke executed on MPS",
        "C++ libtorch payload block smoke executed on CUDA",
        "C++ libtorch payload block smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch payload block smoke failed or returned an unexpected status",
    );
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}

fn torch_payload_online_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch payload online smoke executed on CPU",
            "C++ libtorch payload online smoke executed on MPS",
            "C++ libtorch payload online smoke executed on CUDA",
            "C++ libtorch payload online smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload online smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload online smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks
        .iter()
        .any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim)
    {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload online smoke received inconsistent tensor shapes"
                .to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let code = unsafe {
        hcp_ringattn_torch_payload_online_smoke(
            payload.as_ptr(),
            payload.len(),
            block_lens.as_ptr(),
            block_lens.len(),
            i32::try_from(num_heads).unwrap_or(i32::MAX),
            i32::try_from(head_dim).unwrap_or(i32::MAX),
        )
    };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_payload_online_smoke_message() });
    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch payload online smoke executed on CPU",
        "C++ libtorch payload online smoke executed on MPS",
        "C++ libtorch payload online smoke executed on CUDA",
        "C++ libtorch payload online smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch payload online smoke failed or returned an unexpected status",
    );
    let processed_blocks = if report.status == "pass" {
        requested_blocks
    } else {
        0
    };
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}

fn torch_payload_chunk_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch payload chunk smoke executed on CPU",
            "C++ libtorch payload chunk smoke executed on MPS",
            "C++ libtorch payload chunk smoke executed on CUDA",
            "C++ libtorch payload chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload chunk smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks
        .iter()
        .any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim)
    {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload chunk smoke received inconsistent tensor shapes"
                .to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let code = unsafe {
        hcp_ringattn_torch_payload_chunk_smoke(
            payload.as_ptr(),
            payload.len(),
            block_lens.as_ptr(),
            block_lens.len(),
            PAYLOAD_CHUNK_QUERY_LEN,
            i32::try_from(num_heads).unwrap_or(i32::MAX),
            i32::try_from(head_dim).unwrap_or(i32::MAX),
        )
    };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_payload_chunk_smoke_message() });
    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch payload chunk smoke executed on CPU",
        "C++ libtorch payload chunk smoke executed on MPS",
        "C++ libtorch payload chunk smoke executed on CUDA",
        "C++ libtorch payload chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch payload chunk smoke failed or returned an unexpected status",
    );
    let processed_blocks = if report.status == "pass" {
        requested_blocks
    } else {
        0
    };
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}

fn torch_query_chunk_bridge_report(blocks: &[protocol::CpPayloadBlock]) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch query chunk smoke executed on CPU",
            "C++ libtorch query chunk smoke executed on MPS",
            "C++ libtorch query chunk smoke executed on CUDA",
            "C++ libtorch query chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query chunk smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks
        .iter()
        .any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim)
    {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query chunk smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    if blocks.iter().any(|block| block.query_len() != query_len) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query chunk smoke received inconsistent query shapes".to_string(),
            message: "inconsistent query_len across CP payload blocks".to_string(),
        };
    }

    let mut groups: BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = BTreeMap::new();
    for block in blocks {
        groups
            .entry(block.compute_domain())
            .or_default()
            .push(block);
    }

    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else {
            continue;
        };
        if group_blocks
            .iter()
            .any(|block| block.query_payload() != group_first.query_payload())
        {
            code = -6;
            message = format!("inconsistent query payload for compute_domain={compute_domain}");
            break;
        }
        if group_blocks.iter().any(|block| {
            block.layer_index() != group_first.layer_index()
                || block.output_seq_offset() != group_first.output_seq_offset()
                || block.output_slot_values() != group_first.output_slot_values()
        }) {
            code = -6;
            message = format!("inconsistent output slot for compute_domain={compute_domain}");
            break;
        }

        let mut kv_payload = Vec::new();
        let mut block_lens = Vec::with_capacity(group_blocks.len());
        for block in &group_blocks {
            kv_payload.extend_from_slice(block.payload());
            block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
        }
        code = unsafe {
            hcp_ringattn_torch_query_chunk_smoke(
                group_first.query_payload().as_ptr(),
                group_first.query_payload().len(),
                kv_payload.as_ptr(),
                kv_payload.len(),
                block_lens.as_ptr(),
                block_lens.len(),
                i32::try_from(group_first.query_len()).unwrap_or(i32::MAX),
                i32::try_from(num_heads).unwrap_or(i32::MAX),
                i32::try_from(head_dim).unwrap_or(i32::MAX),
            )
        };
        message = c_string_from_ptr(unsafe { hcp_ringattn_torch_query_chunk_smoke_message() });
        let group_report = torch_report_from_code(
            code,
            message.clone(),
            "C++ libtorch query chunk smoke executed on CPU",
            "C++ libtorch query chunk smoke executed on MPS",
            "C++ libtorch query chunk smoke executed on CUDA",
            "C++ libtorch query chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query chunk smoke failed or returned an unexpected status",
        );
        if group_report.status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
        processed_blocks += group_blocks.len();
    }

    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch query chunk smoke executed on CPU",
        "C++ libtorch query chunk smoke executed on MPS",
        "C++ libtorch query chunk smoke executed on CUDA",
        "C++ libtorch query chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch query chunk smoke failed or returned an unexpected status",
    );
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}

fn torch_query_output_bridge_report(blocks: &[protocol::CpPayloadBlock]) -> TorchQueryOutputReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch query output smoke executed on CPU",
            "C++ libtorch query output smoke executed on MPS",
            "C++ libtorch query output smoke executed on CUDA",
            "C++ libtorch query output smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query output smoke failed or returned an unexpected status",
        );
        return TorchQueryOutputReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
            output_groups: Vec::new(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query output smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
            output_groups: Vec::new(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks.iter().any(|block| {
        block.num_heads() != num_heads
            || block.head_dim() != head_dim
            || block.query_len() != query_len
    }) {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query output smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent query_len/num_heads/head_dim across CP payload blocks"
                .to_string(),
            output_groups: Vec::new(),
        };
    }

    let mut groups: BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = BTreeMap::new();
    for block in blocks {
        groups
            .entry(block.compute_domain())
            .or_default()
            .push(block);
    }

    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    let mut output_groups = Vec::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else {
            continue;
        };
        if group_blocks
            .iter()
            .any(|block| block.query_payload() != group_first.query_payload())
        {
            code = -6;
            message = format!("inconsistent query payload for compute_domain={compute_domain}");
            break;
        }

        let mut kv_payload = Vec::new();
        let mut block_lens = Vec::with_capacity(group_blocks.len());
        for block in &group_blocks {
            kv_payload.extend_from_slice(block.payload());
            block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
        }

        let mut output_checksum = 0.0_f64;
        let mut max_abs_err = 0.0_f64;
        let mut output_values = 0_usize;
        code = unsafe {
            hcp_ringattn_torch_query_chunk_output_smoke(
                group_first.query_payload().as_ptr(),
                group_first.query_payload().len(),
                kv_payload.as_ptr(),
                kv_payload.len(),
                block_lens.as_ptr(),
                block_lens.len(),
                i32::try_from(group_first.query_len()).unwrap_or(i32::MAX),
                i32::try_from(num_heads).unwrap_or(i32::MAX),
                i32::try_from(head_dim).unwrap_or(i32::MAX),
                &mut output_checksum,
                &mut max_abs_err,
                &mut output_values,
            )
        };
        message =
            c_string_from_ptr(unsafe { hcp_ringattn_torch_query_chunk_output_smoke_message() });
        let group_report = torch_report_from_code(
            code,
            message.clone(),
            "C++ libtorch query output smoke executed on CPU",
            "C++ libtorch query output smoke executed on MPS",
            "C++ libtorch query output smoke executed on CUDA",
            "C++ libtorch query output smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query output smoke failed or returned an unexpected status",
        );
        if group_report.status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
        let expected_output_values = query_len * num_heads * head_dim;
        if output_values != expected_output_values {
            code = -6;
            message = format!(
                "compute_domain={compute_domain} output values mismatch expected={expected_output_values} actual={output_values}"
            );
            break;
        }
        output_groups.push(TorchQueryOutputGroup {
            compute_domain: compute_domain.to_string(),
            layer_index: group_first.layer_index(),
            output_seq_offset: group_first.output_seq_offset(),
            query_len: group_first.query_len(),
            output_slot_values: group_first.output_slot_values(),
            blocks: group_blocks.len(),
            output_values,
            output_checksum,
            max_abs_err,
        });
        processed_blocks += group_blocks.len();
    }

    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch query output smoke executed on CPU",
        "C++ libtorch query output smoke executed on MPS",
        "C++ libtorch query output smoke executed on CUDA",
        "C++ libtorch query output smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch query output smoke failed or returned an unexpected status",
    );
    TorchQueryOutputReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
        output_groups,
    }
}

fn torch_report_from_code(
    code: i32,
    message: String,
    cpu_note: &'static str,
    mps_note: &'static str,
    cuda_note: &'static str,
    disabled_note: &'static str,
    generic_fail_note: &'static str,
) -> TorchBridgeReport {
    let requested_device = env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let expected_code = torch_device_success_code(&requested_device);
    let requested_is_cuda = expected_code == Some(3);
    let status = match (
        cfg!(hcp_torch_enabled),
        torch_bridge_enabled_by_env(),
        expected_code,
    ) {
        (false, false, _) => "skipped",
        (false, true, _) => "fail",
        (true, _, Some(expected)) if code == expected => "pass",
        (true, _, _) => "fail",
    };
    let note = match (
        cfg!(hcp_torch_enabled),
        requested_device.as_str(),
        requested_is_cuda,
        code,
    ) {
        (false, _, _, _) => disabled_note.to_string(),
        (true, "cpu", _, 1) => cpu_note.to_string(),
        (true, "mps", _, 2) => mps_note.to_string(),
        (true, _, true, 3) => cuda_note.to_string(),
        (true, _, _, -4) => {
            "Unsupported HCP_TORCH_DEVICE; expected cpu, mps, cuda, or cuda:N".to_string()
        }
        (true, _, true, -5) => {
            "CUDA backend is unavailable in the current libtorch process".to_string()
        }
        (true, _, _, _) => generic_fail_note.to_string(),
    };
    TorchBridgeReport {
        status,
        compiled: cfg!(hcp_torch_enabled),
        requested_device,
        status_code: code,
        note,
        message,
    }
}

fn run(stress_test: bool) -> Result<Report, RingError> {
    let tol = Tolerance {
        max_abs_err: 1e-10,
        mean_abs_err: 1e-12,
        max_rel_err: 1e-7,
    };
    let stress_seeds: Vec<u64> = (0..5).map(|i| 42 + i as u64).collect();
    let cases: Vec<CaseReport> = default_cases()
        .into_iter()
        .enumerate()
        .map(|(index, (name, config))| {
            let seeds = if stress_test && config.global_seq_len <= 256 {
                stress_seeds.clone()
            } else {
                vec![42 + index as u64]
            };
            run_case(name, config, &seeds, tol)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let passed = cases.iter().filter(|case| case.status == "pass").count();
    let failed = cases.len() - passed;
    let protocol_smoke = protocol::run_protocol_smoke()?;
    let cp_ring_smoke = protocol::run_cp_ring_node_smoke()?;
    let tch_compute_output_checksum = cp_ring_smoke.compute_output_checksum();
    let torch_bridge = torch_bridge_report();
    let torch_attention_bridge = torch_attention_bridge_report();
    let tch_attention_bridge = tch_attention_bridge_report();
    let torch_block_update_bridge =
        torch_block_update_bridge_report(cp_ring_smoke.compute_updates());
    let torch_payload_block_bridge =
        torch_payload_block_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_payload_online_bridge =
        torch_payload_online_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_payload_chunk_bridge =
        torch_payload_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_query_chunk_bridge = torch_query_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_query_output_bridge =
        torch_query_output_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_payload_block_bridge =
        tch_payload_block_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_payload_online_bridge =
        tch_payload_online_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_payload_chunk_bridge =
        tch_payload_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_query_chunk_bridge =
        tch_query_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_query_output_bridge =
        tch_query_output_bridge_report(cp_ring_smoke.payload_blocks());
    let status = if failed == 0
        && protocol_smoke.status == "pass"
        && cp_ring_smoke.status == "pass"
        && torch_bridge.status != "fail"
        && torch_attention_bridge.status != "fail"
        && tch_attention_bridge.status != "fail"
        && torch_block_update_bridge.status != "fail"
        && torch_payload_block_bridge.status != "fail"
        && torch_payload_online_bridge.status != "fail"
        && torch_payload_chunk_bridge.status != "fail"
        && torch_query_chunk_bridge.status != "fail"
        && torch_query_output_bridge.status != "fail"
        && tch_payload_block_bridge.status != "fail"
        && tch_payload_online_bridge.status != "fail"
        && tch_payload_chunk_bridge.status != "fail"
        && tch_query_chunk_bridge.status != "fail"
        && tch_query_output_bridge.status != "fail"
    {
        "pass"
    } else {
        "fail"
    };
    Ok(Report {
        status,
        summary: Summary {
            cases: cases.len(),
            passed,
            failed,
        },
        protocol_smoke,
        cp_ring_smoke,
        tch_compute_output_checksum,
        cxx_bridge: CxxBridgeReport {
            status: "ok",
            smoke_domains: unsafe { hcp_ringattn_cxx_smoke_domain_count() },
        },
        torch_bridge,
        torch_attention_bridge,
        tch_attention_bridge,
        torch_block_update_bridge,
        torch_payload_block_bridge,
        torch_payload_online_bridge,
        torch_payload_chunk_bridge,
        torch_query_chunk_bridge,
        torch_query_output_bridge,
        tch_payload_block_bridge,
        tch_payload_online_bridge,
        tch_payload_chunk_bridge,
        tch_query_chunk_bridge,
        tch_query_output_bridge,
        cases,
    })
}

fn write_json_report<T: Serialize>(report_path: &str, report: &T) -> Result<(), RingError> {
    if let Some(parent) = Path::new(report_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(report_path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

fn run_remote_p2p(args: &CliArgs) -> Result<protocol::RemoteP2pReport, RingError> {
    match args.remote_p2p_role.as_deref() {
        Some("server") => {
            let bind_addr = args.bind_addr.as_deref().unwrap_or("0.0.0.0:29172");
            Ok(protocol::run_remote_p2p_server(bind_addr)?)
        }
        Some("client") => {
            let connect_addr = args.connect_addr.as_deref().ok_or_else(|| {
                RingError::InvalidCli("--connect is required for remote client".to_string())
            })?;
            Ok(protocol::run_remote_p2p_client(connect_addr)?)
        }
        Some(role) => Err(RingError::InvalidCli(format!(
            "unsupported --remote-p2p-role {role}; expected server, client, or cp-node"
        ))),
        None => Err(RingError::InvalidCli(
            "--remote-p2p-role is required for remote mode".to_string(),
        )),
    }
}

fn run_remote_cp_node(args: &CliArgs) -> Result<protocol::RemoteCpNodeReport, RingError> {
    let node_index = args
        .node_index
        .ok_or_else(|| RingError::InvalidCli("--node-index is required for cp-node".to_string()))?;
    let bind_addr = args
        .bind_addr
        .as_deref()
        .ok_or_else(|| RingError::InvalidCli("--bind is required for cp-node".to_string()))?;
    let connect_addr = args
        .connect_addr
        .as_deref()
        .ok_or_else(|| RingError::InvalidCli("--connect is required for cp-node".to_string()))?;
    Ok(protocol::run_remote_cp_node(
        node_index,
        bind_addr,
        connect_addr,
    )?)
}

fn main() -> Result<(), RingError> {
    let args = parse_cli_args()?;
    if args.remote_p2p_role.as_deref() == Some("cp-node") {
        let cp_node = run_remote_cp_node(&args)?;
        let torch_payload_block_bridge =
            torch_payload_block_bridge_report(cp_node.payload_blocks());
        let torch_payload_online_bridge =
            torch_payload_online_bridge_report(cp_node.payload_blocks());
        let torch_payload_chunk_bridge =
            torch_payload_chunk_bridge_report(cp_node.payload_blocks());
        let torch_query_chunk_bridge = torch_query_chunk_bridge_report(cp_node.payload_blocks());
        let torch_query_output_bridge = torch_query_output_bridge_report(cp_node.payload_blocks());
        let tch_payload_block_bridge =
            tch_payload_block_bridge_report(cp_node.payload_blocks());
        let tch_payload_online_bridge =
            tch_payload_online_bridge_report(cp_node.payload_blocks());
        let tch_payload_chunk_bridge =
            tch_payload_chunk_bridge_report(cp_node.payload_blocks());
        let tch_query_chunk_bridge =
            tch_query_chunk_bridge_report(cp_node.payload_blocks());
        let tch_query_output_bridge =
            tch_query_output_bridge_report(cp_node.payload_blocks());
        let tch_compute_output_checksum = cp_node.compute_output_checksum();
        let status = if cp_node.status == "pass"
            && torch_payload_block_bridge.status != "fail"
            && torch_payload_online_bridge.status != "fail"
            && torch_payload_chunk_bridge.status != "fail"
            && torch_query_chunk_bridge.status != "fail"
            && torch_query_output_bridge.status != "fail"
            && tch_payload_block_bridge.status != "fail"
            && tch_payload_online_bridge.status != "fail"
            && tch_payload_chunk_bridge.status != "fail"
            && tch_query_chunk_bridge.status != "fail"
            && tch_query_output_bridge.status != "fail"
        {
            "pass"
        } else {
            "fail"
        };
        let report = RemoteCpNodeRunReport {
            status,
            cp_node,
            torch_payload_block_bridge,
            torch_payload_online_bridge,
            torch_payload_chunk_bridge,
            torch_query_chunk_bridge,
            torch_query_output_bridge,
            tch_payload_block_bridge,
            tch_payload_online_bridge,
            tch_payload_chunk_bridge,
            tch_query_chunk_bridge,
            tch_query_output_bridge,
            tch_compute_output_checksum,
        };
        write_json_report(&args.report_path, &report)?;
        println!(
            "[rust-remote-cp-node] status={} role={} transport={} sent={} received={} compute_updates={} torch_payload_block_status={} torch_payload_block_code={} torch_payload_blocks={}/{} torch_payload_online_status={} torch_payload_online_code={} torch_payload_online_blocks={}/{} torch_payload_chunk_status={} torch_payload_chunk_code={} torch_payload_chunk_blocks={}/{} torch_query_chunk_status={} torch_query_chunk_code={} torch_query_chunk_blocks={}/{} torch_query_output_status={} torch_query_output_code={} torch_query_output_groups={} torch_query_output_blocks={}/{} tch_payload_block_status={} tch_payload_block_code={} tch_payload_block_blocks={}/{} tch_payload_online_status={} tch_payload_online_code={} tch_payload_online_blocks={}/{} tch_payload_chunk_status={} tch_payload_chunk_code={} tch_payload_chunk_blocks={}/{} tch_query_chunk_status={} tch_query_chunk_code={} tch_query_chunk_blocks={}/{} tch_query_output_status={} tch_query_output_code={} tch_query_output_groups={} tch_query_output_blocks={}/{} tch_compute_output_checksum={} report={}",
            report.status,
            report.cp_node.role(),
            report.cp_node.transport(),
            report.cp_node.messages_sent(),
            report.cp_node.messages_received(),
            report.cp_node.compute_updates(),
            report.torch_payload_block_bridge.status,
            report.torch_payload_block_bridge.status_code,
            report.torch_payload_block_bridge.processed_blocks,
            report.torch_payload_block_bridge.requested_blocks,
            report.torch_payload_online_bridge.status,
            report.torch_payload_online_bridge.status_code,
            report.torch_payload_online_bridge.processed_blocks,
            report.torch_payload_online_bridge.requested_blocks,
            report.torch_payload_chunk_bridge.status,
            report.torch_payload_chunk_bridge.status_code,
            report.torch_payload_chunk_bridge.processed_blocks,
            report.torch_payload_chunk_bridge.requested_blocks,
            report.torch_query_chunk_bridge.status,
            report.torch_query_chunk_bridge.status_code,
            report.torch_query_chunk_bridge.processed_blocks,
            report.torch_query_chunk_bridge.requested_blocks,
            report.torch_query_output_bridge.status,
            report.torch_query_output_bridge.status_code,
            report.torch_query_output_bridge.output_groups.len(),
            report.torch_query_output_bridge.processed_blocks,
            report.torch_query_output_bridge.requested_blocks,
            report.tch_payload_block_bridge.status,
            report.tch_payload_block_bridge.status_code,
            report.tch_payload_block_bridge.processed_blocks,
            report.tch_payload_block_bridge.requested_blocks,
            report.tch_payload_online_bridge.status,
            report.tch_payload_online_bridge.status_code,
            report.tch_payload_online_bridge.processed_blocks,
            report.tch_payload_online_bridge.requested_blocks,
            report.tch_payload_chunk_bridge.status,
            report.tch_payload_chunk_bridge.status_code,
            report.tch_payload_chunk_bridge.processed_blocks,
            report.tch_payload_chunk_bridge.requested_blocks,
            report.tch_query_chunk_bridge.status,
            report.tch_query_chunk_bridge.status_code,
            report.tch_query_chunk_bridge.processed_blocks,
            report.tch_query_chunk_bridge.requested_blocks,
            report.tch_query_output_bridge.status,
            report.tch_query_output_bridge.status_code,
            report.tch_query_output_bridge.output_groups.len(),
            report.tch_query_output_bridge.processed_blocks,
            report.tch_query_output_bridge.requested_blocks,
            report.tch_compute_output_checksum,
            args.report_path
        );
        if report.torch_payload_block_bridge.status == "fail"
            && !report.torch_payload_block_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_payload_block_message={}",
                compact_message(&report.torch_payload_block_bridge.message, 360)
            );
        }
        if report.torch_payload_online_bridge.status == "fail"
            && !report.torch_payload_online_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_payload_online_message={}",
                compact_message(&report.torch_payload_online_bridge.message, 360)
            );
        }
        if report.torch_payload_chunk_bridge.status == "fail"
            && !report.torch_payload_chunk_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_payload_chunk_message={}",
                compact_message(&report.torch_payload_chunk_bridge.message, 360)
            );
        }
        if report.torch_query_chunk_bridge.status == "fail"
            && !report.torch_query_chunk_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_query_chunk_message={}",
                compact_message(&report.torch_query_chunk_bridge.message, 360)
            );
        }
        if report.torch_query_output_bridge.status == "fail"
            && !report.torch_query_output_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_query_output_message={}",
                compact_message(&report.torch_query_output_bridge.message, 360)
            );
        }
        if report.status == "pass" {
            return Ok(());
        }
        std::process::exit(1);
    }
    if args.remote_p2p_role.is_some() {
        let report = run_remote_p2p(&args)?;
        write_json_report(&args.report_path, &report)?;
        println!(
            "[rust-remote-p2p] status={} role={} transport={} sent={} received={} report={}",
            report.status,
            report.role(),
            report.transport(),
            report.messages_sent(),
            report.messages_received(),
            args.report_path
        );
        return Ok(());
    }

    let report = run(args.stress_test)?;
    write_json_report(&args.report_path, &report)?;
    println!(
        "[rust-ringattn] status={} passed={}/{} protocol_status={} protocol_messages={} cp_ring_status={} cp_ring_messages={} cp_ring_compute_updates={} cxx_domains={} torch_status={} torch_device={} torch_code={} torch_attention_status={} torch_attention_code={} tch_attention_status={} tch_attention_code={} torch_block_update_status={} torch_block_update_code={} torch_block_updates={} torch_payload_block_status={} torch_payload_block_code={} torch_payload_blocks={}/{} torch_payload_online_status={} torch_payload_online_code={} torch_payload_online_blocks={}/{} torch_payload_chunk_status={} torch_payload_chunk_code={} torch_payload_chunk_blocks={}/{} torch_query_chunk_status={} torch_query_chunk_code={} torch_query_chunk_blocks={}/{} torch_query_output_status={} torch_query_output_code={} torch_query_output_groups={} torch_query_output_blocks={}/{} tch_payload_block_status={} tch_payload_block_code={} tch_payload_block_blocks={}/{} tch_payload_online_status={} tch_payload_online_code={} tch_payload_online_blocks={}/{} tch_payload_chunk_status={} tch_payload_chunk_code={} tch_payload_chunk_blocks={}/{} tch_query_chunk_status={} tch_query_chunk_code={} tch_query_chunk_blocks={}/{} tch_query_output_status={} tch_query_output_code={} tch_query_output_groups={} tch_query_output_blocks={}/{} tch_compute_output_checksum={} torch_compiled={} report={}",
        report.status,
        report.summary.passed,
        report.summary.cases,
        report.protocol_smoke.status,
        report.protocol_smoke.messages_sent(),
        report.cp_ring_smoke.status,
        report.cp_ring_smoke.messages_sent(),
        report.cp_ring_smoke.compute_updates(),
        report.cxx_bridge.smoke_domains,
        report.torch_bridge.status,
        report.torch_bridge.requested_device,
        report.torch_bridge.status_code,
        report.torch_attention_bridge.status,
        report.torch_attention_bridge.status_code,
        report.tch_attention_bridge.status,
        report.tch_attention_bridge.status_code,
        report.torch_block_update_bridge.status,
        report.torch_block_update_bridge.status_code,
        report.torch_block_update_bridge.requested_updates,
        report.torch_payload_block_bridge.status,
        report.torch_payload_block_bridge.status_code,
        report.torch_payload_block_bridge.processed_blocks,
        report.torch_payload_block_bridge.requested_blocks,
        report.torch_payload_online_bridge.status,
        report.torch_payload_online_bridge.status_code,
        report.torch_payload_online_bridge.processed_blocks,
        report.torch_payload_online_bridge.requested_blocks,
        report.torch_payload_chunk_bridge.status,
        report.torch_payload_chunk_bridge.status_code,
        report.torch_payload_chunk_bridge.processed_blocks,
        report.torch_payload_chunk_bridge.requested_blocks,
        report.torch_query_chunk_bridge.status,
        report.torch_query_chunk_bridge.status_code,
        report.torch_query_chunk_bridge.processed_blocks,
        report.torch_query_chunk_bridge.requested_blocks,
        report.torch_query_output_bridge.status,
        report.torch_query_output_bridge.status_code,
        report.torch_query_output_bridge.output_groups.len(),
        report.torch_query_output_bridge.processed_blocks,
        report.torch_query_output_bridge.requested_blocks,
        report.tch_payload_block_bridge.status,
        report.tch_payload_block_bridge.status_code,
        report.tch_payload_block_bridge.processed_blocks,
        report.tch_payload_block_bridge.requested_blocks,
        report.tch_payload_online_bridge.status,
        report.tch_payload_online_bridge.status_code,
        report.tch_payload_online_bridge.processed_blocks,
        report.tch_payload_online_bridge.requested_blocks,
        report.tch_payload_chunk_bridge.status,
        report.tch_payload_chunk_bridge.status_code,
        report.tch_payload_chunk_bridge.processed_blocks,
        report.tch_payload_chunk_bridge.requested_blocks,
        report.tch_query_chunk_bridge.status,
        report.tch_query_chunk_bridge.status_code,
        report.tch_query_chunk_bridge.processed_blocks,
        report.tch_query_chunk_bridge.requested_blocks,
        report.tch_query_output_bridge.status,
        report.tch_query_output_bridge.status_code,
        report.tch_query_output_bridge.output_groups.len(),
        report.tch_query_output_bridge.processed_blocks,
        report.tch_query_output_bridge.requested_blocks,
        report.tch_compute_output_checksum,
        report.torch_bridge.compiled,
        args.report_path
    );
    if report.torch_bridge.status == "fail" && !report.torch_bridge.message.is_empty() {
        println!(
            "[rust-ringattn] torch_message={}",
            compact_message(&report.torch_bridge.message, 360)
        );
    }
    if report.torch_attention_bridge.status == "fail"
        && !report.torch_attention_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_attention_message={}",
            compact_message(&report.torch_attention_bridge.message, 360)
        );
    }
    if report.tch_attention_bridge.status == "fail"
        && !report.tch_attention_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_attention_message={}",
            compact_message(&report.tch_attention_bridge.message, 360)
        );
    }
    if report.torch_block_update_bridge.status == "fail"
        && !report.torch_block_update_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_block_update_message={}",
            compact_message(&report.torch_block_update_bridge.message, 360)
        );
    }
    if report.torch_payload_block_bridge.status == "fail"
        && !report.torch_payload_block_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_payload_block_message={}",
            compact_message(&report.torch_payload_block_bridge.message, 360)
        );
    }
    if report.torch_payload_online_bridge.status == "fail"
        && !report.torch_payload_online_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_payload_online_message={}",
            compact_message(&report.torch_payload_online_bridge.message, 360)
        );
    }
    if report.torch_payload_chunk_bridge.status == "fail"
        && !report.torch_payload_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_payload_chunk_message={}",
            compact_message(&report.torch_payload_chunk_bridge.message, 360)
        );
    }
    if report.torch_query_chunk_bridge.status == "fail"
        && !report.torch_query_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_query_chunk_message={}",
            compact_message(&report.torch_query_chunk_bridge.message, 360)
        );
    }
    if report.torch_query_output_bridge.status == "fail"
        && !report.torch_query_output_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_query_output_message={}",
            compact_message(&report.torch_query_output_bridge.message, 360)
        );
    }
    if report.tch_payload_block_bridge.status == "fail"
        && !report.tch_payload_block_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_payload_block_message={}",
            compact_message(&report.tch_payload_block_bridge.message, 360)
        );
    }
    if report.tch_payload_online_bridge.status == "fail"
        && !report.tch_payload_online_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_payload_online_message={}",
            compact_message(&report.tch_payload_online_bridge.message, 360)
        );
    }
    if report.tch_payload_chunk_bridge.status == "fail"
        && !report.tch_payload_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_payload_chunk_message={}",
            compact_message(&report.tch_payload_chunk_bridge.message, 360)
        );
    }
    if report.tch_query_chunk_bridge.status == "fail"
        && !report.tch_query_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_query_chunk_message={}",
            compact_message(&report.tch_query_chunk_bridge.message, 360)
        );
    }
    if report.tch_query_output_bridge.status == "fail"
        && !report.tch_query_output_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_query_output_message={}",
            compact_message(&report.tch_query_output_bridge.message, 360)
        );
    }
    if report.status == "pass" {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
