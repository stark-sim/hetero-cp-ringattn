use serde::Serialize;
use std::env;
use std::fs;
use std::path::Path;
use thiserror::Error;

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
    cxx_bridge: CxxBridgeReport,
    torch_bridge: TorchBridgeReport,
    cases: Vec<CaseReport>,
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
    compiled: bool,
    requested_device: String,
    status_code: i32,
    note: String,
    message: String,
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
}

#[derive(Clone, Copy, Serialize)]
struct Tolerance {
    max_abs_err: f64,
    mean_abs_err: f64,
}

#[derive(Serialize)]
struct Metrics {
    max_abs_err: f64,
    mean_abs_err: f64,
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

#[derive(Serialize)]
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

fn run_case(
    name: &'static str,
    config: CaseConfig,
    seed: u64,
    tol: Tolerance,
) -> Result<CaseReport, RingError> {
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
    for (lhs, rhs) in reference.data.iter().zip(modeled.data.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > max_abs_err {
            max_abs_err = diff;
        }
        total_abs_err += diff;
    }
    let mean_abs_err = total_abs_err / reference.data.len() as f64;
    let status = if max_abs_err <= tol.max_abs_err && mean_abs_err <= tol.mean_abs_err {
        "pass"
    } else {
        "fail"
    };

    Ok(CaseReport {
        name,
        status,
        seed,
        tolerance: tol,
        metrics: Metrics {
            max_abs_err,
            mean_abs_err,
        },
        config,
        ring_trace_summary: traces,
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
    ]
}

fn parse_report_path() -> String {
    let mut args = env::args().skip(1);
    let mut report_path = String::from("reports/rust_ringattn_correctness.json");
    while let Some(arg) = args.next() {
        if arg == "--report-path" {
            if let Some(value) = args.next() {
                report_path = value;
            }
        }
    }
    report_path
}

fn torch_bridge_report() -> TorchBridgeReport {
    let code = unsafe { hcp_ringattn_torch_smoke() };
    let message = unsafe {
        let ptr = hcp_ringattn_torch_smoke_message();
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    let requested_device = env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let requested_is_cuda =
        requested_device == "cuda" || requested_device.strip_prefix("cuda:").is_some();
    let note = match (
        cfg!(hcp_torch_enabled),
        requested_device.as_str(),
        requested_is_cuda,
        code,
    ) {
        (false, _, _, _) => {
            "C++ libtorch bridge is disabled; build with HCP_ENABLE_TORCH=1".to_string()
        }
        (true, "cpu", _, 1) => "C++ libtorch bridge executed on CPU".to_string(),
        (true, "mps", _, 2) => "C++ libtorch bridge executed on MPS".to_string(),
        (true, _, true, 3) => "C++ libtorch bridge executed on CUDA".to_string(),
        (true, _, _, -4) => {
            "Unsupported HCP_TORCH_DEVICE; expected cpu, mps, cuda, or cuda:N".to_string()
        }
        (true, _, _, _) => {
            "C++ libtorch bridge failed or returned an unexpected status".to_string()
        }
    };
    TorchBridgeReport {
        compiled: cfg!(hcp_torch_enabled),
        requested_device,
        status_code: code,
        note,
        message,
    }
}

fn run() -> Result<Report, RingError> {
    let tol = Tolerance {
        max_abs_err: 1e-10,
        mean_abs_err: 1e-12,
    };
    let cases: Vec<CaseReport> = default_cases()
        .into_iter()
        .enumerate()
        .map(|(index, (name, config))| run_case(name, config, 42 + index as u64, tol))
        .collect::<Result<Vec<_>, _>>()?;
    let passed = cases.iter().filter(|case| case.status == "pass").count();
    let failed = cases.len() - passed;
    Ok(Report {
        status: if failed == 0 { "pass" } else { "fail" },
        summary: Summary {
            cases: cases.len(),
            passed,
            failed,
        },
        cxx_bridge: CxxBridgeReport {
            status: "ok",
            smoke_domains: unsafe { hcp_ringattn_cxx_smoke_domain_count() },
        },
        torch_bridge: torch_bridge_report(),
        cases,
    })
}

fn main() -> Result<(), RingError> {
    let report_path = parse_report_path();
    let report = run()?;
    if let Some(parent) = Path::new(&report_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!(
        "[rust-ringattn] status={} passed={}/{} cxx_domains={} torch_compiled={} report={}",
        report.status,
        report.summary.passed,
        report.summary.cases,
        report.cxx_bridge.smoke_domains,
        report.torch_bridge.compiled,
        report_path
    );
    if report.status == "pass" {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
