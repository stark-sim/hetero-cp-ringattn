use crate::error::{RingError, Tolerance, ToleranceTier};
use crate::report::{CaseConfig, CaseReport, DomainConfig, DomainTrace, Metrics, SeedResult};
use crate::smoke::reference_algo::{full_attention_reference, ring_attention_model, Lcg, Tensor3};

pub fn run_case_single_seed(
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
    let (modeled, traces) = ring_attention_model(&q, &k, &v, config)?;
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

pub fn run_case(
    name: &'static str,
    config: CaseConfig,
    seeds: &[u64],
    tolerance_tier: ToleranceTier,
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
        tolerance_tier,
        tolerance: tol,
        metrics,
        config,
        ring_trace_summary: worst_traces,
        seed_results,
    })
}

pub fn case_config(chunks: &[usize], block_sizes: &[usize], heads: usize, dim: usize) -> CaseConfig {
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

pub fn default_cases() -> Vec<(&'static str, CaseConfig)> {
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
