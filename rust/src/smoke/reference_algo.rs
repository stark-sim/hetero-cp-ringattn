use crate::error::RingError;
use crate::report::{BlockTrace, CaseConfig, DomainSpec, DomainTrace};

pub const PAYLOAD_CHUNK_QUERY_LEN: i32 = 4;

#[derive(Clone, Debug)]
pub struct Tensor3 {
    pub seq: usize,
    pub heads: usize,
    pub dim: usize,
    pub data: Vec<f64>,
}

impl Tensor3 {
    pub fn zeros(seq: usize, heads: usize, dim: usize) -> Self {
        Self {
            seq,
            heads,
            dim,
            data: vec![0.0; seq * heads * dim],
        }
    }

    pub fn random(seq: usize, heads: usize, dim: usize, rng: &mut Lcg) -> Self {
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

    pub fn get(&self, q: usize, h: usize, d: usize) -> f64 {
        self.data[((q * self.heads + h) * self.dim) + d]
    }

    pub fn set(&mut self, q: usize, h: usize, d: usize, value: f64) {
        self.data[((q * self.heads + h) * self.dim) + d] = value;
    }

    pub fn copy_seq_from(&mut self, dst_seq_offset: usize, src: &Tensor3) {
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
pub struct Lcg {
    state: u64,
    spare: Option<f64>,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    pub fn next_f64_open01(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = self.state >> 11;
        ((value as f64) + 0.5) / ((1_u64 << 53) as f64)
    }

    pub fn standard_normal(&mut self) -> f64 {
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

pub fn build_specs(config: &CaseConfig) -> Result<Vec<DomainSpec>, RingError> {
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

pub fn ring_source_order(target_index: usize, domain_count: usize) -> impl Iterator<Item = usize> {
    (0..domain_count).map(move |step| (target_index + step) % domain_count)
}

pub fn block_ranges(spec: &DomainSpec) -> impl Iterator<Item = (usize, usize)> + '_ {
    let start = spec.seq_offset;
    let stop = spec.seq_offset + spec.seq_chunk_len;
    (start..stop)
        .step_by(spec.block_size)
        .map(move |block_start| (block_start, usize::min(block_start + spec.block_size, stop)))
}

pub fn attention_scale(head_dim: usize) -> f64 {
    1.0 / (head_dim as f64).sqrt()
}

pub fn full_attention_reference(q: &Tensor3, k: &Tensor3, v: &Tensor3) -> Tensor3 {
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

pub fn ring_domain_output(
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
pub fn online_update_block(
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

pub fn ring_attention_model(
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
