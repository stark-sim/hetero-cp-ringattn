//! 【Ring Attention scheduling strategies】
//!
//! This module defines how a full prompt sequence is partitioned across HCP
//! domains.  All strategies preserve the mathematical equivalence of full
//! attention; they only change the physical layout of tokens so that load
//! balancing or communication properties improve.
//!
//! Strategies:
//! - `Vanilla`: contiguous capacity-aware chunks (the original HCP default).
//! - `Striped`: weighted round-robin permutation of token positions.
//! - `ZigZag`: each domain holds a prefix segment and a suffix segment so that
//!   every domain has both early and late positions.

/// Scheduling strategy for distributed ring attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RingSchedulingStrategy {
    /// Contiguous capacity-aware chunks.
    #[default]
    Vanilla,
    /// Weighted round-robin permutation (Brandon et al., Striped Attention).
    Striped,
    /// Prefix + suffix symmetric assignment (Zhu 2024 / Megatron-Core ZigZag).
    ZigZag,
}

impl RingSchedulingStrategy {
    /// All supported variants, useful for benchmarks.
    #[allow(dead_code)]
    pub fn all() -> &'static [RingSchedulingStrategy] {
        &[
            RingSchedulingStrategy::Vanilla,
            RingSchedulingStrategy::Striped,
            RingSchedulingStrategy::ZigZag,
        ]
    }

    /// Parse from a lowercase string used in CLI / configs.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "vanilla" => Some(RingSchedulingStrategy::Vanilla),
            "striped" => Some(RingSchedulingStrategy::Striped),
            "zigzag" => Some(RingSchedulingStrategy::ZigZag),
            _ => None,
        }
    }
}

/// Build the assignment of each global position to a domain.
///
/// Returns a vector `a` where `a[p]` is the domain id that owns global
/// position `p`.  The total number of positions assigned to domain `i` equals
/// `chunks[i]`.
pub fn build_assignment(chunks: &[usize], strategy: RingSchedulingStrategy) -> Vec<usize> {
    let seq_len: usize = chunks.iter().sum();
    let num_domains = chunks.len();
    match strategy {
        RingSchedulingStrategy::Vanilla => {
            let mut assign = Vec::with_capacity(seq_len);
            for (domain, &chunk) in chunks.iter().enumerate() {
                assign.extend(vec![domain; chunk]);
            }
            assign
        }
        RingSchedulingStrategy::Striped => {
            // Weighted round-robin: schedule one slot per domain until all
            // chunk quotas are consumed, then repeat the schedule.
            let mut sched = Vec::new();
            let mut remaining: Vec<_> = chunks.iter().cloned().enumerate().collect();
            while remaining.iter().any(|(_, c)| *c > 0) {
                for (domain, count) in remaining.iter_mut() {
                    if *count > 0 {
                        sched.push(*domain);
                        *count -= 1;
                    }
                }
            }
            (0..seq_len).map(|p| sched[p % sched.len()]).collect()
        }
        RingSchedulingStrategy::ZigZag => {
            // Capacity-aware zigzag: domain i receives a prefix of length
            // ceil(chunk_i / 2) and a suffix of length floor(chunk_i / 2).
            // Prefixes are placed in domain order; suffixes are placed in
            // reverse domain order.  This generalizes the classic 2N-segment
            // zigzag to uneven capacity-aware chunks.
            let mut prefix_lens = Vec::with_capacity(num_domains);
            let mut suffix_lens = Vec::with_capacity(num_domains);
            for &chunk in chunks {
                let half = chunk / 2;
                let rem = chunk % 2;
                prefix_lens.push(half + rem);
                suffix_lens.push(half);
            }
            let mut assignment = vec![0usize; seq_len];
            let mut pos = 0usize;
            for (domain, &len) in prefix_lens.iter().enumerate() {
                for _ in 0..len {
                    assignment[pos] = domain;
                    pos += 1;
                }
            }
            for (domain, &len) in suffix_lens.iter().enumerate().rev() {
                for _ in 0..len {
                    assignment[pos] = domain;
                    pos += 1;
                }
            }
            assignment
        }
    }
}

/// Given an assignment vector, return the ordered list of global positions
/// owned by each domain (in local storage order).
pub fn build_domain_positions(assignment: &[usize]) -> Vec<Vec<usize>> {
    let num_domains = assignment.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut positions: Vec<Vec<usize>> = vec![Vec::new(); num_domains];
    for (p, &domain) in assignment.iter().enumerate() {
        positions[domain].push(p);
    }
    positions
}

/// Build the inverse permutation used to reconstruct the original sequence
/// order from per-domain outputs concatenated in domain order.
///
/// For each global position `p`, `inverse[p]` is the index in the concatenated
/// `[output_0, output_1, ...]` tensor that corresponds to `p`.
#[allow(dead_code)]
pub fn build_inverse_perm(assignment: &[usize]) -> Vec<i64> {
    let positions = build_domain_positions(assignment);
    let num_domains = positions.len();
    let mut offsets = vec![0usize; num_domains + 1];
    for i in 0..num_domains {
        offsets[i + 1] = offsets[i] + positions[i].len();
    }
    let mut inverse = vec![0i64; assignment.len()];
    for domain in 0..num_domains {
        for (local_idx, &global_pos) in positions[domain].iter().enumerate() {
            inverse[global_pos] = (offsets[domain] + local_idx) as i64;
        }
    }
    inverse
}

#[cfg(feature = "tch-backend")]
/// Convert a list of global positions into a 1-D Int64 tensor on `device`.
#[allow(dead_code)]
pub fn position_ids_tensor(positions: &[usize], device: tch::Device) -> tch::Tensor {
    let ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
    tch::Tensor::from_slice(&ids).to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vanilla_assignment() {
        let chunks = vec![3, 1];
        let a = build_assignment(&chunks, RingSchedulingStrategy::Vanilla);
        assert_eq!(a, vec![0, 0, 0, 1]);
    }

    #[test]
    fn test_striped_assignment() {
        let chunks = vec![3, 1];
        let a = build_assignment(&chunks, RingSchedulingStrategy::Striped);
        assert_eq!(a, vec![0, 1, 0, 0]);
    }

    #[test]
    fn test_zigzag_assignment_uneven() {
        let chunks = vec![3, 1];
        let a = build_assignment(&chunks, RingSchedulingStrategy::ZigZag);
        // domain0 prefix=2, domain1 prefix=1, domain1 suffix=0, domain0 suffix=1
        assert_eq!(a, vec![0, 0, 1, 0]);
        let positions = build_domain_positions(&a);
        assert_eq!(positions[0], vec![0, 1, 3]);
        assert_eq!(positions[1], vec![2]);
    }

    #[test]
    fn test_zigzag_assignment_even() {
        // 2 domains, equal chunks of 4 -> classic zigzag
        let chunks = vec![4, 4];
        let a = build_assignment(&chunks, RingSchedulingStrategy::ZigZag);
        assert_eq!(a, vec![0, 0, 1, 1, 1, 1, 0, 0]);
        let positions = build_domain_positions(&a);
        assert_eq!(positions[0], vec![0, 1, 6, 7]);
        assert_eq!(positions[1], vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_inverse_perm() {
        let chunks = vec![3, 1];
        let a = build_assignment(&chunks, RingSchedulingStrategy::Striped);
        let inv = build_inverse_perm(&a);
        // positions[0] = [0,2,3], positions[1] = [1]
        // inverse[0]=0, inverse[2]=1, inverse[3]=2, inverse[1]=3
        assert_eq!(inv, vec![0, 3, 1, 2]);
    }
}
