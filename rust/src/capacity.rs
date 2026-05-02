//! Cross-platform device capacity query and chunk-size allocation.
//!
//! This module provides:
//! - `query_device_capacity_mb`: Heuristic free-memory estimation per device type.
//! - `allocate_by_capacity`: Integer proportional allocation (largest-remainder method).
//!
//! # Design rationale
//!
//! - **Memory is the hard constraint** (OOM fails immediately); throughput can be
//!   mitigated by scheduling.  Therefore Phase 2 uses memory as the capacity proxy.
//! - **No new cargo dependencies**: We query CUDA via `nvidia-smi` subprocess,
//!   system RAM via `/proc/meminfo` (Linux) or `sysctl` (macOS).
//! - **Conservative heuristics** for MPS (unified memory) and CPU (slower compute).

use tch::Device;

/// Query an approximate free-memory capacity for the given device, in megabytes.
///
/// The value is a heuristic:
/// - **CUDA**: free VRAM reported by `nvidia-smi` (subprocess). Falls back to
///   `u64::MAX` if unavailable, which causes the coordinator to treat the device
///   as infinite-capacity and fall back to even sharding.
/// - **MPS (macOS)**: total physical RAM / 2. Unified memory is shared with the
///   OS, so we use a conservative fraction.
/// - **CPU**: available system RAM / 4. CPU inference is much slower, so we
///   deliberately under-weight RAM to avoid overloading CPU domains.
pub fn query_device_capacity_mb(device: Device) -> u64 {
    match device {
        Device::Cuda(_) => query_cuda_free_memory_mb().unwrap_or(u64::MAX),
        Device::Mps => query_total_ram_mb() / 2,
        Device::Cpu => query_available_ram_mb() / 4,
        _ => query_available_ram_mb() / 4,
    }
}

/// Allocate prompt tokens across domains proportionally to their capacities.
///
/// Uses the **largest-remainder method** to guarantee:
/// - `sum(chunks) == prompt_len`
/// - `chunks[i] >= 1` when `prompt_len >= num_domains`
/// - Monotonicity: if `c_a > c_b` then `chunk_a >= chunk_b` (before remainder tie-break)
///
/// # Arguments
/// - `prompt_len`: total number of tokens to distribute.
/// - `capacities`: free-memory heuristic per domain (in MB, or any proportional unit).
///
/// # Panics
/// - If `capacities` is empty.
/// - If `prompt_len < capacities.len()` (cannot give each domain at least 1 token).
pub fn allocate_by_capacity(prompt_len: usize, capacities: &[u64]) -> Vec<usize> {
    let n = capacities.len();
    assert!(n > 0, "capacities must not be empty");
    assert!(
        prompt_len >= n,
        "prompt_len ({}) must be >= num_domains ({}) to give each domain at least 1 token",
        prompt_len, n
    );

    let total: u128 = capacities.iter().map(|&c| c as u128).sum();

    // Fallback: even distribution when all capacities are zero or query failed.
    if total == 0 {
        let base = prompt_len / n;
        let rem = prompt_len % n;
        return (0..n).map(|i| base + if i < rem { 1 } else { 0 }).collect();
    }

    // 1. Compute base chunks (floor of proportional allocation).
    let mut chunks: Vec<usize> = capacities
        .iter()
        .map(|&c| ((prompt_len as u128 * c as u128) / total) as usize)
        .collect();

    // 2. Ensure every domain gets at least 1 token (ring attention invariant).
    let mut tokens_consumed = 0usize;
    for c in chunks.iter_mut() {
        if *c == 0 {
            *c = 1;
        }
        tokens_consumed = tokens_consumed.saturating_add(*c);
    }

    // 3. Handle deficit (should not normally happen when prompt_len >= n,
    //    but guard against overflow due to the +1 adjustments above).
    if tokens_consumed > prompt_len {
        let mut excess = tokens_consumed - prompt_len;
        while excess > 0 {
            let max_idx = chunks
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap();
            if chunks[max_idx] > 1 {
                chunks[max_idx] -= 1;
                excess -= 1;
            } else {
                // All at minimum; this path is theoretically unreachable
                // when prompt_len >= n, but break to avoid infinite loop.
                break;
            }
        }
    }

    // 4. Distribute remaining tokens by largest fractional remainder.
    let distributed: usize = chunks.iter().sum();
    let remainder = prompt_len.saturating_sub(distributed);

    if remainder > 0 {
        let mut fractions: Vec<(usize, f64)> = capacities
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let exact = prompt_len as f64 * c as f64 / total as f64;
                (i, exact - chunks[i] as f64)
            })
            .collect();
        // Sort descending by fractional remainder.
        fractions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, _) in fractions.iter().take(remainder) {
            chunks[*i] += 1;
        }
    }

    // Final sanity check (debug-only; can be removed in release).
    debug_assert_eq!(
        chunks.iter().sum::<usize>(),
        prompt_len,
        "allocate_by_capacity invariant violated: sum({:?}) != {}",
        chunks,
        prompt_len
    );

    chunks
}

// ---------------------------------------------------------------------------
// Platform-specific memory queries
// ---------------------------------------------------------------------------

/// Query CUDA free memory via `nvidia-smi` subprocess.
fn query_cuda_free_memory_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    s.trim().parse::<u64>().ok()
}

/// Query available system RAM in megabytes.
fn query_available_ram_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    let kb: u64 = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    return kb / 1024;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        // macOS does not expose a reliable "available" counter like Linux.
        // Use total RAM / 4 as a conservative heuristic for CPU workloads.
        if let Ok(bytes) = query_total_ram_bytes() {
            return (bytes / 1024 / 1024) / 4;
        }
    }
    // Conservative fallback.
    4096
}

/// Query total physical RAM in megabytes.
fn query_total_ram_mb() -> u64 {
    query_total_ram_bytes()
        .map(|b| b / 1024 / 1024)
        .unwrap_or(16384)
}

#[cfg(target_os = "macos")]
fn query_total_ram_bytes() -> Result<u64, String> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .map_err(|e| format!("sysctl failed: {e}"))?;
    let s = String::from_utf8(output.stdout).map_err(|e| format!("utf8: {e}"))?;
    s.trim()
        .parse::<u64>()
        .map_err(|e| format!("parse: {e}"))
}

#[cfg(target_os = "linux")]
fn query_total_ram_bytes() -> Result<u64, String> {
    let content =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("read meminfo: {e}"))?;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            let kb: u64 = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return Ok(kb * 1024);
        }
    }
    Err("MemTotal not found in /proc/meminfo".to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn query_total_ram_bytes() -> Result<u64, String> {
    Err("Unsupported platform for RAM query".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_equal_capacity() {
        let chunks = allocate_by_capacity(10, &[4096, 4096]);
        assert_eq!(chunks, vec![5, 5]);
    }

    #[test]
    fn test_allocate_2to1() {
        // 8192 : 4096 = 2 : 1
        // L=9, proportional = [6, 3]
        let chunks = allocate_by_capacity(9, &[8192, 4096]);
        assert_eq!(chunks, vec![6, 3]);
    }

    #[test]
    fn test_allocate_3to1() {
        // L=10, proportional = [7.5, 2.5] → floor [7, 2], remainder 1
        // fractional remainders: 0.5 > 0.5 → tie, first gets +1
        let chunks = allocate_by_capacity(10, &[12288, 4096]);
        assert_eq!(chunks.iter().sum::<usize>(), 10);
        assert_eq!(chunks[0] + chunks[1], 10);
        assert!(chunks[0] >= chunks[1]);
    }

    #[test]
    fn test_allocate_three_domains() {
        // Capacities 4:2:1, L=14
        // Exact: [8, 4, 2] → sum = 14, no remainder
        let chunks = allocate_by_capacity(14, &[4096, 2048, 1024]);
        assert_eq!(chunks, vec![8, 4, 2]);
    }

    #[test]
    fn test_allocate_three_domains_with_remainder() {
        // Capacities 4:2:1, L=10
        // Exact: [5.71, 2.86, 1.43] → floor [5, 2, 1], sum=8, remainder=2
        // Fractional remainders: 0.71, 0.86, 0.43 → order: domain1, domain0, domain2
        let chunks = allocate_by_capacity(10, &[4096, 2048, 1024]);
        assert_eq!(chunks.iter().sum::<usize>(), 10);
        assert!(chunks[0] >= chunks[1] && chunks[1] >= chunks[2]);
    }

    #[test]
    fn test_allocate_one_zero_capacity() {
        // Domain 0 reports 0 capacity, but must still get at least 1 token.
        let chunks = allocate_by_capacity(5, &[0, 4096]);
        assert_eq!(chunks.iter().sum::<usize>(), 5);
        assert_eq!(chunks[0], 1);
        assert_eq!(chunks[1], 4);
    }

    #[test]
    fn test_allocate_min_1_token() {
        // Bare minimum: exactly 3 tokens for 3 domains.
        let chunks = allocate_by_capacity(3, &[100, 100, 100]);
        assert_eq!(chunks, vec![1, 1, 1]);
    }

    #[test]
    fn test_allocate_large_difference() {
        // Extreme ratio: 100:1, L=20
        let chunks = allocate_by_capacity(20, &[100000, 1000]);
        assert_eq!(chunks.iter().sum::<usize>(), 20);
        assert!(chunks[0] > chunks[1]);
        assert!(chunks[1] >= 1);
    }

    #[test]
    fn test_allocate_all_zero_fallback() {
        // All capacities zero → fallback to even distribution.
        let chunks = allocate_by_capacity(8, &[0, 0, 0]);
        assert_eq!(chunks, vec![3, 3, 2]);
    }

    #[test]
    fn test_allocate_uneven_2domain_realistic() {
        // Simulates MPS (16GB RAM /2 = 8GB heuristic) vs CUDA (12GB free).
        let chunks = allocate_by_capacity(11, &[8192, 12288]);
        assert_eq!(chunks.iter().sum::<usize>(), 11);
        // Proportional: [4.4, 6.6] → floor [4, 6], remainder 1
        // fractional: 0.4 < 0.6 → domain1 gets +1 → [4, 7]
        assert_eq!(chunks, vec![4, 7]);
    }

    #[test]
    fn test_allocate_uneven_3domain_realistic() {
        // MPS (8GB) + CUDA0 (12GB) + CUDA1 (6GB)
        let chunks = allocate_by_capacity(11, &[8192, 12288, 6144]);
        assert_eq!(chunks.iter().sum::<usize>(), 11);
        // Exact: [3.385, 5.077, 2.538] → floor [3, 5, 2], remainder 1
        // fractional remainders: d2=0.538, d0=0.385, d1=0.077
        // remainder → d2 gets +1
        assert_eq!(chunks, vec![3, 5, 3]);
    }
}
