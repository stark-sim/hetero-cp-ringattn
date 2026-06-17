use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

/// Pure-Rust sampling from a f32 logits slice (no libtorch required).
#[cfg(not(feature = "tch-backend"))]
pub(crate) fn sample_token_slice(logits: &[f32], temperature: f64, top_p: f64) -> Result<u32, ModelError> {
    // Greedy decoding
    if temperature <= 0.0 {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best = i;
                best_val = v;
            }
        }
        return Ok(best as u32);
    }

    // Temperature scaling + softmax
    let scaled: Vec<f32> = logits.iter().map(|&x| (x as f64 / temperature) as f32).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scaled.iter().map(|&x| (x - max).exp()).sum();
    let mut probs: Vec<f32> = scaled.iter().map(|&x| (x - max).exp() / exp_sum).collect();

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum > top_p as f32 {
                cutoff = i + 1;
                break;
            }
        }
        let mut filtered = vec![0.0f32; probs.len()];
        for (idx, p) in indexed.iter().take(cutoff) {
            filtered[*idx] = *p;
        }
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            probs = filtered.iter().map(|&p| p / sum).collect();
        }
    }

    // Multinomial sampling via cumulative probabilities
    let r: f32 = rand::random::<f32>();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(i as u32);
        }
    }
    Ok((probs.len().saturating_sub(1)) as u32)
}

/// 【从 logits 采样单个 token】
///
/// 这是 LLM 推理的"决策"步骤：模型输出一堆分数（logits），
/// 我们从中选一个 token 作为下一个生成的词。
///
/// 三种采样模式：
/// 1. Greedy（temperature == 0.0）: 直接选分数最高的 token。
///    优点：确定性输出，可复现。缺点：缺乏多样性，容易进入重复循环。
///
/// 2. Temperature scaling（temperature > 0）:
///    - temperature 高 → 概率分布更均匀 → 输出更随机、有创意
///    - temperature 低 → 概率分布更尖锐 → 输出更保守、确定性更强
///
/// 3. Top-p（nucleus）filtering（top_p > 0 且 < 1）:
///    只从累计概率达到 top_p 的那部分 token 中采样。
///    例如 top_p=0.9 表示只考虑概率最高的、累计占 90% 的那部分 token，
///    排除长尾的低概率噪声 token。
#[cfg(feature = "tch-backend")]
pub(crate) fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32, ModelError> {
    // Greedy decoding
    if temperature <= 0.0 {
        return Ok(logits.argmax(-1, false).int64_value(&[]) as u32);
    }

    // Temperature scaling
    let scaled_logits = logits / temperature;

    // Top-p (nucleus) filtering
    let filtered_logits = if top_p > 0.0 && top_p < 1.0 {
        let probs = scaled_logits.softmax(-1, Kind::Float);
        // Sort descending: (values, indices)
        let sorted = probs.sort(-1, true);
        let sorted_probs = sorted.0;
        let sorted_indices = sorted.1;
        // Cumulative sum
        let cumsum = sorted_probs.cumsum(-1, Kind::Float);
        // Find tokens to remove: cumsum > top_p
        let remove_mask = cumsum.gt(top_p);
        // Set removed probabilities to 0
        let filtered_sorted = sorted_probs * remove_mask.logical_not().to_kind(Kind::Float);
        // Scatter back to original positions and renormalize
        let filtered = Tensor::zeros_like(&probs)
            .scatter_add(-1, &sorted_indices, &filtered_sorted);
        // Avoid log(0) by adding epsilon
        let epsilon = 1e-10;
        (filtered.clamp_min(epsilon)).log()
    } else {
        scaled_logits.shallow_clone()
    };

    // Convert to probabilities and sample
    let probs = filtered_logits.softmax(-1, Kind::Float);
    // Clamp to avoid numerical issues
    let safe_probs = probs.clamp_min(1e-10);
    let sampled = safe_probs.multinomial(1, true);
    Ok(sampled.int64_value(&[]) as u32)
}
