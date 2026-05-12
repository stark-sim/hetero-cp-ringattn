use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

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
