use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

/// Sample a single token from logits.
///
/// - `temperature == 0.0`: greedy argmax.
/// - `temperature > 0.0`: temperature scaling + optional top-p filtering + multinomial sampling.
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
