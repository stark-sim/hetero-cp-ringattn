use crate::model::{ModelError, KvCache};

#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// Abstraction over attention computation.
///
/// This is the key decoupling layer between the model and HCP Core.
/// - `LocalAttentionBackend` runs attention on a single device.
/// - `HcpRingAttentionBackend` (Phase 2) distributes attention across domains.
///
/// Future inference engines like vLLM can plug in their own backend
/// or use `HcpRingAttentionBackend` to offload distributed attention.
#[cfg(feature = "tch-backend")]
pub trait AttentionBackend {
    /// Run attention forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - `[batch, seq_len, hidden_size]`
    /// * `position_ids` - `[batch, seq_len]` absolute positions for RoPE
    /// * `kv_cache` - Optional KV cache for incremental decoding
    /// * `attention_mask` - Optional causal/padding mask to add to scores
    ///
    /// # Returns
    /// * `[batch, seq_len, hidden_size]` attention output
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError>;
}

/// Single-device attention backend.
///
/// Wraps `GqaAttention` and runs full attention locally on CPU/MPS/CUDA.
#[cfg(feature = "tch-backend")]
pub struct LocalAttentionBackend {
    pub attention: super::layers::GqaAttention,
}

#[cfg(feature = "tch-backend")]
impl AttentionBackend for LocalAttentionBackend {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        self.attention.forward(hidden_states, position_ids, kv_cache, attention_mask)
    }
}
