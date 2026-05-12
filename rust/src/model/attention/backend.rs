use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::Tensor;
#[cfg(feature = "tch-backend")]
use crate::model::transport::KvTransport;

/// 【Attention 计算后端 Trait】
///
/// HCP 支持多种 attention 实现方式，这个 trait 是统一接口。
/// 目前只有 `HcpRingAttentionBackend` 一个实现（单节点和分布式共用）。
///
/// 设计意图：解耦 Attention 算法与上层模型结构，方便未来扩展
/// （如 FlashAttention、内存优化版等）。
///
/// 关键方法：
/// - forward: 给定 hidden_states，计算 attention 输出
/// - set_distributed: 配置分布式 KV 传输（只有 ring backend 需要）
#[cfg(feature = "tch-backend")]
pub trait AttentionBackend: Send {
    /// Forward pass: compute attention output for the given hidden states.
    ///
    /// `hidden_states`: `[batch, seq_len, hidden_size]`
    /// `position_ids`: `[batch, seq_len]` (Int64)
    /// `kv_cache`: Optional KV cache for autoregressive decoding
    /// `attention_mask`: Optional causal mask for prefill (shape `[1, 1, seq_len, seq_len]`)
    ///
    /// Returns: `[batch, seq_len, hidden_size]`
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError>;

    /// Optional: configure distributed transport, domain id, and sequence offset.
    /// Only `HcpRingAttentionBackend` implements this; others are no-ops.
    #[cfg(feature = "tch-backend")]
    #[allow(dead_code)]
    fn set_distributed(&mut self, _domain_id: usize, _seq_offset: usize, _transport: Option<Box<dyn KvTransport>>) {
        // Local backend 不需要分布式配置，noop
    }
}

/// Local (non-distributed) attention backend using standard GQA.
#[cfg(feature = "tch-backend")]
pub struct LocalAttentionBackend {
    pub attention: crate::model::layers::GqaAttention,
}

#[cfg(feature = "tch-backend")]
impl AttentionBackend for LocalAttentionBackend {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, ModelError> {
        self.attention.forward(hidden_states, position_ids, kv_cache, attention_mask)
    }
}