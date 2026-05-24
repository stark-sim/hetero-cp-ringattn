use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// KV cache trait for decoupled attention backends.
///
/// This trait enables future block-aware implementations (e.g. PagedAttention)
/// without modifying `LlamaModel` or `HcpRingAttentionBackend`.
#[cfg(feature = "tch-backend")]
pub trait KvCache: Send {
    /// Append new K/V tokens and return the full K/V tensors for attention compute.
    ///
    /// `new_k` / `new_v`: [batch, num_kv_heads, new_seq_len, head_dim]
    /// Returns: (k_full, v_full): [batch, num_kv_heads, cache_len + new_seq_len, head_dim]
    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError>;

    /// Current cached sequence length.
    fn seq_len(&self) -> usize;

    /// Reset to empty state.
    fn clear(&mut self);

    /// Whether the cache is empty.
    fn is_empty(&self) -> bool;
}

/// 【连续 KV 缓存】用于自回归生成时复用之前计算好的 Key 和 Value。
///
/// 存储的 K/V tensor 形状：[batch, num_kv_heads, cache_len, head_dim]
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct ContiguousKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    seq_len: usize,
}

#[cfg(feature = "tch-backend")]
impl ContiguousKvCache {
    /// 【创建空缓存】
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            seq_len: 0,
        }
    }
}

#[cfg(feature = "tch-backend")]
impl KvCache for ContiguousKvCache {
    /// 【更新缓存】把新的 K/V 拼接到缓存末尾，返回完整的 K/V（包含历史）。
    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError> {
        let (k_full, v_full) = if let Some(ref k) = self.k {
            let k_cat = Tensor::cat(&[k, new_k], 2);
            let v_cat = Tensor::cat(&[self.v.as_ref().unwrap(), new_v], 2);
            (k_cat, v_cat)
        } else {
            (new_k.shallow_clone(), new_v.shallow_clone())
        };

        self.k = Some(k_full.shallow_clone());
        self.v = Some(v_full.shallow_clone());
        self.seq_len = k_full.size()[2] as usize;

        Ok((k_full, v_full))
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    fn is_empty(&self) -> bool {
        self.k.is_none()
    }
}

/// 【每层一个 KV 缓存】Vec<Option<ContiguousKvCache>> 表示 num_layers 个可选缓存。
/// Option 是为了某些 layer 可能不需要缓存（虽然通常所有 layer 都有）。
#[cfg(feature = "tch-backend")]
pub type KvCaches = Vec<Option<ContiguousKvCache>>;

/// 【创建多层 KV 缓存】为每个 layer 初始化一个空的 ContiguousKvCache。
#[cfg(feature = "tch-backend")]
pub fn create_kv_caches(num_layers: usize) -> KvCaches {
    (0..num_layers).map(|_| Some(ContiguousKvCache::new())).collect()
}
