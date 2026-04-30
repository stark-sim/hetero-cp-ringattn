use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// Key-Value cache for autoregressive generation.
///
/// Stores K and V tensors of shape `[batch, num_kv_heads, cache_len, head_dim]`.
/// Supports both prefill (full sequence) and decode (single token append).
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    /// Current sequence length in the cache.
    pub seq_len: usize,
}

#[cfg(feature = "tch-backend")]
impl KvCache {
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            seq_len: 0,
        }
    }

    /// Update cache with new K/V tensors and return the full K/V including history.
    ///
    /// `new_k` / `new_v` shape: `[batch, num_kv_heads, new_seq_len, head_dim]`
    ///
    /// Returns full K/V shape: `[batch, num_kv_heads, cache_len + new_seq_len, head_dim]`
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError> {
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

    /// Reset cache to empty state.
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    /// Whether the cache has any entries.
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }
}

/// A collection of KV caches, one per layer.
#[cfg(feature = "tch-backend")]
pub type KvCaches = Vec<Option<KvCache>>;

#[cfg(feature = "tch-backend")]
pub fn create_kv_caches(num_layers: usize) -> KvCaches {
    (0..num_layers).map(|_| Some(KvCache::new())).collect()
}
