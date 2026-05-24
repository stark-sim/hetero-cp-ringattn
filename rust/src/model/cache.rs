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

/// 【Block Table KV 缓存】逻辑上将 KV 分成固定大小的 blocks。
///
/// **Current limitation**: Without a custom kernel that reads from non-contiguous
/// blocks, `update()` still concatenates blocks into a contiguous tensor for
/// attention compute. The block table provides the foundation for future
/// kernel-level batching but does not improve throughput today.
///
/// 设计意图：
/// - 为未来的 PagedAttention / custom kernel 提供数据结构基础
/// - 支持 block 级别的内存管理（减少碎片、block 共享、copy-on-write）
/// - 当前通过 `Tensor::cat` 拼接 blocks，与 `ContiguousKvCache` 行为一致
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct BlockTableKvCache {
    /// 每个 block 容纳的 token 数（默认 16）。
    block_size: usize,
    /// K blocks。每个 block 的 shape 为 [batch, num_kv_heads, block_len, head_dim]。
    k_blocks: Vec<Tensor>,
    /// V blocks。
    v_blocks: Vec<Tensor>,
    /// 最后一个 block 中已经使用的 token 数（0..block_size）。
    last_block_used: usize,
    seq_len: usize,
}

#[cfg(feature = "tch-backend")]
impl BlockTableKvCache {
    /// 【创建空 block table 缓存】
    ///
    /// `block_size`: 每个 block 的 token 容量。标准 vLLM 使用 16。
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            k_blocks: Vec::new(),
            v_blocks: Vec::new(),
            last_block_used: 0,
            seq_len: 0,
        }
    }

    /// 返回逻辑 block 列表，供未来的 custom kernel 直接消费。
    ///
    /// 每个 block 的 shape: [batch, num_kv_heads, block_len, head_dim]
    /// 其中最后一个 block 的 `block_len` 可能小于 `block_size`。
    pub fn k_blocks(&self) -> &[Tensor] {
        &self.k_blocks
    }

    pub fn v_blocks(&self) -> &[Tensor] {
        &self.v_blocks
    }
}

#[cfg(feature = "tch-backend")]
impl KvCache for BlockTableKvCache {
    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError> {
        let new_seq_len = new_k.size()[2] as usize;
        let mut remaining = new_seq_len;
        let mut offset = 0i64;

        // Fill the last block if it has remaining space.
        if self.last_block_used > 0 && self.last_block_used < self.block_size && !self.k_blocks.is_empty() {
            let space = self.block_size - self.last_block_used;
            let take = remaining.min(space);
            if take > 0 {
                let mut k_slice = new_k.narrow(2, offset, take as i64);
                let mut v_slice = new_v.narrow(2, offset, take as i64);
                let mut last_k = self.k_blocks.pop().unwrap();
                let mut last_v = self.v_blocks.pop().unwrap();
                let new_k_block = Tensor::cat(&[&mut last_k, &mut k_slice], 2);
                let new_v_block = Tensor::cat(&[&mut last_v, &mut v_slice], 2);
                self.k_blocks.push(new_k_block);
                self.v_blocks.push(new_v_block);
                self.last_block_used += take;
                remaining -= take;
                offset += take as i64;
            }
        }

        // Allocate new blocks for any remaining tokens.
        while remaining > 0 {
            let take = remaining.min(self.block_size);
            let mut k_slice = new_k.narrow(2, offset, take as i64);
            let mut v_slice = new_v.narrow(2, offset, take as i64);
            self.k_blocks.push(k_slice);
            self.v_blocks.push(v_slice);
            self.last_block_used = take;
            remaining -= take;
            offset += take as i64;
        }

        self.seq_len += new_seq_len;

        // Return full K/V by concatenating all blocks along seq_len dimension.
        let k_full = Tensor::cat(
            &self.k_blocks.iter().map(|t| t.shallow_clone()).collect::<Vec<_>>(),
            2,
        );
        let v_full = Tensor::cat(
            &self.v_blocks.iter().map(|t| t.shallow_clone()).collect::<Vec<_>>(),
            2,
        );

        Ok((k_full, v_full))
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.k_blocks.clear();
        self.v_blocks.clear();
        self.last_block_used = 0;
        self.seq_len = 0;
    }

    fn is_empty(&self) -> bool {
        self.k_blocks.is_empty()
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

#[cfg(test)]
#[cfg(feature = "tch-backend")]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_block_table_matches_contiguous() {
        let device = Device::Cpu;
        let batch = 1i64;
        let num_kv_heads = 2i64;
        let head_dim = 8i64;
        let block_size = 4usize;

        // Shared synthetic K/V data
        let k1 = Tensor::randn([batch, num_kv_heads, 3, head_dim], (Kind::Float, device));
        let v1 = Tensor::randn([batch, num_kv_heads, 3, head_dim], (Kind::Float, device));
        let k2 = Tensor::randn([batch, num_kv_heads, 5, head_dim], (Kind::Float, device));
        let v2 = Tensor::randn([batch, num_kv_heads, 5, head_dim], (Kind::Float, device));
        let k3 = Tensor::randn([batch, num_kv_heads, 2, head_dim], (Kind::Float, device));
        let v3 = Tensor::randn([batch, num_kv_heads, 2, head_dim], (Kind::Float, device));

        let mut contiguous = ContiguousKvCache::new();
        let mut block_table = BlockTableKvCache::new(block_size);

        // Step 1: update with 3 tokens
        let (ck1, cv1) = contiguous.update(&k1, &v1).unwrap();
        let (bk1, bv1) = block_table.update(&k1, &v1).unwrap();

        assert_eq!(contiguous.seq_len(), 3);
        assert_eq!(block_table.seq_len(), 3);
        assert_eq!(block_table.k_blocks().len(), 1); // fits in one block

        let diff_k1 = (&ck1 - &bk1).abs().mean(Kind::Float).double_value(&[]);
        let diff_v1 = (&cv1 - &bv1).abs().mean(Kind::Float).double_value(&[]);
        assert!(diff_k1 < 1e-6, "step 1 k diff: {}", diff_k1);
        assert!(diff_v1 < 1e-6, "step 1 v diff: {}", diff_v1);

        // Step 2: update with 5 tokens (crosses block boundary)
        let (ck2, cv2) = contiguous.update(&k2, &v2).unwrap();
        let (bk2, bv2) = block_table.update(&k2, &v2).unwrap();

        assert_eq!(contiguous.seq_len(), 8);
        assert_eq!(block_table.seq_len(), 8);
        assert_eq!(block_table.k_blocks().len(), 2); // 3+1 in first, 4 in second

        let diff_k2 = (&ck2 - &bk2).abs().mean(Kind::Float).double_value(&[]);
        let diff_v2 = (&cv2 - &bv2).abs().mean(Kind::Float).double_value(&[]);
        assert!(diff_k2 < 1e-6, "step 2 k diff: {}", diff_k2);
        assert!(diff_v2 < 1e-6, "step 2 v diff: {}", diff_v2);

        // Step 3: update with 2 tokens
        let (ck3, cv3) = contiguous.update(&k3, &v3).unwrap();
        let (bk3, bv3) = block_table.update(&k3, &v3).unwrap();

        assert_eq!(contiguous.seq_len(), 10);
        assert_eq!(block_table.seq_len(), 10);

        let diff_k3 = (&ck3 - &bk3).abs().mean(Kind::Float).double_value(&[]);
        let diff_v3 = (&cv3 - &bv3).abs().mean(Kind::Float).double_value(&[]);
        assert!(diff_k3 < 1e-6, "step 3 k diff: {}", diff_k3);
        assert!(diff_v3 < 1e-6, "step 3 v diff: {}", diff_v3);
    }

    #[test]
    fn test_block_table_clear_and_is_empty() {
        let device = Device::Cpu;
        let k = Tensor::randn([1, 2, 3, 8], (Kind::Float, device));
        let v = Tensor::randn([1, 2, 3, 8], (Kind::Float, device));

        let mut cache = BlockTableKvCache::new(4);
        assert!(cache.is_empty());

        let _ = cache.update(&k, &v).unwrap();
        assert!(!cache.is_empty());
        assert_eq!(cache.seq_len(), 3);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.k_blocks().is_empty());
    }
}
