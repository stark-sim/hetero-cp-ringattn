use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// KV cache trait for decoupled attention backends.
///
/// This trait enables future block-aware implementations (e.g. PagedAttention)
/// without modifying `LlamaModel` or `HcpRingAttentionBackend`.
#[cfg(feature = "tch-backend")]
#[allow(dead_code)]
pub trait KvCache: Send {
    /// Provide the absolute positions associated with the next K/V update.
    fn prepare_positions(&mut self, _position_ids: &Tensor) -> Result<(), ModelError> {
        Ok(())
    }

    /// Append new K/V tokens and return the full K/V tensors for attention compute.
    ///
    /// `new_k` / `new_v`: [batch, num_kv_heads, new_seq_len, head_dim]
    /// Returns: (k_full, v_full): [batch, num_kv_heads, cache_len + new_seq_len, head_dim]
    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError>;

    /// Current cached sequence length.
    fn seq_len(&self) -> usize;

    /// 【分片更新（decode Q-ring growth 分片用）】
    ///
    /// `keep = true`：与 `update` 完全相同（新 token 持久化到缓存）。
    /// `keep = false`：新 token 不写入缓存（本节点不是该 token 的归属节点，直接丢弃），
    /// 但返回值仍然是 [缓存内容; 新 token]，供当前 step 的 attention 使用。
    ///
    /// 默认实现忽略 `keep`（退化为全量复制的旧行为），保证未适配的缓存实现行为不变。
    fn update_sharded(&mut self, new_k: &Tensor, new_v: &Tensor, keep: bool) -> Result<(Tensor, Tensor), ModelError> {
        let _ = keep;
        self.update(new_k, new_v)
    }

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

    /// 【分片更新】keep=false 时不持久化新 token，仅返回 [缓存内容; 新 token]。
    fn update_sharded(&mut self, new_k: &Tensor, new_v: &Tensor, keep: bool) -> Result<(Tensor, Tensor), ModelError> {
        if keep {
            return self.update(new_k, new_v);
        }
        // 【非归属节点】丢弃新 token，返回拼接结果供本 step attention 使用
        match self.k.as_ref() {
            Some(k) => Ok((
                Tensor::cat(&[k, new_k], 2),
                Tensor::cat(&[self.v.as_ref().unwrap(), new_v], 2),
            )),
            None => Ok((new_k.shallow_clone(), new_v.shallow_clone())),
        }
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
    #[allow(dead_code)]
    pub fn k_blocks(&self) -> &[Tensor] {
        &self.k_blocks
    }

    #[allow(dead_code)]
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
            let k_slice = new_k.narrow(2, offset, take as i64);
            let v_slice = new_v.narrow(2, offset, take as i64);
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

    /// 【分片更新】keep=false 时不持久化新 token，仅返回 [所有 block; 新 token]。
    fn update_sharded(&mut self, new_k: &Tensor, new_v: &Tensor, keep: bool) -> Result<(Tensor, Tensor), ModelError> {
        if keep {
            return self.update(new_k, new_v);
        }
        // 【非归属节点】丢弃新 token，返回拼接结果供本 step attention 使用
        if self.k_blocks.is_empty() {
            return Ok((new_k.shallow_clone(), new_v.shallow_clone()));
        }
        let mut k_parts: Vec<Tensor> = self.k_blocks.iter().map(|t| t.shallow_clone()).collect();
        k_parts.push(new_k.shallow_clone());
        let mut v_parts: Vec<Tensor> = self.v_blocks.iter().map(|t| t.shallow_clone()).collect();
        v_parts.push(new_v.shallow_clone());
        Ok((Tensor::cat(&k_parts, 2), Tensor::cat(&v_parts, 2)))
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

/// 【KV 缓存实现枚举】封装所有可用的 KV 缓存实现。
///
/// 使用 enum 而非 `Box<dyn KvCache>` 以避免 trait object 的生命周期问题
/// 和堆分配开销，同时保持运行时切换能力。
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub enum KvCacheImpl {
    Contiguous(ContiguousKvCache),
    BlockTable(BlockTableKvCache),
    #[allow(dead_code)]
    ReservedPositioned(crate::model::self_driving::ReservedPositionedKvShard),
}

#[cfg(feature = "tch-backend")]
impl KvCacheImpl {
    /// Get the current K and V tensors from the cache.
    pub fn get_kv(&self) -> Option<(Tensor, Tensor)> {
        match self {
            KvCacheImpl::Contiguous(c) => {
                let k = c.k.as_ref()?.shallow_clone();
                let v = c.v.as_ref()?.shallow_clone();
                Some((k, v))
            }
            KvCacheImpl::BlockTable(c) => {
                if c.k_blocks.is_empty() {
                    return None;
                }
                let k = Tensor::cat(
                    &c.k_blocks.iter().map(|t| t.shallow_clone()).collect::<Vec<_>>(),
                    2,
                );
                let v = Tensor::cat(
                    &c.v_blocks.iter().map(|t| t.shallow_clone()).collect::<Vec<_>>(),
                    2,
                );
                Some((k, v))
            }
            KvCacheImpl::ReservedPositioned(c) => {
                if c.is_empty() {
                    None
                } else {
                    Some((c.active_k(), c.active_v()))
                }
            }
        }
    }
}

#[cfg(feature = "tch-backend")]
impl KvCache for KvCacheImpl {
    fn prepare_positions(&mut self, position_ids: &Tensor) -> Result<(), ModelError> {
        match self {
            KvCacheImpl::Contiguous(c) => c.prepare_positions(position_ids),
            KvCacheImpl::BlockTable(c) => c.prepare_positions(position_ids),
            KvCacheImpl::ReservedPositioned(c) => c.prepare_positions(position_ids),
        }
    }

    fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError> {
        match self {
            KvCacheImpl::Contiguous(c) => c.update(new_k, new_v),
            KvCacheImpl::BlockTable(c) => c.update(new_k, new_v),
            KvCacheImpl::ReservedPositioned(c) => c.update(new_k, new_v),
        }
    }

    fn update_sharded(&mut self, new_k: &Tensor, new_v: &Tensor, keep: bool) -> Result<(Tensor, Tensor), ModelError> {
        match self {
            KvCacheImpl::Contiguous(c) => c.update_sharded(new_k, new_v, keep),
            KvCacheImpl::BlockTable(c) => c.update_sharded(new_k, new_v, keep),
            KvCacheImpl::ReservedPositioned(c) => c.update_sharded(new_k, new_v, keep),
        }
    }

    fn seq_len(&self) -> usize {
        match self {
            KvCacheImpl::Contiguous(c) => c.seq_len(),
            KvCacheImpl::BlockTable(c) => c.seq_len(),
            KvCacheImpl::ReservedPositioned(c) => c.seq_len(),
        }
    }

    fn clear(&mut self) {
        match self {
            KvCacheImpl::Contiguous(c) => c.clear(),
            KvCacheImpl::BlockTable(c) => c.clear(),
            KvCacheImpl::ReservedPositioned(c) => c.clear(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            KvCacheImpl::Contiguous(c) => c.is_empty(),
            KvCacheImpl::BlockTable(c) => c.is_empty(),
            KvCacheImpl::ReservedPositioned(c) => c.is_empty(),
        }
    }
}

/// 【每层一个 KV 缓存】Vec<Option<KvCacheImpl>> 表示 num_layers 个可选缓存。
/// Option 是为了某些 layer 可能不需要缓存（虽然通常所有 layer 都有）。
#[cfg(feature = "tch-backend")]
pub type KvCaches = Vec<Option<KvCacheImpl>>;

/// 【创建多层 KV 缓存】为每个 layer 初始化一个缓存实例。
///
/// 缓存类型由环境变量控制：
/// - `HCP_KV_CACHE_BLOCK_TABLE=1`（或 `true`）：使用 `BlockTableKvCache`
/// - 默认（未设置或其他值）：使用 `ContiguousKvCache`
///
/// BlockTable 的 block 大小可通过 `HCP_KV_CACHE_BLOCK_SIZE` 调整（默认 16）。
#[cfg(feature = "tch-backend")]
pub fn create_kv_caches(num_layers: usize) -> KvCaches {
    let use_block_table = std::env::var("HCP_KV_CACHE_BLOCK_TABLE")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let block_size = std::env::var("HCP_KV_CACHE_BLOCK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    (0..num_layers)
        .map(|_| {
            if use_block_table {
                Some(KvCacheImpl::BlockTable(BlockTableKvCache::new(block_size)))
            } else {
                Some(KvCacheImpl::Contiguous(ContiguousKvCache::new()))
            }
        })
        .collect()
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

    /// 【update_sharded keep=true 与 update 行为一致】
    #[test]
    fn test_update_sharded_keep_matches_update() {
        let device = Device::Cpu;
        let k1 = Tensor::randn([1, 2, 3, 8], (Kind::Float, device));
        let v1 = Tensor::randn([1, 2, 3, 8], (Kind::Float, device));
        let k2 = Tensor::randn([1, 2, 1, 8], (Kind::Float, device));
        let v2 = Tensor::randn([1, 2, 1, 8], (Kind::Float, device));

        let mut a = ContiguousKvCache::new();
        let mut b = ContiguousKvCache::new();
        let _ = a.update(&k1, &v1).unwrap();
        let _ = b.update(&k1, &v1).unwrap();

        let (ka, va) = a.update(&k2, &v2).unwrap();
        let (kb, vb) = b.update_sharded(&k2, &v2, true).unwrap();

        assert_eq!(a.seq_len(), 4);
        assert_eq!(b.seq_len(), 4);
        let diff_k = (&ka - &kb).abs().max().double_value(&[]);
        let diff_v = (&va - &vb).abs().max().double_value(&[]);
        assert_eq!(diff_k, 0.0);
        assert_eq!(diff_v, 0.0);
    }

    /// 【growth 分片不变量】3 节点按 global_pos % 3 分片：
    /// - keep=false 的节点不持久化（seq_len 不变），但返回值仍包含当前 token；
    /// - 每个节点最终 durable = prefill chunk + 自己的 p%N 份额（无缺口、无重复）。
    #[test]
    fn test_update_sharded_growth_sharding() {
        let device = Device::Cpu;
        let num_nodes = 3usize;
        let prefill_chunk_len = 2usize; // 每节点 prefill chunk 长度
        let global_prefill_len = prefill_chunk_len * num_nodes; // 6
        let decode_steps = 5usize;

        // 每个节点先写入自己的 prefill chunk
        let mut caches: Vec<ContiguousKvCache> = (0..num_nodes)
            .map(|_| {
                let mut c = ContiguousKvCache::new();
                let k = Tensor::randn([1, 2, prefill_chunk_len as i64, 8], (Kind::Float, device));
                let v = Tensor::randn([1, 2, prefill_chunk_len as i64, 8], (Kind::Float, device));
                let _ = c.update(&k, &v).unwrap();
                c
            })
            .collect();

        for step in 0..decode_steps {
            let global_pos = global_prefill_len + step;
            let new_k = Tensor::randn([1, 2, 1, 8], (Kind::Float, device));
            let new_v = Tensor::randn([1, 2, 1, 8], (Kind::Float, device));
            for (node, cache) in caches.iter_mut().enumerate() {
                let keep = global_pos % num_nodes == node;
                let before = cache.seq_len();
                let (k_full, v_full) = cache.update_sharded(&new_k, &new_v, keep).unwrap();
                // 返回值始终包含当前 token（供本 step attention）
                assert_eq!(k_full.size()[2] as usize, before + 1);
                assert_eq!(v_full.size()[2] as usize, before + 1);
                // 非归属节点不持久化
                assert_eq!(cache.seq_len(), if keep { before + 1 } else { before });
            }
        }

        // 每个节点 durable = prefill chunk + 自己的 p%N 份额
        let mut total = 0usize;
        for (node, cache) in caches.iter().enumerate() {
            let share = (0..decode_steps)
                .filter(|s| (global_prefill_len + s) % num_nodes == node)
                .count();
            assert_eq!(cache.seq_len(), prefill_chunk_len + share, "node {} durable len", node);
            total += cache.seq_len();
        }
        // 无缺口、无重复：所有节点 durable 之和 = 全局 token 数
        assert_eq!(total, global_prefill_len + decode_steps);
    }

    /// 【BlockTable 分片更新】keep=false 时返回完整 KV 但不持久化。
    #[test]
    fn test_update_sharded_block_table() {
        let device = Device::Cpu;
        let k1 = Tensor::randn([1, 2, 3, 8], (Kind::Float, device));
        let v1 = Tensor::randn([1, 2, 3, 8], (Kind::Float, device));
        let k2 = Tensor::randn([1, 2, 1, 8], (Kind::Float, device));
        let v2 = Tensor::randn([1, 2, 1, 8], (Kind::Float, device));

        let mut cache = BlockTableKvCache::new(4);
        let _ = cache.update(&k1, &v1).unwrap();

        let (k_full, _) = cache.update_sharded(&k2, &v2, false).unwrap();
        assert_eq!(k_full.size()[2], 4);
        assert_eq!(cache.seq_len(), 3); // 未持久化

        let (k_full, _) = cache.update_sharded(&k2, &v2, true).unwrap();
        assert_eq!(k_full.size()[2], 4);
        assert_eq!(cache.seq_len(), 4); // 已持久化
    }
}
