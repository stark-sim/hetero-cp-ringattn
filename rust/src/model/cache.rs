use crate::model::ModelError;

#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// 【KV 缓存】用于自回归生成时复用之前计算好的 Key 和 Value。
///
/// 在 Transformer 的 Attention 中，计算每个 token 的注意力时需要用到所有前面 token 的 K 和 V。
/// 如果每次生成一个新 token 都重新计算所有历史 token 的 K/V，会非常慢。
/// KV Cache 就是把这些历史 K/V 存起来，新 token 只需要把自己的 K/V append 到缓存末尾。
///
/// 存储的 K/V tensor 形状：[batch, num_kv_heads, cache_len, head_dim]
/// - batch: 批量大小（同时生成多少条句子）
/// - num_kv_heads: GQA 的 key/value head 数量
/// - cache_len: 当前缓存中已经存储的 token 数量
/// - head_dim: 每个 head 的维度
///
/// 支持两种模式：
/// - prefill（首次处理完整 prompt）：把整段序列的 K/V 一次性存入缓存
/// - decode（逐 token 生成）：每次只 append 一个新 token 的 K/V
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct KvCache {
    // K/V 缓存。Option 表示初始状态下缓存为空（None）。
    // 第一次 update 后变成 Some(tensor)。
    k: Option<Tensor>,
    v: Option<Tensor>,
    /// 当前缓存中的序列长度（已存储多少个 token 的 K/V）。
    pub seq_len: usize,
}

#[cfg(feature = "tch-backend")]
impl KvCache {
    /// 【创建空缓存】
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            seq_len: 0,
        }
    }

    /// 【更新缓存】把新的 K/V 拼接到缓存末尾，返回完整的 K/V（包含历史）。
    ///
    /// 参数：
    /// - new_k / new_v: 新 token(s) 的 K/V，shape [batch, num_kv_heads, new_seq_len, head_dim]
    ///
    /// 返回值：
    /// - (k_full, v_full): 拼接后的完整 K/V，shape [batch, num_kv_heads, cache_len + new_seq_len, head_dim]
    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor), ModelError> {
        let (k_full, v_full) = if let Some(ref k) = self.k {
            // 如果缓存中已有历史 K/V，用 Tensor::cat 在第 2 维（seq_len 维）拼接。
            // cat(&[历史 K, 新 K], 2) → [batch, num_kv_heads, cache_len + new_seq_len, head_dim]
            let k_cat = Tensor::cat(&[k, new_k], 2);
            let v_cat = Tensor::cat(&[self.v.as_ref().unwrap(), new_v], 2);
            (k_cat, v_cat)
        } else {
            // 第一次更新，缓存为空，直接返回新的 K/V（浅拷贝，不复制底层数据）。
            (new_k.shallow_clone(), new_v.shallow_clone())
        };

        // 把拼接后的结果存回缓存，供下次生成使用。
        self.k = Some(k_full.shallow_clone());
        self.v = Some(v_full.shallow_clone());
        // 更新 seq_len 为拼接后的序列长度。
        self.seq_len = k_full.size()[2] as usize;

        Ok((k_full, v_full))
    }

    /// 【清空缓存】重置为空状态，释放内存。
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    /// 【判断是否为空】
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }
}

/// 【每层一个 KV 缓存】Vec<Option<KvCache>> 表示 num_layers 个可选缓存。
/// Option 是为了某些 layer 可能不需要缓存（虽然通常所有 layer 都有）。
#[cfg(feature = "tch-backend")]
pub type KvCaches = Vec<Option<KvCache>>;

/// 【创建多层 KV 缓存】为每个 layer 初始化一个空的 KvCache。
#[cfg(feature = "tch-backend")]
pub fn create_kv_caches(num_layers: usize) -> KvCaches {
    (0..num_layers).map(|_| Some(KvCache::new())).collect()
}
