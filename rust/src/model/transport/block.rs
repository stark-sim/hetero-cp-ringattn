#![allow(dead_code)]

#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// 【KV Block】Ring Attention 中分布式 worker 之间交换的数据单元。
///
/// 每个 block 包含：
/// - layer_idx: 当前属于哪一层（每层独立交换，不跨层混用）
/// - global_seq_start / global_seq_end: 这个 block 覆盖的全局序列范围
/// - k / v: Key 和 Value tensor，shape [batch, num_kv_heads, seq_len, head_dim]
///
/// 【为什么按 layer 独立交换？】
/// Transformer 是逐层计算的：layer0 先算完，才能算 layer1。
/// 所以 layer0 的 KV 需要在 layer0 的 attention 计算完成后立即交换，
/// 不能等到 layer1 再交换（否则 layer0 的 attention 结果已经需要 layer1 用了）。
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct KvBlock {
    pub layer_idx: usize,
    pub global_seq_start: usize,
    pub global_seq_end: usize,
    pub k: Tensor,
    pub v: Tensor,
    /// 【micro block 索引】当 KV block 被切分成更小的 micro blocks 时，
    /// 表示这是第几个 micro block（从 0 开始）。
    /// 默认 0 表示未切分（单个 block）。
    pub micro_block_idx: usize,
    /// 【micro block 总数】该 domain 在这一 round 中总共有多少个 micro blocks。
    /// 默认 1 表示未切分（单个 block）。
    pub total_micro_blocks: usize,
}

impl KvBlock {
    /// 【创建单个未切分的 KV block】向后兼容的便捷构造函数。
    pub fn single(layer_idx: usize, global_seq_start: usize, global_seq_end: usize, k: Tensor, v: Tensor) -> Self {
        Self { layer_idx, global_seq_start, global_seq_end, k, v, micro_block_idx: 0, total_micro_blocks: 1 }
    }
}

impl Clone for KvBlock {
    fn clone(&self) -> Self {
        Self {
            layer_idx: self.layer_idx,
            global_seq_start: self.global_seq_start,
            global_seq_end: self.global_seq_end,
            k: self.k.shallow_clone(),
            v: self.v.shallow_clone(),
            micro_block_idx: self.micro_block_idx,
            total_micro_blocks: self.total_micro_blocks,
        }
    }
}
