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
    /// 【原始位置 id】用于 Striped / 非均等 permutation 场景下的 causal mask。
    /// 如果为 None，则默认使用 [global_seq_start, global_seq_end) 的连续位置。
    pub position_ids: Option<Tensor>,
}

impl KvBlock {
    /// 【创建单个未切分的 KV block】向后兼容的便捷构造函数。
    pub fn single(
        layer_idx: usize,
        global_seq_start: usize,
        global_seq_end: usize,
        k: Tensor,
        v: Tensor,
    ) -> Self {
        Self {
            layer_idx,
            global_seq_start,
            global_seq_end,
            k,
            v,
            micro_block_idx: 0,
            total_micro_blocks: 1,
            position_ids: None,
        }
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
            position_ids: self.position_ids.as_ref().map(|t| t.shallow_clone()),
        }
    }
}

/// 【Decode Q-ring Packet】decode 阶段绕环传输的 (Q, O, LSE) 累加器包。
///
/// LoongServe 风格：decode 时不传 KV，改传单 token 的 Q 和 online softmax
/// 累加器 (O, LSE)。每个节点收到后用本地 KV segment 对 packet.q 计算 partial
/// 并 merge，再转发给 successor；N-1 轮后 packet 完成全环合并。
/// - q: [batch, num_heads, 1, head_dim]
/// - o: [batch, num_heads, 1, head_dim]（已归一化的累加输出）
/// - lse: [batch, num_heads, 1] fp32（log-sum-exp 累加器）
/// - scale: 1/sqrt(head_dim)（协议完整性字段，所有节点取值相同）
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct RingPacket {
    pub layer_idx: usize,
    pub q: Tensor,
    pub o: Tensor,
    pub lse: Tensor,
    pub scale: f64,
}

#[cfg(feature = "tch-backend")]
impl Clone for RingPacket {
    fn clone(&self) -> Self {
        Self {
            layer_idx: self.layer_idx,
            q: self.q.shallow_clone(),
            o: self.o.shallow_clone(),
            lse: self.lse.shallow_clone(),
            scale: self.scale,
        }
    }
}

/// Experimental self-driving layer packet sent after the starter has produced
/// the first local attention partial.
///
/// Unlike `RingPacket`, this packet also carries the activation state needed by
/// the finisher to run output projection, residual, norm, and MLP locally.
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct SelfDrivingPacket {
    pub layer_idx: usize,
    pub residual: Tensor,
    pub normalized: Tensor,
    pub position_ids: Tensor,
    pub q: Tensor,
    pub attention_output: Tensor,
    pub lse: Tensor,
    pub assignee: usize,
    pub current_domain: usize,
    pub domains: usize,
    pub visited_domains: usize,
}

#[cfg(feature = "tch-backend")]
impl Clone for SelfDrivingPacket {
    fn clone(&self) -> Self {
        Self {
            layer_idx: self.layer_idx,
            residual: self.residual.shallow_clone(),
            normalized: self.normalized.shallow_clone(),
            position_ids: self.position_ids.shallow_clone(),
            q: self.q.shallow_clone(),
            attention_output: self.attention_output.shallow_clone(),
            lse: self.lse.shallow_clone(),
            assignee: self.assignee,
            current_domain: self.current_domain,
            domains: self.domains,
            visited_domains: self.visited_domains,
        }
    }
}
