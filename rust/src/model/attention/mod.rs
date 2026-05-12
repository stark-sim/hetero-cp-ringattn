//! 【Attention 模块】
//!
//! HCP 的 Attention 计算核心：
//! - `backend`: AttentionBackend trait 定义（统一接口）
//! - `ring`: HcpRingAttentionBackend（online softmax + KV ring 交换）
//!
//! 所有单节点和分布式推理都使用 HcpRingAttentionBackend，
//! 它通过固定 chunk-size 上限避免 O(seq²) 的 scores 显存爆炸。

pub mod backend;
pub mod ring;

pub use backend::AttentionBackend;
pub use ring::HcpRingAttentionBackend;
