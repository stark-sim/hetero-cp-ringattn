//! 【协议层模块】
//!
//! HCP 的分布式通信协议，负责 worker 之间 KV block 的序列化和传输：
//! - `message`: 协议消息结构（RingAttnMessage、错误类型、序列化）
//! - `framing`: 长度前缀帧格式（TCP 流分割）
//! - `transport`: MessageSender / MessageReceiver trait（TCP / Channel 实现）
//! - `node`: CP Ring Node 逻辑和 smoke test 运行时
//!
//! 协议使用 bincode 序列化（比 JSON 紧凑高效），
//! 配合长度前缀帧解决 TCP 流式传输的消息边界问题。

pub mod message;
pub mod transport;
pub mod framing;
pub mod node;

pub use message::ProtocolError;
pub(crate) use message::RingAttnMessage;
#[cfg(feature = "tch-backend")]
pub(crate) use message::{
    RingAttnMessageKind, FLOAT32_BYTES,
};
pub use node::{
    CpPayloadBlock, CpRingNodeSmokeReport, ProtocolSmokeReport,
    RemoteCpNodeReport, RemoteP2pReport, run_cp_ring_node_smoke,
    run_protocol_smoke, run_remote_cp_node, run_remote_p2p_client,
    run_remote_p2p_server,
};
pub(crate) use node::{
    DomainModelState, OnlineSoftmaxAccumulator,
};
