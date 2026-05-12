#![allow(dead_code)]

use super::block::KvBlock;

/// 【KV 传输层 Trait】定义分布式 worker 之间交换 KV block 的接口。
///
/// HCP 支持多种传输实现：
/// - `QuicKvTransport`: 生产环境使用，基于 QUIC，支持高延迟网络（如 VPN）
/// - `TcpKvTransport`: 简单 TCP 传输，用于测试
/// - `MockKvTransport` / `LinkedMockKvTransport`: 内存传输，用于单元测试
///
/// 【设计要点】
/// - `Send` trait: transport 需要跨线程发送（tokio runtime 和工作线程之间）
/// - `exchange_kv_block`: 默认实现是先 send 再 recv，但 QUIC 覆盖为并发执行
///   （防止大 KV block 同时双向发送时死锁）
#[cfg(feature = "tch-backend")]
pub trait KvTransport: Send {
    /// Send a KV block to the next peer in the ring.
    fn send_kv_block(&mut self, block: &KvBlock) -> Result<(), String>;

    /// Receive a KV block from the previous peer in the ring.
    /// Returns `None` when the peer has closed the connection (no more blocks).
    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String>;

    /// Atomically send a block and receive a block in the same round.
    ///
    /// Default implementation is sequential (send then recv). QUIC transport
    /// overrides this to run both directions concurrently, preventing deadlock
    /// when both peers simultaneously send large KV blocks that exceed the
    /// stream receive window.
    fn exchange_kv_block(&mut self, block: &KvBlock) -> Result<Option<KvBlock>, String> {
        self.send_kv_block(block)?;
        self.recv_kv_block()
    }
}
