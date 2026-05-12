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
    /// 【提交异步发送】把 KV block 放入发送队列，立即返回。
    ///
    /// 对于 QUIC 实现，这会序列化 block 并推入内部 channel，
    /// 真正的网络发送在 background async task 中进行。
    /// 对于 TCP/Mock 实现，这可能同步完成（立即写入或放入内存队列）。
    fn submit_send(&mut self, block: &KvBlock) -> Result<(), String>;

    /// 【轮询接收】非阻塞地检查是否有 peer KV block 到达。
    ///
    /// - `Some(block)`: 收到一个 block
    /// - `None`: 暂时没有数据（调用者可以继续做其他事，稍后再 poll）
    fn poll_recv(&mut self) -> Result<Option<KvBlock>, String>;

    /// 【刷新发送】等待所有已 submit 的 send 完成。
    ///
    /// 在 attention 计算结束前调用，确保所有 KV block 都已成功发送到 peer。
    fn flush_send(&mut self) -> Result<(), String>;

    // ====== 旧同步 API：基于 split-phase 方法提供默认实现 ======
    // 这样现有代码（测试、简单场景）可以继续使用同步方法，无需改动。

    /// Send a KV block to the next peer in the ring.
    /// 默认实现：submit_send + flush_send（阻塞到发送完成）。
    fn send_kv_block(&mut self, block: &KvBlock) -> Result<(), String> {
        self.submit_send(block)?;
        self.flush_send()
    }

    /// Receive a KV block from the previous peer in the ring.
    /// 默认实现：轮询 poll_recv 直到有数据（忙等，1ms 间隔）。
    /// Returns `None` when the peer has closed the connection (no more blocks).
    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        loop {
            match self.poll_recv()? {
                Some(block) => return Ok(Some(block)),
                None => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
    }

    /// Atomically send a block and receive a block in the same round.
    ///
    /// 默认实现：submit_send 后立即开始 recv_kv_block 忙等。
    /// 在 QUIC 实现中，send 和 recv 在内部是并发的（不同的 async task），
    /// 所以这里虽然调用顺序是串行的，实际网络 I/O 是并行的。
    fn exchange_kv_block(&mut self, block: &KvBlock) -> Result<Option<KvBlock>, String> {
        self.submit_send(block)?;
        let result = self.recv_kv_block();
        self.flush_send()?;
        result
    }
}
