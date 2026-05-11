#![allow(dead_code)]

use super::block::KvBlock;

/// Transport for exchanging KV blocks between distributed attention workers.
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
