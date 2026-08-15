#![allow(dead_code)]

//! Block-level transport abstraction for zero-copy device-memory transfer.
//!
//! This is the data-plane contract for form-B (block-direct) transport — NIXL
//! as a third transport alongside QUIC/TCP. It models NIXL's semantics
//! (register memory → exchange metadata via side channel → async transfer →
//! poll completion), NOT the byte-stream semantics of the existing KvTransport.
//!
//! The descriptor model (addr + len + dev_id + metadata blob) is exactly
//! NIXL's nixlBasicDesc/nixlBlobDesc shape and also vLLM's physical-block
//! shape, so this abstraction is the single block data-plane that later serves
//! vLLM paged-KV integration (docs/BLOCK_RING_FUSION.md).

#[cfg(feature = "tch-backend")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// A registered memory block descriptor: start address, byte length, device
/// id, and an opaque metadata blob (shape/dtype/etc). Serializes so it can be
/// exchanged over the side channel.
#[cfg(feature = "tch-backend")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockDesc {
    pub addr: u64,
    pub len: u64,
    pub dev_id: u64,
    pub meta: Vec<u8>,
}

/// A remote block descriptor received over the side channel, tagged with the
/// peer agent name and the peer-local block id it refers to.
#[cfg(feature = "tch-backend")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteBlockDesc {
    pub agent: String,
    pub block_id: u64,
    pub desc: BlockDesc,
}

/// A local registration handle: the stable id plus the descriptor that was
/// advertised to peers via the side channel.
#[cfg(feature = "tch-backend")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockHandle {
    pub id: u64,
    pub desc: BlockDesc,
}

/// A completed transfer, reported by poll_transfers. bytes feeds the K10
/// wire-byte ledger.
#[cfg(feature = "tch-backend")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferCompletion {
    pub block_id: u64,
    pub bytes: u64,
}

/// Serialized agent metadata advertised over the side channel. The peer calls
/// load_remote_metadata with this blob.
#[cfg(feature = "tch-backend")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub agent: String,
    pub blocks: Vec<RemoteBlockDesc>,
}

/// Block-direct transport: moves registered device-memory blocks with zero
/// serialization (in the NIXL case) instead of streaming serialized frames.
///
/// Lifecycle (both sides register, then exchange metadata, then the sender
/// posts an async transfer into the receiver's registered block):
///
/// 1. register_block — register a local tensor's device memory.
/// 2. local_metadata — serialize agent name + block descriptors; hand to the
///    side channel (in HCP, the coordinator control plane).
/// 3. load_remote_metadata — ingest a peer's metadata blob.
/// 4. submit_transfer(local, remote) — post an async transfer from a local
///    registered block into the peer's registered block described by remote.
/// 5. poll_transfers — collect completed transfers (and their byte counts).
#[cfg(feature = "tch-backend")]
pub trait KvBlockTransport: Send {
    /// Register a local tensor's device memory for zero-copy transfer.
    fn register_block(&mut self, tensor: &Tensor) -> Result<BlockHandle, String>;

    /// Deregister a previously registered block.
    fn deregister_block(&mut self, handle: &BlockHandle) -> Result<(), String>;

    /// Serialized local agent metadata (name + block descriptors) for the side
    /// channel.
    fn local_metadata(&self) -> Result<Vec<u8>, String>;

    /// Ingest a peer's metadata blob; returns the peer agent name.
    fn load_remote_metadata(&mut self, blob: &[u8]) -> Result<String, String>;

    /// Post an async transfer from a local registered block into the peer's
    /// registered block. Returns immediately; completion is observed via
    /// poll_transfers.
    fn submit_transfer(
        &mut self,
        local: &BlockHandle,
        remote: &RemoteBlockDesc,
    ) -> Result<(), String>;

    /// Non-blocking poll of completed transfers.
    fn poll_transfers(&mut self) -> Result<Vec<TransferCompletion>, String>;

    /// Cumulative bytes moved in the send direction (K10 wire-byte caliber).
    fn wire_bytes_sent(&self) -> u64;

    /// Cumulative bytes moved in the recv direction (K10 wire-byte caliber).
    fn wire_bytes_recv(&self) -> u64;
}
