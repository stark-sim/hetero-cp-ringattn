#![allow(dead_code)]

//! In-memory block transport implementing [crate::model::transport::block_transport::KvBlockTransport].
//!
//! This is the Mac-testable reference baseline for the NIXL block-direct
//! transport: it keeps the same register → metadata → transfer → poll
//! lifecycle and descriptor model, but moves bytes through shared inboxes
//! instead of UCX. It exists to (a) prove the trait surface and (b) give the
//! NIXL implementation a semantics contract to match before any GPU work.

#[cfg(feature = "tch-backend")]
use super::block_transport::{
    AgentMetadata, BlockDesc, BlockHandle, KvBlockTransport, RemoteBlockDesc, TransferCompletion,
};
#[cfg(feature = "tch-backend")]
use std::collections::HashMap;
#[cfg(feature = "tch-backend")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// A serialized block payload plus routing metadata.
#[cfg(feature = "tch-backend")]
struct BlockFrame {
    /// Peer-local block id this frame is destined for.
    block_id: u64,
    /// Serialized tensor bytes.
    bytes: Vec<u8>,
}

#[cfg(feature = "tch-backend")]
fn tensor_to_bytes(t: &Tensor) -> Result<Vec<u8>, String> {
    let flat = t.contiguous().view(-1).to_kind(tch::Kind::Float);
    let values: Vec<f32> =
        Vec::try_from(&flat).map_err(|e| format!("tensor to vec failed: {e}"))?;
    Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

#[cfg(feature = "tch-backend")]
fn bytes_to_tensor(bytes: &[u8], shape: &[i64], device: tch::Device) -> Result<Tensor, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("byte length not f32-aligned: {}", bytes.len()));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_slice(&values).reshape(shape).to_device(device))
}

/// In-memory block transport. A pair is linked via shared inboxes so a
/// transfer posted on one side lands in the peer's registered block, mirroring
/// NIXL's registered-buffer handoff.
#[cfg(feature = "tch-backend")]
pub struct SerializedBlockTransport {
    agent: String,
    next_block_id: u64,
    /// block id -> (registered tensor shallow clone, advertised descriptor).
    blocks: HashMap<u64, (Tensor, BlockDesc)>,
    /// peer inbox: frames pushed by the peer toward this agent.
    peer_inbox: Arc<Mutex<Vec<BlockFrame>>>,
    /// self inbox: frames this agent pushes toward the peer.
    self_inbox: Arc<Mutex<Vec<BlockFrame>>>,
    wire_sent: u64,
    wire_recv: u64,
}

#[cfg(feature = "tch-backend")]
impl SerializedBlockTransport {
    /// Create a linked pair (a, b): a's transfers land in b's blocks and vice
    /// versa.
    pub fn create_pair() -> (Self, Self) {
        let a_inbox = Arc::new(Mutex::new(Vec::<BlockFrame>::new()));
        let b_inbox = Arc::new(Mutex::new(Vec::<BlockFrame>::new()));
        (
            Self::new("agent-a".to_string(), b_inbox.clone(), a_inbox.clone()),
            Self::new("agent-b".to_string(), a_inbox, b_inbox),
        )
    }

    fn new(
        agent: String,
        peer_inbox: Arc<Mutex<Vec<BlockFrame>>>,
        self_inbox: Arc<Mutex<Vec<BlockFrame>>>,
    ) -> Self {
        Self {
            agent,
            next_block_id: 0,
            blocks: HashMap::new(),
            peer_inbox,
            self_inbox,
            wire_sent: 0,
            wire_recv: 0,
        }
    }
}

#[cfg(feature = "tch-backend")]
impl KvBlockTransport for SerializedBlockTransport {
    fn register_block(&mut self, tensor: &Tensor) -> Result<BlockHandle, String> {
        let id = self.next_block_id;
        self.next_block_id += 1;
        let bytes = tensor_to_bytes(tensor)?;
        let desc = BlockDesc {
            addr: tensor.data_ptr() as u64,
            len: bytes.len() as u64,
            dev_id: 0,
            meta: serde_json::to_vec(&tensor.size())
                .map_err(|e| format!("serialize shape meta failed: {e}"))?,
        };
        self.blocks
            .insert(id, (tensor.shallow_clone(), desc.clone()));
        Ok(BlockHandle { id, desc })
    }

    fn deregister_block(&mut self, handle: &BlockHandle) -> Result<(), String> {
        self.blocks.remove(&handle.id);
        Ok(())
    }

    fn local_metadata(&self) -> Result<Vec<u8>, String> {
        let blocks = self
            .blocks
            .iter()
            .map(|(&id, (_, desc))| RemoteBlockDesc {
                agent: self.agent.clone(),
                block_id: id,
                desc: desc.clone(),
            })
            .collect();
        bincode::serialize(&AgentMetadata {
            agent: self.agent.clone(),
            blocks,
        })
        .map_err(|e| format!("serialize agent metadata failed: {e}"))
    }

    fn load_remote_metadata(&mut self, blob: &[u8]) -> Result<String, String> {
        let meta: AgentMetadata = bincode::deserialize(blob)
            .map_err(|e| format!("deserialize agent metadata failed: {e}"))?;
        Ok(meta.agent)
    }

    fn submit_transfer(
        &mut self,
        local: &BlockHandle,
        remote: &RemoteBlockDesc,
    ) -> Result<(), String> {
        let (tensor, _) = self
            .blocks
            .get(&local.id)
            .ok_or_else(|| format!("local block {} not registered", local.id))?;
        let bytes = tensor_to_bytes(tensor)?;
        self.wire_sent += bytes.len() as u64;
        self.self_inbox.lock().unwrap().push(BlockFrame {
            block_id: remote.block_id,
            bytes,
        });
        Ok(())
    }

    fn poll_transfers(&mut self) -> Result<Vec<TransferCompletion>, String> {
        let mut completions = Vec::new();
        let mut inbox = self.peer_inbox.lock().unwrap();
        while let Some(frame) = inbox.pop() {
            let (tensor, _) = self
                .blocks
                .get_mut(&frame.block_id)
                .ok_or_else(|| format!("received frame for unknown block {}", frame.block_id))?;
            let shape = tensor.size();
            let received = bytes_to_tensor(&frame.bytes, &shape, tensor.device())?;
            let _ = &*tensor;
            *tensor = received;
            self.wire_recv += frame.bytes.len() as u64;
            completions.push(TransferCompletion {
                block_id: frame.block_id,
                bytes: frame.bytes.len() as u64,
            });
        }
        Ok(completions)
    }

    fn wire_bytes_sent(&self) -> u64 {
        self.wire_sent
    }

    fn wire_bytes_recv(&self) -> u64 {
        self.wire_recv
    }
}

#[cfg(all(test, feature = "tch-backend"))]
mod tests {
    use super::*;
    use crate::model::transport::block_transport::KvBlockTransport;
    use tch::{Device, Kind, Tensor};

    /// A pair of linked in-memory block transports round-trips a registered
    /// block: sender registers → exchanges metadata → submit_transfer →
    /// receiver poll_transfers → receiver's block now holds the sender bytes.
    /// This proves the block-direct lifecycle that NIXL must match.
    #[test]
    fn serialized_block_transport_roundtrips_registered_block() {
        let device = Device::Cpu;
        let (mut sender, mut receiver) = SerializedBlockTransport::create_pair();

        // Receiver registers the destination block (empty), sender registers
        // the source block (data). In the real ring the destination is a
        // pre-allocated receive buffer.
        let dest_tensor = Tensor::zeros([1, 2, 3, 4], (Kind::Float, device));
        let dest_handle = receiver.register_block(&dest_tensor).unwrap();

        let src_tensor = Tensor::arange(24, (Kind::Float, device)).reshape([1, 2, 3, 4]);
        let src_handle = sender.register_block(&src_tensor).unwrap();

        // Exchange metadata via the side channel (here: direct call).
        let sender_md = sender.local_metadata().unwrap();
        let receiver_md = receiver.local_metadata().unwrap();
        let _sender_agent = receiver.load_remote_metadata(&sender_md).unwrap();
        let _receiver_agent = sender.load_remote_metadata(&receiver_md).unwrap();

        // Sender transfers its source block into the receiver's dest block.
        let remote = RemoteBlockDesc {
            agent: "agent-b".to_string(),
            block_id: dest_handle.id,
            desc: dest_handle.desc.clone(),
        };
        sender.submit_transfer(&src_handle, &remote).unwrap();

        // Receiver polls and observes completion with the exact byte count.
        let completions = receiver.poll_transfers().unwrap();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].block_id, dest_handle.id);

        // Receiver's block now holds the source bytes.
        let (received, _) = receiver.blocks.get(&dest_handle.id).unwrap();
        let diff = (&src_tensor - received).abs().max().double_value(&[]);
        assert_eq!(diff, 0.0, "block bytes changed across in-memory transfer");

        // K10 wire-byte accounting: sent == recv == serialized size.
        let expected_bytes = 24 * 4; // 24 f32 elements
        assert_eq!(sender.wire_bytes_sent(), expected_bytes);
        assert_eq!(receiver.wire_bytes_recv(), expected_bytes);
        assert_eq!(completions[0].bytes, expected_bytes as u64);
    }
}
