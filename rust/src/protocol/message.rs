use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SCHEMA_VERSION: u16 = 1;
pub(crate) const FLOAT32_BYTES: usize = 4;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("serialization error: {0}")]
    Serialize(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("domain list must not be empty")]
    EmptyDomains,
    #[error("domain {domain_id} has invalid seq_chunk_len or block_size")]
    InvalidDomain { domain_id: String },
    #[error("invalid socket address {address}: {reason}")]
    InvalidSocketAddress { address: String, reason: String },
    #[error("invalid node index {node_index} for domain_count={domain_count}")]
    InvalidNodeIndex {
        node_index: usize,
        domain_count: usize,
    },
    #[error("unsupported HCP_REMOTE_CP_DOMAINS={value}; expected 2 or 3")]
    UnsupportedRemoteCpDomainCount { value: String },
    #[error("frame too large: {bytes} bytes")]
    FrameTooLarge { bytes: usize },
    #[error("transport has no peer for domain={domain_id}")]
    MissingPeer { domain_id: String },
    #[error("transport inbox is empty for domain={domain_id}")]
    EmptyInbox { domain_id: String },
    #[error("message mismatch: {field} expected={expected} actual={actual}")]
    MessageMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("channel send failed from {sender_domain} to {receiver_domain}: {reason}")]
    ChannelSend {
        sender_domain: String,
        receiver_domain: String,
        reason: String,
    },
    #[error("channel recv failed for domain={domain_id}: {reason}")]
    ChannelRecv { domain_id: String, reason: String },
    #[error("node thread panicked for domain={domain_id}")]
    NodeThreadPanic { domain_id: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RingAttnMessageKind {
    KvBlock,
    SoftmaxState,
    Terminate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PayloadKind {
    KvBlock,
    SoftmaxState,
    Control,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct BlockMetadata {
    pub(crate) global_offset: usize,
    pub(crate) block_len: usize,
    pub(crate) source_seq_offset: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct TensorMetadata {
    pub(crate) dtype: String,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) payload_bytes: usize,
    pub(crate) checksum: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct RingAttnMessage {
    pub(crate) schema_version: u16,
    pub(crate) sequence_id: u64,
    pub(crate) layer_index: i32,
    pub(crate) ring_step: usize,
    pub(crate) source_domain: String,
    pub(crate) sender_domain: String,
    pub(crate) receiver_domain: String,
    pub(crate) message_kind: RingAttnMessageKind,
    pub(crate) payload_kind: PayloadKind,
    pub(crate) block: Option<BlockMetadata>,
    pub(crate) tensor: Option<TensorMetadata>,
    pub(crate) payload: Vec<u8>,
}

pub(crate) fn message_kind_name(kind: &RingAttnMessageKind) -> &'static str {
    match kind {
        RingAttnMessageKind::KvBlock => "kv_block",
        RingAttnMessageKind::SoftmaxState => "softmax_state",
        RingAttnMessageKind::Terminate => "terminate",
    }
}

pub(crate) fn payload_kind_name(kind: &PayloadKind) -> &'static str {
    match kind {
        PayloadKind::KvBlock => "kv_block",
        PayloadKind::SoftmaxState => "softmax_state",
        PayloadKind::Control => "control",
    }
}

pub(crate) fn serialize_message(message: &RingAttnMessage) -> Result<Vec<u8>, ProtocolError> {
    bincode::serialize(message)
        .map_err(|e| ProtocolError::Serialize(e.to_string()))
}

pub(crate) fn deserialize_message(frame: &[u8]) -> Result<RingAttnMessage, ProtocolError> {
    bincode::deserialize(frame)
        .map_err(|e| ProtocolError::Serialize(e.to_string()))
}

pub(crate) fn validate_message(
    expected: &RingAttnMessage,
    actual: &RingAttnMessage,
) -> Result<(), ProtocolError> {
    if expected == actual {
        return Ok(());
    }
    macro_rules! check_field {
        ($field:ident) => {
            if expected.$field != actual.$field {
                return Err(ProtocolError::MessageMismatch {
                    field: stringify!($field),
                    expected: format!("{:?}", expected.$field),
                    actual: format!("{:?}", actual.$field),
                });
            }
        };
    }
    check_field!(schema_version);
    check_field!(sequence_id);
    check_field!(layer_index);
    check_field!(ring_step);
    check_field!(source_domain);
    check_field!(sender_domain);
    check_field!(receiver_domain);
    check_field!(message_kind);
    check_field!(payload_kind);
    check_field!(block);
    check_field!(tensor);
    check_field!(payload);
    Ok(())
}

pub(crate) fn checksum(payload: &[u8]) -> u64 {
    payload.iter().fold(0_u64, |acc, byte| {
        acc.wrapping_mul(131).wrapping_add(*byte as u64)
    })
}

pub(crate) fn read_f32(bytes: &[u8], value_index: usize) -> f32 {
    let start = value_index * FLOAT32_BYTES;
    f32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ])
}

/// 构造一个完整的 RingAttnMessage（KV block 类型），用于序列化测试。
pub(crate) fn sample_kv_message() -> RingAttnMessage {
    RingAttnMessage {
        schema_version: SCHEMA_VERSION,
        sequence_id: 42,
        layer_index: 3,
        ring_step: 2,
        source_domain: "domain-a".to_string(),
        sender_domain: "domain-b".to_string(),
        receiver_domain: "domain-c".to_string(),
        message_kind: RingAttnMessageKind::KvBlock,
        payload_kind: PayloadKind::KvBlock,
        block: Some(BlockMetadata {
            global_offset: 16,
            block_len: 8,
            source_seq_offset: 16,
        }),
        tensor: Some(TensorMetadata {
            dtype: "float32".to_string(),
            num_heads: 4,
            head_dim: 16,
            payload_bytes: 1024,
            checksum: 0xDEAD_BEEF,
        }),
        payload: vec![0x01, 0x02, 0x03, 0x04, 0x05],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let original = sample_kv_message();
        let bytes = serialize_message(&original).unwrap();
        let restored = deserialize_message(&bytes).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_serialize_deserialize_payload_integrity() {
        let mut original = sample_kv_message();
        // 用更大的 payload 验证二进制完整性
        original.payload = (0..256u16)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        original.tensor = Some(TensorMetadata {
            dtype: "float32".to_string(),
            num_heads: 4,
            head_dim: 16,
            payload_bytes: original.payload.len(),
            checksum: 0,
        });

        let bytes = serialize_message(&original).unwrap();
        let restored = deserialize_message(&bytes).unwrap();
        assert_eq!(restored.payload, original.payload);
    }

    #[test]
    fn test_schema_version_field_roundtrip() {
        let mut original = sample_kv_message();
        original.schema_version = 999;
        let bytes = serialize_message(&original).unwrap();
        let restored = deserialize_message(&bytes).unwrap();
        assert_eq!(restored.schema_version, 999);
        assert_ne!(restored.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn test_all_message_kinds_roundtrip() {
        let base = sample_kv_message();

        let kinds = vec![
            (RingAttnMessageKind::KvBlock, PayloadKind::KvBlock),
            (RingAttnMessageKind::SoftmaxState, PayloadKind::SoftmaxState),
            (RingAttnMessageKind::Terminate, PayloadKind::Control),
        ];

        for (msg_kind, payload_kind) in kinds {
            let mut msg = base.clone();
            msg.message_kind = msg_kind;
            msg.payload_kind = payload_kind;
            let bytes = serialize_message(&msg).unwrap();
            let restored = deserialize_message(&bytes).unwrap();
            assert_eq!(msg, restored);
        }
    }
}
