use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use thiserror::Error;

const SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("domain list must not be empty")]
    EmptyDomains,
    #[error("domain {domain_id} has invalid seq_chunk_len or block_size")]
    InvalidDomain { domain_id: String },
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
}

#[derive(Clone, Debug)]
struct DomainSpec {
    domain_id: String,
    seq_offset: usize,
    seq_chunk_len: usize,
    block_size: usize,
    device: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RingAttnMessageKind {
    KvBlock,
    SoftmaxState,
    Terminate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PayloadKind {
    KvBlock,
    SoftmaxState,
    Control,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct BlockMetadata {
    global_offset: usize,
    block_len: usize,
    source_seq_offset: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct TensorMetadata {
    dtype: String,
    num_heads: usize,
    head_dim: usize,
    payload_bytes: usize,
    checksum: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct RingAttnMessage {
    schema_version: u16,
    sequence_id: u64,
    layer_index: i32,
    ring_step: usize,
    source_domain: String,
    sender_domain: String,
    receiver_domain: String,
    message_kind: RingAttnMessageKind,
    payload_kind: PayloadKind,
    block: Option<BlockMetadata>,
    tensor: Option<TensorMetadata>,
    payload: Vec<u8>,
}

#[derive(Serialize)]
pub struct ProtocolSmokeReport {
    pub status: &'static str,
    transport: &'static str,
    schema_version: u16,
    domains: usize,
    source_blocks: usize,
    summary: ProtocolSummary,
    links: Vec<LinkReport>,
    route_preview: Vec<RoutePreview>,
}

impl ProtocolSmokeReport {
    pub fn messages_sent(&self) -> usize {
        self.summary.messages_sent
    }
}

#[derive(Serialize)]
struct ProtocolSummary {
    kv_block_messages: usize,
    softmax_state_messages: usize,
    terminate_messages: usize,
    messages_sent: usize,
    messages_received: usize,
    bytes_sent: usize,
    bytes_received: usize,
}

#[derive(Clone, Serialize)]
struct LinkReport {
    sender_domain: String,
    receiver_domain: String,
    sender_device: String,
    receiver_device: String,
}

#[derive(Clone, Serialize)]
struct RoutePreview {
    sequence_id: u64,
    source_domain: String,
    sender_domain: String,
    receiver_domain: String,
    block_start: usize,
    block_len: usize,
    ring_step: usize,
}

#[derive(Default)]
struct TransportMetrics {
    messages_sent: usize,
    messages_received: usize,
    bytes_sent: usize,
    bytes_received: usize,
}

struct Envelope {
    sender_domain: String,
    frame: Vec<u8>,
}

struct LocalP2pTransport {
    inboxes: BTreeMap<String, VecDeque<Envelope>>,
    metrics: TransportMetrics,
}

impl LocalP2pTransport {
    fn new(domains: &[DomainSpec]) -> Self {
        let inboxes = domains
            .iter()
            .map(|domain| (domain.domain_id.clone(), VecDeque::new()))
            .collect();
        Self {
            inboxes,
            metrics: TransportMetrics::default(),
        }
    }

    fn send(
        &mut self,
        sender_domain: &str,
        receiver_domain: &str,
        frame: Vec<u8>,
    ) -> Result<(), ProtocolError> {
        if !self.inboxes.contains_key(sender_domain) {
            return Err(ProtocolError::MissingPeer {
                domain_id: sender_domain.to_string(),
            });
        }
        let Some(inbox) = self.inboxes.get_mut(receiver_domain) else {
            return Err(ProtocolError::MissingPeer {
                domain_id: receiver_domain.to_string(),
            });
        };
        self.metrics.messages_sent += 1;
        self.metrics.bytes_sent += frame.len();
        inbox.push_back(Envelope {
            sender_domain: sender_domain.to_string(),
            frame,
        });
        Ok(())
    }

    fn recv(
        &mut self,
        expected_sender: &str,
        receiver_domain: &str,
    ) -> Result<Vec<u8>, ProtocolError> {
        let Some(inbox) = self.inboxes.get_mut(receiver_domain) else {
            return Err(ProtocolError::MissingPeer {
                domain_id: receiver_domain.to_string(),
            });
        };
        let Some(envelope) = inbox.pop_front() else {
            return Err(ProtocolError::EmptyInbox {
                domain_id: receiver_domain.to_string(),
            });
        };
        if envelope.sender_domain != expected_sender {
            return Err(ProtocolError::MessageMismatch {
                field: "sender_domain",
                expected: expected_sender.to_string(),
                actual: envelope.sender_domain,
            });
        }
        self.metrics.messages_received += 1;
        self.metrics.bytes_received += envelope.frame.len();
        Ok(envelope.frame)
    }
}

pub fn run_protocol_smoke() -> Result<ProtocolSmokeReport, ProtocolError> {
    let domains = default_domains()?;
    let mut transport = LocalP2pTransport::new(&domains);
    let mut route_preview = Vec::new();
    let mut sequence_id = 1_u64;
    let mut source_blocks = 0_usize;
    let mut kv_block_messages = 0_usize;

    for source_index in 0..domains.len() {
        for (block_start, block_stop) in block_ranges(&domains[source_index]) {
            source_blocks += 1;
            for ring_step in 1..domains.len() {
                let sender_index = (source_index + ring_step - 1) % domains.len();
                let receiver_index = (source_index + ring_step) % domains.len();
                let message = kv_block_message(
                    sequence_id,
                    ring_step,
                    &domains[source_index],
                    &domains[sender_index],
                    &domains[receiver_index],
                    block_start,
                    block_stop,
                );
                let frame = serialize_message(&message)?;
                transport.send(&message.sender_domain, &message.receiver_domain, frame)?;
                let decoded = deserialize_message(
                    &transport.recv(&message.sender_domain, &message.receiver_domain)?,
                )?;
                validate_message(&message, &decoded)?;
                if route_preview.len() < 8 {
                    route_preview.push(RoutePreview {
                        sequence_id,
                        source_domain: message.source_domain.clone(),
                        sender_domain: message.sender_domain.clone(),
                        receiver_domain: message.receiver_domain.clone(),
                        block_start,
                        block_len: block_stop - block_start,
                        ring_step,
                    });
                }
                kv_block_messages += 1;
                sequence_id += 1;
            }
        }
    }

    let softmax_state_messages = send_control_message(
        &mut transport,
        sequence_id,
        RingAttnMessageKind::SoftmaxState,
        PayloadKind::SoftmaxState,
        &domains[0],
        &domains[1],
        make_softmax_state_payload(&domains[0]),
    )?;
    sequence_id += 1;
    let terminate_messages = send_control_message(
        &mut transport,
        sequence_id,
        RingAttnMessageKind::Terminate,
        PayloadKind::Control,
        &domains[domains.len() - 1],
        &domains[0],
        Vec::new(),
    )?;

    Ok(ProtocolSmokeReport {
        status: "pass",
        transport: "local_p2p_queue",
        schema_version: SCHEMA_VERSION,
        domains: domains.len(),
        source_blocks,
        summary: ProtocolSummary {
            kv_block_messages,
            softmax_state_messages,
            terminate_messages,
            messages_sent: transport.metrics.messages_sent,
            messages_received: transport.metrics.messages_received,
            bytes_sent: transport.metrics.bytes_sent,
            bytes_received: transport.metrics.bytes_received,
        },
        links: ring_links(&domains),
        route_preview,
    })
}

fn default_domains() -> Result<Vec<DomainSpec>, ProtocolError> {
    let chunks = [64_usize, 40, 56];
    let block_sizes = [32_usize, 10, 14];
    let devices = ["mps", "cuda:0", "cpu"];
    let mut offset = 0_usize;
    let mut domains = Vec::with_capacity(chunks.len());
    for (index, ((seq_chunk_len, block_size), device)) in chunks
        .iter()
        .zip(block_sizes.iter())
        .zip(devices.iter())
        .enumerate()
    {
        if *seq_chunk_len == 0 || *block_size == 0 {
            return Err(ProtocolError::InvalidDomain {
                domain_id: format!("domain-{index}"),
            });
        }
        domains.push(DomainSpec {
            domain_id: format!("domain-{index}"),
            seq_offset: offset,
            seq_chunk_len: *seq_chunk_len,
            block_size: *block_size,
            device: (*device).to_string(),
        });
        offset += *seq_chunk_len;
    }
    if domains.is_empty() {
        return Err(ProtocolError::EmptyDomains);
    }
    Ok(domains)
}

fn ring_links(domains: &[DomainSpec]) -> Vec<LinkReport> {
    domains
        .iter()
        .enumerate()
        .map(|(index, sender)| {
            let receiver = &domains[(index + 1) % domains.len()];
            LinkReport {
                sender_domain: sender.domain_id.clone(),
                receiver_domain: receiver.domain_id.clone(),
                sender_device: sender.device.clone(),
                receiver_device: receiver.device.clone(),
            }
        })
        .collect()
}

fn block_ranges(spec: &DomainSpec) -> impl Iterator<Item = (usize, usize)> + '_ {
    let start = spec.seq_offset;
    let stop = spec.seq_offset + spec.seq_chunk_len;
    (start..stop)
        .step_by(spec.block_size)
        .map(move |block_start| (block_start, usize::min(block_start + spec.block_size, stop)))
}

fn kv_block_message(
    sequence_id: u64,
    ring_step: usize,
    source: &DomainSpec,
    sender: &DomainSpec,
    receiver: &DomainSpec,
    block_start: usize,
    block_stop: usize,
) -> RingAttnMessage {
    let block_len = block_stop - block_start;
    let payload = make_kv_payload(sequence_id, source, block_start, block_len);
    RingAttnMessage {
        schema_version: SCHEMA_VERSION,
        sequence_id,
        layer_index: 0,
        ring_step,
        source_domain: source.domain_id.clone(),
        sender_domain: sender.domain_id.clone(),
        receiver_domain: receiver.domain_id.clone(),
        message_kind: RingAttnMessageKind::KvBlock,
        payload_kind: PayloadKind::KvBlock,
        block: Some(BlockMetadata {
            global_offset: block_start,
            block_len,
            source_seq_offset: source.seq_offset,
        }),
        tensor: Some(TensorMetadata {
            dtype: "float32".to_string(),
            num_heads: 3,
            head_dim: 24,
            payload_bytes: payload.len(),
            checksum: checksum(&payload),
        }),
        payload,
    }
}

fn send_control_message(
    transport: &mut LocalP2pTransport,
    sequence_id: u64,
    message_kind: RingAttnMessageKind,
    payload_kind: PayloadKind,
    sender: &DomainSpec,
    receiver: &DomainSpec,
    payload: Vec<u8>,
) -> Result<usize, ProtocolError> {
    let message = RingAttnMessage {
        schema_version: SCHEMA_VERSION,
        sequence_id,
        layer_index: 0,
        ring_step: 0,
        source_domain: sender.domain_id.clone(),
        sender_domain: sender.domain_id.clone(),
        receiver_domain: receiver.domain_id.clone(),
        message_kind,
        payload_kind,
        block: None,
        tensor: Some(TensorMetadata {
            dtype: "float32".to_string(),
            num_heads: 3,
            head_dim: 24,
            payload_bytes: payload.len(),
            checksum: checksum(&payload),
        }),
        payload,
    };
    let frame = serialize_message(&message)?;
    transport.send(&message.sender_domain, &message.receiver_domain, frame)?;
    let decoded =
        deserialize_message(&transport.recv(&message.sender_domain, &message.receiver_domain)?)?;
    validate_message(&message, &decoded)?;
    Ok(1)
}

fn make_kv_payload(
    sequence_id: u64,
    source: &DomainSpec,
    block_start: usize,
    block_len: usize,
) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&sequence_id.to_le_bytes());
    payload.extend_from_slice(&(source.seq_offset as u64).to_le_bytes());
    payload.extend_from_slice(&(block_start as u64).to_le_bytes());
    payload.extend_from_slice(&(block_len as u64).to_le_bytes());
    payload.extend_from_slice(source.domain_id.as_bytes());
    payload.extend_from_slice(source.device.as_bytes());
    payload
}

fn make_softmax_state_payload(source: &DomainSpec) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&(source.seq_chunk_len as u64).to_le_bytes());
    payload.extend_from_slice(&(3_u64).to_le_bytes());
    payload.extend_from_slice(&(24_u64).to_le_bytes());
    payload.extend_from_slice(source.domain_id.as_bytes());
    payload
}

fn serialize_message(message: &RingAttnMessage) -> Result<Vec<u8>, ProtocolError> {
    Ok(serde_json::to_vec(message)?)
}

fn deserialize_message(frame: &[u8]) -> Result<RingAttnMessage, ProtocolError> {
    Ok(serde_json::from_slice(frame)?)
}

fn validate_message(
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

fn checksum(payload: &[u8]) -> u64 {
    payload.iter().fold(0_u64, |acc, byte| {
        acc.wrapping_mul(131).wrapping_add(*byte as u64)
    })
}
