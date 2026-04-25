use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;
use thiserror::Error;

const SCHEMA_VERSION: u16 = 1;
const FRAME_LEN_BYTES: usize = 4;
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const REMOTE_IO_TIMEOUT: Duration = Duration::from_secs(10);
const REMOTE_CLIENT_DOMAIN: &str = "mac-mps";
const REMOTE_SERVER_DOMAIN: &str = "gpu-cuda";

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("domain list must not be empty")]
    EmptyDomains,
    #[error("domain {domain_id} has invalid seq_chunk_len or block_size")]
    InvalidDomain { domain_id: String },
    #[error("invalid socket address {address}: {reason}")]
    InvalidSocketAddress { address: String, reason: String },
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
pub struct RemoteP2pReport {
    pub status: &'static str,
    role: &'static str,
    transport: &'static str,
    bind_addr: Option<String>,
    connect_addr: Option<String>,
    local_addr: Option<String>,
    peer_addr: Option<String>,
    summary: RemoteP2pSummary,
    validated_messages: Vec<RemoteValidatedMessage>,
}

impl RemoteP2pReport {
    pub fn role(&self) -> &'static str {
        self.role
    }

    pub fn transport(&self) -> &'static str {
        self.transport
    }

    pub fn messages_sent(&self) -> usize {
        self.summary.messages_sent
    }

    pub fn messages_received(&self) -> usize {
        self.summary.messages_received
    }
}

#[derive(Default, Serialize)]
struct RemoteP2pSummary {
    kv_block_messages: usize,
    softmax_state_messages: usize,
    terminate_messages: usize,
    messages_sent: usize,
    messages_received: usize,
    bytes_sent: usize,
    bytes_received: usize,
}

#[derive(Serialize)]
struct RemoteValidatedMessage {
    direction: &'static str,
    sequence_id: u64,
    message_kind: &'static str,
    payload_kind: &'static str,
    source_domain: String,
    sender_domain: String,
    receiver_domain: String,
    payload_bytes: usize,
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

pub fn run_remote_p2p_server(bind_addr: &str) -> Result<RemoteP2pReport, ProtocolError> {
    let bind_socket = parse_socket_addr(bind_addr)?;
    let listener = TcpListener::bind(bind_socket)?;
    let actual_bind_addr = listener.local_addr()?.to_string();
    let (mut stream, _) = listener.accept()?;
    configure_stream(&stream)?;
    let local_addr = stream.local_addr()?.to_string();
    let peer_addr = stream.peer_addr()?.to_string();

    let client = remote_client_domain();
    let server = remote_server_domain();
    let mut summary = RemoteP2pSummary::default();
    let mut validated_messages = Vec::new();

    let expected_kv = remote_kv_block_message(&client, &server);
    let kv_message = read_message_frame(&mut stream, &mut summary)?;
    validate_message(&expected_kv, &kv_message)?;
    validated_messages.push(remote_validated_message("recv", &kv_message));

    let ack = remote_softmax_ack_message(&server, &client);
    write_message_frame(&mut stream, &ack, &mut summary)?;
    validated_messages.push(remote_validated_message("send", &ack));

    let expected_terminate = remote_terminate_message(&client, &server);
    let terminate = read_message_frame(&mut stream, &mut summary)?;
    validate_message(&expected_terminate, &terminate)?;
    validated_messages.push(remote_validated_message("recv", &terminate));

    Ok(RemoteP2pReport {
        status: "pass",
        role: "server",
        transport: "tcp_remote_pair",
        bind_addr: Some(actual_bind_addr),
        connect_addr: None,
        local_addr: Some(local_addr),
        peer_addr: Some(peer_addr),
        summary,
        validated_messages,
    })
}

pub fn run_remote_p2p_client(connect_addr: &str) -> Result<RemoteP2pReport, ProtocolError> {
    let connect_socket = parse_socket_addr(connect_addr)?;
    let mut stream = TcpStream::connect(connect_socket)?;
    configure_stream(&stream)?;
    let local_addr = stream.local_addr()?.to_string();
    let peer_addr = stream.peer_addr()?.to_string();

    let client = remote_client_domain();
    let server = remote_server_domain();
    let mut summary = RemoteP2pSummary::default();
    let mut validated_messages = Vec::new();

    let kv_message = remote_kv_block_message(&client, &server);
    write_message_frame(&mut stream, &kv_message, &mut summary)?;
    validated_messages.push(remote_validated_message("send", &kv_message));

    let expected_ack = remote_softmax_ack_message(&server, &client);
    let ack = read_message_frame(&mut stream, &mut summary)?;
    validate_message(&expected_ack, &ack)?;
    validated_messages.push(remote_validated_message("recv", &ack));

    let terminate = remote_terminate_message(&client, &server);
    write_message_frame(&mut stream, &terminate, &mut summary)?;
    validated_messages.push(remote_validated_message("send", &terminate));

    Ok(RemoteP2pReport {
        status: "pass",
        role: "client",
        transport: "tcp_remote_pair",
        bind_addr: None,
        connect_addr: Some(connect_addr.to_string()),
        local_addr: Some(local_addr),
        peer_addr: Some(peer_addr),
        summary,
        validated_messages,
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

fn remote_client_domain() -> DomainSpec {
    DomainSpec {
        domain_id: REMOTE_CLIENT_DOMAIN.to_string(),
        seq_offset: 0,
        seq_chunk_len: 32,
        block_size: 32,
        device: "mps".to_string(),
    }
}

fn remote_server_domain() -> DomainSpec {
    DomainSpec {
        domain_id: REMOTE_SERVER_DOMAIN.to_string(),
        seq_offset: 32,
        seq_chunk_len: 32,
        block_size: 32,
        device: "cuda:0".to_string(),
    }
}

fn remote_kv_block_message(client: &DomainSpec, server: &DomainSpec) -> RingAttnMessage {
    kv_block_message(
        1,
        1,
        client,
        client,
        server,
        client.seq_offset,
        client.seq_offset + client.block_size,
    )
}

fn remote_softmax_ack_message(server: &DomainSpec, client: &DomainSpec) -> RingAttnMessage {
    control_message(
        2,
        RingAttnMessageKind::SoftmaxState,
        PayloadKind::SoftmaxState,
        server,
        client,
        make_softmax_state_payload(server),
    )
}

fn remote_terminate_message(client: &DomainSpec, server: &DomainSpec) -> RingAttnMessage {
    control_message(
        3,
        RingAttnMessageKind::Terminate,
        PayloadKind::Control,
        client,
        server,
        Vec::new(),
    )
}

fn parse_socket_addr(address: &str) -> Result<SocketAddr, ProtocolError> {
    address
        .parse::<SocketAddr>()
        .map_err(|error| ProtocolError::InvalidSocketAddress {
            address: address.to_string(),
            reason: error.to_string(),
        })
}

fn configure_stream(stream: &TcpStream) -> Result<(), ProtocolError> {
    stream.set_read_timeout(Some(REMOTE_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(REMOTE_IO_TIMEOUT))?;
    stream.set_nodelay(true)?;
    Ok(())
}

fn write_message_frame(
    stream: &mut TcpStream,
    message: &RingAttnMessage,
    summary: &mut RemoteP2pSummary,
) -> Result<(), ProtocolError> {
    let frame = serialize_message(message)?;
    if frame.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge { bytes: frame.len() });
    }
    let frame_len = u32::try_from(frame.len())
        .map_err(|_| ProtocolError::FrameTooLarge { bytes: frame.len() })?;
    stream.write_all(&frame_len.to_be_bytes())?;
    stream.write_all(&frame)?;
    summary.messages_sent += 1;
    summary.bytes_sent += FRAME_LEN_BYTES + frame.len();
    count_remote_message(summary, message);
    Ok(())
}

fn read_message_frame(
    stream: &mut TcpStream,
    summary: &mut RemoteP2pSummary,
) -> Result<RingAttnMessage, ProtocolError> {
    let mut len_bytes = [0_u8; FRAME_LEN_BYTES];
    stream.read_exact(&mut len_bytes)?;
    let frame_len = u32::from_be_bytes(len_bytes) as usize;
    if frame_len > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge { bytes: frame_len });
    }
    let mut frame = vec![0_u8; frame_len];
    stream.read_exact(&mut frame)?;
    let message = deserialize_message(&frame)?;
    summary.messages_received += 1;
    summary.bytes_received += FRAME_LEN_BYTES + frame.len();
    count_remote_message(summary, &message);
    Ok(message)
}

fn count_remote_message(summary: &mut RemoteP2pSummary, message: &RingAttnMessage) {
    match message.message_kind {
        RingAttnMessageKind::KvBlock => summary.kv_block_messages += 1,
        RingAttnMessageKind::SoftmaxState => summary.softmax_state_messages += 1,
        RingAttnMessageKind::Terminate => summary.terminate_messages += 1,
    }
}

fn remote_validated_message(
    direction: &'static str,
    message: &RingAttnMessage,
) -> RemoteValidatedMessage {
    RemoteValidatedMessage {
        direction,
        sequence_id: message.sequence_id,
        message_kind: message_kind_name(&message.message_kind),
        payload_kind: payload_kind_name(&message.payload_kind),
        source_domain: message.source_domain.clone(),
        sender_domain: message.sender_domain.clone(),
        receiver_domain: message.receiver_domain.clone(),
        payload_bytes: message.payload.len(),
    }
}

fn message_kind_name(kind: &RingAttnMessageKind) -> &'static str {
    match kind {
        RingAttnMessageKind::KvBlock => "kv_block",
        RingAttnMessageKind::SoftmaxState => "softmax_state",
        RingAttnMessageKind::Terminate => "terminate",
    }
}

fn payload_kind_name(kind: &PayloadKind) -> &'static str {
    match kind {
        PayloadKind::KvBlock => "kv_block",
        PayloadKind::SoftmaxState => "softmax_state",
        PayloadKind::Control => "control",
    }
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
    let message = control_message(
        sequence_id,
        message_kind,
        payload_kind,
        sender,
        receiver,
        payload,
    );
    let frame = serialize_message(&message)?;
    transport.send(&message.sender_domain, &message.receiver_domain, frame)?;
    let decoded =
        deserialize_message(&transport.recv(&message.sender_domain, &message.receiver_domain)?)?;
    validate_message(&message, &decoded)?;
    Ok(1)
}

fn control_message(
    sequence_id: u64,
    message_kind: RingAttnMessageKind,
    payload_kind: PayloadKind,
    sender: &DomainSpec,
    receiver: &DomainSpec,
    payload: Vec<u8>,
) -> RingAttnMessage {
    RingAttnMessage {
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
    }
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
