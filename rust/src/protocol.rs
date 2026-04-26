use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{
    mpsc::{self, Receiver, Sender},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;
use thiserror::Error;

const SCHEMA_VERSION: u16 = 1;
const FRAME_LEN_BYTES: usize = 4;
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const REMOTE_IO_TIMEOUT: Duration = Duration::from_secs(10);
const REMOTE_CONNECT_RETRY_DELAY: Duration = Duration::from_millis(200);
const REMOTE_CONNECT_ATTEMPTS: usize = 300;
const REMOTE_ACCEPT_RETRY_DELAY: Duration = Duration::from_millis(200);
const REMOTE_ACCEPT_ATTEMPTS: usize = 300;
const REMOTE_CLIENT_DOMAIN: &str = "mac-mps";
const REMOTE_SERVER_DOMAIN: &str = "gpu-cuda";
const KV_NUM_HEADS: usize = 3;
const KV_HEAD_DIM: usize = 24;
const KV_TENSOR_COUNT: usize = 2;
const QUERY_CHUNK_LEN: usize = 4;
const FLOAT32_BYTES: usize = 4;

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

#[derive(Clone, Debug)]
struct DomainSpec {
    domain_id: String,
    seq_offset: usize,
    seq_chunk_len: usize,
    block_size: usize,
    device: String,
}

#[derive(Clone, Debug)]
struct DomainModelState {
    seq_offset: usize,
    seq_chunk_len: usize,
    query_len: usize,
    num_heads: usize,
    head_dim: usize,
    q_chunk: Vec<u8>,
    kv_storage: Vec<u8>,
}

impl DomainModelState {
    fn new(domain: &DomainSpec) -> Self {
        let query_len = usize::min(QUERY_CHUNK_LEN, domain.seq_chunk_len);
        let q_values = query_len * KV_NUM_HEADS * KV_HEAD_DIM;
        let kv_values = KV_TENSOR_COUNT * domain.seq_chunk_len * KV_NUM_HEADS * KV_HEAD_DIM;
        let mut q_chunk = Vec::with_capacity(q_values * FLOAT32_BYTES);
        let mut kv_storage = Vec::with_capacity(kv_values * FLOAT32_BYTES);

        for token_offset in 0..query_len {
            let global_token = domain.seq_offset + token_offset;
            for head in 0..KV_NUM_HEADS {
                for dim in 0..KV_HEAD_DIM {
                    let value = model_q_value(domain, global_token, head, dim);
                    q_chunk.extend_from_slice(&value.to_le_bytes());
                }
            }
        }

        for tensor_index in 0..KV_TENSOR_COUNT {
            for token_offset in 0..domain.seq_chunk_len {
                let global_token = domain.seq_offset + token_offset;
                for head in 0..KV_NUM_HEADS {
                    for dim in 0..KV_HEAD_DIM {
                        let value = model_kv_value(domain, tensor_index, global_token, head, dim);
                        kv_storage.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }

        Self {
            seq_offset: domain.seq_offset,
            seq_chunk_len: domain.seq_chunk_len,
            query_len,
            num_heads: KV_NUM_HEADS,
            head_dim: KV_HEAD_DIM,
            q_chunk,
            kv_storage,
        }
    }

    fn build_all(domains: &[DomainSpec]) -> Vec<Self> {
        domains.iter().map(Self::new).collect()
    }

    fn kv_block_payload(&self, block_start: usize, block_stop: usize) -> Vec<u8> {
        debug_assert!(block_start >= self.seq_offset);
        debug_assert!(block_stop <= self.seq_offset + self.seq_chunk_len);
        let block_len = block_stop - block_start;
        let values_per_token = self.num_heads * self.head_dim;
        let bytes_per_token = values_per_token * FLOAT32_BYTES;
        let tensor_stride_bytes = self.seq_chunk_len * bytes_per_token;
        let local_start = block_start - self.seq_offset;
        let mut payload =
            Vec::with_capacity(KV_TENSOR_COUNT * block_len * values_per_token * FLOAT32_BYTES);

        for tensor_index in 0..KV_TENSOR_COUNT {
            let offset = tensor_index * tensor_stride_bytes + local_start * bytes_per_token;
            let bytes = block_len * bytes_per_token;
            payload.extend_from_slice(&self.kv_storage[offset..offset + bytes]);
        }
        payload
    }

    fn query_payload(&self) -> &[u8] {
        &self.q_chunk
    }
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

#[derive(Serialize)]
pub struct CpRingNodeSmokeReport {
    pub status: &'static str,
    transport: &'static str,
    schema_version: u16,
    domains: usize,
    source_blocks: usize,
    expected_kv_messages: usize,
    summary: CpRingNodeSummary,
    nodes: Vec<CpRingNodeReport>,
    #[serde(skip_serializing)]
    payload_blocks: Vec<CpPayloadBlock>,
}

impl CpRingNodeSmokeReport {
    pub fn messages_sent(&self) -> usize {
        self.summary.messages_sent
    }

    pub fn compute_updates(&self) -> usize {
        self.summary.compute_updates
    }

    pub fn payload_blocks(&self) -> &[CpPayloadBlock] {
        &self.payload_blocks
    }
}

#[derive(Serialize)]
pub struct RemoteCpNodeReport {
    pub status: &'static str,
    role: &'static str,
    transport: &'static str,
    node_index: usize,
    domain_id: String,
    bind_addr: String,
    connect_addr: String,
    inbound_peer: String,
    outbound_peer: String,
    local_listener_addr: String,
    outbound_local_addr: String,
    outbound_peer_addr: String,
    inbound_peer_addr: String,
    summary: RemoteCpNodeSummary,
    first_routes: Vec<CpRingRoutePreview>,
    #[serde(skip_serializing)]
    payload_blocks: Vec<CpPayloadBlock>,
}

impl RemoteCpNodeReport {
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

    pub fn compute_updates(&self) -> usize {
        self.summary.compute_updates
    }

    pub fn payload_blocks(&self) -> &[CpPayloadBlock] {
        &self.payload_blocks
    }
}

#[derive(Default, Serialize)]
struct RemoteCpNodeSummary {
    source_blocks: usize,
    payload_blocks_captured: usize,
    initial_messages_sent: usize,
    forwarded_messages_sent: usize,
    messages_sent: usize,
    messages_received: usize,
    compute_updates: usize,
    bytes_sent: usize,
    bytes_received: usize,
}

#[derive(Default, Serialize)]
struct CpRingNodeSummary {
    node_threads: usize,
    source_blocks: usize,
    payload_blocks_captured: usize,
    initial_messages_sent: usize,
    forwarded_messages_sent: usize,
    messages_sent: usize,
    messages_received: usize,
    compute_updates: usize,
    bytes_sent: usize,
    bytes_received: usize,
}

#[derive(Serialize)]
struct CpRingNodeReport {
    domain_id: String,
    role: &'static str,
    inbound_peer: String,
    outbound_peer: String,
    source_blocks: usize,
    initial_messages_sent: usize,
    forwarded_messages_sent: usize,
    messages_received: usize,
    compute_updates: usize,
    bytes_sent: usize,
    bytes_received: usize,
    first_routes: Vec<CpRingRoutePreview>,
    #[serde(skip_serializing)]
    payload_blocks: Vec<CpPayloadBlock>,
}

#[derive(Clone, Serialize)]
struct CpRingRoutePreview {
    sequence_id: u64,
    source_domain: String,
    sender_domain: String,
    receiver_domain: String,
    next_receiver_domain: Option<String>,
    block_start: usize,
    block_len: usize,
    ring_step: usize,
}

#[derive(Clone, Debug)]
pub struct CpPayloadBlock {
    sequence_id: u64,
    compute_domain: String,
    block_len: usize,
    query_len: usize,
    num_heads: usize,
    head_dim: usize,
    query_payload: Vec<u8>,
    payload: Vec<u8>,
}

impl CpPayloadBlock {
    pub fn sequence_id(&self) -> u64 {
        self.sequence_id
    }

    pub fn compute_domain(&self) -> &str {
        &self.compute_domain
    }

    pub fn block_len(&self) -> usize {
        self.block_len
    }

    pub fn query_len(&self) -> usize {
        self.query_len
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn query_payload(&self) -> &[u8] {
        &self.query_payload
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
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
    let model_states = DomainModelState::build_all(&domains);
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
                    &model_states[source_index],
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

pub fn run_cp_ring_node_smoke() -> Result<CpRingNodeSmokeReport, ProtocolError> {
    let domains = default_domains()?;
    let domain_count = domains.len();
    let source_block_counts = domains
        .iter()
        .map(|domain| block_ranges(domain).count())
        .collect::<Vec<_>>();
    let total_source_blocks = source_block_counts.iter().sum::<usize>();
    let expected_kv_messages = total_source_blocks * (domain_count - 1);

    let mut senders = Vec::with_capacity(domain_count);
    let mut receivers = Vec::with_capacity(domain_count);
    for _ in 0..domain_count {
        let (sender, receiver) = mpsc::channel::<Vec<u8>>();
        senders.push(sender);
        receivers.push(Some(receiver));
    }

    let mut handles = Vec::with_capacity(domain_count);
    for domain_index in 0..domain_count {
        let domain_id = domains[domain_index].domain_id.clone();
        let receiver =
            receivers[domain_index]
                .take()
                .ok_or_else(|| ProtocolError::MissingPeer {
                    domain_id: domain_id.clone(),
                })?;
        let next_sender = senders[(domain_index + 1) % domain_count].clone();
        let thread_domains = domains.clone();
        let expected_inbound_messages = total_source_blocks - source_block_counts[domain_index];
        handles.push((
            domain_id.clone(),
            thread::spawn(move || {
                run_cp_ring_node(
                    domain_index,
                    thread_domains,
                    receiver,
                    next_sender,
                    expected_inbound_messages,
                )
            }),
        ));
    }
    drop(senders);

    let mut nodes = Vec::with_capacity(domain_count);
    for (domain_id, handle) in handles {
        let node = handle
            .join()
            .map_err(|_| ProtocolError::NodeThreadPanic { domain_id })??;
        nodes.push(node);
    }
    nodes.sort_by(|lhs, rhs| lhs.domain_id.cmp(&rhs.domain_id));
    let payload_blocks = nodes
        .iter()
        .flat_map(|node| node.payload_blocks.iter().cloned())
        .collect::<Vec<_>>();

    let summary = CpRingNodeSummary {
        node_threads: nodes.len(),
        source_blocks: total_source_blocks,
        payload_blocks_captured: payload_blocks.len(),
        initial_messages_sent: nodes.iter().map(|node| node.initial_messages_sent).sum(),
        forwarded_messages_sent: nodes.iter().map(|node| node.forwarded_messages_sent).sum(),
        messages_sent: nodes
            .iter()
            .map(|node| node.initial_messages_sent + node.forwarded_messages_sent)
            .sum(),
        messages_received: nodes.iter().map(|node| node.messages_received).sum(),
        compute_updates: nodes.iter().map(|node| node.compute_updates).sum(),
        bytes_sent: nodes.iter().map(|node| node.bytes_sent).sum(),
        bytes_received: nodes.iter().map(|node| node.bytes_received).sum(),
    };
    let expected_compute_updates = total_source_blocks * domain_count;
    let status = if summary.node_threads == domain_count
        && summary.initial_messages_sent == total_source_blocks
        && summary.messages_sent == expected_kv_messages
        && summary.messages_received == expected_kv_messages
        && summary.compute_updates == expected_compute_updates
        && summary.payload_blocks_captured == expected_compute_updates
    {
        "pass"
    } else {
        "fail"
    };

    Ok(CpRingNodeSmokeReport {
        status,
        transport: "cp_ring_node_runtime",
        schema_version: SCHEMA_VERSION,
        domains: domain_count,
        source_blocks: total_source_blocks,
        expected_kv_messages,
        summary,
        nodes,
        payload_blocks,
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

pub fn run_remote_cp_node(
    node_index: usize,
    bind_addr: &str,
    connect_addr: &str,
) -> Result<RemoteCpNodeReport, ProtocolError> {
    let domains = remote_cp_domains()?;
    if node_index >= domains.len() {
        return Err(ProtocolError::InvalidNodeIndex {
            node_index,
            domain_count: domains.len(),
        });
    }
    let domain_count = domains.len();
    let local_model_state = DomainModelState::new(&domains[node_index]);
    let source_block_counts = domains
        .iter()
        .map(|domain| block_ranges(domain).count())
        .collect::<Vec<_>>();
    let total_source_blocks = source_block_counts.iter().sum::<usize>();
    let domain = domains[node_index].clone();
    let inbound_peer = domains[(node_index + domain_count - 1) % domain_count].clone();
    let outbound_peer = domains[(node_index + 1) % domain_count].clone();
    let expected_inbound_messages = total_source_blocks - source_block_counts[node_index];
    let expected_forwarded_messages =
        expected_inbound_messages - source_block_counts[(node_index + 1) % domain_count];
    let bind_socket = parse_socket_addr(bind_addr)?;
    let connect_socket = parse_socket_addr(connect_addr)?;

    let listener = TcpListener::bind(bind_socket)?;
    let local_listener_addr = listener.local_addr()?.to_string();

    let outbound_stream = connect_with_retry(connect_socket)?;
    configure_stream(&outbound_stream)?;
    let outbound_local_addr = outbound_stream.local_addr()?.to_string();
    let outbound_peer_addr = outbound_stream.peer_addr()?.to_string();
    let outbound_stream = Arc::new(Mutex::new(outbound_stream));

    let recv_domains = domains.clone();
    let recv_domain_id = domain.domain_id.clone();
    let recv_model_state = local_model_state.clone();
    let recv_outbound_peer = outbound_peer.clone();
    let recv_outbound_stream = Arc::clone(&outbound_stream);
    let recv_handle = thread::spawn(move || {
        receive_remote_cp_node_messages(
            node_index,
            recv_domain_id,
            recv_domains,
            recv_model_state,
            listener,
            expected_inbound_messages,
            recv_outbound_peer,
            recv_outbound_stream,
        )
    });

    let mut summary = RemoteCpNodeSummary::default();
    let mut first_routes = Vec::new();
    let mut payload_blocks = Vec::new();
    for (block_index, (block_start, block_stop)) in block_ranges(&domain).enumerate() {
        let message = kv_block_message(
            cp_ring_sequence_id(node_index, block_index),
            1,
            &domain,
            &local_model_state,
            &domain,
            &outbound_peer,
            block_start,
            block_stop,
        );
        let bytes_sent = {
            let mut stream =
                outbound_stream
                    .lock()
                    .map_err(|error| ProtocolError::ChannelSend {
                        sender_domain: domain.domain_id.clone(),
                        receiver_domain: outbound_peer.domain_id.clone(),
                        reason: error.to_string(),
                    })?;
            write_raw_message_frame(&mut stream, &message)?
        };
        summary.source_blocks += 1;
        summary.initial_messages_sent += 1;
        summary.messages_sent += 1;
        summary.bytes_sent += bytes_sent;
        summary.compute_updates += 1;
        push_payload_block(&mut payload_blocks, &domain, &local_model_state, &message);
        push_remote_cp_route_preview(&mut first_routes, &message, Some(&outbound_peer.domain_id));
    }

    let recv_report = recv_handle
        .join()
        .map_err(|_| ProtocolError::NodeThreadPanic {
            domain_id: domain.domain_id.clone(),
        })??;
    summary.messages_received = recv_report.messages_received;
    summary.bytes_received = recv_report.bytes_received;
    summary.forwarded_messages_sent = recv_report.forwarded_messages_sent;
    summary.messages_sent += recv_report.forwarded_messages_sent;
    summary.bytes_sent += recv_report.bytes_forwarded;
    summary.compute_updates += recv_report.compute_updates;
    payload_blocks.extend(recv_report.payload_blocks);
    summary.payload_blocks_captured = payload_blocks.len();
    for route in recv_report.first_routes {
        if first_routes.len() >= 8 {
            break;
        }
        first_routes.push(route);
    }

    let status = if summary.messages_sent == summary.source_blocks + summary.forwarded_messages_sent
        && summary.initial_messages_sent == source_block_counts[node_index]
        && summary.forwarded_messages_sent == expected_forwarded_messages
        && summary.messages_received == expected_inbound_messages
        && summary.compute_updates == summary.source_blocks + expected_inbound_messages
        && summary.payload_blocks_captured == summary.compute_updates
    {
        "pass"
    } else {
        "fail"
    };

    Ok(RemoteCpNodeReport {
        status,
        role: "listener_and_outbound_peer",
        transport: "tcp_remote_cp_node",
        node_index,
        domain_id: domain.domain_id,
        bind_addr: bind_addr.to_string(),
        connect_addr: connect_addr.to_string(),
        inbound_peer: inbound_peer.domain_id,
        outbound_peer: outbound_peer.domain_id,
        local_listener_addr,
        outbound_local_addr,
        outbound_peer_addr,
        inbound_peer_addr: recv_report.inbound_peer_addr,
        summary,
        first_routes,
        payload_blocks,
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

fn run_cp_ring_node(
    domain_index: usize,
    domains: Vec<DomainSpec>,
    receiver: Receiver<Vec<u8>>,
    next_sender: Sender<Vec<u8>>,
    expected_inbound_messages: usize,
) -> Result<CpRingNodeReport, ProtocolError> {
    let domain_count = domains.len();
    let domain = domains[domain_index].clone();
    let model_state = DomainModelState::new(&domain);
    let inbound_peer = domains[(domain_index + domain_count - 1) % domain_count].clone();
    let outbound_peer = domains[(domain_index + 1) % domain_count].clone();
    let mut report = CpRingNodeReport {
        domain_id: domain.domain_id.clone(),
        role: "listener_and_outbound_peer",
        inbound_peer: inbound_peer.domain_id.clone(),
        outbound_peer: outbound_peer.domain_id.clone(),
        source_blocks: 0,
        initial_messages_sent: 0,
        forwarded_messages_sent: 0,
        messages_received: 0,
        compute_updates: 0,
        bytes_sent: 0,
        bytes_received: 0,
        first_routes: Vec::new(),
        payload_blocks: Vec::new(),
    };

    for (block_index, (block_start, block_stop)) in block_ranges(&domain).enumerate() {
        let message = kv_block_message(
            cp_ring_sequence_id(domain_index, block_index),
            1,
            &domain,
            &model_state,
            &domain,
            &outbound_peer,
            block_start,
            block_stop,
        );
        send_cp_node_frame(&next_sender, &message, &mut report, true)?;
        report.source_blocks += 1;
        report.compute_updates += 1;
        push_payload_block(&mut report.payload_blocks, &domain, &model_state, &message);
        push_cp_route_preview(&mut report, &message, Some(&outbound_peer.domain_id));
    }

    while report.messages_received < expected_inbound_messages {
        let frame = receiver
            .recv()
            .map_err(|error| ProtocolError::ChannelRecv {
                domain_id: domain.domain_id.clone(),
                reason: error.to_string(),
            })?;
        report.bytes_received += frame.len();
        report.messages_received += 1;
        let message = deserialize_message(&frame)?;
        validate_cp_ring_message(domain_index, &domains, &message)?;
        report.compute_updates += 1;
        push_payload_block(&mut report.payload_blocks, &domain, &model_state, &message);

        let should_forward = message.ring_step < domain_count - 1;
        if should_forward {
            let forwarded = forward_kv_message(&message, &domain, &outbound_peer);
            push_cp_route_preview(&mut report, &message, Some(&outbound_peer.domain_id));
            send_cp_node_frame(&next_sender, &forwarded, &mut report, false)?;
        } else {
            push_cp_route_preview(&mut report, &message, None);
        }
    }

    Ok(report)
}

fn cp_ring_sequence_id(domain_index: usize, block_index: usize) -> u64 {
    100_000 + (domain_index as u64) * 10_000 + block_index as u64
}

fn send_cp_node_frame(
    sender: &Sender<Vec<u8>>,
    message: &RingAttnMessage,
    report: &mut CpRingNodeReport,
    is_initial_message: bool,
) -> Result<(), ProtocolError> {
    let frame = serialize_message(message)?;
    let frame_len = frame.len();
    sender
        .send(frame)
        .map_err(|error| ProtocolError::ChannelSend {
            sender_domain: message.sender_domain.clone(),
            receiver_domain: message.receiver_domain.clone(),
            reason: error.to_string(),
        })?;
    if is_initial_message {
        report.initial_messages_sent += 1;
    } else {
        report.forwarded_messages_sent += 1;
    }
    report.bytes_sent += frame_len;
    Ok(())
}

fn forward_kv_message(
    message: &RingAttnMessage,
    sender: &DomainSpec,
    receiver: &DomainSpec,
) -> RingAttnMessage {
    let mut forwarded = message.clone();
    forwarded.ring_step += 1;
    forwarded.sender_domain = sender.domain_id.clone();
    forwarded.receiver_domain = receiver.domain_id.clone();
    forwarded
}

fn push_cp_route_preview(
    report: &mut CpRingNodeReport,
    message: &RingAttnMessage,
    next_receiver_domain: Option<&str>,
) {
    if report.first_routes.len() >= 6 {
        return;
    }
    let Some(block) = &message.block else {
        return;
    };
    report.first_routes.push(CpRingRoutePreview {
        sequence_id: message.sequence_id,
        source_domain: message.source_domain.clone(),
        sender_domain: message.sender_domain.clone(),
        receiver_domain: message.receiver_domain.clone(),
        next_receiver_domain: next_receiver_domain.map(ToOwned::to_owned),
        block_start: block.global_offset,
        block_len: block.block_len,
        ring_step: message.ring_step,
    });
}

fn push_payload_block(
    payload_blocks: &mut Vec<CpPayloadBlock>,
    compute_domain: &DomainSpec,
    model_state: &DomainModelState,
    message: &RingAttnMessage,
) {
    if message.message_kind != RingAttnMessageKind::KvBlock {
        return;
    }
    let (Some(block), Some(tensor)) = (&message.block, &message.tensor) else {
        return;
    };
    payload_blocks.push(CpPayloadBlock {
        sequence_id: message.sequence_id,
        compute_domain: compute_domain.domain_id.clone(),
        block_len: block.block_len,
        query_len: model_state.query_len,
        num_heads: tensor.num_heads,
        head_dim: tensor.head_dim,
        query_payload: model_state.query_payload().to_vec(),
        payload: message.payload.clone(),
    });
}

fn validate_cp_ring_message(
    receiver_index: usize,
    domains: &[DomainSpec],
    message: &RingAttnMessage,
) -> Result<(), ProtocolError> {
    if message.schema_version != SCHEMA_VERSION {
        return Err(ProtocolError::MessageMismatch {
            field: "schema_version",
            expected: SCHEMA_VERSION.to_string(),
            actual: message.schema_version.to_string(),
        });
    }
    if message.message_kind != RingAttnMessageKind::KvBlock {
        return Err(ProtocolError::MessageMismatch {
            field: "message_kind",
            expected: format!("{:?}", RingAttnMessageKind::KvBlock),
            actual: format!("{:?}", message.message_kind),
        });
    }
    if message.payload_kind != PayloadKind::KvBlock {
        return Err(ProtocolError::MessageMismatch {
            field: "payload_kind",
            expected: format!("{:?}", PayloadKind::KvBlock),
            actual: format!("{:?}", message.payload_kind),
        });
    }

    let receiver = &domains[receiver_index];
    if message.receiver_domain != receiver.domain_id {
        return Err(ProtocolError::MessageMismatch {
            field: "receiver_domain",
            expected: receiver.domain_id.clone(),
            actual: message.receiver_domain.clone(),
        });
    }
    let expected_sender = &domains[(receiver_index + domains.len() - 1) % domains.len()];
    if message.sender_domain != expected_sender.domain_id {
        return Err(ProtocolError::MessageMismatch {
            field: "sender_domain",
            expected: expected_sender.domain_id.clone(),
            actual: message.sender_domain.clone(),
        });
    }

    let source_index = domain_index_by_id(domains, &message.source_domain)?;
    let expected_ring_step = (receiver_index + domains.len() - source_index) % domains.len();
    if expected_ring_step == 0 || expected_ring_step >= domains.len() {
        return Err(ProtocolError::MessageMismatch {
            field: "ring_step",
            expected: "1..domain_count-1".to_string(),
            actual: expected_ring_step.to_string(),
        });
    }
    if message.ring_step != expected_ring_step {
        return Err(ProtocolError::MessageMismatch {
            field: "ring_step",
            expected: expected_ring_step.to_string(),
            actual: message.ring_step.to_string(),
        });
    }

    let source = &domains[source_index];
    let Some(block) = &message.block else {
        return Err(ProtocolError::MessageMismatch {
            field: "block",
            expected: "Some(BlockMetadata)".to_string(),
            actual: "None".to_string(),
        });
    };
    if block.source_seq_offset != source.seq_offset {
        return Err(ProtocolError::MessageMismatch {
            field: "block.source_seq_offset",
            expected: source.seq_offset.to_string(),
            actual: block.source_seq_offset.to_string(),
        });
    }
    let source_stop = source.seq_offset + source.seq_chunk_len;
    if block.block_len == 0
        || block.global_offset < source.seq_offset
        || block.global_offset + block.block_len > source_stop
    {
        return Err(ProtocolError::MessageMismatch {
            field: "block.range",
            expected: format!("{}..={source_stop}", source.seq_offset),
            actual: format!("{}+{}", block.global_offset, block.block_len),
        });
    }
    let Some(tensor) = &message.tensor else {
        return Err(ProtocolError::MessageMismatch {
            field: "tensor",
            expected: "Some(TensorMetadata)".to_string(),
            actual: "None".to_string(),
        });
    };
    if tensor.payload_bytes != message.payload.len() {
        return Err(ProtocolError::MessageMismatch {
            field: "tensor.payload_bytes",
            expected: message.payload.len().to_string(),
            actual: tensor.payload_bytes.to_string(),
        });
    }
    let actual_checksum = checksum(&message.payload);
    if tensor.checksum != actual_checksum {
        return Err(ProtocolError::MessageMismatch {
            field: "tensor.checksum",
            expected: actual_checksum.to_string(),
            actual: tensor.checksum.to_string(),
        });
    }
    Ok(())
}

fn domain_index_by_id(domains: &[DomainSpec], domain_id: &str) -> Result<usize, ProtocolError> {
    domains
        .iter()
        .position(|domain| domain.domain_id == domain_id)
        .ok_or_else(|| ProtocolError::MissingPeer {
            domain_id: domain_id.to_string(),
        })
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

fn remote_cp_domains() -> Result<Vec<DomainSpec>, ProtocolError> {
    let domain_count_value = env::var("HCP_REMOTE_CP_DOMAINS").unwrap_or_else(|_| "2".to_string());
    let domain_count = domain_count_value.parse::<usize>().map_err(|_| {
        ProtocolError::UnsupportedRemoteCpDomainCount {
            value: domain_count_value.clone(),
        }
    })?;
    let domains = match domain_count {
        2 => vec![
            DomainSpec {
                domain_id: REMOTE_CLIENT_DOMAIN.to_string(),
                seq_offset: 0,
                seq_chunk_len: 32,
                block_size: 8,
                device: "mps".to_string(),
            },
            DomainSpec {
                domain_id: REMOTE_SERVER_DOMAIN.to_string(),
                seq_offset: 32,
                seq_chunk_len: 32,
                block_size: 8,
                device: "cuda:0".to_string(),
            },
        ],
        3 => vec![
            DomainSpec {
                domain_id: REMOTE_CLIENT_DOMAIN.to_string(),
                seq_offset: 0,
                seq_chunk_len: 32,
                block_size: 8,
                device: "mps".to_string(),
            },
            DomainSpec {
                domain_id: REMOTE_SERVER_DOMAIN.to_string(),
                seq_offset: 32,
                seq_chunk_len: 32,
                block_size: 8,
                device: "cuda:0".to_string(),
            },
            DomainSpec {
                domain_id: "mac-mps-2".to_string(),
                seq_offset: 64,
                seq_chunk_len: 32,
                block_size: 8,
                device: "mps".to_string(),
            },
        ],
        _ => {
            return Err(ProtocolError::UnsupportedRemoteCpDomainCount {
                value: domain_count_value,
            });
        }
    };
    Ok(domains)
}

struct RemoteCpRecvReport {
    messages_received: usize,
    forwarded_messages_sent: usize,
    compute_updates: usize,
    bytes_received: usize,
    bytes_forwarded: usize,
    inbound_peer_addr: String,
    first_routes: Vec<CpRingRoutePreview>,
    payload_blocks: Vec<CpPayloadBlock>,
}

#[allow(clippy::too_many_arguments)]
fn receive_remote_cp_node_messages(
    node_index: usize,
    domain_id: String,
    domains: Vec<DomainSpec>,
    model_state: DomainModelState,
    listener: TcpListener,
    expected_messages: usize,
    outbound_peer: DomainSpec,
    outbound_stream: Arc<Mutex<TcpStream>>,
) -> Result<RemoteCpRecvReport, ProtocolError> {
    let mut stream = accept_with_retry(listener)?;
    configure_stream(&stream)?;
    let inbound_peer_addr = stream.peer_addr()?.to_string();
    let mut report = RemoteCpRecvReport {
        messages_received: 0,
        forwarded_messages_sent: 0,
        compute_updates: 0,
        bytes_received: 0,
        bytes_forwarded: 0,
        inbound_peer_addr,
        first_routes: Vec::new(),
        payload_blocks: Vec::new(),
    };
    while report.messages_received < expected_messages {
        let (message, bytes_received) = read_raw_message_frame(&mut stream)?;
        validate_cp_ring_message(node_index, &domains, &message)?;
        report.messages_received += 1;
        report.compute_updates += 1;
        report.bytes_received += bytes_received;
        push_payload_block(
            &mut report.payload_blocks,
            &domains[node_index],
            &model_state,
            &message,
        );
        let should_forward = message.ring_step < domains.len() - 1;
        if should_forward {
            let forwarded = forward_kv_message(&message, &domains[node_index], &outbound_peer);
            let bytes_forwarded = {
                let mut stream =
                    outbound_stream
                        .lock()
                        .map_err(|error| ProtocolError::ChannelSend {
                            sender_domain: domains[node_index].domain_id.clone(),
                            receiver_domain: outbound_peer.domain_id.clone(),
                            reason: error.to_string(),
                        })?;
                write_raw_message_frame(&mut stream, &forwarded)?
            };
            report.forwarded_messages_sent += 1;
            report.bytes_forwarded += bytes_forwarded;
            push_remote_cp_route_preview(
                &mut report.first_routes,
                &message,
                Some(&outbound_peer.domain_id),
            );
        } else {
            push_remote_cp_route_preview(&mut report.first_routes, &message, None);
        }
    }
    if report.messages_received != expected_messages {
        return Err(ProtocolError::ChannelRecv {
            domain_id,
            reason: format!(
                "expected {expected_messages} messages, received {}",
                report.messages_received
            ),
        });
    }
    Ok(report)
}

fn accept_with_retry(listener: TcpListener) -> Result<TcpStream, ProtocolError> {
    listener.set_nonblocking(true)?;
    let mut last_error = None;
    for _ in 0..REMOTE_ACCEPT_ATTEMPTS {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false)?;
                return Ok(stream);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(REMOTE_ACCEPT_RETRY_DELAY);
            }
            Err(error) => {
                last_error = Some(error);
                thread::sleep(REMOTE_ACCEPT_RETRY_DELAY);
            }
        }
    }
    Err(ProtocolError::Io(last_error.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::TimedOut, "accept retry exhausted")
    })))
}

fn connect_with_retry(connect_socket: SocketAddr) -> Result<TcpStream, ProtocolError> {
    let mut last_error = None;
    for _ in 0..REMOTE_CONNECT_ATTEMPTS {
        match TcpStream::connect(connect_socket) {
            Ok(stream) => return Ok(stream),
            Err(error) => {
                last_error = Some(error);
                thread::sleep(REMOTE_CONNECT_RETRY_DELAY);
            }
        }
    }
    Err(ProtocolError::Io(last_error.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::TimedOut, "connect retry exhausted")
    })))
}

fn write_raw_message_frame(
    stream: &mut TcpStream,
    message: &RingAttnMessage,
) -> Result<usize, ProtocolError> {
    let frame = serialize_message(message)?;
    if frame.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge { bytes: frame.len() });
    }
    let frame_len = u32::try_from(frame.len())
        .map_err(|_| ProtocolError::FrameTooLarge { bytes: frame.len() })?;
    stream.write_all(&frame_len.to_be_bytes())?;
    stream.write_all(&frame)?;
    Ok(FRAME_LEN_BYTES + frame.len())
}

fn read_raw_message_frame(
    stream: &mut TcpStream,
) -> Result<(RingAttnMessage, usize), ProtocolError> {
    let mut len_bytes = [0_u8; FRAME_LEN_BYTES];
    stream.read_exact(&mut len_bytes)?;
    let frame_len = u32::from_be_bytes(len_bytes) as usize;
    if frame_len > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge { bytes: frame_len });
    }
    let mut frame = vec![0_u8; frame_len];
    stream.read_exact(&mut frame)?;
    Ok((deserialize_message(&frame)?, FRAME_LEN_BYTES + frame.len()))
}

fn push_remote_cp_route_preview(
    routes: &mut Vec<CpRingRoutePreview>,
    message: &RingAttnMessage,
    next_receiver_domain: Option<&str>,
) {
    if routes.len() >= 8 {
        return;
    }
    let Some(block) = &message.block else {
        return;
    };
    routes.push(CpRingRoutePreview {
        sequence_id: message.sequence_id,
        source_domain: message.source_domain.clone(),
        sender_domain: message.sender_domain.clone(),
        receiver_domain: message.receiver_domain.clone(),
        next_receiver_domain: next_receiver_domain.map(ToOwned::to_owned),
        block_start: block.global_offset,
        block_len: block.block_len,
        ring_step: message.ring_step,
    });
}

fn remote_kv_block_message(client: &DomainSpec, server: &DomainSpec) -> RingAttnMessage {
    let client_model_state = DomainModelState::new(client);
    kv_block_message(
        1,
        1,
        client,
        &client_model_state,
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

#[allow(clippy::too_many_arguments)]
fn kv_block_message(
    sequence_id: u64,
    ring_step: usize,
    source: &DomainSpec,
    source_model_state: &DomainModelState,
    sender: &DomainSpec,
    receiver: &DomainSpec,
    block_start: usize,
    block_stop: usize,
) -> RingAttnMessage {
    let block_len = block_stop - block_start;
    let payload = source_model_state.kv_block_payload(block_start, block_stop);
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
            num_heads: KV_NUM_HEADS,
            head_dim: KV_HEAD_DIM,
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
            num_heads: KV_NUM_HEADS,
            head_dim: KV_HEAD_DIM,
            payload_bytes: payload.len(),
            checksum: checksum(&payload),
        }),
        payload,
    }
}

fn model_q_value(source: &DomainSpec, global_token: usize, head: usize, dim: usize) -> f32 {
    let source_bias = (source.seq_offset as f32 + 1.0) * 0.0001;
    (global_token as f32 + 1.0) * 0.011
        + head as f32 * 0.007_812_5
        + dim as f32 * 0.000_976_562_5
        + source_bias
        + 0.0625
}

fn model_kv_value(
    source: &DomainSpec,
    tensor_index: usize,
    global_token: usize,
    head: usize,
    dim: usize,
) -> f32 {
    let source_bias = (source.seq_offset as f32 + 1.0) * 0.0001;
    let tensor_bias = tensor_index as f32 * 0.031;
    (global_token as f32 + 1.0) * 0.013
        + head as f32 * 0.017
        + dim as f32 * 0.0019
        + source_bias
        + tensor_bias
}

fn make_softmax_state_payload(source: &DomainSpec) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&(source.seq_chunk_len as u64).to_le_bytes());
    payload.extend_from_slice(&(KV_NUM_HEADS as u64).to_le_bytes());
    payload.extend_from_slice(&(KV_HEAD_DIM as u64).to_le_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_domain(seq_offset: usize, seq_chunk_len: usize) -> DomainSpec {
        DomainSpec {
            domain_id: format!("test-domain-{seq_offset}"),
            seq_offset,
            seq_chunk_len,
            block_size: 2,
            device: "cpu".to_string(),
        }
    }

    #[test]
    fn domain_model_state_slices_kv_blocks_from_owned_storage() {
        let domain = test_domain(8, 6);
        let state = DomainModelState::new(&domain);
        let block_start = 10;
        let block_stop = 13;
        let payload = state.kv_block_payload(block_start, block_stop);

        let mut expected = Vec::new();
        for tensor_index in 0..KV_TENSOR_COUNT {
            for global_token in block_start..block_stop {
                for head in 0..KV_NUM_HEADS {
                    for dim in 0..KV_HEAD_DIM {
                        let value = model_kv_value(&domain, tensor_index, global_token, head, dim);
                        expected.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }

        assert_eq!(payload, expected);
    }

    #[test]
    fn domain_model_state_query_payload_is_domain_local() {
        let lhs_domain = test_domain(0, 8);
        let rhs_domain = test_domain(32, 8);
        let lhs = DomainModelState::new(&lhs_domain);
        let rhs = DomainModelState::new(&rhs_domain);

        assert_eq!(
            lhs.query_payload().len(),
            QUERY_CHUNK_LEN * KV_NUM_HEADS * KV_HEAD_DIM * FLOAT32_BYTES
        );
        assert_ne!(lhs.query_payload(), rhs.query_payload());
    }
}
