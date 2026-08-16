//! Control protocol for distributed multi-process inference.
//!
//! Defines messages exchanged between coordinator and workers,
//! plus frame I/O helpers (length-prefixed bytes over TcpStream or QUIC streams).

use quinn::{RecvStream, SendStream};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use tokio::runtime::Handle;

/// Control messages sent from coordinator to worker.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum WorkerCommand {
    /// Run prefill on the given token IDs for a specific request.
    /// `seq_offset` is the global start position of this chunk (domain0=0, domain1=chunk0_len, etc.)
    /// `position_ids` optionally overrides the default [seq_offset, seq_offset+1, ...) ordering
    /// for non-contiguous scheduling strategies such as Striped or ZigZag.
    /// `layer_kv_capacities` optionally provides this worker's finite-horizon
    /// reserved KV capacity for every model layer.
    Prefill {
        request_id: u64,
        chunk: Vec<i64>,
        seq_offset: i64,
        position_ids: Option<Vec<i64>>,
        layer_kv_capacities: Option<Vec<usize>>,
    },
    /// Run single-token decode for a specific request.
    Decode { request_id: u64, token: i64 },
    /// Run batch decode for multiple requests in a single forward pass.
    /// Each tuple is (request_id, token_to_decode).
    DecodeBatch { request_tokens: Vec<(u64, i64)> },
    /// Run route-B stationary continuation for a specific request:
    /// the continuation segment's historical KV never moves; each worker
    /// projects/appends only its own position offsets from the frozen plan
    /// and the LayerPacket travels hop-by-hop around the ring.
    /// `capacity_tickets` carries the whole-ring tickets so workers stay
    /// stateless about cluster topology; `starter_domain` is selected by the
    /// coordinator (correctness is starter-agnostic).
    StationaryContinuation {
        request_id: u64,
        tokens: Vec<i64>,
        position_ids: Vec<i64>,
        capacity_tickets: Vec<u64>,
        starter_domain: usize,
    },
    /// Mainline self-driving decode step for a specific request: one token is
    /// driven through every layer as a single packet (N-1 hops per layer) with
    /// per-layer KV assignees drawn from the request's frozen decode schedule.
    /// `token_offset` is the 0-based decode step within the request's decode
    /// horizon (`decode_horizon` tokens total); the frozen plan spans
    /// `decode_horizon * layers` units so growth KV stays capacity-balanced
    /// across the whole decode phase.
    StationaryDecode {
        request_id: u64,
        token: i64,
        position: i64,
        capacity_tickets: Vec<u64>,
        starter_domain: usize,
        token_offset: usize,
        decode_horizon: usize,
    },
    /// Synchronize global sequence length before decode.
    SyncGlobalSeqLen { request_id: u64, len: usize },
    /// Release per-request state (KV cache, past_key_values, etc.) for a completed request.
    ReleaseRequest { request_id: u64 },
    /// S3b: request each worker to report its NIXL block-transport metadata for
    /// the coordinator-mediated side channel. Workers reply with NixlMetadata.
    /// The coordinator is the single source of topology knowledge, so NIXL
    /// agent metadata + block descriptors travel over the existing control
    /// plane instead of a separate side-channel port.
    NixlExchange,
    /// S3b: broadcast every peer's NIXL metadata + block descriptors to every
    /// worker. Each entry is (domain_id, nixl_metadata_blob,
    /// serialized_block_descs). Pure bytes: the coordinator relays without
    /// deserializing, so this stays independent of the nixl-backend feature.
    NixlPeers { peers: Vec<(u64, Vec<u8>, Vec<u8>)> },
    /// Shutdown the worker.
    Shutdown,
}

/// Response messages sent from worker to coordinator.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum WorkerResponse {
    /// Prefill completed; includes last-token logits as f32 bytes and the
    /// worker's global_seq_len so the coordinator can sync across domains.
    PrefillDone {
        request_id: u64,
        last_logits_bytes: Vec<u8>,
        global_seq_len: usize,
    },
    /// Decode completed; includes logits as f32 bytes.
    DecodeDone {
        request_id: u64,
        logits_bytes: Vec<u8>,
    },
    /// Batch decode completed; includes logits for each request.
    /// Each tuple is (request_id, logits_bytes).
    DecodeBatchDone { request_logits: Vec<(u64, Vec<u8>)> },
    /// Stationary continuation completed; `logits_bytes` is `Some` only on
    /// the finisher domain (all other workers acknowledge with `None`).
    StationaryContinuationDone {
        request_id: u64,
        logits_bytes: Option<Vec<u8>>,
    },
    /// Worker encountered an error.
    Error { request_id: u64, message: String },
    /// S3b: worker's NIXL metadata (get_local_md blob) + serialized block
    /// descriptors, reported in response to NixlExchange.
    NixlMetadata { metadata: Vec<u8>, block_descs: Vec<u8> },
}

/// Validate the frozen-plan fields of a `StationaryContinuation` command.
/// Pure and transport-free so both coordinator (before broadcast) and every
/// worker (before execution) can run the same check.
pub fn validate_stationary_continuation(
    domains: usize,
    tokens: &[i64],
    position_ids: &[i64],
    capacity_tickets: &[u64],
    starter_domain: usize,
) -> Result<(), String> {
    if domains == 0 {
        return Err("stationary continuation requires at least one domain".to_string());
    }
    if tokens.is_empty() {
        return Err("stationary continuation requires a non-empty segment".to_string());
    }
    if tokens.len() != position_ids.len() {
        return Err(format!(
            "stationary continuation segment/position length mismatch: {} vs {}",
            tokens.len(),
            position_ids.len()
        ));
    }
    if capacity_tickets.len() != domains {
        return Err(format!(
            "stationary continuation tickets/domains mismatch: {} vs {}",
            capacity_tickets.len(),
            domains
        ));
    }
    if capacity_tickets.iter().all(|&t| t == 0) {
        return Err("stationary continuation tickets must not be all zero".to_string());
    }
    if starter_domain >= domains {
        return Err(format!(
            "stationary continuation starter {starter_domain} out of {domains} domains"
        ));
    }
    Ok(())
}

/// Validate the frozen-plan fields of a `StationaryDecode` command.
/// Pure and transport-free so both coordinator (before broadcast) and every
/// worker (before execution) can run the same check.
pub fn validate_stationary_decode(
    domains: usize,
    capacity_tickets: &[u64],
    starter_domain: usize,
    token_offset: usize,
    decode_horizon: usize,
) -> Result<(), String> {
    if domains == 0 {
        return Err("stationary decode requires at least one domain".to_string());
    }
    if capacity_tickets.len() != domains {
        return Err(format!(
            "stationary decode tickets/domains mismatch: {} vs {}",
            capacity_tickets.len(),
            domains
        ));
    }
    if capacity_tickets.iter().all(|&t| t == 0) {
        return Err("stationary decode tickets must not be all zero".to_string());
    }
    if starter_domain >= domains {
        return Err(format!(
            "stationary decode starter {starter_domain} out of {domains} domains"
        ));
    }
    if decode_horizon == 0 {
        return Err("stationary decode horizon must be non-zero".to_string());
    }
    if token_offset >= decode_horizon {
        return Err(format!(
            "stationary decode token_offset {token_offset} out of horizon {decode_horizon}"
        ));
    }
    Ok(())
}

/// Serialize a message to bytes using bincode.
pub fn serialize<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, String> {
    bincode::serialize(value).map_err(|e| format!("serialize failed: {e}"))
}

/// Deserialize bytes to a message using bincode.
pub fn deserialize<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    bincode::deserialize(bytes).map_err(|e| format!("deserialize failed: {e}"))
}

/// Write a length-prefixed frame to a stream.
/// Frame format: [4-byte BE length][payload bytes]
///
/// Uses a manual retry loop instead of `write_all` to handle
/// `ErrorKind::WouldBlock` / `ErrorKind::Interrupted` on high-latency
/// links where TCP send buffers may temporarily stall.
#[allow(dead_code)]
pub fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<(), String> {
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(payload);

    let mut written = 0usize;
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(600);

    while written < buf.len() {
        if start.elapsed() > timeout {
            return Err(format!("write_frame timeout after {:?}", timeout));
        }
        match stream.write(&buf[written..]) {
            Ok(0) => {
                return Err("write_frame: peer closed connection".to_string());
            }
            Ok(n) => {
                written += n;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {
                continue;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
            Err(e) => {
                return Err(format!("write_frame failed: {e}"));
            }
        }
    }
    stream
        .flush()
        .map_err(|e| format!("write_frame flush failed: {e}"))?;
    Ok(())
}

/// Read a length-prefixed frame from a stream.
#[allow(dead_code)]
pub fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let mut len_bytes = [0u8; 4];
    stream
        .read_exact(&mut len_bytes)
        .map_err(|e| format!("read_frame length failed: {e}"))?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(format!("read_frame: frame too large ({len} bytes)"));
    }
    let mut payload = vec![0u8; len];
    stream
        .read_exact(&mut payload)
        .map_err(|e| format!("read_frame payload failed: {e}"))?;
    Ok(payload)
}

/// Send a command to a stream.
#[allow(dead_code)]
pub fn send_command(stream: &mut TcpStream, cmd: &WorkerCommand) -> Result<(), String> {
    let bytes = serialize(cmd)?;
    write_frame(stream, &bytes)
}

/// Receive a command from a stream.
#[allow(dead_code)]
pub fn recv_command(stream: &mut TcpStream) -> Result<WorkerCommand, String> {
    let bytes = read_frame(stream)?;
    deserialize(&bytes)
}

/// Send a response to a stream.
#[allow(dead_code)]
pub fn send_response(stream: &mut TcpStream, resp: &WorkerResponse) -> Result<(), String> {
    let bytes = serialize(resp)?;
    write_frame(stream, &bytes)
}

/// Receive a response from a stream.
#[allow(dead_code)]
pub fn recv_response(stream: &mut TcpStream) -> Result<WorkerResponse, String> {
    let bytes = read_frame(stream)?;
    deserialize(&bytes)
}

/// Handshake payload sent by worker immediately after connecting to coordinator.
///
/// Fixed 16-byte layout (little-endian):
/// - bytes [0..8): domain_id (u64)
/// - bytes [8..16): capacity_score in MB (u64)
#[derive(Debug, Clone, Copy)]
pub struct WorkerHandshake {
    pub domain_id: u64,
    pub capacity_mb: u64,
}

impl WorkerHandshake {
    pub const SIZE: usize = 16;

    pub fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.domain_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.capacity_mb.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        Self {
            domain_id: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            capacity_mb: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        }
    }
}

/// Write a handshake to a stream.
#[allow(dead_code)]
pub fn write_handshake(stream: &mut TcpStream, handshake: &WorkerHandshake) -> Result<(), String> {
    stream
        .write_all(&handshake.to_bytes())
        .map_err(|e| format!("write_handshake failed: {e}"))
}

/// Read a handshake from a stream.
#[allow(dead_code)]
pub fn read_handshake(stream: &mut TcpStream) -> Result<WorkerHandshake, String> {
    let mut buf = [0u8; WorkerHandshake::SIZE];
    stream
        .read_exact(&mut buf)
        .map_err(|e| format!("read_handshake failed: {e}"))?;
    Ok(WorkerHandshake::from_bytes(&buf))
}

// ---------------------------------------------------------------------------
// QUIC variants
// ---------------------------------------------------------------------------

/// Returns the default QUIC frame/command timeout in seconds.
/// Controlled by `HCP_QUIC_TIMEOUT_SECS` environment variable (default: 600).
pub fn default_quic_timeout_secs() -> u64 {
    std::env::var("HCP_QUIC_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600)
}

/// Write a length-prefixed frame to a QUIC send stream.
pub fn write_frame_quic(send: &mut SendStream, payload: &[u8], rt: &Handle) -> Result<(), String> {
    write_frame_quic_timeout(send, payload, rt, default_quic_timeout_secs())
}

/// Write a length-prefixed frame with an explicit timeout.
pub fn write_frame_quic_timeout(
    send: &mut SendStream,
    payload: &[u8],
    rt: &Handle,
    timeout_secs: u64,
) -> Result<(), String> {
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(payload);

    rt.block_on(async {
        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            send.write_all(&buf),
        )
        .await
        .map_err(|_| format!("write_frame_quic timeout after {timeout_secs}s"))?
        .map_err(|e| format!("write_frame_quic failed: {e}"))
    })
}

/// Read a length-prefixed frame from a QUIC recv stream.
pub fn read_frame_quic(recv: &mut RecvStream, rt: &Handle) -> Result<Vec<u8>, String> {
    read_frame_quic_timeout(recv, rt, default_quic_timeout_secs())
}

/// Read a length-prefixed frame with an explicit timeout.
pub fn read_frame_quic_timeout(
    recv: &mut RecvStream,
    rt: &Handle,
    timeout_secs: u64,
) -> Result<Vec<u8>, String> {
    let mut len_bytes = [0u8; 4];
    rt.block_on(async {
        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            crate::distributed::transport::quic::read_exact(recv, &mut len_bytes),
        )
        .await
        .map_err(|_| format!("read_frame_quic length timeout after {timeout_secs}s"))?
        .map_err(|e| format!("read_frame_quic length failed: {e}"))?;
        let len = u32::from_be_bytes(len_bytes) as usize;
        if len > 64 * 1024 * 1024 {
            return Err(format!("read_frame_quic: frame too large ({len} bytes)"));
        }
        let mut payload = vec![0u8; len];
        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            crate::distributed::transport::quic::read_exact(recv, &mut payload),
        )
        .await
        .map_err(|_| format!("read_frame_quic payload timeout after {timeout_secs}s"))?
        .map_err(|e| format!("read_frame_quic payload failed: {e}"))?;
        Ok(payload)
    })
}

/// Send a command over a QUIC send stream.
pub fn send_command_quic(
    send: &mut SendStream,
    cmd: &WorkerCommand,
    rt: &Handle,
) -> Result<(), String> {
    send_command_quic_timeout(send, cmd, rt, default_quic_timeout_secs())
}

/// Send a command with an explicit timeout.
pub fn send_command_quic_timeout(
    send: &mut SendStream,
    cmd: &WorkerCommand,
    rt: &Handle,
    timeout_secs: u64,
) -> Result<(), String> {
    let bytes = serialize(cmd)?;
    write_frame_quic_timeout(send, &bytes, rt, timeout_secs)
}

/// Receive a command from a QUIC recv stream.
pub fn recv_command_quic(recv: &mut RecvStream, rt: &Handle) -> Result<WorkerCommand, String> {
    recv_command_quic_timeout(recv, rt, default_quic_timeout_secs())
}

/// Receive a command with an explicit timeout.
pub fn recv_command_quic_timeout(
    recv: &mut RecvStream,
    rt: &Handle,
    timeout_secs: u64,
) -> Result<WorkerCommand, String> {
    let bytes = read_frame_quic_timeout(recv, rt, timeout_secs)?;
    deserialize(&bytes)
}

/// Send a response over a QUIC send stream.
pub fn send_response_quic(
    send: &mut SendStream,
    resp: &WorkerResponse,
    rt: &Handle,
) -> Result<(), String> {
    let bytes = serialize(resp)?;
    write_frame_quic(send, &bytes, rt)
}

/// Receive a response from a QUIC recv stream.
pub fn recv_response_quic(recv: &mut RecvStream, rt: &Handle) -> Result<WorkerResponse, String> {
    let bytes = read_frame_quic(recv, rt)?;
    deserialize(&bytes)
}

/// Write a handshake to a QUIC send stream.
pub fn write_handshake_quic(
    send: &mut SendStream,
    handshake: &WorkerHandshake,
    rt: &Handle,
) -> Result<(), String> {
    rt.block_on(async {
        send.write_all(&handshake.to_bytes())
            .await
            .map_err(|e| format!("write_handshake_quic failed: {e}"))
    })
}

/// Read a handshake from a QUIC recv stream.
pub fn read_handshake_quic(recv: &mut RecvStream, rt: &Handle) -> Result<WorkerHandshake, String> {
    let mut buf = [0u8; WorkerHandshake::SIZE];
    rt.block_on(async {
        crate::distributed::transport::quic::read_exact(recv, &mut buf)
            .await
            .map_err(|e| format!("read_handshake_quic failed: {e}"))
    })?;
    Ok(WorkerHandshake::from_bytes(&buf))
}

/// Connect to an address with retry.
#[allow(dead_code)]
pub fn connect_with_retry(addr: &str, attempts: usize, delay_ms: u64) -> Result<TcpStream, String> {
    for i in 0..attempts {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
                return Ok(stream);
            }
            Err(e) => {
                if i == attempts - 1 {
                    return Err(format!(
                        "failed to connect to {addr} after {attempts} attempts: {e}"
                    ));
                }
                std::thread::sleep(Duration::from_millis(delay_ms));
            }
        }
    }
    unreachable!()
}

/// Accept a connection with retry (polls non-blocking listener).
#[allow(dead_code)]
pub fn accept_with_retry(
    listener: &std::net::TcpListener,
    attempts: usize,
    delay_ms: u64,
) -> Result<TcpStream, String> {
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("set_nonblocking failed: {e}"))?;
    for i in 0..attempts {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
                return Ok(stream);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if i == attempts - 1 {
                    return Err(format!("accept timeout after {attempts} attempts"));
                }
                std::thread::sleep(Duration::from_millis(delay_ms));
            }
            Err(e) => {
                return Err(format!("accept failed: {e}"));
            }
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bincode_format() {
        let cmd = WorkerCommand::Prefill {
            request_id: 1,
            chunk: vec![1, 2, 3],
            seq_offset: 0,
            position_ids: None,
            layer_kv_capacities: Some(vec![5, 6]),
        };
        let bytes = bincode::serialize(&cmd).unwrap();
        println!("Prefill cmd: {:?}", bytes);
        let decoded: WorkerCommand = bincode::deserialize(&bytes).unwrap();
        let WorkerCommand::Prefill {
            layer_kv_capacities,
            ..
        } = decoded
        else {
            panic!("expected Prefill command");
        };
        assert_eq!(layer_kv_capacities, Some(vec![5, 6]));

        let cmd2 = WorkerCommand::Decode {
            request_id: 1,
            token: 42,
        };
        let bytes2 = bincode::serialize(&cmd2).unwrap();
        println!("Decode cmd: {:?}", bytes2);

        let cmd3 = WorkerCommand::SyncGlobalSeqLen {
            request_id: 1,
            len: 11,
        };
        let bytes3 = bincode::serialize(&cmd3).unwrap();
        println!("SyncGlobalSeqLen cmd: {:?}", bytes3);

        let cmd4 = WorkerCommand::Shutdown;
        let bytes4 = bincode::serialize(&cmd4).unwrap();
        println!("Shutdown cmd: {:?}", bytes4);

        let resp = WorkerResponse::PrefillDone {
            request_id: 1,
            last_logits_bytes: vec![0xAB, 0xCD],
            global_seq_len: 11,
        };
        let rbytes = bincode::serialize(&resp).unwrap();
        println!("PrefillDone resp: {:?}", rbytes);

        // WorkerHandshake: domain_id(u64 LE) + capacity_mb(u64 LE) = 16 bytes
        let hs_bytes: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0];
        println!("Handshake (expected): {:?}", hs_bytes);
    }

    #[test]
    fn stationary_continuation_roundtrips_bincode() {
        let cmd = WorkerCommand::StationaryContinuation {
            request_id: 75,
            tokens: vec![11, 13, 17, 19],
            position_ids: vec![5, 6, 7, 8],
            capacity_tickets: vec![1, 2, 3],
            starter_domain: 2,
        };
        let bytes = serialize(&cmd).unwrap();
        let decoded: WorkerCommand = deserialize(&bytes).unwrap();
        let WorkerCommand::StationaryContinuation {
            request_id,
            tokens,
            position_ids,
            capacity_tickets,
            starter_domain,
        } = decoded
        else {
            panic!("expected StationaryContinuation command");
        };
        assert_eq!(request_id, 75);
        assert_eq!(tokens, vec![11, 13, 17, 19]);
        assert_eq!(position_ids, vec![5, 6, 7, 8]);
        assert_eq!(capacity_tickets, vec![1, 2, 3]);
        assert_eq!(starter_domain, 2);

        let resp = WorkerResponse::StationaryContinuationDone {
            request_id: 75,
            logits_bytes: Some(vec![0xAB, 0xCD]),
        };
        let bytes = serialize(&resp).unwrap();
        let WorkerResponse::StationaryContinuationDone { logits_bytes, .. } =
            deserialize(&bytes).unwrap()
        else {
            panic!("expected StationaryContinuationDone response");
        };
        assert_eq!(logits_bytes, Some(vec![0xAB, 0xCD]));

        let ack = WorkerResponse::StationaryContinuationDone {
            request_id: 75,
            logits_bytes: None,
        };
        let bytes = serialize(&ack).unwrap();
        let WorkerResponse::StationaryContinuationDone { logits_bytes, .. } =
            deserialize(&bytes).unwrap()
        else {
            panic!("expected StationaryContinuationDone ack");
        };
        assert_eq!(logits_bytes, None);
    }

    #[test]
    fn stationary_decode_roundtrips_bincode_and_validates() {
        let cmd = WorkerCommand::StationaryDecode {
            request_id: 76,
            token: 42,
            position: 7,
            capacity_tickets: vec![1, 2, 3],
            starter_domain: 2,
            token_offset: 3,
            decode_horizon: 8,
        };
        let bytes = serialize(&cmd).unwrap();
        let decoded: WorkerCommand = deserialize(&bytes).unwrap();
        let WorkerCommand::StationaryDecode {
            request_id,
            token,
            position,
            capacity_tickets,
            starter_domain,
            token_offset,
            decode_horizon,
        } = decoded
        else {
            panic!("expected StationaryDecode command");
        };
        assert_eq!(request_id, 76);
        assert_eq!(token, 42);
        assert_eq!(position, 7);
        assert_eq!(capacity_tickets, vec![1, 2, 3]);
        assert_eq!(starter_domain, 2);
        assert_eq!(token_offset, 3);
        assert_eq!(decode_horizon, 8);

        let ok = validate_stationary_decode(3, &[1, 2, 3], 2, 3, 8);
        assert!(ok.is_ok());
        // tickets/domains mismatch
        assert!(validate_stationary_decode(2, &[1, 2, 3], 1, 0, 8).is_err());
        // all-zero tickets
        assert!(validate_stationary_decode(2, &[0, 0], 1, 0, 8).is_err());
        // starter out of range
        assert!(validate_stationary_decode(2, &[1, 3], 2, 0, 8).is_err());
        // zero horizon
        assert!(validate_stationary_decode(2, &[1, 3], 1, 0, 0).is_err());
        // token_offset out of horizon
        assert!(validate_stationary_decode(2, &[1, 3], 1, 8, 8).is_err());
        // zero domains
        assert!(validate_stationary_decode(0, &[], 0, 0, 8).is_err());
    }

    #[test]
    fn validate_stationary_continuation_rejects_bad_plans() {
        let ok = validate_stationary_continuation(2, &[11, 13], &[5, 6], &[1, 3], 1);
        assert!(ok.is_ok());
        // empty segment
        assert!(validate_stationary_continuation(2, &[], &[], &[1, 3], 1).is_err());
        // segment/position length mismatch
        assert!(validate_stationary_continuation(2, &[11], &[5, 6], &[1, 3], 1).is_err());
        // tickets/domains mismatch
        assert!(validate_stationary_continuation(2, &[11], &[5], &[1, 3, 2], 1).is_err());
        // all-zero tickets
        assert!(validate_stationary_continuation(2, &[11], &[5], &[0, 0], 1).is_err());
        // starter out of range
        assert!(validate_stationary_continuation(2, &[11], &[5], &[1, 3], 2).is_err());
        // zero domains
        assert!(validate_stationary_continuation(0, &[11], &[5], &[], 0).is_err());
    }

    #[test]
    fn nixl_side_channel_roundtrips_bincode() {
        // Coordinator -> worker: request metadata exchange.
        let exchange = WorkerCommand::NixlExchange;
        let bytes = serialize(&exchange).unwrap();
        assert!(matches!(deserialize::<WorkerCommand>(&bytes).unwrap(), WorkerCommand::NixlExchange));

        // Worker -> coordinator: report metadata + block descriptors (opaque bytes).
        let meta = WorkerResponse::NixlMetadata {
            metadata: vec![0xAB, 0xCD, 0x01, 0x02],
            block_descs: vec![0x10, 0x20, 0x30],
        };
        let bytes = serialize(&meta).unwrap();
        let decoded: WorkerResponse = deserialize(&bytes).unwrap();
        let WorkerResponse::NixlMetadata {
            metadata,
            block_descs,
        } = decoded
        else {
            panic!("expected NixlMetadata response");
        };
        assert_eq!(metadata, vec![0xAB, 0xCD, 0x01, 0x02]);
        assert_eq!(block_descs, vec![0x10, 0x20, 0x30]);

        // Coordinator -> worker: broadcast all peers' metadata.
        let peers = WorkerCommand::NixlPeers {
            peers: vec![(0, vec![1, 2, 3], vec![9, 8]), (1, vec![4, 5], vec![7, 6])],
        };
        let bytes = serialize(&peers).unwrap();
        let decoded: WorkerCommand = deserialize(&bytes).unwrap();
        let WorkerCommand::NixlPeers { peers: p } = decoded else {
            panic!("expected NixlPeers command");
        };
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], (0, vec![1, 2, 3], vec![9, 8]));
        assert_eq!(p[1], (1, vec![4, 5], vec![7, 6]));
    }
}
