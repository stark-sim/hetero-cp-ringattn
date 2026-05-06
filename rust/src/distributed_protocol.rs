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
    /// Run prefill on the given token IDs.
    /// `seq_offset` is the global start position of this chunk (domain0=0, domain1=chunk0_len, etc.)
    Prefill {
        chunk: Vec<i64>,
        seq_offset: i64,
    },
    /// Run single-token decode.
    Decode(i64),
    /// Synchronize global sequence length before decode.
    SyncGlobalSeqLen(usize),
    /// Shutdown the worker.
    Shutdown,
}

/// Response messages sent from worker to coordinator.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum WorkerResponse {
    /// Prefill completed; includes last-token logits as f32 bytes and the
    /// worker's global_seq_len so the coordinator can sync across domains.
    PrefillDone {
        last_logits_bytes: Vec<u8>,
        global_seq_len: usize,
    },
    /// Decode completed; includes logits as f32 bytes.
    DecodeDone {
        logits_bytes: Vec<u8>,
    },
    /// Worker encountered an error.
    Error(String),
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

/// Write a length-prefixed frame to a QUIC send stream.
pub fn write_frame_quic(send: &mut SendStream, payload: &[u8], rt: &Handle) -> Result<(), String> {
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(payload);

    rt.block_on(async {
        send.write_all(&buf).await
            .map_err(|e| format!("write_frame_quic failed: {e}"))
    })
}

/// Read a length-prefixed frame from a QUIC recv stream.
pub fn read_frame_quic(recv: &mut RecvStream, rt: &Handle) -> Result<Vec<u8>, String> {
    let mut len_bytes = [0u8; 4];
    rt.block_on(async {
        crate::quic_transport::read_exact(recv, &mut len_bytes).await
            .map_err(|e| format!("read_frame_quic length failed: {e}"))?;
        let len = u32::from_be_bytes(len_bytes) as usize;
        if len > 64 * 1024 * 1024 {
            return Err(format!("read_frame_quic: frame too large ({len} bytes)"));
        }
        let mut payload = vec![0u8; len];
        crate::quic_transport::read_exact(recv, &mut payload).await
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
    let bytes = serialize(cmd)?;
    write_frame_quic(send, &bytes, rt)
}

/// Receive a command from a QUIC recv stream.
pub fn recv_command_quic(recv: &mut RecvStream, rt: &Handle) -> Result<WorkerCommand, String> {
    let bytes = read_frame_quic(recv, rt)?;
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
        send.write_all(&handshake.to_bytes()).await
            .map_err(|e| format!("write_handshake_quic failed: {e}"))
    })
}

/// Read a handshake from a QUIC recv stream.
pub fn read_handshake_quic(recv: &mut RecvStream, rt: &Handle) -> Result<WorkerHandshake, String> {
    let mut buf = [0u8; WorkerHandshake::SIZE];
    rt.block_on(async {
        crate::quic_transport::read_exact(recv, &mut buf).await
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
                    return Err(format!("failed to connect to {addr} after {attempts} attempts: {e}"));
                }
                std::thread::sleep(Duration::from_millis(delay_ms));
            }
        }
    }
    unreachable!()
}

/// Accept a connection with retry (polls non-blocking listener).
#[allow(dead_code)]
pub fn accept_with_retry(listener: &std::net::TcpListener, attempts: usize, delay_ms: u64) -> Result<TcpStream, String> {
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
            chunk: vec![1, 2, 3],
            seq_offset: 0,
        };
        let bytes = bincode::serialize(&cmd).unwrap();
        println!("Prefill cmd: {:?}", bytes);

        let cmd2 = WorkerCommand::Decode(42);
        let bytes2 = bincode::serialize(&cmd2).unwrap();
        println!("Decode cmd: {:?}", bytes2);

        let cmd3 = WorkerCommand::SyncGlobalSeqLen(11);
        let bytes3 = bincode::serialize(&cmd3).unwrap();
        println!("SyncGlobalSeqLen cmd: {:?}", bytes3);

        let cmd4 = WorkerCommand::Shutdown;
        let bytes4 = bincode::serialize(&cmd4).unwrap();
        println!("Shutdown cmd: {:?}", bytes4);

        let resp = WorkerResponse::PrefillDone {
            last_logits_bytes: vec![0xAB, 0xCD],
            global_seq_len: 11,
        };
        let rbytes = bincode::serialize(&resp).unwrap();
        println!("PrefillDone resp: {:?}", rbytes);

        // WorkerHandshake: domain_id(u64 LE) + capacity_mb(u64 LE) = 16 bytes
        let hs_bytes: Vec<u8> = vec![0,0,0,0,0,0,0,0, 0,16,0,0,0,0,0,0];
        println!("Handshake (expected): {:?}", hs_bytes);
    }
}
