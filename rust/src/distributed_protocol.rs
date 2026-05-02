//! Control protocol for distributed multi-process inference.
//!
//! Defines messages exchanged between coordinator and workers,
//! plus frame I/O helpers (length-prefixed bytes over TcpStream).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Control messages sent from coordinator to worker.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum WorkerCommand {
    /// Run prefill on the given token IDs.
    Prefill(Vec<i64>),
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
pub fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<(), String> {
    let len = payload.len() as u32;
    stream
        .write_all(&len.to_be_bytes())
        .map_err(|e| format!("write_frame length failed: {e}"))?;
    stream
        .write_all(payload)
        .map_err(|e| format!("write_frame payload failed: {e}"))?;
    Ok(())
}

/// Read a length-prefixed frame from a stream.
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
pub fn send_command(stream: &mut TcpStream, cmd: &WorkerCommand) -> Result<(), String> {
    let bytes = serialize(cmd)?;
    write_frame(stream, &bytes)
}

/// Receive a command from a stream.
pub fn recv_command(stream: &mut TcpStream) -> Result<WorkerCommand, String> {
    let bytes = read_frame(stream)?;
    deserialize(&bytes)
}

/// Send a response to a stream.
pub fn send_response(stream: &mut TcpStream, resp: &WorkerResponse) -> Result<(), String> {
    let bytes = serialize(resp)?;
    write_frame(stream, &bytes)
}

/// Receive a response from a stream.
pub fn recv_response(stream: &mut TcpStream) -> Result<WorkerResponse, String> {
    let bytes = read_frame(stream)?;
    deserialize(&bytes)
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
