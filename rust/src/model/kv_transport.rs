#![allow(dead_code)]

#[cfg(feature = "tch-backend")]
use std::io::{Read, Write};
#[cfg(feature = "tch-backend")]
use std::net::TcpStream;
#[cfg(feature = "tch-backend")]
use tch::{Device, Tensor};

/// A Key/Value block exchanged between distributed workers during ring attention.
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct KvBlock {
    pub layer_idx: usize,
    pub global_seq_start: usize,
    pub global_seq_end: usize,
    pub k: Tensor,
    pub v: Tensor,
}

/// Transport for exchanging KV blocks between distributed attention workers.
#[cfg(feature = "tch-backend")]
pub trait KvTransport {
    /// Send a KV block to the next peer in the ring.
    fn send_kv_block(&mut self, block: &KvBlock) -> Result<(), String>;

    /// Receive a KV block from the previous peer in the ring.
    /// Returns `None` when the peer has closed the connection (no more blocks).
    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String>;
}

/// TCP-based KV block transport using length-prefixed JSON metadata + raw f32 bytes.
#[cfg(feature = "tch-backend")]
pub struct TcpKvTransport {
    stream: TcpStream,
    device: Device,
}

#[cfg(feature = "tch-backend")]
impl TcpKvTransport {
    pub fn new(stream: TcpStream, device: Device) -> Result<Self, String> {
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
            .map_err(|e| format!("set_read_timeout failed: {e}"))?;
        stream
            .set_write_timeout(Some(std::time::Duration::from_secs(30)))
            .map_err(|e| format!("set_write_timeout failed: {e}"))?;
        Ok(Self { stream, device })
    }

    fn tensor_to_bytes(t: &Tensor) -> Result<Vec<u8>, String> {
        let flat = t.contiguous().view(-1);
        let values: Vec<f32> =
            Vec::try_from(&flat).map_err(|e| format!("tensor to vec failed: {e}"))?;
        Ok(values.iter().flat_map(|&v| v.to_le_bytes()).collect())
    }

    fn bytes_to_tensor(bytes: &[u8], shape: &[i64], device: Device) -> Result<Tensor, String> {
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let expected = shape.iter().product::<i64>() as usize;
        if values.len() != expected {
            return Err(format!(
                "byte length mismatch: expected {} floats, got {}",
                expected,
                values.len()
            ));
        }
        Ok(Tensor::from_slice(&values)
            .reshape(shape)
            .to_device(device))
    }
}

#[cfg(feature = "tch-backend")]
impl KvTransport for TcpKvTransport {
    fn send_kv_block(&mut self, block: &KvBlock) -> Result<(), String> {
        let k_bytes = Self::tensor_to_bytes(&block.k)?;
        let v_bytes = Self::tensor_to_bytes(&block.v)?;
        let k_shape: Vec<i64> = block.k.size();
        let v_shape: Vec<i64> = block.v.size();

        let meta = serde_json::json!({
            "layer_idx": block.layer_idx,
            "global_seq_start": block.global_seq_start,
            "global_seq_end": block.global_seq_end,
            "k_shape": k_shape,
            "v_shape": v_shape,
            "k_bytes": k_bytes.len(),
            "v_bytes": v_bytes.len(),
        });
        let meta_bytes = meta.to_string().into_bytes();
        let meta_len = meta_bytes.len() as u32;

        // Frame: [meta_len: u32 BE] [meta_bytes] [k_bytes] [v_bytes]
        let total_len = 4 + meta_bytes.len() + k_bytes.len() + v_bytes.len();
        let mut frame = Vec::with_capacity(total_len);
        frame.extend_from_slice(&meta_len.to_be_bytes());
        frame.extend_from_slice(&meta_bytes);
        frame.extend_from_slice(&k_bytes);
        frame.extend_from_slice(&v_bytes);

        self.stream
            .write_all(&frame)
            .map_err(|e| format!("send_kv_block write failed: {e}"))?;
        Ok(())
    }

    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        // Read meta_len
        let mut len_bytes = [0u8; 4];
        match self.stream.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(format!("recv_kv_block read meta_len failed: {e}")),
        }
        let meta_len = u32::from_be_bytes(len_bytes) as usize;

        // Read meta
        let mut meta_bytes = vec![0u8; meta_len];
        self.stream
            .read_exact(&mut meta_bytes)
            .map_err(|e| format!("recv_kv_block read meta failed: {e}"))?;
        let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
            .map_err(|e| format!("recv_kv_block parse meta failed: {e}"))?;

        let layer_idx = meta["layer_idx"]
            .as_u64()
            .ok_or("missing layer_idx")? as usize;
        let global_seq_start = meta["global_seq_start"]
            .as_u64()
            .ok_or("missing global_seq_start")? as usize;
        let global_seq_end = meta["global_seq_end"]
            .as_u64()
            .ok_or("missing global_seq_end")? as usize;
        let k_bytes_len = meta["k_bytes"].as_u64().ok_or("missing k_bytes")? as usize;
        let v_bytes_len = meta["v_bytes"].as_u64().ok_or("missing v_bytes")? as usize;
        let k_shape: Vec<i64> = meta["k_shape"]
            .as_array()
            .ok_or("missing k_shape")?
            .iter()
            .map(|v| v.as_i64().ok_or("invalid k_shape"))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{e}"))?;
        let v_shape: Vec<i64> = meta["v_shape"]
            .as_array()
            .ok_or("missing v_shape")?
            .iter()
            .map(|v| v.as_i64().ok_or("invalid v_shape"))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{e}"))?;

        // Read k_bytes
        let mut k_bytes = vec![0u8; k_bytes_len];
        self.stream
            .read_exact(&mut k_bytes)
            .map_err(|e| format!("recv_kv_block read k_bytes failed: {e}"))?;

        // Read v_bytes
        let mut v_bytes = vec![0u8; v_bytes_len];
        self.stream
            .read_exact(&mut v_bytes)
            .map_err(|e| format!("recv_kv_block read v_bytes failed: {e}"))?;

        let k = Self::bytes_to_tensor(&k_bytes, &k_shape, self.device)?;
        let v = Self::bytes_to_tensor(&v_bytes, &v_shape, self.device)?;

        Ok(Some(KvBlock {
            layer_idx,
            global_seq_start,
            global_seq_end,
            k,
            v,
        }))
    }
}

/// In-memory transport for unit testing distributed attention logic.
#[cfg(feature = "tch-backend")]
pub struct MockKvTransport {
    queue: std::collections::VecDeque<KvBlock>,
}

#[cfg(feature = "tch-backend")]
impl MockKvTransport {
    pub fn new() -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
        }
    }

    pub fn push(&mut self, block: KvBlock) {
        self.queue.push_back(block);
    }
}

#[cfg(feature = "tch-backend")]
impl KvTransport for MockKvTransport {
    fn send_kv_block(&mut self, _block: &KvBlock) -> Result<(), String> {
        Ok(())
    }

    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        Ok(self.queue.pop_front())
    }
}
