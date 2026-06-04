#![allow(dead_code)]

#[cfg(feature = "tch-backend")]
use std::io::{Read, Write};
#[cfg(feature = "tch-backend")]
use std::net::TcpStream;
#[cfg(feature = "tch-backend")]
use tch::{Device, Tensor};

use super::block::KvBlock;
use super::r#trait::KvTransport;

/// 【基于 TCP 的 KV Block 传输】
///
/// 帧格式（length-prefixed）：
/// ```text
/// [meta_len: u32 BE] [meta_json] [k_raw_bytes] [v_raw_bytes]
/// ```
///
/// meta_json 包含：layer_idx, global_seq_start/end, k/v shape, k/v bytes 长度, k/v dtype。
/// raw bytes 是 f32 的小端序二进制表示（支持 Float/Half/BFloat16/Double，传输时统一为 f32）。
///
/// 【为什么不直接用 bincode 序列化整个 KvBlock？】
/// - JSON meta 便于人工调试和抓包分析
/// - raw bytes 避免 JSON 对大浮点数组的编码开销（JSON 编码 f32 数组体积大 2~3 倍）
///
/// 【局限性】
/// - TCP 没有内置流控和拥塞控制优化，大 KV block 可能阻塞
/// - 没有加密（生产环境应使用 QUIC 或 TLS）
/// - 超时较短（30s），不适合超慢网络
#[cfg(feature = "tch-backend")]
pub struct TcpKvTransport {
    stream: TcpStream,
    device: Device,
    /// 【内部发送缓冲区】用于 submit_send 的异步化。
    /// TCP 本身是同步流，submit_send 会把完整 frame 先序列化到 buffer，
    /// 在 flush_send 时才一次性写入 stream。
    send_buffer: Vec<u8>,
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
        Ok(Self { stream, device, send_buffer: Vec::new() })
    }

    fn tensor_to_bytes(t: &Tensor) -> Result<(Vec<u8>, String), String> {
        let flat = t.contiguous().view(-1).to_kind(tch::Kind::Float);
        let values: Vec<f32> =
            Vec::try_from(&flat).map_err(|e| format!("tensor to vec failed: {e}"))?;
        let bytes = values.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let dtype = match t.kind() {
            tch::Kind::Float => "float32",
            tch::Kind::Half => "float16",
            tch::Kind::BFloat16 => "bfloat16",
            tch::Kind::Double => "float64",
            _ => "float32",
        };
        Ok((bytes, dtype.to_string()))
    }

    fn bytes_to_tensor(
        bytes: &[u8],
        shape: &[i64],
        device: Device,
        dtype_str: &str,
    ) -> Result<Tensor, String> {
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
        let t = Tensor::from_slice(&values)
            .reshape(shape)
            .to_device(device);
        let kind = match dtype_str {
            "float16" => tch::Kind::Half,
            "bfloat16" => tch::Kind::BFloat16,
            "float64" => tch::Kind::Double,
            _ => tch::Kind::Float,
        };
        Ok(t.to_kind(kind))
    }
}

/// 【把序列化 KV block 追加到 buffer，但不实际写入网络】
#[cfg(feature = "tch-backend")]
fn serialize_block_to_buffer(
    send_buffer: &mut Vec<u8>,
    block: &KvBlock,
) -> Result<(), String> {
    let (k_bytes, k_dtype) = TcpKvTransport::tensor_to_bytes(&block.k)?;
    let (v_bytes, v_dtype) = TcpKvTransport::tensor_to_bytes(&block.v)?;
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
        "k_dtype": k_dtype,
        "v_dtype": v_dtype,
    });
    let meta_bytes = meta.to_string().into_bytes();
    let meta_len = meta_bytes.len() as u32;

    // Frame: [meta_len: u32 BE] [meta_bytes] [k_bytes] [v_bytes]
    send_buffer.extend_from_slice(&meta_len.to_be_bytes());
    send_buffer.extend_from_slice(&meta_bytes);
    send_buffer.extend_from_slice(&k_bytes);
    send_buffer.extend_from_slice(&v_bytes);
    Ok(())
}

#[cfg(feature = "tch-backend")]
impl KvTransport for TcpKvTransport {
    fn submit_send(&mut self, block: &KvBlock) -> Result<(), String> {
        serialize_block_to_buffer(&mut self.send_buffer, block)
    }

    fn poll_recv(&mut self) -> Result<Option<KvBlock>, String> {
        // TCP 是同步流，poll_recv 需要非阻塞读取一个完整 frame。
        // 为了简化实现，这里切换到非阻塞模式尝试读取，如果读不到就切回阻塞模式返回 None。
        // 实际测试中使用 Mock 或 QUIC，TCP transport 使用较少。
        self.stream
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking failed: {e}"))?;

        let result = self.recv_kv_block();

        // 恢复阻塞模式（不影响 send）
        let _ = self.stream.set_nonblocking(false);

        match result {
            Ok(opt) => Ok(opt),
            Err(e) if e.contains("WouldBlock") => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn flush_send(&mut self) -> Result<(), String> {
        if !self.send_buffer.is_empty() {
            self.stream
                .write_all(&self.send_buffer)
                .map_err(|e| format!("flush_send write failed: {e}"))?;
            self.send_buffer.clear();
        }
        Ok(())
    }

    /// 【保留阻塞接收实现】避免 trait 默认实现的 1ms 忙等。
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
        let micro_block_idx = meta["micro_block_idx"].as_u64().unwrap_or(0) as usize;
        let total_micro_blocks = meta["total_micro_blocks"].as_u64().unwrap_or(1) as usize;
        let k_bytes_len = meta["k_bytes"].as_u64().ok_or("missing k_bytes")? as usize;
        let v_bytes_len = meta["v_bytes"].as_u64().ok_or("missing v_bytes")? as usize;
        let k_shape: Vec<i64> = meta["k_shape"]
            .as_array()
            .ok_or("missing k_shape")?
            .iter()
            .map(|v| v.as_i64().ok_or("invalid k_shape"))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let v_shape: Vec<i64> = meta["v_shape"]
            .as_array()
            .ok_or("missing v_shape")?
            .iter()
            .map(|v| v.as_i64().ok_or("invalid v_shape"))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

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

        let k_dtype = meta["k_dtype"].as_str().unwrap_or("float32");
        let v_dtype = meta["v_dtype"].as_str().unwrap_or("float32");
        let k = Self::bytes_to_tensor(&k_bytes, &k_shape, self.device, k_dtype)?;
        let v = Self::bytes_to_tensor(&v_bytes, &v_shape, self.device, v_dtype)?;

        Ok(Some(KvBlock {
            layer_idx,
            global_seq_start,
            global_seq_end,
            k,
            v,
            micro_block_idx,
            total_micro_blocks,
        }))
    }
}
