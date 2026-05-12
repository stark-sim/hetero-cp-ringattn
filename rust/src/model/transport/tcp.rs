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
/// meta_json 包含：layer_idx, global_seq_start/end, k/v shape, k/v bytes 长度。
/// raw bytes 是 f32 的小端序二进制表示。
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
