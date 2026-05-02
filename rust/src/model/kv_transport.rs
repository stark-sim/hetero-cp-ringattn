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

/// 【双向内存传输通道】用于单进程测试中模拟两个分布式 worker 交换 KV block。
/// 
/// 在真实的分布式环境里，domain0 和 domain1 运行在两台不同的机器上，通过网络互相发送 KV。
/// 在单进程测试里，我们用这个结构模拟网络：
/// - domain0 发送的 KV block 会进入 domain1 的接收队列
/// - domain1 发送的 KV block 会进入 domain0 的接收队列
#[cfg(feature = "tch-backend")]
#[derive(Clone)]
pub struct LinkedMockKvTransport {
    // 【peer_inbox：对方的收件箱】
    // 当我们调用 send 时，数据会被推入这个队列。
    // 这个队列实际上是对端（peer）的 self_inbox，所以对方调用 recv 时就能读到。
    // 
    // 用 Arc<Mutex<VecDeque>> 的原因是：
    // - VecDeque：双端队列，支持从尾部 push、从头部 pop（先进先出）。
    // - Mutex：多线程锁，保证同一时间只有一个线程能读写队列。
    // - Arc：原子引用计数，让多个 transport 实例共享同一个队列（不拷贝数据）。
    peer_inbox: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<KvBlock>>>,

    // 【self_inbox：自己的收件箱】
    // 当我们调用 recv 时，数据会从这个队列弹出。
    // 这个队列由对端在 send 时写入。
    self_inbox: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<KvBlock>>>,
}

#[cfg(feature = "tch-backend")]
impl LinkedMockKvTransport {
    /// 【创建一对互通的传输通道】返回 (t0, t1)。
    /// 
    /// 核心设计：交叉共享队列，让 t0 的发送等于 t1 的接收，反之亦然。
    /// 
    /// 具体做法：
    /// - 创建两个空队列 q0 和 q1。
    /// - t0.peer_inbox = q1（t0 发送 → 写入 q1）
    /// - t0.self_inbox = q0（t0 接收 → 从 q0 读取）
    /// - t1.peer_inbox = q0（t1 发送 → 写入 q0）
    /// - t1.self_inbox = q1（t1 接收 → 从 q1 读取）
    /// 
    /// 结果：
    /// - t0.send() 的数据会被 t1.recv() 读到
    /// - t1.send() 的数据会被 t0.recv() 读到
    pub fn create_pair() -> (Self, Self) {
        let q0 = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
        let q1 = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
        (
            Self { peer_inbox: q1.clone(), self_inbox: q0.clone() },
            Self { peer_inbox: q0.clone(), self_inbox: q1.clone() },
        )
    }
}

#[cfg(feature = "tch-backend")]
impl KvTransport for LinkedMockKvTransport {
    /// 【发送 KV block】把 block 的副本放入对方的收件箱（peer_inbox）。
    /// 
    // shallow_clone() 不拷贝底层浮点数据，只增加引用计数，所以很高效。
    fn send_kv_block(&mut self, block: &KvBlock) -> Result<(), String> {
        let cloned = KvBlock {
            layer_idx: block.layer_idx,
            global_seq_start: block.global_seq_start,
            global_seq_end: block.global_seq_end,
            k: block.k.shallow_clone(),
            v: block.v.shallow_clone(),
        };
        self.peer_inbox.lock().unwrap().push_back(cloned);
        Ok(())
    }

    /// 【接收 KV block】从自己的收件箱（self_inbox）头部取出一个 block。
    /// 
    /// 如果队列为空，返回 Ok(None)，表示暂时没有数据。
    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        Ok(self.self_inbox.lock().unwrap().pop_front())
    }
}

#[cfg(test)]
#[cfg(feature = "tch-backend")]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use tch::{Device, Kind, Tensor};

    /// 验证 TcpKvTransport 通过本地 loopback 发送和接收 KV block 后，
    /// tensor 值与原始值完全一致（float32 原始字节传输，无精度损失）。
    #[test]
    fn test_tcp_kv_transport_roundtrip() {
        let device = Device::Cpu;
        let k = Tensor::randn([1, 4, 8, 16], (Kind::Float, device));
        let v = Tensor::randn([1, 4, 8, 16], (Kind::Float, device));

        let block = KvBlock {
            layer_idx: 2,
            global_seq_start: 16,
            global_seq_end: 24,
            k: k.shallow_clone(),
            v: v.shallow_clone(),
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        // Server: receive the block and return it
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut transport = TcpKvTransport::new(stream, device).unwrap();
            transport.recv_kv_block().unwrap().unwrap()
        });

        // Client: send the block
        let client = thread::spawn(move || {
            let stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            let mut transport = TcpKvTransport::new(stream, device).unwrap();
            transport.send_kv_block(&block).unwrap();
        });

        client.join().unwrap();
        let received = server.join().unwrap();

        // Verify metadata
        assert_eq!(received.layer_idx, 2);
        assert_eq!(received.global_seq_start, 16);
        assert_eq!(received.global_seq_end, 24);
        assert_eq!(received.k.size(), vec![1, 4, 8, 16]);
        assert_eq!(received.v.size(), vec![1, 4, 8, 16]);

        // Verify tensor values: float32 raw bytes round-trip should be exact
        let k_diff = (&k - &received.k).abs().max().double_value(&[]);
        let v_diff = (&v - &received.v).abs().max().double_value(&[]);
        println!("TcpKvTransport roundtrip k_diff={} v_diff={}", k_diff, v_diff);
        assert_eq!(k_diff, 0.0, "K tensor changed after TCP roundtrip");
        assert_eq!(v_diff, 0.0, "V tensor changed after TCP roundtrip");
    }
}
