//! QUIC-based KV transport for distributed ring attention.
//!
//! 【Step 2 架构：Async Task + Channel Split-Phase】
//!
//! 内部维护两个独立的 tokio spawned tasks：
//! - **send task**：从 mpsc channel 接收序列化后的 frame，写入 QUIC send stream
//! - **recv task**：从 QUIC recv stream 读取 frame，反序列化后推入 mpsc channel
//!
//! 主线程通过 split-phase API 与 tasks 交互：
//! - `submit_send()`：序列化 block → 推入 send channel（不等待网络写入完成）
//! - `poll_recv()`：非阻塞检查 recv channel，有数据就返回
//! - `flush_send()`：发送 flush marker，等待 send task 确认所有之前的数据已交给 QUIC
//!
//! 这种架构使得 attention 计算可以与 KV 传输完全重叠：
//! 主线程在 `process_kv_block()` 计算的同时，send task 在后台把下一个 block
//! 写入网络，recv task 在后台等待接收 peer block。
#[cfg(feature = "tch-backend")]
use crate::model::transport::{KvBlock, KvTransport};
#[cfg(feature = "tch-backend")]
use quinn::SendStream;
use quinn::{ClientConfig, Endpoint, RecvStream, ServerConfig};
use rustls::client::danger::{ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use std::net::SocketAddr;
use std::sync::Arc;
#[cfg(feature = "tch-backend")]
use tch::{Device, Tensor};
#[cfg(feature = "tch-backend")]
use tokio::runtime::Handle;
#[cfg(feature = "tch-backend")]
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
        ]
    }
}

pub fn create_endpoint(listen_addr: SocketAddr) -> Result<Endpoint, String> {
    // Self-signed cert for server side
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])
        .map_err(|e| format!("cert generation failed: {e}"))?;
    let cert_der = cert.cert.der().clone();
    let key_der = cert.key_pair.serialize_der();

    let cert_chain = vec![cert_der];
    let key = rustls::pki_types::PrivateKeyDer::try_from(key_der)
        .map_err(|e| format!("key conversion failed: {e}"))?;

    let mut server_config = ServerConfig::with_single_cert(cert_chain, key)
        .map_err(|e| format!("server config failed: {e}"))?;
    let transport_config = Arc::get_mut(&mut server_config.transport).unwrap();
    transport_config.max_concurrent_bidi_streams(256u32.into());
    transport_config.max_concurrent_uni_streams(256u32.into());
    // Aggressive keep-alive to prevent NAT/firewall from dropping idle UDP mappings.
    // With 1.2s RTT cross-VPN, NAT idle timeouts (often 30-60s) can expire during
    // long prefill computation gaps. Keep-alive every 1s ensures NAT table refresh.
    transport_config.keep_alive_interval(Some(std::time::Duration::from_secs(1)));
    transport_config.max_idle_timeout(Some(std::time::Duration::from_secs(3600).try_into().unwrap()));
    // Disable MTU discovery: Tailscale WireGuard MTU is 1280, and PMTUD may probe
    // larger sizes that get dropped by intermediate devices. Stick to conservative
    // 1200 bytes to avoid fragmentation-related packet loss on high-RTT paths.
    transport_config.mtu_discovery_config(None);
    transport_config.initial_mtu(1200);
    // Increase stream window to accommodate large KV blocks (e.g. 1.3MB for 1365 tokens).
    // Default ~1.2MB is insufficient for ring-KV exchange deadlocking.
    // GQA repeat 后 KV block 大小 = 2 * num_heads * seq * head_dim * 4 bytes.
    // 8192 tokens → ~58.7MB, 16384 tokens → ~117MB, 32768 tokens → ~224MB.
    // 必须同时增大 send_window 和 receive_window，否则 ring 中双方同时 write_all
    // 大 block 时会因为发送端窗口耗尽而互相死锁。
    transport_config.stream_receive_window((512u64 * 1024 * 1024).try_into().unwrap());
    transport_config.receive_window((1024u64 * 1024 * 1024).try_into().unwrap());
    // 1GB send_window to cover 64K+ seq distributed prefill:
    // 32K seq KV block = ~224MB (K+V), two domains send simultaneously = ~448MB.
    // 256MB was insufficient and caused deadlock. 1GB provides headroom for 128K.
    transport_config.send_window(1024u64 * 1024 * 1024);

    let mut endpoint = Endpoint::server(server_config, listen_addr)
        .map_err(|e| format!("bind failed: {e}"))?;

    // Client config so this endpoint can also dial outbound
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    let quic_client_config = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .map_err(|e| format!("quic client config failed: {e}"))?,
    ));
    endpoint.set_default_client_config(quic_client_config);

    Ok(endpoint)
}

#[derive(Debug)]
pub enum ReadExactError {
    Closed,
    ReadError(quinn::ReadError),
}

impl std::fmt::Display for ReadExactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadExactError::Closed => write!(f, "stream closed"),
            ReadExactError::ReadError(e) => write!(f, "{e}"),
        }
    }
}

pub async fn read_exact(stream: &mut RecvStream, buf: &mut [u8]) -> Result<(), ReadExactError> {
    let mut offset = 0;
    while offset < buf.len() {
        match stream.read(&mut buf[offset..]).await {
            Ok(Some(n)) => offset += n,
            Ok(None) => return Err(ReadExactError::Closed),
            Err(e) => return Err(ReadExactError::ReadError(e)),
        }
    }
    Ok(())
}

/// 【发送命令】Data = 序列化后的 frame；Flush = 要求 send task 确认所有数据已提交。
#[cfg(feature = "tch-backend")]
enum SendCmd {
    Data(Vec<u8>),
    Flush(oneshot::Sender<()>),
}

/// 【QUIC KV Transport — Split-Phase 实现】
///
/// 内部包含两个独立的 tokio tasks（send / recv），主线程通过 channel 与之交互。
/// 所有 Tensor 序列化/反序列化发生在主线程（submit_send）和 recv task 中，
/// channel 中只传递 `Vec<u8>`，避免 Tensor 跨线程移动的问题。
#[cfg(feature = "tch-backend")]
pub struct QuicKvTransport {
    /// 向 send task 发送命令（序列化 frame 或 flush marker）
    send_tx: mpsc::Sender<SendCmd>,
    /// 从 recv task 接收反序列化后的 KvBlock
    recv_rx: mpsc::Receiver<KvBlock>,
    /// send task 的 JoinHandle（Drop 时需要 abort）
    #[allow(dead_code)]
    send_task: tokio::task::JoinHandle<()>,
    /// recv task 的 JoinHandle（Drop 时需要 abort）
    #[allow(dead_code)]
    recv_task: tokio::task::JoinHandle<()>,
    rt: Handle,
    device: Device,
}

#[cfg(feature = "tch-backend")]
impl QuicKvTransport {
    pub fn new(send: SendStream, recv: RecvStream, rt: Handle, device: Device) -> Self {
        // Channel buffer = 1024：允许网络传输多个 block 的同时，主线程序列化后续 block。
        // 这是 N-domain Serial 模式必需的：一次性 submit N 个 layer 的 blocks 时，
        // 如果 buffer 太小（如 2），send_task 和 recv_task 会在网络/缓冲区阻塞时互相死锁。
        // 对于 1M context，micro block 数量可达数百个，64 不够；1024 提供足够 headroom。
        let (send_tx, send_rx) = mpsc::channel::<SendCmd>(1024);
        let (recv_tx, recv_rx) = mpsc::channel::<KvBlock>(1024);

        let send_task = rt.spawn(send_task_loop(send, send_rx));
        let recv_task = rt.spawn(recv_task_loop(recv, recv_tx, device));

        Self {
            send_tx,
            recv_rx,
            send_task,
            recv_task,
            rt,
            device,
        }
    }
}

/// 【Send Task】从 channel 接收序列化 frame，写入 QUIC send stream。
///
/// 这个 task 独立运行，即使主线程在进行 attention 计算，它也在后台
/// 把 KV block 写入网络，实现计算-通信重叠。
#[cfg(feature = "tch-backend")]
async fn send_task_loop(mut send: SendStream, mut cmd_rx: mpsc::Receiver<SendCmd>) {
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            SendCmd::Data(frame) => {
                if let Err(e) = send.write_all(&frame).await {
                    eprintln!("[quic send_task] write_all failed: {e}");
                    break;
                }
            }
            SendCmd::Flush(ack) => {
                // 所有之前的数据已经 write_all 进入 QUIC 发送缓冲区，直接 ack。
                // 注意：不调用 send.finish()，那样会关闭整个 stream。
                let _ = ack.send(());
            }
        }
    }
    // channel 关闭或出错，优雅退出。recv_task 会自行处理 stream 的另一端。
}

/// 【Recv Task】从 QUIC recv stream 读取 frame，反序列化后推入 channel。
///
/// 这个 task 独立运行，即使主线程在进行 attention 计算，它也在后台
/// 等待接收 peer block，一有数据就推入 channel 供 poll_recv() 消费。
#[cfg(feature = "tch-backend")]
async fn recv_task_loop(mut recv: RecvStream, block_tx: mpsc::Sender<KvBlock>, device: Device) {
    let mut handshake_done = false;
    loop {
        match recv_kv_block_from_stream(&mut recv, &mut handshake_done, device).await {
            Ok(Some(block)) => {
                if block_tx.send(block).await.is_err() {
                    break; // 主线程已 drop recv_rx，不需要继续接收
                }
            }
            Ok(None) => break, // stream 正常关闭（peer 发送了 FIN）
            Err(e) => {
                eprintln!("[quic recv_task] error: {e}");
                break;
            }
        }
    }
    // block_tx 在这里被 drop，recv_rx.recv() 会返回 None，通知主线程 stream 已关闭
}

/// 【序列化 KV block 为 Vec<u8> frame】
#[cfg(feature = "tch-backend")]
fn serialize_kv_block(block: &KvBlock) -> Result<Vec<u8>, String> {
    let (k_bytes, k_dtype) = tensor_to_bytes(&block.k)?;
    let (v_bytes, v_dtype) = tensor_to_bytes(&block.v)?;
    let k_shape: Vec<i64> = block.k.size();
    let v_shape: Vec<i64> = block.v.size();

    let meta = serde_json::json!({
        "layer_idx": block.layer_idx,
        "global_seq_start": block.global_seq_start,
        "global_seq_end": block.global_seq_end,
        "micro_block_idx": block.micro_block_idx,
        "total_micro_blocks": block.total_micro_blocks,
        "k_shape": k_shape,
        "v_shape": v_shape,
        "k_bytes": k_bytes.len(),
        "v_bytes": v_bytes.len(),
        "k_dtype": k_dtype,
        "v_dtype": v_dtype,
    });
    let meta_bytes = meta.to_string().into_bytes();
    let meta_len = meta_bytes.len() as u32;

    let mut frame = Vec::with_capacity(4 + meta_bytes.len() + k_bytes.len() + v_bytes.len());
    frame.extend_from_slice(&meta_len.to_be_bytes());
    frame.extend_from_slice(&meta_bytes);
    frame.extend_from_slice(&k_bytes);
    frame.extend_from_slice(&v_bytes);
    Ok(frame)
}

/// 【接收一个 KV block 从 QUIC recv stream】（async，可被 recv task 调用）
#[cfg(feature = "tch-backend")]
async fn recv_kv_block_from_stream(
    recv: &mut RecvStream,
    handshake_done: &mut bool,
    device: Device,
) -> Result<Option<KvBlock>, String> {
    // Skip the 1-byte dummy written during stream setup (once per stream)
    if !*handshake_done {
        let mut dummy = [0u8; 1];
        match read_exact(recv, &mut dummy).await {
            Ok(()) => *handshake_done = true,
            Err(ReadExactError::Closed) => return Ok(None),
            Err(ReadExactError::ReadError(e)) => {
                return Err(format!("quic recv dummy failed: {e}"));
            }
        }
    }

    let mut len_bytes = [0u8; 4];
    match read_exact(recv, &mut len_bytes).await {
        Ok(()) => {}
        Err(ReadExactError::Closed) => return Ok(None),
        Err(ReadExactError::ReadError(e)) => {
            return Err(format!("quic recv meta_len failed: {e}"));
        }
    }
    let meta_len = u32::from_be_bytes(len_bytes) as usize;

    let mut meta_bytes = vec![0u8; meta_len];
    read_exact(recv, &mut meta_bytes).await
        .map_err(|e| format!("quic recv meta failed: {e}"))?;
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
        .map_err(|e| format!("quic parse meta failed: {e}"))?;

    let layer_idx = meta["layer_idx"].as_u64().ok_or("missing layer_idx")? as usize;
    let global_seq_start = meta["global_seq_start"].as_u64().ok_or("missing global_seq_start")? as usize;
    let global_seq_end = meta["global_seq_end"].as_u64().ok_or("missing global_seq_end")? as usize;
    let micro_block_idx = meta["micro_block_idx"].as_u64().unwrap_or(0) as usize;
    let total_micro_blocks = meta["total_micro_blocks"].as_u64().unwrap_or(1) as usize;
    let k_bytes_len = meta["k_bytes"].as_u64().ok_or("missing k_bytes")? as usize;
    let v_bytes_len = meta["v_bytes"].as_u64().ok_or("missing v_bytes")? as usize;
    let k_shape: Vec<i64> = meta["k_shape"].as_array().ok_or("missing k_shape")?
        .iter().map(|v| v.as_i64().ok_or("invalid k_shape"))
        .collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
    let v_shape: Vec<i64> = meta["v_shape"].as_array().ok_or("missing v_shape")?
        .iter().map(|v| v.as_i64().ok_or("invalid v_shape"))
        .collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

    let mut k_bytes = vec![0u8; k_bytes_len];
    read_exact(recv, &mut k_bytes).await
        .map_err(|e| format!("quic recv k_bytes failed: {e}"))?;
    let mut v_bytes = vec![0u8; v_bytes_len];
    read_exact(recv, &mut v_bytes).await
        .map_err(|e| format!("quic recv v_bytes failed: {e}"))?;

    let k_dtype = meta["k_dtype"].as_str().unwrap_or("float32");
    let v_dtype = meta["v_dtype"].as_str().unwrap_or("float32");
    let k = bytes_to_tensor(&k_bytes, &k_shape, device, k_dtype)?;
    let v = bytes_to_tensor(&v_bytes, &v_shape, device, v_dtype)?;

    Ok(Some(KvBlock { layer_idx, global_seq_start, global_seq_end, k, v, micro_block_idx, total_micro_blocks }))
}

#[cfg(feature = "tch-backend")]
#[cfg(feature = "tch-backend")]
impl KvTransport for QuicKvTransport {
    /// 【提交异步发送】序列化 block 后推入 send channel，立即返回。
    ///
    /// send task 在后台把 frame 写入 QUIC stream，主线程无需等待。
    /// 如果 channel 已满（send task 还在传输前一个 block），这里会 block_on
    /// 直到有空间，这是自然的 backpressure。
    fn submit_send(&mut self, block: &KvBlock) -> Result<(), String> {
        let frame = serialize_kv_block(block)?;
        self.rt.block_on(async {
            self.send_tx.send(SendCmd::Data(frame)).await
                .map_err(|e| format!("quic send channel closed: {e}"))
        })
    }

    /// 【轮询接收】非阻塞检查 recv channel。
    ///
    /// - Some(block): peer block 已到达
    /// - None: 暂时没有数据（主线程应继续做其他计算，稍后重试 poll）
    fn poll_recv(&mut self) -> Result<Option<KvBlock>, String> {
        match self.recv_rx.try_recv() {
            Ok(block) => Ok(Some(block)),
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => Ok(None),
        }
    }

    /// 【刷新发送】等待所有已 submit 的数据被 send task 处理。
    ///
    /// 发送一个 Flush marker 到 send channel，等待 send task ack。
    /// 因为 channel 是有序的，当 ack 返回时，所有之前的数据都已经 write_all。
    fn flush_send(&mut self) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.rt.block_on(async {
            self.send_tx.send(SendCmd::Flush(tx)).await
                .map_err(|e| format!("quic send channel closed during flush: {e}"))?;
            rx.await.map_err(|e| format!("quic flush ack dropped: {e}"))
        })
    }

    /// 【覆盖默认 recv_kv_block】避免 trait 默认的 1ms 忙等循环。
    ///
    /// 直接使用 block_on + recv() 阻塞等待，效率更高。
    /// 默认 600s 超时防止永久挂起，可通过 HCP_QUIC_TIMEOUT_SECS 覆盖。
    /// 大 KV block（4K+ seq）在跨 VPN 慢网络下传输可能超过 120s，需要更长的超时。
    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        let timeout_secs = std::env::var("HCP_QUIC_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600);
        self.rt.block_on(async {
            match tokio::time::timeout(
                std::time::Duration::from_secs(timeout_secs),
                self.recv_rx.recv()
            ).await {
                Ok(Some(block)) => Ok(Some(block)),
                Ok(None) => Ok(None), // channel closed（stream 已关闭）
                Err(_) => Err(format!("recv_kv_block timeout after {timeout_secs}s")),
            }
        })
    }
}

#[cfg(feature = "tch-backend")]
#[cfg(feature = "tch-backend")]
fn tensor_to_bytes(t: &Tensor) -> Result<(Vec<u8>, String), String> {
    let flat = t.contiguous().view(-1).to_kind(tch::Kind::Float);
    let values: Vec<f32> = Vec::try_from(&flat).map_err(|e| format!("tensor to vec failed: {e}"))?;
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

#[cfg(feature = "tch-backend")]
fn bytes_to_tensor(bytes: &[u8], shape: &[i64], device: Device, dtype_str: &str) -> Result<Tensor, String> {
    let values: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let expected = shape.iter().product::<i64>() as usize;
    if values.len() != expected {
        return Err(format!("byte length mismatch: expected {} floats, got {}", expected, values.len()));
    }
    let t = Tensor::from_slice(&values).reshape(shape).to_device(device);
    let kind = match dtype_str {
        "float16" => tch::Kind::Half,
        "bfloat16" => tch::Kind::BFloat16,
        "float64" => tch::Kind::Double,
        _ => tch::Kind::Float,
    };
    Ok(t.to_kind(kind))
}
