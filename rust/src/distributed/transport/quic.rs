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
use crate::model::transport::{KvBlock, KvTransport, RingMessage, RingPacket, SelfDrivingPacket};
#[cfg(feature = "tch-backend")]
use quinn::SendStream;
use quinn::{ClientConfig, Endpoint, RecvStream, ServerConfig};
use rustls::client::danger::{ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
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
    transport_config.max_idle_timeout(Some(
        std::time::Duration::from_secs(3600).try_into().unwrap(),
    ));
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

    let mut endpoint =
        Endpoint::server(server_config, listen_addr).map_err(|e| format!("bind failed: {e}"))?;

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
    /// 从 recv task 接收反序列化后的消息（KV block 或 Q-ring packet）
    recv_rx: mpsc::Receiver<RingMessage>,
    /// 【交叉暂存】同一 stream 复用两类 frame：poll_recv 只取 KV、
    /// poll_recv_packet 只取 packet，另一类先暂存等对应调用取出。
    pending_kv: std::collections::VecDeque<KvBlock>,
    pending_packets: std::collections::VecDeque<RingPacket>,
    pending_self_driving_packets: std::collections::VecDeque<SelfDrivingPacket>,
    /// send task 的 JoinHandle（Drop 时需要 abort）
    #[allow(dead_code)]
    send_task: tokio::task::JoinHandle<()>,
    /// recv task 的 JoinHandle（Drop 时需要 abort）
    #[allow(dead_code)]
    recv_task: tokio::task::JoinHandle<()>,
    rt: Handle,
    device: Device,
    /// K10 wire-byte accounting. Sent bytes are accumulated synchronously on
    /// the caller thread at submit time (frame length is known after
    /// serialization); recv bytes are accumulated in the background recv task
    /// and shared back through an atomic.
    wire_sent: u64,
    wire_recv: Arc<AtomicU64>,
}

#[cfg(feature = "tch-backend")]
impl QuicKvTransport {
    pub fn new(send: SendStream, recv: RecvStream, rt: Handle, device: Device) -> Self {
        // Channel buffer：允许网络传输多个 block 的同时，主线程序列化后续 block。
        // 这是 N-domain Serial 模式必需的：一次性 submit N 个 layer 的 blocks 时，
        // 如果 buffer 太小（如 2），send_task 和 recv_task 会在网络/缓冲区阻塞时互相死锁。
        // 对于 1M context，micro block 数量可达数百个，64 不够；但 1024 在 16GB GPU 上太占显存。
        // 默认 512，可通过 HCP_KV_CHANNEL_BUFFER_SIZE 覆盖。
        let buffer_size: usize = std::env::var("HCP_KV_CHANNEL_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let (send_tx, send_rx) = mpsc::channel::<SendCmd>(buffer_size);
        let (recv_tx, recv_rx) = mpsc::channel::<RingMessage>(buffer_size);

        let send_task = rt.spawn(send_task_loop(send, send_rx));
        let wire_recv = Arc::new(AtomicU64::new(0));
        let recv_task = rt.spawn(recv_task_loop(recv, recv_tx, device, wire_recv.clone()));

        Self {
            send_tx,
            recv_rx,
            pending_kv: std::collections::VecDeque::new(),
            pending_packets: std::collections::VecDeque::new(),
            pending_self_driving_packets: std::collections::VecDeque::new(),
            send_task,
            recv_task,
            rt,
            device,
            wire_sent: 0,
            wire_recv,
        }
    }

    /// 【排空 recv channel】把已到达的消息按类型放入对应暂存队列。
    fn drain_recv_channel(&mut self) {
        while let Ok(msg) = self.recv_rx.try_recv() {
            match msg {
                RingMessage::KvBlock(block) => self.pending_kv.push_back(block),
                RingMessage::RingPacket(packet) => self.pending_packets.push_back(packet),
                RingMessage::SelfDrivingPacket(packet) => {
                    self.pending_self_driving_packets.push_back(packet)
                }
            }
        }
    }
}

/// 【Send Task】从 channel 接收序列化 frame，写入 QUIC send stream。
///
/// 这个 task 独立运行，即使主线程在进行 attention 计算，它也在后台
/// 把 KV block 写入网络，实现计算-通信重叠。
#[cfg(feature = "tch-backend")]
async fn send_task_loop(mut send: SendStream, mut cmd_rx: mpsc::Receiver<SendCmd>) {
    let stream_id = send.id();
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            SendCmd::Data(frame) => {
                if let Err(e) = send.write_all(&frame).await {
                    eprintln!("[quic send_task] write_all failed (stream {stream_id:?}): {e}");
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
    // SendStream 在此 drop；记录退出原因以便诊断意外的 stream 关闭。
    eprintln!("[quic send_task] exiting (stream {stream_id:?}): send channel closed");
}

/// 【Recv Task】从 QUIC recv stream 读取 frame，反序列化后推入 channel。
///
/// 这个 task 独立运行，即使主线程在进行 attention 计算，它也在后台
/// 等待接收 peer 消息（KV block 或 Q-ring packet），一有数据就推入 channel 供 poll 消费。
#[cfg(feature = "tch-backend")]
async fn recv_task_loop(
    mut recv: RecvStream,
    msg_tx: mpsc::Sender<RingMessage>,
    device: Device,
    wire_recv: Arc<AtomicU64>,
) {
    let stream_id = recv.id();
    let mut handshake_done = false;
    loop {
        match recv_frame_from_stream(&mut recv, &mut handshake_done, device).await {
            Ok(Some((msg, frame_len))) => {
                wire_recv.fetch_add(frame_len, Ordering::Relaxed);
                if msg_tx.send(msg).await.is_err() {
                    eprintln!(
                        "[quic recv_task] exiting (stream {stream_id:?}): message channel closed (receiver dropped)"
                    );
                    break; // 主线程已 drop recv_rx，不需要继续接收
                }
            }
            Ok(None) => {
                eprintln!("[quic recv_task] exiting (stream {stream_id:?}): peer sent FIN (stream closed cleanly)");
                break; // stream 正常关闭（peer 发送了 FIN）
            }
            Err(e) => {
                eprintln!("[quic recv_task] error (stream {stream_id:?}): {e}");
                break;
            }
        }
    }
    // msg_tx 在这里被 drop，recv_rx.recv() 会返回 None，通知主线程 stream 已关闭
}

/// 【序列化 KV block 为 Vec<u8> frame】
/// Payload 顺序为 K、V、可选 Int64 position_ids；旧 frame 可省略位置 metadata。
#[cfg(feature = "tch-backend")]
fn serialize_kv_block(block: &KvBlock) -> Result<Vec<u8>, String> {
    let (k_bytes, k_dtype) = tensor_to_bytes(&block.k)?;
    let (v_bytes, v_dtype) = tensor_to_bytes(&block.v)?;
    let k_shape: Vec<i64> = block.k.size();
    let v_shape: Vec<i64> = block.v.size();
    let position_payload = block
        .position_ids
        .as_ref()
        .map(|positions| {
            let (bytes, dtype) = tensor_to_bytes(positions)?;
            let meta = serde_json::json!({
                "shape": positions.size(),
                "bytes": bytes.len(),
                "dtype": dtype,
            });
            Ok::<_, String>((meta, bytes))
        })
        .transpose()?;
    let position_meta = position_payload.as_ref().map(|(meta, _)| meta);

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
        "position_ids": position_meta,
    });
    let meta_bytes = meta.to_string().into_bytes();
    let meta_len = meta_bytes.len() as u32;

    let position_bytes_len = position_payload
        .as_ref()
        .map_or(0, |(_, bytes)| bytes.len());
    let mut frame = Vec::with_capacity(
        4 + meta_bytes.len() + k_bytes.len() + v_bytes.len() + position_bytes_len,
    );
    frame.extend_from_slice(&meta_len.to_be_bytes());
    frame.extend_from_slice(&meta_bytes);
    frame.extend_from_slice(&k_bytes);
    frame.extend_from_slice(&v_bytes);
    if let Some((_, bytes)) = position_payload {
        frame.extend_from_slice(&bytes);
    }
    Ok(frame)
}

/// 【序列化 Q-ring packet 为 Vec<u8> frame】
///
/// 与 KV block 共用同一 framing 约定（length-prefixed JSON meta + raw f32 bytes），
/// meta 中 "type": "ring_packet" 用于区分 payload 类型。
#[cfg(feature = "tch-backend")]
fn serialize_ring_packet(packet: &RingPacket) -> Result<Vec<u8>, String> {
    let (q_bytes, q_dtype) = tensor_to_bytes(&packet.q)?;
    let (o_bytes, o_dtype) = tensor_to_bytes(&packet.o)?;
    let (lse_bytes, lse_dtype) = tensor_to_bytes(&packet.lse)?;

    let meta = serde_json::json!({
        "type": "ring_packet",
        "layer_idx": packet.layer_idx,
        "scale": packet.scale,
        "q_shape": packet.q.size(),
        "o_shape": packet.o.size(),
        "lse_shape": packet.lse.size(),
        "q_bytes": q_bytes.len(),
        "o_bytes": o_bytes.len(),
        "lse_bytes": lse_bytes.len(),
        "q_dtype": q_dtype,
        "o_dtype": o_dtype,
        "lse_dtype": lse_dtype,
    });
    let meta_bytes = meta.to_string().into_bytes();
    let meta_len = meta_bytes.len() as u32;

    let mut frame =
        Vec::with_capacity(4 + meta_bytes.len() + q_bytes.len() + o_bytes.len() + lse_bytes.len());
    frame.extend_from_slice(&meta_len.to_be_bytes());
    frame.extend_from_slice(&meta_bytes);
    frame.extend_from_slice(&q_bytes);
    frame.extend_from_slice(&o_bytes);
    frame.extend_from_slice(&lse_bytes);
    Ok(frame)
}

#[cfg(feature = "tch-backend")]
fn serialize_self_driving_packet(packet: &SelfDrivingPacket) -> Result<Vec<u8>, String> {
    let tensors = [
        ("residual", &packet.residual),
        ("normalized", &packet.normalized),
        ("position_ids", &packet.position_ids),
        ("q", &packet.q),
        ("attention_output", &packet.attention_output),
        ("lse", &packet.lse),
    ];
    let mut payloads = Vec::with_capacity(tensors.len());
    let mut tensor_meta = serde_json::Map::new();
    for (name, tensor) in tensors {
        let (bytes, dtype) = tensor_to_bytes(tensor)?;
        tensor_meta.insert(
            name.to_string(),
            serde_json::json!({
                "shape": tensor.size(),
                "bytes": bytes.len(),
                "dtype": dtype,
            }),
        );
        payloads.push(bytes);
    }

    let meta = serde_json::json!({
        "type": "self_driving_packet",
        "layer_idx": packet.layer_idx,
        "assignee": packet.assignee,
        "current_domain": packet.current_domain,
        "domains": packet.domains,
        "visited_domains": packet.visited_domains,
        "tensors": tensor_meta,
    });
    let meta_bytes = meta.to_string().into_bytes();
    let mut frame =
        Vec::with_capacity(4 + meta_bytes.len() + payloads.iter().map(Vec::len).sum::<usize>());
    frame.extend_from_slice(&(meta_bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(&meta_bytes);
    for payload in payloads {
        frame.extend_from_slice(&payload);
    }
    Ok(frame)
}

/// 【接收一个 frame 从 QUIC recv stream】（async，可被 recv task 调用）
///
/// 按 meta["type"] 分发：无 "type" 字段的 frame 一律按 KV block 解析（向后兼容），
/// "ring_packet" 解析为 Q-ring packet。
#[cfg(feature = "tch-backend")]
async fn recv_frame_from_stream(
    recv: &mut RecvStream,
    handshake_done: &mut bool,
    device: Device,
) -> Result<Option<(RingMessage, u64)>, String> {
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
    read_exact(recv, &mut meta_bytes)
        .await
        .map_err(|e| format!("quic recv meta failed: {e}"))?;
    let meta: serde_json::Value =
        serde_json::from_slice(&meta_bytes).map_err(|e| format!("quic parse meta failed: {e}"))?;

    let layer_idx = meta["layer_idx"].as_u64().ok_or("missing layer_idx")? as usize;

    if meta["type"].as_str() == Some("self_driving_packet") {
        let residual = recv_named_tensor(recv, &meta, "residual", device).await?;
        let normalized = recv_named_tensor(recv, &meta, "normalized", device).await?;
        let position_ids = recv_named_tensor(recv, &meta, "position_ids", device).await?;
        let q = recv_named_tensor(recv, &meta, "q", device).await?;
        let attention_output = recv_named_tensor(recv, &meta, "attention_output", device).await?;
        let lse = recv_named_tensor(recv, &meta, "lse", device).await?;
        let assignee = meta["assignee"].as_u64().ok_or("missing assignee")? as usize;
        let current_domain = meta["current_domain"]
            .as_u64()
            .ok_or("missing current_domain")? as usize;
        let domains = meta["domains"].as_u64().ok_or("missing domains")? as usize;
        let visited_domains = meta["visited_domains"]
            .as_u64()
            .ok_or("missing visited_domains")? as usize;
        let frame_len = quic_frame_wire_len(&meta, meta_len);
        return Ok(Some((
            RingMessage::SelfDrivingPacket(SelfDrivingPacket {
                layer_idx,
                residual,
                normalized,
                position_ids,
                q,
                attention_output,
                lse,
                assignee,
                current_domain,
                domains,
                visited_domains,
            }),
            frame_len,
        )));
    }

    if meta["type"].as_str() == Some("ring_packet") {
        let scale = meta["scale"].as_f64().ok_or("missing scale")?;
        let read_tensor = |prefix: &str| -> Result<(Vec<u8>, Vec<i64>, String), String> {
            let bytes_len = meta[format!("{prefix}_bytes")]
                .as_u64()
                .ok_or_else(|| format!("missing {prefix}_bytes"))?
                as usize;
            let shape: Vec<i64> = meta[format!("{prefix}_shape")]
                .as_array()
                .ok_or_else(|| format!("missing {prefix}_shape"))?
                .iter()
                .map(|v| v.as_i64().ok_or("invalid shape"))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e: &str| e.to_string())?;
            let dtype = meta[format!("{prefix}_dtype")]
                .as_str()
                .unwrap_or("float32")
                .to_string();
            Ok((vec![0u8; bytes_len], shape, dtype))
        };
        let (mut q_bytes, q_shape, q_dtype) = read_tensor("q")?;
        read_exact(recv, &mut q_bytes)
            .await
            .map_err(|e| format!("quic recv q_bytes failed: {e}"))?;
        let (mut o_bytes, o_shape, o_dtype) = read_tensor("o")?;
        read_exact(recv, &mut o_bytes)
            .await
            .map_err(|e| format!("quic recv o_bytes failed: {e}"))?;
        let (mut lse_bytes, lse_shape, lse_dtype) = read_tensor("lse")?;
        read_exact(recv, &mut lse_bytes)
            .await
            .map_err(|e| format!("quic recv lse_bytes failed: {e}"))?;
        let q = bytes_to_tensor(&q_bytes, &q_shape, device, &q_dtype)?;
        let o = bytes_to_tensor(&o_bytes, &o_shape, device, &o_dtype)?;
        let lse = bytes_to_tensor(&lse_bytes, &lse_shape, device, &lse_dtype)?;
        let frame_len = quic_frame_wire_len(&meta, meta_len);
        return Ok(Some((
            RingMessage::RingPacket(RingPacket {
                layer_idx,
                q,
                o,
                lse,
                scale,
            }),
            frame_len,
        )));
    }

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

    let mut k_bytes = vec![0u8; k_bytes_len];
    read_exact(recv, &mut k_bytes)
        .await
        .map_err(|e| format!("quic recv k_bytes failed: {e}"))?;
    let mut v_bytes = vec![0u8; v_bytes_len];
    read_exact(recv, &mut v_bytes)
        .await
        .map_err(|e| format!("quic recv v_bytes failed: {e}"))?;

    let k_dtype = meta["k_dtype"].as_str().unwrap_or("float32");
    let v_dtype = meta["v_dtype"].as_str().unwrap_or("float32");
    let k = bytes_to_tensor(&k_bytes, &k_shape, device, k_dtype)?;
    let v = bytes_to_tensor(&v_bytes, &v_shape, device, v_dtype)?;
    let position_ids = match meta.get("position_ids").filter(|value| !value.is_null()) {
        Some(position_meta) => {
            let bytes_len = position_meta["bytes"]
                .as_u64()
                .ok_or("missing position_ids bytes")? as usize;
            let shape = position_meta["shape"]
                .as_array()
                .ok_or("missing position_ids shape")?
                .iter()
                .map(|value| value.as_i64().ok_or("invalid position_ids shape"))
                .collect::<Result<Vec<_>, _>>()?;
            let dtype = position_meta["dtype"]
                .as_str()
                .ok_or("missing position_ids dtype")?;
            let mut bytes = vec![0_u8; bytes_len];
            read_exact(recv, &mut bytes)
                .await
                .map_err(|e| format!("quic recv position_ids bytes failed: {e}"))?;
            Some(bytes_to_tensor(&bytes, &shape, device, dtype)?)
        }
        None => None,
    };

    let frame_len = quic_frame_wire_len(&meta, meta_len);
    Ok(Some((
        RingMessage::KvBlock(KvBlock {
            layer_idx,
            global_seq_start,
            global_seq_end,
            k,
            v,
            micro_block_idx,
            total_micro_blocks,
            position_ids,
        }),
        frame_len,
    )))
}

/// 【K10 wire-byte accounting】compute the full serialized frame length
/// (4-byte meta_len prefix + meta JSON + raw tensor payload) from the parsed
/// meta. This mirrors the send-side frame layout exactly, so recv-side bytes
/// are byte-for-byte comparable with send-side bytes.
#[cfg(feature = "tch-backend")]
fn quic_frame_wire_len(meta: &serde_json::Value, meta_len: usize) -> u64 {
    let payload: usize = match meta["type"].as_str() {
        Some("self_driving_packet") => {
            let tensors = &meta["tensors"];
            [
                "residual",
                "normalized",
                "position_ids",
                "q",
                "attention_output",
                "lse",
            ]
            .iter()
            .map(|name| tensors[name]["bytes"].as_u64().unwrap_or(0) as usize)
            .sum()
        }
        Some("ring_packet") => ["q_bytes", "o_bytes", "lse_bytes"]
            .iter()
            .map(|key| meta[key].as_u64().unwrap_or(0) as usize)
            .sum(),
        _ => {
            let mut total = meta["k_bytes"].as_u64().unwrap_or(0) as usize
                + meta["v_bytes"].as_u64().unwrap_or(0) as usize;
            if let Some(pos) = meta.get("position_ids").filter(|v| !v.is_null()) {
                total += pos["bytes"].as_u64().unwrap_or(0) as usize;
            }
            total
        }
    };
    (4 + meta_len + payload) as u64
}

#[cfg(feature = "tch-backend")]
async fn recv_named_tensor(
    recv: &mut RecvStream,
    meta: &serde_json::Value,
    name: &str,
    device: Device,
) -> Result<Tensor, String> {
    let tensor_meta = &meta["tensors"][name];
    let bytes_len = tensor_meta["bytes"]
        .as_u64()
        .ok_or_else(|| format!("missing {name} bytes"))? as usize;
    let shape = tensor_meta["shape"]
        .as_array()
        .ok_or_else(|| format!("missing {name} shape"))?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| format!("invalid {name} shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dtype = tensor_meta["dtype"]
        .as_str()
        .ok_or_else(|| format!("missing {name} dtype"))?;
    let mut bytes = vec![0_u8; bytes_len];
    read_exact(recv, &mut bytes)
        .await
        .map_err(|error| format!("quic recv {name} bytes failed: {error}"))?;
    bytes_to_tensor(&bytes, &shape, device, dtype)
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
        self.wire_sent += frame.len() as u64;
        self.rt.block_on(async {
            self.send_tx
                .send(SendCmd::Data(frame))
                .await
                .map_err(|e| format!("quic send channel closed: {e}"))
        })
    }

    /// 【轮询接收】非阻塞检查 recv channel。
    ///
    /// - Some(block): peer block 已到达
    /// - None: 暂时没有数据（主线程应继续做其他计算，稍后重试 poll）
    /// 交叉到达的 Q-ring packet 会被暂存，等 poll_recv_packet / recv_packet 取出。
    fn poll_recv(&mut self) -> Result<Option<KvBlock>, String> {
        self.drain_recv_channel();
        Ok(self.pending_kv.pop_front())
    }

    /// 【刷新发送】等待所有已 submit 的数据被 send task 处理。
    ///
    /// 发送一个 Flush marker 到 send channel，等待 send task ack。
    /// 因为 channel 是有序的，当 ack 返回时，所有之前的数据都已经 write_all。
    fn flush_send(&mut self) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.rt.block_on(async {
            self.send_tx
                .send(SendCmd::Flush(tx))
                .await
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
        if let Some(block) = self.pending_kv.pop_front() {
            return Ok(Some(block));
        }
        let timeout_secs = std::env::var("HCP_QUIC_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600);
        self.rt.block_on(async {
            loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(timeout_secs),
                    self.recv_rx.recv(),
                )
                .await
                {
                    Ok(Some(RingMessage::KvBlock(block))) => return Ok(Some(block)),
                    // 交叉流量：packet 暂存，继续等 KV block
                    Ok(Some(RingMessage::RingPacket(packet))) => {
                        self.pending_packets.push_back(packet);
                    }
                    Ok(Some(RingMessage::SelfDrivingPacket(packet))) => {
                        self.pending_self_driving_packets.push_back(packet);
                    }
                    Ok(None) => return Ok(None), // channel closed（stream 已关闭）
                    Err(_) => return Err(format!("recv_kv_block timeout after {timeout_secs}s")),
                }
            }
        })
    }

    fn supports_ring_packets(&self) -> bool {
        true
    }

    /// 【提交异步发送 Q-ring packet】与 KV block 共用 send channel / send task。
    fn submit_send_packet(&mut self, packet: &RingPacket) -> Result<(), String> {
        let frame = serialize_ring_packet(packet)?;
        self.wire_sent += frame.len() as u64;
        self.rt.block_on(async {
            self.send_tx
                .send(SendCmd::Data(frame))
                .await
                .map_err(|e| format!("quic send channel closed: {e}"))
        })
    }

    /// 【轮询接收 packet】非阻塞；交叉到达的 KV block 会被暂存。
    fn poll_recv_packet(&mut self) -> Result<Option<RingPacket>, String> {
        self.drain_recv_channel();
        Ok(self.pending_packets.pop_front())
    }

    /// 【阻塞接收 packet】与 recv_kv_block 相同的超时语义。
    fn recv_packet(&mut self) -> Result<Option<RingPacket>, String> {
        if let Some(packet) = self.pending_packets.pop_front() {
            return Ok(Some(packet));
        }
        let timeout_secs = std::env::var("HCP_QUIC_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600);
        self.rt.block_on(async {
            loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(timeout_secs),
                    self.recv_rx.recv(),
                )
                .await
                {
                    Ok(Some(RingMessage::RingPacket(packet))) => return Ok(Some(packet)),
                    // 交叉流量：KV block 暂存，继续等 packet
                    Ok(Some(RingMessage::KvBlock(block))) => {
                        self.pending_kv.push_back(block);
                    }
                    Ok(Some(RingMessage::SelfDrivingPacket(packet))) => {
                        self.pending_self_driving_packets.push_back(packet);
                    }
                    Ok(None) => return Ok(None),
                    Err(_) => return Err(format!("recv_packet timeout after {timeout_secs}s")),
                }
            }
        })
    }

    fn supports_self_driving_packets(&self) -> bool {
        true
    }

    fn wire_bytes_sent(&self) -> u64 {
        self.wire_sent
    }

    fn wire_bytes_recv(&self) -> u64 {
        self.wire_recv.load(Ordering::Relaxed)
    }

    fn submit_send_self_driving_packet(
        &mut self,
        packet: &SelfDrivingPacket,
    ) -> Result<(), String> {
        let frame = serialize_self_driving_packet(packet)?;
        self.wire_sent += frame.len() as u64;
        self.rt.block_on(async {
            self.send_tx
                .send(SendCmd::Data(frame))
                .await
                .map_err(|error| format!("quic send channel closed: {error}"))
        })
    }

    fn poll_recv_self_driving_packet(&mut self) -> Result<Option<SelfDrivingPacket>, String> {
        self.drain_recv_channel();
        Ok(self.pending_self_driving_packets.pop_front())
    }

    fn recv_self_driving_packet(&mut self) -> Result<Option<SelfDrivingPacket>, String> {
        if let Some(packet) = self.pending_self_driving_packets.pop_front() {
            return Ok(Some(packet));
        }
        let timeout_secs = std::env::var("HCP_QUIC_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(600);
        self.rt.block_on(async {
            loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(timeout_secs),
                    self.recv_rx.recv(),
                )
                .await
                {
                    Ok(Some(RingMessage::SelfDrivingPacket(packet))) => return Ok(Some(packet)),
                    Ok(Some(RingMessage::KvBlock(block))) => self.pending_kv.push_back(block),
                    Ok(Some(RingMessage::RingPacket(packet))) => {
                        self.pending_packets.push_back(packet)
                    }
                    Ok(None) => return Ok(None),
                    Err(_) => {
                        return Err(format!(
                            "recv_self_driving_packet timeout after {timeout_secs}s"
                        ))
                    }
                }
            }
        })
    }
}

#[cfg(feature = "tch-backend")]
#[cfg(feature = "tch-backend")]
fn tensor_to_bytes(t: &Tensor) -> Result<(Vec<u8>, String), String> {
    if t.kind() == tch::Kind::Int64 {
        let flat = t.contiguous().view(-1);
        let values: Vec<i64> =
            Vec::try_from(&flat).map_err(|e| format!("tensor to vec failed: {e}"))?;
        let bytes = values
            .iter()
            .flat_map(|&value| value.to_le_bytes())
            .collect();
        return Ok((bytes, "int64".to_string()));
    }
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

#[cfg(feature = "tch-backend")]
fn bytes_to_tensor(
    bytes: &[u8],
    shape: &[i64],
    device: Device,
    dtype_str: &str,
) -> Result<Tensor, String> {
    if dtype_str == "int64" {
        if !bytes.len().is_multiple_of(8) {
            return Err(format!("int64 byte length is not aligned: {}", bytes.len()));
        }
        let values: Vec<i64> = bytes
            .chunks_exact(8)
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();
        let expected = shape.iter().product::<i64>() as usize;
        if values.len() != expected {
            return Err(format!(
                "byte length mismatch: expected {expected} int64 values, got {}",
                values.len()
            ));
        }
        return Ok(Tensor::from_slice(&values).reshape(shape).to_device(device));
    }
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
    let t = Tensor::from_slice(&values).reshape(shape).to_device(device);
    let kind = match dtype_str {
        "float16" => tch::Kind::Half,
        "bfloat16" => tch::Kind::BFloat16,
        "float64" => tch::Kind::Double,
        _ => tch::Kind::Float,
    };
    Ok(t.to_kind(kind))
}

#[cfg(all(test, feature = "tch-backend"))]
mod tests {
    use super::*;
    use crate::model::transport::SelfDrivingPacket;
    use tch::Kind;

    struct TestQuicStreams {
        _client_endpoint: Endpoint,
        _server_endpoint: Endpoint,
        client_send: SendStream,
        client_recv: RecvStream,
        server_send: SendStream,
        server_recv: RecvStream,
    }

    fn connected_quic_streams(runtime: &tokio::runtime::Runtime) -> TestQuicStreams {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let (client_endpoint, server_endpoint, client_connection, server_connection) = runtime
            .block_on(async {
                let client_endpoint = create_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
                let server_endpoint = create_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
                let server_addr = server_endpoint.local_addr().unwrap();
                let client_connect = client_endpoint.connect(server_addr, "localhost").unwrap();
                let server_incoming = server_endpoint.accept().await.unwrap();
                let (client, server) = tokio::join!(client_connect, server_incoming);
                (
                    client_endpoint,
                    server_endpoint,
                    client.unwrap(),
                    server.unwrap(),
                )
            });
        let ((client_send, client_recv), (mut server_send, server_recv)) =
            runtime.block_on(async {
                let (mut client_send, client_recv) = client_connection.open_bi().await.unwrap();
                client_send.write_all(b"\x00").await.unwrap();
                let (server_send, server_recv) = server_connection.accept_bi().await.unwrap();
                ((client_send, client_recv), (server_send, server_recv))
            });
        runtime.block_on(async {
            server_send.write_all(b"\x00").await.unwrap();
        });
        TestQuicStreams {
            _client_endpoint: client_endpoint,
            _server_endpoint: server_endpoint,
            client_send,
            client_recv,
            server_send,
            server_recv,
        }
    }

    fn without_position_metadata(frame: Vec<u8>) -> Vec<u8> {
        let old_meta_len = u32::from_be_bytes(frame[..4].try_into().unwrap()) as usize;
        let mut meta: serde_json::Value =
            serde_json::from_slice(&frame[4..4 + old_meta_len]).unwrap();
        meta.as_object_mut().unwrap().remove("position_ids");
        let meta_bytes = meta.to_string().into_bytes();
        let mut legacy = Vec::with_capacity(4 + meta_bytes.len() + frame.len() - 4 - old_meta_len);
        legacy.extend_from_slice(&(meta_bytes.len() as u32).to_be_bytes());
        legacy.extend_from_slice(&meta_bytes);
        legacy.extend_from_slice(&frame[4 + old_meta_len..]);
        legacy
    }

    fn test_self_driving_packet(device: Device) -> SelfDrivingPacket {
        SelfDrivingPacket {
            layer_idx: 7,
            residual: Tensor::arange(8, (Kind::Float, device))
                .reshape([1, 1, 8])
                .to_kind(Kind::BFloat16),
            normalized: (Tensor::arange(8, (Kind::Float, device)) * 0.5)
                .reshape([1, 1, 8])
                .to_kind(Kind::BFloat16),
            position_ids: Tensor::from_slice(&[16_777_217_i64]).reshape([1, 1]),
            q: Tensor::arange(32, (Kind::Float, device))
                .reshape([1, 4, 1, 8])
                .to_kind(Kind::BFloat16),
            attention_output: (Tensor::arange(32, (Kind::Float, device)) * 0.25)
                .reshape([1, 4, 1, 8])
                .to_kind(Kind::BFloat16),
            lse: Tensor::arange(4, (Kind::Float, device)).reshape([1, 4, 1]),
            assignee: 1,
            current_domain: 1,
            domains: 2,
            visited_domains: 1,
        }
    }

    /// m>1 stationary continuation packet with real Qwen2-0.5B GQA shapes
    /// (hidden 896, 24 heads, head_dim 64, 2 kv heads -> packet heads 24).
    /// `positions` selects the query count m and proves the codec makes no
    /// position-contiguity assumption.
    fn multi_query_self_driving_packet(device: Device, positions: &[i64]) -> SelfDrivingPacket {
        let m = positions.len() as i64;
        let hidden = 896_i64;
        let heads = 24_i64;
        let head_dim = 64_i64;
        let residual_elements = m * hidden;
        let q_elements = heads * m * head_dim;
        SelfDrivingPacket {
            layer_idx: 11,
            residual: (Tensor::arange(residual_elements, (Kind::Float, device)) * 0.001)
                .reshape([1, m, hidden])
                .to_kind(Kind::BFloat16),
            normalized: (Tensor::arange(residual_elements, (Kind::Float, device)) * -0.0005)
                .reshape([1, m, hidden])
                .to_kind(Kind::BFloat16),
            position_ids: Tensor::from_slice(positions).reshape([1, m]),
            q: (Tensor::arange(q_elements, (Kind::Float, device)) * 0.00025)
                .reshape([1, heads, m, head_dim])
                .to_kind(Kind::BFloat16),
            attention_output: (Tensor::arange(q_elements, (Kind::Float, device)) * 0.000125)
                .reshape([1, heads, m, head_dim])
                .to_kind(Kind::BFloat16),
            lse: (Tensor::arange(heads * m, (Kind::Float, device)) * 0.01).reshape([1, heads, m]),
            assignee: 2,
            current_domain: 1,
            domains: 3,
            visited_domains: 1,
        }
    }

    fn assert_self_driving_packet_eq(actual: &SelfDrivingPacket, expected: &SelfDrivingPacket) {
        assert_eq!(actual.layer_idx, expected.layer_idx);
        assert_eq!(actual.assignee, expected.assignee);
        assert_eq!(actual.current_domain, expected.current_domain);
        assert_eq!(actual.domains, expected.domains);
        assert_eq!(actual.visited_domains, expected.visited_domains);
        for (name, actual, wanted) in [
            ("residual", &actual.residual, &expected.residual),
            ("normalized", &actual.normalized, &expected.normalized),
            ("position_ids", &actual.position_ids, &expected.position_ids),
            ("q", &actual.q, &expected.q),
            (
                "attention_output",
                &actual.attention_output,
                &expected.attention_output,
            ),
            ("lse", &actual.lse, &expected.lse),
        ] {
            assert_eq!(actual.kind(), wanted.kind(), "{name} dtype changed");
            assert_eq!(actual.size(), wanted.size(), "{name} shape changed");
            let diff = (actual - wanted)
                .abs()
                .to_kind(Kind::Float)
                .max()
                .double_value(&[]);
            assert_eq!(diff, 0.0, "{name} changed after QUIC roundtrip");
        }
    }

    #[test]
    fn quic_kv_transport_trait_roundtrips_positioned_kv_block() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let TestQuicStreams {
            _client_endpoint,
            _server_endpoint,
            client_send,
            client_recv,
            server_send,
            server_recv,
        } = connected_quic_streams(&runtime);

        let device = Device::Cpu;
        let mut client: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            client_send,
            client_recv,
            runtime.handle().clone(),
            device,
        ));
        let mut server: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            server_send,
            server_recv,
            runtime.handle().clone(),
            device,
        ));
        let expected_positions = Tensor::from_slice(&[0_i64, 9, 16_777_217]);
        let block = KvBlock {
            layer_idx: 5,
            global_seq_start: 0,
            global_seq_end: 3,
            k: Tensor::arange(6, (Kind::Float, device)).reshape([1, 1, 3, 2]),
            v: (Tensor::arange(6, (Kind::Float, device)) * 0.5).reshape([1, 1, 3, 2]),
            micro_block_idx: 0,
            total_micro_blocks: 1,
            position_ids: Some(expected_positions.shallow_clone()),
        };

        client.submit_send(&block).unwrap();
        client.flush_send().unwrap();
        let received = server.recv_kv_block().unwrap().unwrap();

        let received_positions = received
            .position_ids
            .expect("positioned KV block lost position_ids across QUIC");
        assert_eq!(received_positions.kind(), Kind::Int64);
        assert_eq!(received_positions.size(), expected_positions.size());
        assert_eq!(
            Vec::<i64>::try_from(&received_positions).unwrap(),
            Vec::<i64>::try_from(&expected_positions).unwrap()
        );
    }

    #[test]
    fn quic_kv_transport_accepts_legacy_kv_frame_without_position_metadata() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let TestQuicStreams {
            _client_endpoint,
            _server_endpoint,
            mut client_send,
            client_recv: _client_recv,
            server_send,
            server_recv,
        } = connected_quic_streams(&runtime);

        let device = Device::Cpu;
        let block = KvBlock::single(
            2,
            4,
            6,
            Tensor::arange(4, (Kind::Float, device)).reshape([1, 1, 2, 2]),
            Tensor::arange(4, (Kind::Float, device)).reshape([1, 1, 2, 2]),
        );
        let legacy_frame = without_position_metadata(serialize_kv_block(&block).unwrap());
        runtime.block_on(async {
            client_send.write_all(&legacy_frame).await.unwrap();
        });

        let mut server: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            server_send,
            server_recv,
            runtime.handle().clone(),
            device,
        ));
        let received = server.recv_kv_block().unwrap().unwrap();

        assert!(received.position_ids.is_none());
        assert_eq!(received.layer_idx, 2);
        assert_eq!((received.global_seq_start, received.global_seq_end), (4, 6));
    }

    #[test]
    fn quic_kv_transport_trait_roundtrips_self_driving_packet() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let TestQuicStreams {
            _client_endpoint,
            _server_endpoint,
            client_send,
            client_recv,
            server_send,
            server_recv,
        } = connected_quic_streams(&runtime);

        let device = Device::Cpu;
        let mut client: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            client_send,
            client_recv,
            runtime.handle().clone(),
            device,
        ));
        let mut server: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            server_send,
            server_recv,
            runtime.handle().clone(),
            device,
        ));
        assert!(client.supports_self_driving_packets());
        assert!(server.supports_self_driving_packets());

        let expected = test_self_driving_packet(device);
        client.submit_send_self_driving_packet(&expected).unwrap();
        client.flush_send().unwrap();
        let received = server.recv_self_driving_packet().unwrap().unwrap();
        assert_self_driving_packet_eq(&received, &expected);

        server.submit_send_self_driving_packet(&received).unwrap();
        server.flush_send().unwrap();
        let echoed = client.recv_self_driving_packet().unwrap().unwrap();
        assert_self_driving_packet_eq(&echoed, &expected);
    }

    #[test]
    fn quic_kv_transport_roundtrips_multi_query_self_driving_packet() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let TestQuicStreams {
            _client_endpoint,
            _server_endpoint,
            client_send,
            client_recv,
            server_send,
            server_recv,
        } = connected_quic_streams(&runtime);

        let device = Device::Cpu;
        let mut client: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            client_send,
            client_recv,
            runtime.handle().clone(),
            device,
        ));
        let mut server: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            server_send,
            server_recv,
            runtime.handle().clone(),
            device,
        ));

        // Real continuation shapes: m=4 queries at positions [5,6,7,8].
        let contiguous = multi_query_self_driving_packet(device, &[5, 6, 7, 8]);
        // Non-contiguous position ids: m=2 queries at positions [5,7] prove the
        // codec carries position_ids verbatim without a contiguity assumption.
        let strided = multi_query_self_driving_packet(device, &[5, 7]);

        for expected in [&contiguous, &strided] {
            client.submit_send_self_driving_packet(expected).unwrap();
            client.flush_send().unwrap();
            let received = server.recv_self_driving_packet().unwrap().unwrap();
            assert_self_driving_packet_eq(&received, expected);

            server.submit_send_self_driving_packet(&received).unwrap();
            server.flush_send().unwrap();
            let echoed = client.recv_self_driving_packet().unwrap().unwrap();
            assert_self_driving_packet_eq(&echoed, expected);
        }
    }

    /// K10: QUIC transport wire-byte accounting. Sent bytes are counted at
    /// submit time (frame length known after serialization); recv bytes are
    /// counted in the background recv task and shared back. After a single
    /// self-driving packet round-trips, both sides must report the same
    /// non-zero serialized frame length.
    #[test]
    fn quic_wire_bytes_account_for_self_driving_packet() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let TestQuicStreams {
            _client_endpoint,
            _server_endpoint,
            client_send,
            client_recv,
            server_send,
            server_recv,
        } = connected_quic_streams(&runtime);

        let device = Device::Cpu;
        let mut client: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            client_send,
            client_recv,
            runtime.handle().clone(),
            device,
        ));
        let mut server: Box<dyn KvTransport> = Box::new(QuicKvTransport::new(
            server_send,
            server_recv,
            runtime.handle().clone(),
            device,
        ));

        assert_eq!(client.wire_bytes_sent(), 0);
        assert_eq!(server.wire_bytes_recv(), 0);

        let expected = test_self_driving_packet(device);
        client.submit_send_self_driving_packet(&expected).unwrap();
        client.flush_send().unwrap();
        let sent = client.wire_bytes_sent();
        assert!(sent > 0, "sent wire bytes must be non-zero");

        let received = server.recv_self_driving_packet().unwrap().unwrap();
        assert_self_driving_packet_eq(&received, &expected);

        // The recv task increments the shared counter before handing the frame
        // to the channel, so it is already visible once recv returns.
        let recv = server.wire_bytes_recv();
        assert_eq!(
            recv, sent,
            "recv wire bytes must equal sent wire bytes for one frame"
        );
        assert_eq!(
            sent,
            serialize_self_driving_packet(&expected).unwrap().len() as u64
        );
    }
}
