//! QUIC-based KV transport for distributed ring attention.
use crate::model::kv_transport::{KvBlock, KvTransport};
use quinn::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig};
use rustls::client::danger::{ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use std::net::SocketAddr;
use std::sync::Arc;
use tch::{Device, Tensor};
use tokio::runtime::Handle;

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
    transport_config.keep_alive_interval(Some(std::time::Duration::from_secs(1)));
    // Increase stream window to accommodate large KV blocks (e.g. 1.3MB for 1365 tokens).
    // Default ~1.2MB is insufficient for ring-KV exchange deadlocking.
    // GQA repeat 后 KV block 大小 = 2 * num_heads * seq * head_dim * 4 bytes.
    // 8192 tokens → ~58.7MB, 16384 tokens → ~117MB.
    // 必须同时增大 send_window 和 receive_window，否则 ring 中双方同时 write_all
    // 大 block 时会因为发送端窗口耗尽而互相死锁。
    transport_config.stream_receive_window((128u64 * 1024 * 1024).try_into().unwrap());
    transport_config.receive_window((256u64 * 1024 * 1024).try_into().unwrap());
    transport_config.send_window(256u64 * 1024 * 1024);

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

pub struct QuicKvTransport {
    send: SendStream,
    recv: RecvStream,
    rt: Handle,
    device: Device,
    handshake_done: bool,
}

impl QuicKvTransport {
    pub fn new(send: SendStream, recv: RecvStream, rt: Handle, device: Device) -> Self {
        Self { send, recv, rt, device, handshake_done: false }
    }
}

impl KvTransport for QuicKvTransport {
    fn send_kv_block(&mut self, block: &KvBlock) -> Result<(), String> {
        self.rt.block_on(async {
            let k_bytes = tensor_to_bytes(&block.k)?;
            let v_bytes = tensor_to_bytes(&block.v)?;
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

            let mut frame = Vec::with_capacity(4 + meta_bytes.len() + k_bytes.len() + v_bytes.len());
            frame.extend_from_slice(&meta_len.to_be_bytes());
            frame.extend_from_slice(&meta_bytes);
            frame.extend_from_slice(&k_bytes);
            frame.extend_from_slice(&v_bytes);

            self.send.write_all(&frame).await
                .map_err(|e| format!("quic send failed: {e}"))?;
            Ok(())
        })
    }

    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        self.rt.block_on(async {
            // Skip the 1-byte dummy written during stream setup (once per stream)
            if !self.handshake_done {
                let mut dummy = [0u8; 1];
                match read_exact(&mut self.recv, &mut dummy).await {
                    Ok(()) => self.handshake_done = true,
                    Err(ReadExactError::Closed) => return Ok(None),
                    Err(ReadExactError::ReadError(e)) => {
                        return Err(format!("quic recv dummy failed: {e}"));
                    }
                }
            }

            let mut len_bytes = [0u8; 4];
            match read_exact(&mut self.recv, &mut len_bytes).await {
                Ok(()) => {}
                Err(ReadExactError::Closed) => return Ok(None),
                Err(ReadExactError::ReadError(e)) => {
                    return Err(format!("quic recv meta_len failed: {e}"));
                }
            }
            let meta_len = u32::from_be_bytes(len_bytes) as usize;

            let mut meta_bytes = vec![0u8; meta_len];
            read_exact(&mut self.recv, &mut meta_bytes).await
                .map_err(|e| format!("quic recv meta failed: {e}"))?;
            let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
                .map_err(|e| format!("quic parse meta failed: {e}"))?;

            let layer_idx = meta["layer_idx"].as_u64().ok_or("missing layer_idx")? as usize;
            let global_seq_start = meta["global_seq_start"].as_u64().ok_or("missing global_seq_start")? as usize;
            let global_seq_end = meta["global_seq_end"].as_u64().ok_or("missing global_seq_end")? as usize;
            let k_bytes_len = meta["k_bytes"].as_u64().ok_or("missing k_bytes")? as usize;
            let v_bytes_len = meta["v_bytes"].as_u64().ok_or("missing v_bytes")? as usize;
            let k_shape: Vec<i64> = meta["k_shape"].as_array().ok_or("missing k_shape")?
                .iter().map(|v| v.as_i64().ok_or("invalid k_shape"))
                .collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
            let v_shape: Vec<i64> = meta["v_shape"].as_array().ok_or("missing v_shape")?
                .iter().map(|v| v.as_i64().ok_or("invalid v_shape"))
                .collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

            let mut k_bytes = vec![0u8; k_bytes_len];
            read_exact(&mut self.recv, &mut k_bytes).await
                .map_err(|e| format!("quic recv k_bytes failed: {e}"))?;
            let mut v_bytes = vec![0u8; v_bytes_len];
            read_exact(&mut self.recv, &mut v_bytes).await
                .map_err(|e| format!("quic recv v_bytes failed: {e}"))?;

            let k = bytes_to_tensor(&k_bytes, &k_shape, self.device)?;
            let v = bytes_to_tensor(&v_bytes, &v_shape, self.device)?;

            Ok(Some(KvBlock { layer_idx, global_seq_start, global_seq_end, k, v }))
        })
    }
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

fn tensor_to_bytes(t: &Tensor) -> Result<Vec<u8>, String> {
    let flat = t.contiguous().view(-1);
    let values: Vec<f32> = Vec::try_from(&flat).map_err(|e| format!("tensor to vec failed: {e}"))?;
    Ok(values.iter().flat_map(|&v| v.to_le_bytes()).collect())
}

fn bytes_to_tensor(bytes: &[u8], shape: &[i64], device: Device) -> Result<Tensor, String> {
    let values: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let expected = shape.iter().product::<i64>() as usize;
    if values.len() != expected {
        return Err(format!("byte length mismatch: expected {} floats, got {}", expected, values.len()));
    }
    Ok(Tensor::from_slice(&values).reshape(shape).to_device(device))
}
