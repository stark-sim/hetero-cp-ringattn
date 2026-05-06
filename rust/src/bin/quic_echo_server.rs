//! 最小 QUIC echo server — 接收 KV block，k/v 各加 1.0 后回显。
//! 用于验证 Python aioquic client 与 Rust quinn server 的互通性。

use quinn::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig};
use rustls::client::danger::{ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;

#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self, _: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>,
        _: &[u8], _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &rustls::DigitallySignedStruct) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &rustls::DigitallySignedStruct) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
        ]
    }
}

fn create_endpoint(listen_addr: SocketAddr) -> Result<Endpoint, String> {
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
    transport_config.max_idle_timeout(Some(std::time::Duration::from_secs(3600).try_into().unwrap()));
    transport_config.mtu_discovery_config(None);
    transport_config.initial_mtu(1200);
    transport_config.stream_receive_window((512u64 * 1024 * 1024).try_into().unwrap());
    transport_config.receive_window((1024u64 * 1024 * 1024).try_into().unwrap());
    transport_config.send_window(1024u64 * 1024 * 1024);

    let mut endpoint = Endpoint::server(server_config, listen_addr)
        .map_err(|e| format!("bind failed: {e}"))?;

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

async fn read_exact(stream: &mut RecvStream, buf: &mut [u8]) -> Result<(), String> {
    let mut offset = 0;
    while offset < buf.len() {
        match stream.read(&mut buf[offset..]).await {
            Ok(Some(n)) => offset += n,
            Ok(None) => return Err("stream closed".to_string()),
            Err(e) => return Err(format!("read error: {e}")),
        }
    }
    Ok(())
}

async fn recv_kv_block(stream: &mut RecvStream, handshake_done: &mut bool) -> Result<Option<(serde_json::Value, Vec<f32>, Vec<f32>)>, String> {
    // skip dummy
    if !*handshake_done {
        let mut dummy = [0u8; 1];
        match read_exact(stream, &mut dummy).await {
            Ok(()) => *handshake_done = true,
            Err(_) => return Ok(None),
        }
    }

    let mut len_bytes = [0u8; 4];
    read_exact(stream, &mut len_bytes).await?;
    let meta_len = u32::from_be_bytes(len_bytes) as usize;

    let mut meta_bytes = vec![0u8; meta_len];
    read_exact(stream, &mut meta_bytes).await?;
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
        .map_err(|e| format!("parse meta failed: {e}"))?;

    let k_bytes_len = meta["k_bytes"].as_u64().ok_or("missing k_bytes")? as usize;
    let v_bytes_len = meta["v_bytes"].as_u64().ok_or("missing v_bytes")? as usize;

    let mut k_bytes = vec![0u8; k_bytes_len];
    read_exact(stream, &mut k_bytes).await?;
    let mut v_bytes = vec![0u8; v_bytes_len];
    read_exact(stream, &mut v_bytes).await?;

    let k_vals: Vec<f32> = k_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let v_vals: Vec<f32> = v_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(Some((meta, k_vals, v_vals)))
}

async fn send_kv_block(stream: &mut SendStream, meta: &serde_json::Value, k_vals: &[f32], v_vals: &[f32], handshake_done: &mut bool) -> Result<(), String> {
    if !*handshake_done {
        stream.write_all(b"\x00").await.map_err(|e| format!("dummy write failed: {e}"))?;
        *handshake_done = true;
    }

    let k_bytes: Vec<u8> = k_vals.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let v_bytes: Vec<u8> = v_vals.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let mut meta_obj = meta.clone();
    meta_obj["k_bytes"] = serde_json::json!(k_bytes.len());
    meta_obj["v_bytes"] = serde_json::json!(v_bytes.len());

    let meta_bytes = meta_obj.to_string().into_bytes();
    let meta_len = meta_bytes.len() as u32;

    let mut frame = Vec::with_capacity(4 + meta_bytes.len() + k_bytes.len() + v_bytes.len());
    frame.extend_from_slice(&meta_len.to_be_bytes());
    frame.extend_from_slice(&meta_bytes);
    frame.extend_from_slice(&k_bytes);
    frame.extend_from_slice(&v_bytes);

    stream.write_all(&frame).await.map_err(|e| format!("send failed: {e}"))?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustls::crypto::ring::default_provider().install_default().unwrap();
    let listen_addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:29590".to_string())
        .parse()?;

    let endpoint = create_endpoint(listen_addr)?;
    println!("[rust server] listening on {}", listen_addr);

    let incoming = endpoint.accept().await.ok_or("no incoming connection")?;
    let conn = incoming.await?;
    println!("[rust server] connection accepted");

    let (mut send, mut recv) = conn.accept_bi().await?;
    println!("[rust server] bidirectional stream accepted");

    // Send dummy immediately after stream setup (Rust quinn workaround)
    send.write_all(b"\x00").await?;
    let mut handshake_done = false;

    let (meta, mut k_vals, mut v_vals) = recv_kv_block(&mut recv, &mut handshake_done)
        .await?
        .expect("no block received");

    println!("[rust server] received: layer={}, k_shape={:?}",
        meta["layer_idx"], meta["k_shape"]);

    // k/v + 1.0
    for v in &mut k_vals { *v += 1.0; }
    for v in &mut v_vals { *v += 1.0; }

    send_kv_block(&mut send, &meta, &k_vals, &v_vals, &mut handshake_done).await?;
    println!("[rust server] sent back");

    // Wait for client to read all data before closing
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    send.finish()?;
    conn.close(0u32.into(), b"done");
    endpoint.close(0u32.into(), b"done");

    Ok(())
}
