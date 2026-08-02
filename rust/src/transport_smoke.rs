use crate::distributed::transport::quic::{create_endpoint, QuicKvTransport};
use crate::model::transport::{KvTransport, SelfDrivingPacket};
use quinn::{Endpoint, RecvStream, SendStream};
use std::net::SocketAddr;
use tch::{Device, Kind, Tensor};
use tokio::runtime::Runtime;

pub fn run_self_driving_quic_smoke(args: &[String]) -> Result<(), String> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let (mode, bind_addr, peer_addr) = parse_args(args)?;
    let runtime = Runtime::new().map_err(|error| format!("create runtime failed: {error}"))?;
    let endpoint = runtime.block_on(async { create_endpoint(bind_addr) })?;

    match mode {
        "server" => run_server(&runtime, endpoint),
        "client" => run_client(
            &runtime,
            endpoint,
            peer_addr.expect("client peer address was validated"),
        ),
        _ => unreachable!(),
    }
}

fn parse_args(args: &[String]) -> Result<(&str, SocketAddr, Option<SocketAddr>), String> {
    let usage =
        "usage: self_driving_quic_smoke server <bind_addr> | client <bind_addr> <peer_addr>";
    let mode = args.first().map(String::as_str).ok_or(usage)?;
    let bind_addr = args
        .get(1)
        .ok_or(usage)?
        .parse()
        .map_err(|error| format!("invalid bind address: {error}"))?;
    match mode {
        "server" if args.len() == 2 => Ok((mode, bind_addr, None)),
        "client" if args.len() == 3 => {
            let peer_addr = args[2]
                .parse()
                .map_err(|error| format!("invalid peer address: {error}"))?;
            Ok((mode, bind_addr, Some(peer_addr)))
        }
        _ => Err(usage.to_string()),
    }
}

fn run_server(runtime: &Runtime, endpoint: Endpoint) -> Result<(), String> {
    let (connection, send, recv) = runtime.block_on(async {
        let incoming = endpoint
            .accept()
            .await
            .ok_or_else(|| "QUIC endpoint closed before connection".to_string())?;
        let connection = incoming
            .await
            .map_err(|error| format!("accept connection failed: {error}"))?;
        let (mut send, recv) = connection
            .accept_bi()
            .await
            .map_err(|error| format!("accept stream failed: {error}"))?;
        send.write_all(b"\x00")
            .await
            .map_err(|error| format!("write handshake failed: {error}"))?;
        Ok::<_, String>((connection, send, recv))
    })?;
    let mut transport = transport(runtime, send, recv);
    let packet = transport
        .recv_self_driving_packet()?
        .ok_or_else(|| "peer closed before sending packet".to_string())?;
    assert_packet(&packet)?;
    transport.submit_send_self_driving_packet(&packet)?;
    transport.flush_send()?;
    println!(
        "self-driving QUIC server roundtrip ok: peer={}, layer={}, position={}",
        connection.remote_address(),
        packet.layer_idx,
        packet.position_ids.int64_value(&[0, 0])
    );
    Ok(())
}

fn run_client(runtime: &Runtime, endpoint: Endpoint, peer_addr: SocketAddr) -> Result<(), String> {
    let (connection, send, recv) = runtime.block_on(async {
        let connection = endpoint
            .connect(peer_addr, "localhost")
            .map_err(|error| format!("start connection failed: {error}"))?
            .await
            .map_err(|error| format!("connect failed: {error}"))?;
        let (mut send, recv) = connection
            .open_bi()
            .await
            .map_err(|error| format!("open stream failed: {error}"))?;
        send.write_all(b"\x00")
            .await
            .map_err(|error| format!("write handshake failed: {error}"))?;
        Ok::<_, String>((connection, send, recv))
    })?;
    let mut transport = transport(runtime, send, recv);
    let packet = packet();
    transport.submit_send_self_driving_packet(&packet)?;
    transport.flush_send()?;
    let echoed = transport
        .recv_self_driving_packet()?
        .ok_or_else(|| "peer closed before echoing packet".to_string())?;
    assert_packet(&echoed)?;
    println!(
        "self-driving QUIC client roundtrip ok: peer={}, layer={}, position={}",
        connection.remote_address(),
        echoed.layer_idx,
        echoed.position_ids.int64_value(&[0, 0])
    );
    Ok(())
}

fn transport(runtime: &Runtime, send: SendStream, recv: RecvStream) -> Box<dyn KvTransport> {
    Box::new(QuicKvTransport::new(
        send,
        recv,
        runtime.handle().clone(),
        Device::Cpu,
    ))
}

fn packet() -> SelfDrivingPacket {
    let device = Device::Cpu;
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

fn assert_packet(actual: &SelfDrivingPacket) -> Result<(), String> {
    let expected = packet();
    if (
        actual.layer_idx,
        actual.assignee,
        actual.current_domain,
        actual.domains,
        actual.visited_domains,
    ) != (7, 1, 1, 2, 1)
    {
        return Err("self-driving route metadata changed".to_string());
    }
    for (name, actual, expected) in [
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
        if actual.kind() != expected.kind() {
            return Err(format!("{name} dtype changed"));
        }
        let diff = (actual - expected)
            .abs()
            .to_kind(Kind::Float)
            .max()
            .double_value(&[]);
        if diff != 0.0 {
            return Err(format!("{name} changed: max diff {diff}"));
        }
    }
    Ok(())
}
