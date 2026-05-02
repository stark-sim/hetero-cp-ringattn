//! Distributed inference worker process.
//!
//! Loads a LlamaModel, connects to a peer for KV exchange and to a
//! coordinator for control messages, then waits for Prefill/Decode/Shutdown
//! commands.

use crate::distributed_protocol::{recv_command, send_response, WorkerCommand, WorkerResponse};
use crate::model::model::LlamaModel;
use crate::model::{ModelConfig, ModelWeights};
use std::io::Write;
use std::path::Path;
use tch::{Device, Tensor};

#[derive(Debug)]
struct WorkerArgs {
    domain_id: usize,
    seq_offset: i64,
    model_dir: String,
    listen_addr: String,
    next_peer_addr: String,
    coordinator_addr: String,
    num_domains: usize,
}

fn parse_args() -> WorkerArgs {
    let mut domain_id = 0usize;
    let mut seq_offset = 0i64;
    let mut model_dir = String::new();
    let mut listen_addr = String::new();
    let mut next_peer_addr = String::new();
    let mut coordinator_addr = String::new();
    let mut num_domains = 2usize;

    let mut args = std::env::args().skip(1); // skip binary name
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--distributed-role" => { let _ = args.next(); } // consumed by main.rs, skip here
            "--domain-id" => domain_id = args.next().unwrap().parse().unwrap(),
            "--seq-offset" => seq_offset = args.next().unwrap().parse().unwrap(),
            "--model-dir" => model_dir = args.next().unwrap(),
            "--listen-addr" => listen_addr = args.next().unwrap(),
            "--next-peer-addr" => next_peer_addr = args.next().unwrap(),
            "--coordinator-addr" => coordinator_addr = args.next().unwrap(),
            "--num-domains" => num_domains = args.next().unwrap().parse().unwrap(),
            _ => eprintln!("[worker] unknown arg: {arg}"),
        }
    }

    WorkerArgs {
        domain_id,
        seq_offset,
        model_dir,
        listen_addr,
        next_peer_addr,
        coordinator_addr,
        num_domains,
    }
}

fn select_device() -> Device {
    if let Ok(name) = std::env::var("HCP_TORCH_DEVICE").or_else(|_| std::env::var("HCP_TCH_DEVICE")) {
        match name.as_str() {
            "cpu" => Device::Cpu,
            "mps" => Device::Mps,
            "cuda" => Device::Cuda(0),
            _ => {
                if let Some(idx) = name.strip_prefix("cuda:") {
                    if let Ok(i) = idx.parse::<usize>() {
                        Device::Cuda(i)
                    } else {
                        Device::Cpu
                    }
                } else {
                    Device::Cpu
                }
            }
        }
    } else if cfg!(target_os = "macos") && tch::utils::has_mps() {
        Device::Mps
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

fn connect_with_retry(addr: &str, attempts: usize, delay_ms: u64) -> Result<std::net::TcpStream, String> {
    for i in 0..attempts {
        match std::net::TcpStream::connect(addr) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
                let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
                return Ok(stream);
            }
            Err(e) => {
                if i == attempts - 1 {
                    return Err(format!("failed to connect to {addr} after {attempts} attempts: {e}"));
                }
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
    }
    unreachable!()
}

pub fn run() {
    // Install rustls ring crypto provider once per process (required by rustls 0.23)
    let _ = rustls::crypto::ring::default_provider().install_default();

    let args = parse_args();
    println!("[worker {}] starting, seq_offset={}, listen={}, next_peer={}, coordinator={}",
             args.domain_id, args.seq_offset, args.listen_addr, args.next_peer_addr, args.coordinator_addr);

    let device = select_device();
    println!("[worker {}] device: {:?}", args.domain_id, device);

    // Load model
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config = ModelConfig::from_file(&config_path).expect("load config failed");
    let weights = ModelWeights::from_dir(&args.model_dir, device).expect("load weights failed");
    let mut model = LlamaModel::from_weights(config.clone(), &weights, device, args.num_domains)
        .expect("build model failed");
    let mut kv_caches = model.create_kv_caches();

    // Setup QUIC transport for peer KV ring
    println!("[worker {}] setting up QUIC transport", args.domain_id);
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime failed");
    let listen_addr: std::net::SocketAddr = args.listen_addr.parse()
        .expect("invalid listen_addr");
    let next_peer_addr: std::net::SocketAddr = args.next_peer_addr.parse()
        .expect("invalid next_peer_addr");

    let domain_id = args.domain_id;
    let num_domains = args.num_domains;
    let num_layers = config.num_layers;
    let (mut outbound_streams, mut inbound_streams) = rt.block_on(async move {
        let endpoint = crate::quic_transport::create_endpoint(listen_addr)
            .expect("QUIC endpoint bind failed");
        println!("[worker {domain_id}] QUIC endpoint bound to {listen_addr}");

        // Connect to next peer and accept from prev peer concurrently
        println!("[worker {domain_id}] connecting to next peer {next_peer_addr}");
        let dial_fut = endpoint.connect(next_peer_addr, "localhost")
            .unwrap();

        let accept_fut = async move {
            loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    endpoint.accept()
                ).await {
                    Ok(Some(incoming)) => break incoming.await.expect("prev peer connection failed"),
                    Ok(None) => panic!("endpoint closed"),
                    Err(_) => {
                        println!("[worker {domain_id}] accept timeout, retrying...");
                        continue;
                    }
                }
            }
        };

        // For 2-domain ring, next == prev; avoid symmetric connection deadlocks
        // by only having domain 0 dial and domain 1 accept.
        let (conn, prev_conn) = if num_domains == 2 {
            if domain_id == 0 {
                let c = dial_fut.await.expect("connect to next peer failed");
                println!("[worker {domain_id}] QUIC connection to next peer established");
                (c.clone(), c)
            } else {
                let c = accept_fut.await;
                println!("[worker {domain_id}] QUIC connection from prev peer established");
                (c.clone(), c)
            }
        } else {
            let dial_handle = tokio::spawn(dial_fut);
            let accept_handle = tokio::spawn(accept_fut);
            let c = dial_handle.await.expect("dial task panicked")
                .expect("connect to next peer failed");
            println!("[worker {domain_id}] QUIC connection to next peer established");
            let p = accept_handle.await.expect("accept task panicked");
            println!("[worker {domain_id}] QUIC connection from prev peer established");
            (c, p)
        };

        // Open and accept bidirectional streams per layer.
        // Write a 1-byte dummy after open_bi so the peer's accept_bi can observe
        // the new stream immediately (quinn only creates the stream on first write).
        let mut outbound = Vec::with_capacity(num_layers);
        let mut inbound = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let (mut send, recv) = conn.open_bi().await
                .unwrap_or_else(|e| panic!("layer {layer_idx} open_bi failed: {e}"));
            send.write_all(b"\x00").await
                .unwrap_or_else(|e| panic!("layer {layer_idx} dummy write failed: {e}"));
            let (peer_send, peer_recv) = prev_conn.accept_bi().await
                .unwrap_or_else(|e| panic!("layer {layer_idx} accept_bi failed: {e}"));
            outbound.push((send, recv));
            inbound.push((peer_send, peer_recv));
            if layer_idx % 4 == 0 {
                println!("[worker {domain_id}] opened outbound & accepted inbound streams {layer_idx}/{num_layers}");
            }
        }

        (outbound, inbound)
    });

    // Set up distributed domain: each layer gets a QuicKvTransport
    // We use the send side of outbound connection and recv side of inbound connection.
    model.setup_distributed_domain(args.domain_id, args.seq_offset, |_layer_idx| {
        let (out_send, _out_recv) = outbound_streams.remove(0);
        let (_in_send, in_recv) = inbound_streams.remove(0);
        Some(Box::new(crate::quic_transport::QuicKvTransport::new(
            out_send, in_recv, rt.handle().clone(), device,
        )))
    });
    println!("[worker {}] distributed domain setup complete", args.domain_id);

    // Connect to coordinator
    println!("[worker {}] connecting to coordinator {}", args.domain_id, args.coordinator_addr);
    let mut coord_stream = connect_with_retry(&args.coordinator_addr, 300, 200)
        .expect("coordinator connect failed");
    println!("[worker {}] connected to coordinator", args.domain_id);

    // Handshake: send domain_id so coordinator knows who we are
    let handshake = (args.domain_id as u64).to_le_bytes();
    coord_stream.write_all(&handshake).expect("handshake write failed");

    // Command loop
    loop {
        let cmd = recv_command(&mut coord_stream).expect("recv_command failed");
        match cmd {
            WorkerCommand::Prefill(chunk_ids) => {
                let input = Tensor::from_slice(&chunk_ids).unsqueeze(0).to_device(device);
                let logits = model.forward(&input, &mut kv_caches).expect("prefill forward failed");
                // Extract last token logits
                let last_logits = logits.narrow(1, logits.size()[1] - 1, 1).squeeze();
                let logits_vec: Vec<f32> = Vec::try_from(&last_logits).expect("logits to vec failed");
                let logits_bytes: Vec<u8> = logits_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                let resp = WorkerResponse::PrefillDone {
                    last_logits_bytes: logits_bytes,
                    global_seq_len: model.global_seq_len,
                };
                send_response(&mut coord_stream, &resp).expect("send_response failed");
            }
            WorkerCommand::Decode(token) => {
                let input = Tensor::from_slice(&[token]).unsqueeze(0).to_device(device);
                let logits = model.forward(&input, &mut kv_caches).expect("decode forward failed");
                let logits_vec: Vec<f32> = Vec::try_from(&logits.squeeze()).expect("logits to vec failed");
                let logits_bytes: Vec<u8> = logits_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                let resp = WorkerResponse::DecodeDone { logits_bytes };
                send_response(&mut coord_stream, &resp).expect("send_response failed");
            }
            WorkerCommand::SyncGlobalSeqLen(len) => {
                model.global_seq_len = len;
                println!("[worker {}] synced global_seq_len = {}", args.domain_id, len);
            }
            WorkerCommand::Shutdown => {
                println!("[worker {}] shutting down", args.domain_id);
                break;
            }
        }
    }
}


