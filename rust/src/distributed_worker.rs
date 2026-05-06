//! Distributed inference worker process.
//!
//! Supports both single-domain and multi-domain modes.
//! In multi-domain mode (--local-domain-ids), a single process hosts multiple
//! logical workers that share model weights (via shallow_clone) while each
//! maintains its own KV cache and coordinator connection.
//!
//! A process-wide Mutex serializes forward() calls across domains to avoid
//! GPU memory contention when multiple domains share the same physical card.

use crate::distributed_protocol::{
    recv_command_quic, send_response_quic, WorkerCommand, WorkerResponse,
};
use crate::model::model::LlamaModel;
use crate::model::{ModelConfig, ModelWeights};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use tch::{Device, Tensor};

#[derive(Debug)]
struct DomainConfig {
    domain_id: usize,
    listen_addr: String,
    next_peer_addr: String,
}

/// A reusable barrier for synchronizing domains within the same process.
/// All domains must arrive before any domain proceeds, then the barrier resets
/// automatically for the next synchronization point.
struct ResetBarrier {
    arrived: AtomicUsize,
    generation: AtomicUsize,
    target: usize,
}

impl ResetBarrier {
    fn new(target: usize) -> Self {
        Self {
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            target,
        }
    }

    fn wait(&self) {
        let my_gen = self.generation.load(Ordering::Relaxed);
        let count = self.arrived.fetch_add(1, Ordering::SeqCst) + 1;
        if count >= self.target {
            self.arrived.store(0, Ordering::SeqCst);
            self.generation.fetch_add(1, Ordering::SeqCst);
        } else {
            while self.generation.load(Ordering::Relaxed) == my_gen {
                std::thread::yield_now();
            }
        }
    }
}

#[derive(Debug)]
struct WorkerArgs {
    domain_configs: Vec<DomainConfig>,
    model_dir: String,
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

    let mut local_domain_ids: Option<Vec<usize>> = None;
    let mut listen_addrs: Option<Vec<String>> = None;
    let mut next_peer_addrs: Option<Vec<String>> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--distributed-role" => { let _ = args.next(); }
            "--domain-id" => domain_id = args.next().unwrap().parse().unwrap(),
            "--seq-offset" => seq_offset = args.next().unwrap().parse().unwrap(),
            "--model-dir" => model_dir = args.next().unwrap(),
            "--listen-addr" => listen_addr = args.next().unwrap(),
            "--next-peer-addr" => next_peer_addr = args.next().unwrap(),
            "--coordinator-addr" => coordinator_addr = args.next().unwrap(),
            "--num-domains" => num_domains = args.next().unwrap().parse().unwrap(),
            "--local-domain-ids" => {
                local_domain_ids = Some(args.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect());
            }
            "--listen-addrs" => {
                listen_addrs = Some(args.next().unwrap().split(',').map(|s| s.to_string()).collect());
            }
            "--next-peer-addrs" => {
                next_peer_addrs = Some(args.next().unwrap().split(',').map(|s| s.to_string()).collect());
            }
            _ => eprintln!("[worker] unknown arg: {arg}"),
        }
    }

    let domain_configs = if let Some(ids) = local_domain_ids {
        let listens = listen_addrs.expect("--listen-addrs required when --local-domain-ids is set");
        let peers = next_peer_addrs.expect("--next-peer-addrs required when --local-domain-ids is set");
        assert_eq!(ids.len(), listens.len(), "--local-domain-ids and --listen-addrs must have same length");
        assert_eq!(ids.len(), peers.len(), "--local-domain-ids and --next-peer-addrs must have same length");
        ids.into_iter()
            .zip(listens.into_iter().zip(peers.into_iter()))
            .map(|(id, (l, p))| DomainConfig { domain_id: id, listen_addr: l, next_peer_addr: p })
            .collect()
    } else {
        vec![DomainConfig {
            domain_id,
            listen_addr,
            next_peer_addr,
        }]
    };

    WorkerArgs {
        domain_configs,
        model_dir,
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

/// Run a single domain's command loop.
fn domain_worker_loop(
    domain_config: DomainConfig,
    config: ModelConfig,
    weights: &ModelWeights,
    device: Device,
    num_domains: usize,
    coordinator_addr: String,
    barrier: Arc<ResetBarrier>,
) {
    let domain_id = domain_config.domain_id;
    println!("[worker {domain_id}] starting, listen={}, next_peer={}, coordinator={}",
             domain_config.listen_addr, domain_config.next_peer_addr, coordinator_addr);
    println!("[worker {domain_id}] device: {:?}", device);

    // Each domain gets its own LlamaModel instance. Weights are shallow_clone'd
    // so the underlying tensor data is shared across domains in the same process.
    let mut model = LlamaModel::from_weights(config.clone(), weights, device, num_domains)
        .expect("build model failed");
    let mut kv_caches = model.create_kv_caches();

    // Setup QUIC transport for peer KV ring and coordinator control plane
    println!("[worker {domain_id}] setting up QUIC transport");
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime failed");
    let listen_addr: std::net::SocketAddr = domain_config.listen_addr.parse()
        .expect("invalid listen_addr");
    let next_peer_addr: std::net::SocketAddr = domain_config.next_peer_addr.parse()
        .expect("invalid next_peer_addr");
    let coord_addr: std::net::SocketAddr = coordinator_addr.parse()
        .expect("invalid coordinator_addr");

    let num_layers = config.num_layers;
    let (mut outbound_streams, mut inbound_streams, mut coord_send, mut coord_recv) = rt.block_on(async move {
        let endpoint = crate::quic_transport::create_endpoint(listen_addr)
            .expect("QUIC endpoint bind failed");
        let endpoint_for_accept = endpoint.clone();
        println!("[worker {domain_id}] QUIC endpoint bound to {listen_addr}");

        let dial_fut = endpoint.connect(next_peer_addr, "localhost").unwrap();

        let accept_fut = async move {
            loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    endpoint_for_accept.accept()
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

        // Connect to coordinator via QUIC
        println!("[worker {domain_id}] connecting to coordinator {coordinator_addr}");
        let coord_conn = endpoint.connect(coord_addr, "localhost")
            .expect("connect to coordinator failed")
            .await
            .expect("coordinator connection failed");
        println!("[worker {domain_id}] QUIC connection to coordinator established");
        let (cs, cr) = coord_conn.open_bi().await
            .expect("open_bi to coordinator failed");

        (outbound, inbound, cs, cr)
    });

    model.setup_distributed_domain(domain_id, 0, |_layer_idx| {
        let (out_send, _out_recv) = outbound_streams.remove(0);
        let (_in_send, in_recv) = inbound_streams.remove(0);
        Some(Box::new(crate::quic_transport::QuicKvTransport::new(
            out_send, in_recv, rt.handle().clone(), device,
        )))
    });
    println!("[worker {domain_id}] distributed domain setup complete");

    let capacity_mb = crate::capacity::query_device_capacity_mb(device);
    println!("[worker {domain_id}] capacity: {} MB", capacity_mb);

    let handshake = crate::distributed_protocol::WorkerHandshake {
        domain_id: domain_id as u64,
        capacity_mb,
    };
    crate::distributed_protocol::write_handshake_quic(&mut coord_send, &handshake, rt.handle())
        .expect("handshake write failed");
    println!("[worker {domain_id}] handshake sent to coordinator");

    loop {
        println!("[worker {domain_id}] waiting for command...");
        let cmd = recv_command_quic(&mut coord_recv, rt.handle()).expect("recv_command failed");
        println!("[worker {domain_id}] received command");
        match cmd {
            WorkerCommand::Prefill { chunk, seq_offset } => {
                model.seq_offset = seq_offset;
                for layer in model.layers.iter_mut() {
                    layer.attention.set_distributed(domain_id, seq_offset as usize, None);
                }
                let input = Tensor::from_slice(&chunk).unsqueeze(0).to_device(device);
                let logits = model.forward(&input, &mut kv_caches).expect("prefill forward failed");
                let last_logits = logits.narrow(1, logits.size()[1] - 1, 1).squeeze();
                let logits_vec: Vec<f32> = Vec::try_from(&last_logits).expect("logits to vec failed");
                let logits_bytes: Vec<u8> = logits_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                let resp = WorkerResponse::PrefillDone {
                    last_logits_bytes: logits_bytes,
                    global_seq_len: model.global_seq_len,
                };
                send_response_quic(&mut coord_send, &resp, rt.handle()).expect("send_response failed");
            }
            WorkerCommand::Decode(token) => {
                let input = Tensor::from_slice(&[token]).unsqueeze(0).to_device(device);
                let logits = model.forward(&input, &mut kv_caches).expect("decode forward failed");
                let logits_vec: Vec<f32> = Vec::try_from(&logits.squeeze()).expect("logits to vec failed");
                let logits_bytes: Vec<u8> = logits_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                let resp = WorkerResponse::DecodeDone { logits_bytes };
                send_response_quic(&mut coord_send, &resp, rt.handle()).expect("send_response failed");
            }
            WorkerCommand::SyncGlobalSeqLen(len) => {
                model.global_seq_len = len;
                println!("[worker {domain_id}] synced global_seq_len = {}", len);
            }
            WorkerCommand::Shutdown => {
                println!("[worker {domain_id}] shutting down");
                break;
            }
        }
    }
}

pub fn run() {
    // Install rustls ring crypto provider once per process (required by rustls 0.23)
    let _ = rustls::crypto::ring::default_provider().install_default();

    let args = parse_args();
    let device = select_device();

    // Load model once per process. All domains in this process share the weights
    // via shallow_clone (O(1) reference to the same underlying GPU memory).
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config = ModelConfig::from_file(&config_path).expect("load config failed");
    let weights = ModelWeights::from_dir(&args.model_dir, device).expect("load weights failed");
    println!("[multi-worker] loaded model weights once for {} domain(s)", args.domain_configs.len());

    // Barrier to synchronize all domains before forward (ring KV requires all active).
    let barrier = Arc::new(ResetBarrier::new(args.domain_configs.len()));

    let num_domains = args.num_domains;
    let coordinator_addr = args.coordinator_addr;

    let mut handles = Vec::new();
    for domain_config in args.domain_configs {
        let b = barrier.clone();
        let cfg = config.clone();
        let coord = coordinator_addr.clone();
        let w = ModelWeights {
            #[cfg(feature = "tch-backend")]
            tensors: weights.tensors.iter().map(|(k, v)| (k.clone(), v.shallow_clone())).collect(),
            #[cfg(not(feature = "tch-backend"))]
            tensors: weights.tensors.clone(),
        };
        let handle = std::thread::spawn(move || {
            domain_worker_loop(domain_config, cfg, &w, device, num_domains, coord, b);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("domain worker thread panicked");
    }
    println!("[multi-worker] all domains finished");
}
