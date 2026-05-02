//! Distributed inference worker process.
//!
//! Loads a LlamaModel, connects to a peer for KV exchange and to a
//! coordinator for control messages, then waits for Prefill/Decode/Shutdown
//! commands.

use crate::distributed_protocol::{recv_command, send_response, WorkerCommand, WorkerResponse};
use crate::model::model::LlamaModel;
use crate::model::{ModelConfig, ModelWeights};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
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

    let mut args = std::env::args().skip(2); // skip binary name + "worker"
    while let Some(arg) = args.next() {
        match arg.as_str() {
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

fn connect_with_retry(addr: &str, attempts: usize, delay_ms: u64) -> Result<TcpStream, String> {
    for i in 0..attempts {
        match TcpStream::connect(addr) {
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

fn accept_with_retry(listener: &TcpListener, attempts: usize, delay_ms: u64) -> Result<TcpStream, String> {
    let _ = listener.set_nonblocking(true);
    for i in 0..attempts {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
                let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
                return Ok(stream);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if i == attempts - 1 {
                    return Err(format!("accept timeout after {attempts} attempts"));
                }
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
            Err(e) => return Err(format!("accept failed: {e}")),
        }
    }
    unreachable!()
}

pub fn run() {
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

    // Start listener for peer KV (ring inbound)
    let listener = TcpListener::bind(&args.listen_addr).expect("bind listen_addr failed");
    println!("[worker {}] listening on {}", args.domain_id, args.listen_addr);

    // Connect to peer (ring outbound) for each layer
    println!("[worker {}] connecting to next peer {}", args.domain_id, args.next_peer_addr);
    let num_layers = config.num_layers;
    let mut peer_streams = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let stream = connect_with_retry(&args.next_peer_addr, 300, 200)
            .unwrap_or_else(|e| panic!("layer {layer_idx} next peer connect failed: {e}"));
        peer_streams.push(stream);
        if layer_idx % 4 == 0 {
            println!("[worker {}] connected peer layer {}/{}", args.domain_id, layer_idx, num_layers);
        }
    }

    // Accept inbound peer connections for each layer
    println!("[worker {}] waiting for peer inbound connections", args.domain_id);
    let mut inbound_streams = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let stream = accept_with_retry(&listener, 300, 200)
            .unwrap_or_else(|e| panic!("layer {layer_idx} accept failed: {e}"));
        inbound_streams.push(stream);
        if layer_idx % 4 == 0 {
            println!("[worker {}] accepted peer layer {}/{}", args.domain_id, layer_idx, num_layers);
        }
    }

    // Set up distributed domain: each layer gets a TcpKvTransport
    // For 2-domain ring: we send on peer_streams and receive on inbound_streams.
    model.setup_distributed_domain(args.domain_id, args.seq_offset, |layer_idx| {
        let outbound = peer_streams[layer_idx].try_clone().expect("stream clone failed");
        let inbound = inbound_streams[layer_idx].try_clone().expect("stream clone failed");
        // Create a bidirectional transport: send uses outbound, recv uses inbound.
        // TcpKvTransport only supports one direction per instance, so we need a wrapper.
        Some(Box::new(BidirectionalTcpKvTransport::new(outbound, inbound, device)))
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

/// A bidirectional KvTransport that sends on one stream and receives on another.
/// For a 2-domain ring, each worker sends to peer via outbound stream and
/// receives from peer via inbound stream.
struct BidirectionalTcpKvTransport {
    outbound: std::sync::Mutex<TcpStream>,
    inbound: std::sync::Mutex<TcpStream>,
    device: Device,
}

impl BidirectionalTcpKvTransport {
    fn new(outbound: TcpStream, inbound: TcpStream, device: Device) -> Self {
        Self {
            outbound: std::sync::Mutex::new(outbound),
            inbound: std::sync::Mutex::new(inbound),
            device,
        }
    }
}

impl crate::model::KvTransport for BidirectionalTcpKvTransport {
    fn send_kv_block(&mut self, block: &crate::model::kv_transport::KvBlock) -> Result<(), String> {
        let mut guard = self.outbound.lock().unwrap();
        // Reuse TcpKvTransport's serialization logic inline
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

        guard.write_all(&frame).map_err(|e| format!("send_kv_block write failed: {e}"))?;
        Ok(())
    }

    fn recv_kv_block(&mut self) -> Result<Option<crate::model::kv_transport::KvBlock>, String> {
        let mut guard = self.inbound.lock().unwrap();
        let mut len_bytes = [0u8; 4];
        match guard.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(format!("recv_kv_block read meta_len failed: {e}")),
        }
        let meta_len = u32::from_be_bytes(len_bytes) as usize;

        let mut meta_bytes = vec![0u8; meta_len];
        guard.read_exact(&mut meta_bytes).map_err(|e| format!("recv_kv_block read meta failed: {e}"))?;
        let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
            .map_err(|e| format!("recv_kv_block parse meta failed: {e}"))?;

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
        guard.read_exact(&mut k_bytes).map_err(|e| format!("recv_kv_block read k_bytes failed: {e}"))?;
        let mut v_bytes = vec![0u8; v_bytes_len];
        guard.read_exact(&mut v_bytes).map_err(|e| format!("recv_kv_block read v_bytes failed: {e}"))?;

        let k = bytes_to_tensor(&k_bytes, &k_shape, self.device)?;
        let v = bytes_to_tensor(&v_bytes, &v_shape, self.device)?;

        Ok(Some(crate::model::kv_transport::KvBlock {
            layer_idx,
            global_seq_start,
            global_seq_end,
            k,
            v,
        }))
    }
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
