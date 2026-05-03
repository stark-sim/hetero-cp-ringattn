//! Distributed inference coordinator process.
//!
//! Orchestrates prefill and decode across multiple workers.
//! Does NOT load model weights; only needs tokenizer and config.

use crate::distributed_protocol::{
    recv_response_quic, send_command_quic, WorkerCommand, WorkerResponse,
};
use crate::model::config::ModelConfig;
use crate::model::generate::sample_token;
use std::net::SocketAddr;
use std::path::Path;
use tch::Tensor;

#[derive(Debug)]
struct CoordinatorArgs {
    model_dir: String,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    num_domains: usize,
    worker_addrs: Vec<String>,
    listen_addr: String,
    /// Optional explicit chunk sizes for uneven sharding.
    /// If provided, length must equal num_domains and sum must equal prompt length.
    chunk_sizes: Option<Vec<usize>>,
    /// Enable capacity-aware automatic chunk sharding.
    /// Workers report free memory; coordinator allocates proportionally.
    capacity_aware: bool,
    /// Read prompt from file instead of inline --prompt.
    prompt_file: Option<String>,
}

fn parse_args() -> CoordinatorArgs {
    let mut model_dir = String::new();
    let mut prompt = String::new();
    let mut max_tokens = 20usize;
    let mut temperature = 0.0f64;
    let mut top_p = 1.0f64;
    let mut num_domains = 2usize;
    let mut worker_addrs = Vec::new();
    let mut listen_addr = String::new();
    let mut chunk_sizes: Option<Vec<usize>> = None;
    let mut capacity_aware = false;
    let mut prompt_file = None;

    let mut args = std::env::args().skip(1); // skip binary name
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--distributed-role" => { let _ = args.next(); } // consumed by main.rs, skip here
            "--model-dir" => model_dir = args.next().unwrap(),
            "--prompt" => prompt = args.next().unwrap(),
            "--max-tokens" => max_tokens = args.next().unwrap().parse().unwrap(),
            "--temperature" => temperature = args.next().unwrap().parse().unwrap(),
            "--top-p" => top_p = args.next().unwrap().parse().unwrap(),
            "--num-domains" => num_domains = args.next().unwrap().parse().unwrap(),
            "--worker-addrs" => {
                worker_addrs = args.next().unwrap().split(',').map(|s| s.to_string()).collect();
            }
            "--listen-addr" => listen_addr = args.next().unwrap(),
            "--chunk-sizes" => {
                let s = args.next().unwrap();
                chunk_sizes = Some(s.split(',').map(|x| x.parse().unwrap()).collect());
            }
            "--capacity-aware" => capacity_aware = true,
            "--prompt-file" => prompt_file = Some(args.next().unwrap()),
            _ => eprintln!("[coordinator] unknown arg: {arg}"),
        }
    }

    CoordinatorArgs {
        model_dir,
        prompt,
        max_tokens,
        temperature,
        top_p,
        num_domains,
        worker_addrs,
        listen_addr,
        chunk_sizes,
        capacity_aware,
        prompt_file,
    }
}

pub fn run() {
    // Install rustls ring crypto provider once per process (required by rustls 0.23)
    let _ = rustls::crypto::ring::default_provider().install_default();

    let args = parse_args();
    println!("[coordinator] starting, num_domains={}, workers={:?}, listen={}",
             args.num_domains, args.worker_addrs, args.listen_addr);

    // Load tokenizer and config
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config = ModelConfig::from_file(&config_path).expect("load config failed");
    let tokenizer_path = Path::new(&args.model_dir).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("load tokenizer failed");

    // Load prompt from file or use inline string
    let prompt_text = if let Some(ref path) = args.prompt_file {
        std::fs::read_to_string(path).expect("read prompt-file failed")
    } else {
        args.prompt.clone()
    };
    println!("[coordinator] prompt length: {} chars", prompt_text.len());

    // Tokenize prompt
    let encoding = tokenizer.encode(prompt_text.as_str(), true).expect("encode failed");
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    println!("[coordinator] prompt tokens: {}", prompt_ids.len());

    // Create QUIC endpoint and wait for workers to connect
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime failed");
    let listen_addr: SocketAddr = args.listen_addr.parse().expect("invalid listen_addr");

    let endpoint = rt.block_on(async {
        crate::quic_transport::create_endpoint(listen_addr)
    }).expect("create_endpoint failed");
    println!("[coordinator] QUIC endpoint listening on {}", args.listen_addr);

    let mut worker_handshakes: Vec<(usize, u64, quinn::SendStream, quinn::RecvStream)> =
        Vec::with_capacity(args.num_domains);
    for i in 0..args.num_domains {
        let (send, mut recv) = rt.block_on(async {
            let incoming = loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(600),
                    endpoint.accept()
                ).await {
                    Ok(Some(incoming)) => break incoming,
                    Ok(None) => return Err("endpoint closed".to_string()),
                    Err(_) => return Err("accept timeout after 600s".to_string()),
                }
            };
            let conn = incoming.await.map_err(|e| format!("connection failed: {e}"))?;
            println!("[coordinator] worker connection established (accept order {i})");
            let (send, recv) = conn.accept_bi().await
                .map_err(|e| format!("accept_bi failed: {e}"))?;
            Ok::<_, String>((send, recv))
        }).unwrap_or_else(|e| panic!("accept worker {i} failed: {e}"));

        let handshake = crate::distributed_protocol::read_handshake_quic(&mut recv, rt.handle())
            .expect("handshake read failed");
        println!("[coordinator] worker {} connected (accept order {i}), capacity={} MB",
                 handshake.domain_id, handshake.capacity_mb);
        worker_handshakes.push((
            handshake.domain_id as usize,
            handshake.capacity_mb,
            send,
            recv,
        ));
    }
    worker_handshakes.sort_by_key(|(domain_id, _, _, _)| *domain_id);
    let worker_capacities: Vec<u64> = worker_handshakes.iter().map(|(_, cap, _, _)| *cap).collect();
    let mut worker_streams: Vec<(quinn::SendStream, quinn::RecvStream)> = worker_handshakes
        .into_iter()
        .map(|(_, _, send, recv)| (send, recv))
        .collect();

    // Prefill: split prompt into chunks and send to workers
    let seq_len = prompt_ids.len() as i64;

    // Three-tier allocation priority:
    //   1. --chunk-sizes (exact manual override)
    //   2. --capacity-aware (proportional to worker free memory)
    //   3. even sharding (default fallback)
    let chunk_sizes: Vec<usize> = if let Some(ref sizes) = args.chunk_sizes {
        assert_eq!(sizes.len(), args.num_domains,
            "--chunk-sizes length ({}) must match --num-domains ({})",
            sizes.len(), args.num_domains);
        let sum: usize = sizes.iter().sum();
        assert_eq!(sum, seq_len as usize,
            "--chunk-sizes sum ({}) must equal prompt length ({})",
            sum, seq_len);
        sizes.clone()
    } else if args.capacity_aware {
        let chunks = crate::capacity::allocate_by_capacity(seq_len as usize, &worker_capacities);
        println!("[coordinator] capacity-aware chunks: {:?} (capacities: {:?} MB)",
                 chunks, worker_capacities);
        chunks
    } else {
        let chunk_size = (seq_len as usize).div_ceil(args.num_domains).max(1);
        let mut chunks = Vec::with_capacity(args.num_domains);
        let mut offset = 0usize;
        for i in 0..args.num_domains {
            let end = if i == args.num_domains - 1 {
                seq_len as usize
            } else {
                (offset + chunk_size).min(seq_len as usize)
            };
            chunks.push(end - offset);
            offset = end;
        }
        chunks
    };

    // Build boundary offsets from chunk sizes
    let mut chunk_boundaries = vec![0usize];
    for size in &chunk_sizes {
        chunk_boundaries.push(chunk_boundaries.last().unwrap() + size);
    }

    for (domain_id, (send, _recv)) in worker_streams.iter_mut().enumerate() {
        let start = chunk_boundaries[domain_id];
        let end = chunk_boundaries[domain_id + 1];
        let chunk = &prompt_ids[start..end];
        let cmd = WorkerCommand::Prefill {
            chunk: chunk.to_vec(),
            seq_offset: start as i64,
        };
        send_command_quic(send, &cmd, rt.handle()).expect("send Prefill failed");
        println!("[coordinator] sent Prefill chunk [{}, {}) to worker {}", start, end, domain_id);
    }

    // Collect prefill responses
    let mut max_global_seq_len = 0usize;
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    for (domain_id, (_send, recv)) in worker_streams.iter_mut().enumerate() {
        let resp = recv_response_quic(recv, rt.handle()).expect("recv PrefillDone failed");
        match resp {
            WorkerResponse::PrefillDone { last_logits_bytes: bytes, global_seq_len } => {
                println!("[coordinator] worker {} prefill done, global_seq_len={}", domain_id, global_seq_len);
                max_global_seq_len = max_global_seq_len.max(global_seq_len);
                if domain_id == args.num_domains - 1 {
                    last_logits_bytes = bytes;
                }
            }
            WorkerResponse::Error(e) => panic!("worker {} prefill error: {e}", domain_id),
            _ => panic!("unexpected response from worker {}: {:?}", domain_id, resp),
        }
    }

    // Sync global_seq_len to all workers
    println!("[coordinator] max_global_seq_len = {}", max_global_seq_len);
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::SyncGlobalSeqLen(max_global_seq_len);
        send_command_quic(send, &cmd, rt.handle()).expect("send SyncGlobalSeqLen failed");
    }

    // Deserialize last logits and sample first token
    let vocab_size = config.vocab_size as usize;
    let logits_vec: Vec<f32> = last_logits_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if logits_vec.len() != vocab_size {
        panic!("logits size mismatch: expected {}, got {}", vocab_size, logits_vec.len());
    }
    let logits_tensor = Tensor::from_slice(&logits_vec);
    let mut next_token = sample_token(&logits_tensor, args.temperature, args.top_p)
        .expect("sample_token failed") as i64;

    let eos_token = config.eos_token_id();
    let mut generated_ids: Vec<u32> = Vec::new();

    // Decode loop
    for step in 0..args.max_tokens {
        let token = next_token as u32;
        generated_ids.push(token);

        if Some(token) == eos_token {
            println!("[coordinator] EOS at step {}", step);
            break;
        }

        // Broadcast token to all workers
        for (send, _recv) in worker_streams.iter_mut() {
            let cmd = WorkerCommand::Decode(next_token);
            send_command_quic(send, &cmd, rt.handle()).expect("send Decode failed");
        }

        // Receive decode response from first worker (all should be identical)
        let resp = recv_response_quic(&mut worker_streams[0].1, rt.handle())
            .expect("recv DecodeDone failed");
        let logits_bytes = match resp {
            WorkerResponse::DecodeDone { logits_bytes } => logits_bytes,
            WorkerResponse::Error(e) => panic!("worker 0 decode error: {e}"),
            _ => panic!("unexpected response from worker 0: {:?}", resp),
        };

        // Optionally drain responses from other workers to keep streams in sync
        for (_send, recv) in worker_streams.iter_mut().skip(1) {
            let _ = recv_response_quic(recv, rt.handle()).expect("recv DecodeDone failed");
        }

        let decode_logits: Vec<f32> = logits_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let decode_tensor = Tensor::from_slice(&decode_logits);
        next_token = sample_token(&decode_tensor, args.temperature, args.top_p)
            .expect("sample_token failed") as i64;
    }

    // Shutdown workers
    for (send, _recv) in worker_streams.iter_mut() {
        let _ = send_command_quic(send, &WorkerCommand::Shutdown, rt.handle());
    }

    // Decode to text
    let text = tokenizer.decode(&generated_ids, true).expect("decode failed");
    println!("[coordinator] generated: {}", text);
}
