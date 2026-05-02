//! Distributed inference coordinator process.
//!
//! Orchestrates prefill and decode across multiple workers.
//! Does NOT load model weights; only needs tokenizer and config.

use crate::distributed_protocol::{recv_response, send_command, WorkerCommand, WorkerResponse};
use crate::model::config::ModelConfig;
use crate::model::generate::sample_token;
use std::io::Read;
use std::net::{TcpListener, TcpStream};
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

    let mut args = std::env::args().skip(2); // skip binary name + "coordinator"
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
    }
}

fn accept_with_retry(listener: &TcpListener, attempts: usize, delay_ms: u64) -> Result<TcpStream, String> {
    listener.set_nonblocking(true).map_err(|e| format!("set_nonblocking failed: {e}"))?;
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
    println!("[coordinator] starting, num_domains={}, workers={:?}, listen={}",
             args.num_domains, args.worker_addrs, args.listen_addr);

    // Load tokenizer and config
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config = ModelConfig::from_file(&config_path).expect("load config failed");
    let tokenizer_path = Path::new(&args.model_dir).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("load tokenizer failed");

    // Tokenize prompt
    let encoding = tokenizer.encode(args.prompt.as_str(), true).expect("encode failed");
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    println!("[coordinator] prompt tokens: {}", prompt_ids.len());

    // Start listener and wait for workers to connect
    let listener = TcpListener::bind(&args.listen_addr).expect("bind listen_addr failed");
    println!("[coordinator] listening on {}", args.listen_addr);

    let mut worker_handshakes: Vec<(usize, TcpStream)> = Vec::with_capacity(args.num_domains);
    for i in 0..args.num_domains {
        let mut stream = accept_with_retry(&listener, 300, 200)
            .unwrap_or_else(|e| panic!("accept worker {i} failed: {e}"));
        let mut handshake = [0u8; 8];
        stream.read_exact(&mut handshake).expect("handshake read failed");
        let domain_id = u64::from_le_bytes(handshake) as usize;
        println!("[coordinator] worker {domain_id} connected (accept order {i})");
        worker_handshakes.push((domain_id, stream));
    }
    worker_handshakes.sort_by_key(|(domain_id, _)| *domain_id);
    let mut worker_streams: Vec<TcpStream> = worker_handshakes.into_iter().map(|(_, s)| s).collect();

    // Prefill: split prompt into chunks and send to workers
    let seq_len = prompt_ids.len() as i64;
    let chunk_size = (seq_len as usize).div_ceil(args.num_domains).max(1) as i64;

    for (domain_id, stream) in worker_streams.iter_mut().enumerate() {
        let start = (domain_id as i64 * chunk_size).min(seq_len);
        let end = ((domain_id as i64 + 1) * chunk_size).min(seq_len);
        let chunk = &prompt_ids[start as usize..end as usize];
        let cmd = WorkerCommand::Prefill(chunk.to_vec());
        send_command(stream, &cmd).expect("send Prefill failed");
        println!("[coordinator] sent Prefill chunk [{}, {}) to worker {}", start, end, domain_id);
    }

    // Collect prefill responses
    let mut max_global_seq_len = 0usize;
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    for (domain_id, stream) in worker_streams.iter_mut().enumerate() {
        let resp = recv_response(stream).expect("recv PrefillDone failed");
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
    for stream in worker_streams.iter_mut() {
        let cmd = WorkerCommand::SyncGlobalSeqLen(max_global_seq_len);
        send_command(stream, &cmd).expect("send SyncGlobalSeqLen failed");
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
        for stream in worker_streams.iter_mut() {
            let cmd = WorkerCommand::Decode(next_token);
            send_command(stream, &cmd).expect("send Decode failed");
        }

        // Receive decode response from first worker (all should be identical)
        let resp = recv_response(&mut worker_streams[0]).expect("recv DecodeDone failed");
        let logits_bytes = match resp {
            WorkerResponse::DecodeDone { logits_bytes } => logits_bytes,
            WorkerResponse::Error(e) => panic!("worker 0 decode error: {e}"),
            _ => panic!("unexpected response from worker 0: {:?}", resp),
        };

        // Optionally drain responses from other workers to keep streams in sync
        for stream in worker_streams.iter_mut().skip(1) {
            let _ = recv_response(stream).expect("recv DecodeDone failed");
        }

        let decode_logits: Vec<f32> = logits_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let decode_tensor = Tensor::from_slice(&decode_logits);
        next_token = sample_token(&decode_tensor, args.temperature, args.top_p)
            .expect("sample_token failed") as i64;
    }

    // Shutdown workers
    for stream in worker_streams.iter_mut() {
        let _ = send_command(stream, &WorkerCommand::Shutdown);
    }

    // Decode to text
    let text = tokenizer.decode(&generated_ids, true).expect("decode failed");
    println!("[coordinator] generated: {}", text);
}
