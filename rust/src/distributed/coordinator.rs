//! Distributed inference coordinator process.
//!
//! Orchestrates prefill and decode across multiple workers.
//! Does NOT load model weights; only needs tokenizer and config.
//!
//! Two serving modes:
//! 1. **Batch mode**: `--prompts-file` (one per line) — process all prompts then exit.
//! 2. **HTTP API mode**: default when no `--prompts-file`/`--prompt-file`/`--prompt` is given.
//!    Starts an OpenAI-compatible HTTP server on `--http-addr` (default 0.0.0.0:8080)
//!    and serves `/v1/completions`, `/health`, `/metrics`.

use crate::api::types::{InferenceJob, InferenceResult, StreamChunk};
use crate::api::{build_router, ApiState};
use crate::distributed::protocol::{
    recv_response_quic, send_command_quic, WorkerCommand, WorkerResponse,
};
use crate::distributed::scheduler::{BatchScheduler, ActiveRequest};
use crate::model::config::ModelConfig;
use crate::model::sampling::sample_token;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
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
    /// HTTP API bind address. Default "0.0.0.0:8080" when in HTTP mode.
    http_addr: String,
    /// Optional explicit chunk sizes for uneven sharding.
    chunk_sizes: Option<Vec<usize>>,
    /// Enable capacity-aware automatic chunk sharding.
    capacity_aware: bool,
    /// Read prompt from file instead of inline --prompt.
    prompt_file: Option<String>,
    /// Read multiple prompts from file (one per line) for batch serving.
    prompts_file: Option<String>,
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
    let mut http_addr = "0.0.0.0:8080".to_string();
    let mut chunk_sizes: Option<Vec<usize>> = None;
    let mut capacity_aware = false;
    let mut prompt_file = None;
    let mut prompts_file = None;

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
            "--http-addr" => http_addr = args.next().unwrap(),
            "--chunk-sizes" => {
                let s = args.next().unwrap();
                chunk_sizes = Some(s.split(',').map(|x| x.parse().unwrap()).collect());
            }
            "--capacity-aware" => capacity_aware = true,
            "--prompt-file" => prompt_file = Some(args.next().unwrap()),
            "--prompts-file" => prompts_file = Some(args.next().unwrap()),
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
        http_addr,
        chunk_sizes,
        capacity_aware,
        prompt_file,
        prompts_file,
    }
}

/// Process a single inference request against the connected workers.
///
/// Returns `InferenceResult` on success, `String` error message on failure.
#[allow(clippy::too_many_arguments)]
fn process_single_request(
    request_id: u64,
    prompt_text: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    tokenizer: &tokenizers::Tokenizer,
    config: &ModelConfig,
    worker_streams: &mut [(quinn::SendStream, quinn::RecvStream)],
    chunk_sizes_override: &Option<Vec<usize>>,
    capacity_aware: bool,
    worker_capacities: &[u64],
    rt: &tokio::runtime::Runtime,
) -> Result<InferenceResult, String> {
    let eos_token = config.eos_token_id();
    let vocab_size = config.vocab_size;
    let num_domains = worker_streams.len();

    let encoding = tokenizer
        .encode(prompt_text, true)
        .map_err(|e| format!("encode failed: {e}"))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let prompt_tokens = prompt_ids.len();

    let seq_len = prompt_ids.len() as i64;

    // Three-tier allocation priority
    let chunk_sizes: Vec<usize> = if let Some(ref sizes) = chunk_sizes_override {
        if sizes.len() != num_domains {
            return Err(format!(
                "--chunk-sizes length ({}) must match num_domains ({})",
                sizes.len(), num_domains
            ));
        }
        let sum: usize = sizes.iter().sum();
        if sum != seq_len as usize {
            return Err(format!(
                "--chunk-sizes sum ({}) must equal prompt length ({})",
                sum, seq_len
            ));
        }
        sizes.clone()
    } else if capacity_aware {
        crate::capacity::allocate_by_capacity(seq_len as usize, worker_capacities)
    } else {
        let chunk_size = (seq_len as usize).div_ceil(num_domains).max(1);
        let mut chunks = Vec::with_capacity(num_domains);
        let mut offset = 0usize;
        for i in 0..num_domains {
            let end = if i == num_domains - 1 {
                seq_len as usize
            } else {
                (offset + chunk_size).min(seq_len as usize)
            };
            chunks.push(end - offset);
            offset = end;
        }
        chunks
    };

    for (i, size) in chunk_sizes.iter().enumerate() {
        if *size == 0 {
            return Err(format!(
                "prompt too short: domain {} received 0 tokens (total {} tokens, {} domains). \
                 Each domain needs at least 1 token.",
                i, prompt_ids.len(), num_domains
            ));
        }
    }

    let mut chunk_boundaries = vec![0usize];
    for size in &chunk_sizes {
        chunk_boundaries.push(chunk_boundaries.last().unwrap() + size);
    }

    // Prefill
    for (domain_id, (send, _recv)) in worker_streams.iter_mut().enumerate() {
        let start = chunk_boundaries[domain_id];
        let end = chunk_boundaries[domain_id + 1];
        let chunk = &prompt_ids[start..end];
        let cmd = WorkerCommand::Prefill {
            request_id,
            chunk: chunk.to_vec(),
            seq_offset: start as i64,
        };
        send_command_quic(send, &cmd, rt.handle()).map_err(|e| format!("send Prefill failed: {e}"))?;
    }

    let mut max_global_seq_len = 0usize;
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    for (domain_id, (_send, recv)) in worker_streams.iter_mut().enumerate() {
        let resp = recv_response_quic(recv, rt.handle())
            .map_err(|e| format!("recv PrefillDone failed: {e}"))?;
        match resp {
            WorkerResponse::PrefillDone { last_logits_bytes: bytes, global_seq_len, .. } => {
                max_global_seq_len = max_global_seq_len.max(global_seq_len);
                if domain_id == num_domains - 1 {
                    last_logits_bytes = bytes;
                }
            }
            WorkerResponse::Error { message, .. } => {
                return Err(format!("worker {domain_id} prefill error: {message}"));
            }
            _ => return Err(format!("unexpected response from worker {domain_id}: {resp:?}")),
        }
    }

    // Sync global_seq_len
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::SyncGlobalSeqLen { request_id, len: max_global_seq_len };
        let _ = send_command_quic(send, &cmd, rt.handle());
    }

    // Sample first token from last worker's logits
    let logits_vec: Vec<f32> = last_logits_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if logits_vec.len() != vocab_size {
        return Err(format!(
            "logits size mismatch: expected {}, got {}",
            vocab_size, logits_vec.len()
        ));
    }
    let logits_tensor = Tensor::from_slice(&logits_vec);
    let mut next_token = match sample_token(&logits_tensor, temperature, top_p) {
        Ok(t) => t as i64,
        Err(e) => return Err(format!("sample_token failed: {e}")),
    };

    let mut generated_ids: Vec<u32> = Vec::new();

    // Decode loop
    let mut finish_reason = None;
    for step in 0..max_tokens {
        let token = next_token as u32;
        generated_ids.push(token);

        if Some(token) == eos_token {
            finish_reason = Some("stop".to_string());
            break;
        }

        for (send, _recv) in worker_streams.iter_mut() {
            let cmd = WorkerCommand::Decode { request_id, token: next_token };
            let _ = send_command_quic(send, &cmd, rt.handle());
        }

        let resp = recv_response_quic(&mut worker_streams[0].1, rt.handle())
            .map_err(|e| format!("recv DecodeDone failed: {e}"))?;
        let logits_bytes = match resp {
            WorkerResponse::DecodeDone { logits_bytes, .. } => logits_bytes,
            WorkerResponse::Error { message, .. } => {
                return Err(format!("worker 0 decode error: {message}"));
            }
            _ => return Err(format!("unexpected response from worker 0: {resp:?}")),
        };

        for (_send, recv) in worker_streams.iter_mut().skip(1) {
            let _ = recv_response_quic(recv, rt.handle());
        }

        let decode_logits: Vec<f32> = logits_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let decode_tensor = Tensor::from_slice(&decode_logits);
        next_token = match sample_token(&decode_tensor, temperature, top_p) {
            Ok(t) => t as i64,
            Err(e) => return Err(format!("sample_token failed at step {step}: {e}")),
        };
    }

    if finish_reason.is_none() && !generated_ids.is_empty() {
        finish_reason = Some("length".to_string());
    }

    // Release per-request state on workers to prevent memory leak.
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::ReleaseRequest { request_id };
        let _ = send_command_quic(send, &cmd, rt.handle());
    }

    let text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| format!("decode failed: {e}"))?;

    Ok(InferenceResult {
        text,
        prompt_tokens,
        completion_tokens: generated_ids.len(),
        finish_reason,
    })
}

/// Prefill a single request and return an `ActiveRequest` ready for decode batch.
///
/// On prefill failure, sends an error result via `job.tx` and returns `Err`.
#[allow(clippy::too_many_arguments)]
fn prefill_single_request(
    job: InferenceJob,
    tokenizer: &tokenizers::Tokenizer,
    config: &ModelConfig,
    worker_streams: &mut [(quinn::SendStream, quinn::RecvStream)],
    chunk_sizes_override: &Option<Vec<usize>>,
    capacity_aware: bool,
    worker_capacities: &[u64],
    rt: &tokio::runtime::Runtime,
) -> Result<ActiveRequest, String> {
    let eos_token = config.eos_token_id();
    let vocab_size = config.vocab_size;
    let num_domains = worker_streams.len();

    // Tokenize
    let encoding = tokenizer
        .encode(job.prompt.as_str(), true)
        .map_err(|e| format!("encode failed: {e}"))?;
    let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let prompt_tokens = prompt_ids.len();
    let seq_len = prompt_ids.len() as i64;

    // Chunk allocation (same three-tier logic as process_single_request)
    let chunk_sizes: Vec<usize> = if let Some(ref sizes) = chunk_sizes_override {
        if sizes.len() != num_domains {
            let _ = job.tx.send(InferenceResult {
                text: format!("[error: --chunk-sizes length ({}) must match num_domains ({})]", sizes.len(), num_domains),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err("--chunk-sizes length must match num_domains".to_string());
        }
        let sum: usize = sizes.iter().sum();
        if sum != seq_len as usize {
            let _ = job.tx.send(InferenceResult {
                text: format!("[error: --chunk-sizes sum ({}) must equal prompt length ({})]", sum, seq_len),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err("--chunk-sizes sum must equal prompt length".to_string());
        }
        sizes.clone()
    } else if capacity_aware {
        crate::capacity::allocate_by_capacity(seq_len as usize, worker_capacities)
    } else {
        let chunk_size = (seq_len as usize).div_ceil(num_domains).max(1);
        let mut chunks = Vec::with_capacity(num_domains);
        let mut offset = 0usize;
        for i in 0..num_domains {
            let end = if i == num_domains - 1 {
                seq_len as usize
            } else {
                (offset + chunk_size).min(seq_len as usize)
            };
            chunks.push(end - offset);
            offset = end;
        }
        chunks
    };

    for (i, size) in chunk_sizes.iter().enumerate() {
        if *size == 0 {
            let _ = job.tx.send(InferenceResult {
                text: format!(
                    "[error: prompt too short: domain {} received 0 tokens (total {} tokens, {} domains). Each domain needs at least 1 token.]",
                    i, prompt_ids.len(), num_domains
                ),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err(format!("prompt too short for {} domains", num_domains));
        }
    }

    let mut chunk_boundaries = vec![0usize];
    for size in &chunk_sizes {
        chunk_boundaries.push(chunk_boundaries.last().unwrap() + size);
    }

    // Prefill
    for (domain_id, (send, _recv)) in worker_streams.iter_mut().enumerate() {
        let start = chunk_boundaries[domain_id];
        let end = chunk_boundaries[domain_id + 1];
        let chunk = &prompt_ids[start..end];
        let cmd = WorkerCommand::Prefill {
            request_id: job.request_id,
            chunk: chunk.to_vec(),
            seq_offset: start as i64,
        };
        if let Err(e) = send_command_quic(send, &cmd, rt.handle()) {
            let _ = job.tx.send(InferenceResult {
                text: format!("[error: send Prefill failed: {e}]"),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err(format!("send Prefill failed: {e}"));
        }
    }

    let mut max_global_seq_len = 0usize;
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    for (domain_id, (_send, recv)) in worker_streams.iter_mut().enumerate() {
        let resp = match recv_response_quic(recv, rt.handle()) {
            Ok(r) => r,
            Err(e) => {
                let _ = job.tx.send(InferenceResult {
                    text: format!("[error: recv PrefillDone failed: {e}]"),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    finish_reason: Some("error".to_string()),
                });
                return Err(format!("recv PrefillDone failed: {e}"));
            }
        };
        match resp {
            WorkerResponse::PrefillDone { last_logits_bytes: bytes, global_seq_len, .. } => {
                max_global_seq_len = max_global_seq_len.max(global_seq_len);
                if domain_id == num_domains - 1 {
                    last_logits_bytes = bytes;
                }
            }
            WorkerResponse::Error { message, .. } => {
                let _ = job.tx.send(InferenceResult {
                    text: format!("[error: worker {domain_id} prefill error: {message}]"),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    finish_reason: Some("error".to_string()),
                });
                return Err(format!("worker {domain_id} prefill error: {message}"));
            }
            _ => {
                let _ = job.tx.send(InferenceResult {
                    text: format!("[error: unexpected response from worker {domain_id}: {resp:?}]"),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    finish_reason: Some("error".to_string()),
                });
                return Err(format!("unexpected response from worker {domain_id}: {resp:?}"));
            }
        }
    }

    // Sync global_seq_len
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::SyncGlobalSeqLen { request_id: job.request_id, len: max_global_seq_len };
        let _ = send_command_quic(send, &cmd, rt.handle());
    }

    // Sample first token from last worker's logits
    let logits_vec: Vec<f32> = last_logits_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if logits_vec.len() != vocab_size {
        let _ = job.tx.send(InferenceResult {
            text: format!("[error: logits size mismatch: expected {}, got {}]", vocab_size, logits_vec.len()),
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: Some("error".to_string()),
        });
        return Err(format!("logits size mismatch: expected {}, got {}", vocab_size, logits_vec.len()));
    }
    let logits_tensor = Tensor::from_slice(&logits_vec);
    let first_token = match sample_token(&logits_tensor, job.temperature, job.top_p) {
        Ok(t) => t as i64,
        Err(e) => {
            let _ = job.tx.send(InferenceResult {
                text: format!("[error: sample_token failed: {e}]"),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err(format!("sample_token failed: {e}"));
        }
    };

    let mut generated_ids: Vec<u32> = Vec::new();
    let mut finish_reason = None;

    let token = first_token as u32;
    generated_ids.push(token);
    if Some(token) == eos_token {
        finish_reason = Some("stop".to_string());
    }

    Ok(ActiveRequest {
        request_id: job.request_id,
        prompt: job.prompt,
        max_tokens: job.max_tokens,
        temperature: job.temperature,
        top_p: job.top_p,
        prompt_ids,
        prompt_tokens,
        chunk_boundaries,
        generated_ids,
        next_token: first_token,
        finish_reason,
        result_tx: job.tx,
        stream_tx: job.stream_tx,
    })
}

/// Execute one decode iteration for all active requests in the scheduler.
///
/// Returns the list of request IDs that have completed (EOS or max_tokens).
fn decode_iteration(
    scheduler: &mut BatchScheduler,
    worker_streams: &mut [(quinn::SendStream, quinn::RecvStream)],
    eos_token: Option<u32>,
    vocab_size: usize,
    rt: &tokio::runtime::Runtime,
) -> Result<Vec<u64>, String> {
    let _num_domains = worker_streams.len();

    // Collect next tokens from all active requests
    let request_tokens: Vec<(u64, i64)> = scheduler.active_requests()
        .values()
        .map(|req| (req.request_id, req.next_token))
        .collect();

    if request_tokens.is_empty() {
        return Ok(Vec::new());
    }

    // Send DecodeBatch to all workers
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::DecodeBatch { request_tokens: request_tokens.clone() };
        send_command_quic(send, &cmd, rt.handle())
            .map_err(|e| format!("send DecodeBatch failed: {e}"))?;
    }

    // Receive DecodeBatchDone from worker 0 (it has the logits)
    let resp = recv_response_quic(&mut worker_streams[0].1, rt.handle())
        .map_err(|e| format!("recv DecodeBatchDone failed: {e}"))?;
    let request_logits = match resp {
        WorkerResponse::DecodeBatchDone { request_logits } => request_logits,
        WorkerResponse::Error { message, .. } => {
            return Err(format!("worker 0 decode batch error: {message}"));
        }
        _ => return Err(format!("unexpected response from worker 0: {resp:?}")),
    };

    // Drain responses from other workers (they participate in KV ring but logits come from worker 0)
    for (_send, recv) in worker_streams.iter_mut().skip(1) {
        let _ = recv_response_quic(recv, rt.handle());
    }

    // Sample next tokens and update states
    let mut completed = Vec::new();
    for (request_id, logits_bytes) in request_logits {
        let req = match scheduler.get_active_mut(request_id) {
            Some(r) => r,
            None => continue, // request may have already been removed
        };

        let logits_vec: Vec<f32> = logits_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if logits_vec.len() != vocab_size {
            eprintln!("[coordinator] request {request_id} logits size mismatch: expected {vocab_size}, got {}", logits_vec.len());
            continue;
        }
        let logits_tensor = Tensor::from_slice(&logits_vec);
        let next_token = match sample_token(&logits_tensor, req.temperature, req.top_p) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[coordinator] request {request_id} sample_token failed: {e}");
                continue;
            }
        };

        req.generated_ids.push(next_token);
        req.next_token = next_token as i64;

        if Some(next_token) == eos_token {
            req.finish_reason = Some("stop".to_string());
            completed.push(request_id);
        } else if req.generated_ids.len() >= req.max_tokens {
            req.finish_reason = Some("length".to_string());
            completed.push(request_id);
        }
    }

    Ok(completed)
}

/// Coordinator 主入口。
pub fn run() {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let args = parse_args();
    println!("[coordinator] starting, num_domains={}, workers={:?}, listen={}",
             args.num_domains, args.worker_addrs, args.listen_addr);

    // Load tokenizer and config
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config = ModelConfig::from_file(&config_path).expect("load config failed");
    let tokenizer_path = Path::new(&args.model_dir).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("load tokenizer failed");

    // Determine serving mode
    let has_cli_prompts = args.prompts_file.is_some()
        || args.prompt_file.is_some()
        || !args.prompt.is_empty();

    let cli_prompts: Vec<String> = if let Some(ref path) = args.prompts_file {
        let content = std::fs::read_to_string(path).expect("read prompts-file failed");
        content.lines().map(|s| s.to_string()).filter(|s| !s.is_empty()).collect()
    } else if let Some(ref path) = args.prompt_file {
        vec![std::fs::read_to_string(path).expect("read prompt-file failed")]
    } else {
        vec![args.prompt.clone()]
    };

    // Create QUIC endpoint and wait for workers
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime failed");
    let listen_addr: SocketAddr = args.listen_addr.parse().expect("invalid listen_addr");

    let endpoint = rt.block_on(async {
        crate::distributed::transport::quic::create_endpoint(listen_addr)
    }).expect("create_endpoint failed");
    println!("[coordinator] QUIC endpoint listening on {}", args.listen_addr);

    let mut worker_handshakes: Vec<(usize, u64, quinn::SendStream, quinn::RecvStream)> =
        Vec::with_capacity(args.num_domains);
    for i in 0..args.num_domains {
        let (send, mut recv) = rt.block_on(async {
            let incoming = match tokio::time::timeout(
                std::time::Duration::from_secs(600),
                endpoint.accept()
            ).await {
                Ok(Some(incoming)) => incoming,
                Ok(None) => return Err("endpoint closed".to_string()),
                Err(_) => return Err("accept timeout after 600s".to_string()),
            };
            let conn = incoming.await.map_err(|e| format!("connection failed: {e}"))?;
            println!("[coordinator] worker connection established (accept order {i})");
            let (send, recv) = conn.accept_bi().await
                .map_err(|e| format!("accept_bi failed: {e}"))?;
            Ok::<_, String>((send, recv))
        }).unwrap_or_else(|e| panic!("accept worker {i} failed: {e}"));

        let handshake = crate::distributed::protocol::read_handshake_quic(&mut recv, rt.handle())
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
    let worker_streams: Vec<(quinn::SendStream, quinn::RecvStream)> = worker_handshakes
        .into_iter()
        .map(|(_, _, send, recv)| (send, recv))
        .collect();

    // Wrap worker_streams in Arc<Mutex> for shared access between concurrent requests.
    let worker_streams = Arc::new(std::sync::Mutex::new(worker_streams));

    if has_cli_prompts && !cli_prompts.is_empty() {
        // Batch mode: process CLI prompts then exit (serial, no concurrency needed)
        println!("[coordinator] loaded {} prompt(s)", cli_prompts.len());
        for (req_idx, prompt_text) in cli_prompts.iter().enumerate() {
            let request_id = (req_idx + 1) as u64;
            println!("\n[coordinator] === Request {} / {} ===", request_id, cli_prompts.len());
            let mut guard = worker_streams.lock().unwrap_or_else(|e| e.into_inner());
            match process_single_request(
                request_id,
                prompt_text,
                args.max_tokens,
                args.temperature,
                args.top_p,
                &tokenizer,
                &config,
                &mut guard,
                &args.chunk_sizes,
                args.capacity_aware,
                &worker_capacities,
                &rt,
            ) {
                Ok(result) => {
                    println!("[coordinator] generated: {}", result.text);
                }
                Err(e) => {
                    eprintln!("[coordinator] request {request_id} failed: {e}");
                }
            }
        }
        println!("\n[coordinator] all requests done, shutting down workers");
        let mut worker_streams = match Arc::try_unwrap(worker_streams) {
            Ok(mutex) => mutex.into_inner().unwrap_or_else(|e| e.into_inner()),
            Err(_) => {
                eprintln!("[coordinator] warning: worker_streams still shared, cannot shutdown cleanly");
                return;
            }
        };
        shutdown_workers(&mut worker_streams, &endpoint, &rt);
        return;
    }

    // HTTP API mode
    let (job_tx, mut job_rx) = tokio::sync::mpsc::unbounded_channel::<InferenceJob>();

    let queued_counter = Arc::new(AtomicU64::new(0));
    let active_counter = Arc::new(AtomicU64::new(0));

    let api_state = ApiState {
        job_tx,
        request_counter: Arc::new(AtomicU64::new(0)),
        completed_counter: Arc::new(AtomicU64::new(0)),
        failed_counter: Arc::new(AtomicU64::new(0)),
        workers_connected: Arc::new(AtomicU64::new(args.num_domains as u64)),
        num_domains: args.num_domains,
        model_name: "qwen2-0.5b".to_string(), // TODO: derive from config
        queued_counter: queued_counter.clone(),
        active_counter: active_counter.clone(),
    };

    let http_addr = args.http_addr.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("http tokio runtime failed");
        rt.block_on(async {
            let app = build_router(api_state);
            let listener = match tokio::net::TcpListener::bind(&http_addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[coordinator] failed to bind HTTP server on {http_addr}: {e}");
                    return;
                }
            };
            println!("[coordinator] HTTP API listening on {http_addr}");
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("[coordinator] HTTP server error: {e}");
            }
        });
    });

    let max_batch_size = 4usize;
    let mut scheduler = BatchScheduler::new(max_batch_size);
    println!("[coordinator] entering HTTP iterative scheduling mode (max_batch_size={max_batch_size}). Press Ctrl+C to exit.");

    // Iterative scheduling loop: each iteration may prefill new requests and/or
    // decode all active requests.  This replaces the request-level spawn_blocking
    // model with an iteration-level scheduler.
    let eos_token = config.eos_token_id();
    let vocab_size = config.vocab_size as usize;

    loop {
        // Phase 1: Receive new jobs (non-blocking)
        let mut channel_closed = false;
        loop {
            match job_rx.try_recv() {
                Ok(job) => {
                    queued_counter.fetch_sub(1, Ordering::SeqCst);
                    scheduler.enqueue(job);
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    channel_closed = true;
                    break;
                }
            }
        }

        // Phase 2: Execute one scheduling iteration
        {
            let mut guard = worker_streams.lock().unwrap_or_else(|e| e.into_inner());

            // 2a: Prefill a pending request if batch has room
            if scheduler.can_admit() && !scheduler.pending_is_empty() {
                if let Some(job) = scheduler.try_dequeue_pending() {
                    match prefill_single_request(job, &tokenizer, &config, &mut guard, &args.chunk_sizes, args.capacity_aware, &worker_capacities, &rt) {
                        Ok(active_req) => {
                            active_counter.fetch_add(1, Ordering::SeqCst);
                            scheduler.add_active(active_req);
                        }
                        Err(e) => {
                            eprintln!("[coordinator] prefill failed: {e}");
                            // Error result already sent via job.tx in prefill_single_request
                        }
                    }
                }
            }

            // 2b: Decode all active requests
            if !scheduler.active_is_empty() {
                match decode_iteration(&mut scheduler, &mut guard, eos_token, vocab_size, &rt) {
                    Ok(completed) => {
                        // Emit streaming chunks for all active requests.
                        for req in scheduler.active_requests_mut().values_mut() {
                            if let Some(ref chunk_tx) = req.stream_tx {
                                if let Some(&token_id) = req.generated_ids.last() {
                                    let delta = tokenizer.decode(&[token_id], false)
                                        .unwrap_or_else(|e| {
                                            eprintln!("[coordinator] token decode failed for request {}: {e}", req.request_id);
                                            String::new()
                                        });
                                    let _ = chunk_tx.send(StreamChunk {
                                        delta,
                                        token_id,
                                        finish_reason: None,
                                    });
                                }
                            }
                        }

                        // Release per-request state on workers for completed requests.
                        for request_id in &completed {
                            for (send, _recv) in guard.iter_mut() {
                                let cmd = WorkerCommand::ReleaseRequest { request_id: *request_id };
                                let _ = send_command_quic(send, &cmd, rt.handle());
                            }
                        }
                        for request_id in completed {
                            if let Some(req) = scheduler.remove_active(request_id) {
                                active_counter.fetch_sub(1, Ordering::SeqCst);

                                if let Some(ref chunk_tx) = req.stream_tx {
                                    // Streaming: send final chunk with finish_reason.
                                    let _ = chunk_tx.send(StreamChunk {
                                        delta: "".to_string(),
                                        token_id: 0,
                                        finish_reason: req.finish_reason.clone(),
                                    });
                                } else {
                                    // Non-streaming: send full result via oneshot.
                                    let text = tokenizer.decode(&req.generated_ids, true)
                                        .unwrap_or_else(|e| {
                                            eprintln!("[coordinator] decode failed for request {request_id}: {e}");
                                            String::new()
                                        });
                                    let result = InferenceResult {
                                        text,
                                        prompt_tokens: req.prompt_tokens,
                                        completion_tokens: req.generated_ids.len(),
                                        finish_reason: req.finish_reason,
                                    };
                                    let _ = req.result_tx.send(result);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[coordinator] decode iteration failed: {e}");
                        // All active requests fail
                        for request_id in scheduler.active_request_ids() {
                            if let Some(req) = scheduler.remove_active(request_id) {
                                active_counter.fetch_sub(1, Ordering::SeqCst);
                                let _ = req.result_tx.send(InferenceResult {
                                    text: format!("[error: decode batch failed: {e}]"),
                                    prompt_tokens: req.prompt_tokens,
                                    completion_tokens: req.generated_ids.len(),
                                    finish_reason: Some("error".to_string()),
                                });
                            }
                        }
                    }
                }
            }
        } // drop guard

        // Phase 3: If channel closed and no work remains, exit
        if channel_closed && !scheduler.has_work() {
            println!("[coordinator] job channel closed and all requests done, exiting");
            break;
        }

        // Phase 4: If no active or pending work, block until new job arrives
        if !scheduler.has_work() {
            match rt.block_on(job_rx.recv()) {
                Some(job) => {
                    queued_counter.fetch_sub(1, Ordering::SeqCst);
                    scheduler.enqueue(job);
                }
                None => {
                    println!("[coordinator] job channel closed, exiting");
                    break;
                }
            }
        }
    }

    println!("[coordinator] scheduler exited, shutting down workers");

    let mut worker_streams = match Arc::try_unwrap(worker_streams) {
        Ok(mutex) => mutex.into_inner().unwrap_or_else(|e| e.into_inner()),
        Err(_) => {
            eprintln!("[coordinator] warning: worker_streams still shared, using best-effort shutdown");
            return;
        }
    };
    println!("\n[coordinator] shutting down workers");
    shutdown_workers(&mut worker_streams, &endpoint, &rt);
}

/// RAII guard that decrements the active request counter on drop.
struct ActiveRequestGuard(Arc<AtomicU64>);

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Gracefully shutdown all workers with timeout protection.
///
/// 1. Try to send Shutdown command to each worker with a short timeout.
/// 2. Finish send streams so workers see EOF.
/// 3. Close the QUIC endpoint explicitly.
/// 4. Sleep briefly to let connections clean up before runtime drop.
fn shutdown_workers(
    worker_streams: &mut [(quinn::SendStream, quinn::RecvStream)],
    endpoint: &quinn::Endpoint,
    rt: &tokio::runtime::Runtime,
) {
    for (send, _recv) in worker_streams.iter_mut() {
        let _ = crate::distributed::protocol::send_command_quic_timeout(
            send, &WorkerCommand::Shutdown, rt.handle(), 10,
        );
        let _ = send.finish();
    }
    endpoint.close(0u32.into(), b"coordinator shutdown");
    rt.block_on(async {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    });
    println!("[coordinator] shutdown complete");
}
