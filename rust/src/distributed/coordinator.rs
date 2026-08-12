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
use crate::capacity::{admit_reserved_kv_bytes, capacity_mb_to_bytes};
use crate::distributed::protocol::{
    recv_response_quic, send_command_quic, validate_stationary_continuation, WorkerCommand,
    WorkerResponse,
};
use crate::distributed::scheduler::{ActiveRequest, BatchScheduler};
use crate::model::attention::strategy::{
    build_assignment, build_domain_positions, RingSchedulingStrategy,
};
#[cfg(feature = "tch-backend")]
use crate::model::sampling::sample_token;
#[cfg(not(feature = "tch-backend"))]
use crate::model::sampling::sample_token_slice;
use crate::model::self_driving::{stationary_layer_starters, FrozenKvAssigneeSchedule};
use crate::model::ModelConfig;
use std::collections::HashMap;
use std::io::Write;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cfg(feature = "tch-backend")]
fn sample_from_logits_vec(logits: &[f32], temperature: f64, top_p: f64) -> Result<i64, String> {
    let tensor = tch::Tensor::from_slice(logits);
    sample_token(&tensor, temperature, top_p)
        .map(|t| t as i64)
        .map_err(|e| format!("{e}"))
}

#[cfg(not(feature = "tch-backend"))]
fn sample_from_logits_vec(logits: &[f32], temperature: f64, top_p: f64) -> Result<i64, String> {
    sample_token_slice(logits, temperature, top_p)
        .map(|t| t as i64)
        .map_err(|e| format!("{e}"))
}

/// Apply a ring scheduling strategy to a prompt and return per-domain inputs.
///
/// Returns a vector of `(chunk, position_ids, seq_offset)` for each domain,
/// where `chunk` holds the token ids in local storage order, `position_ids`
/// holds the corresponding global positions, and `seq_offset` is the first
/// global position (used for KV metadata and guards).
fn apply_ring_strategy(
    prompt_ids: &[i64],
    chunk_sizes: &[usize],
    strategy: RingSchedulingStrategy,
) -> Vec<(Vec<i64>, Vec<i64>, i64)> {
    let assignment = build_assignment(chunk_sizes, strategy);
    let positions = build_domain_positions(&assignment);
    positions
        .iter()
        .map(|pos| {
            let chunk: Vec<i64> = pos.iter().map(|&p| prompt_ids[p]).collect();
            let pos_ids: Vec<i64> = pos.iter().map(|&p| p as i64).collect();
            let seq_offset = pos.first().copied().unwrap_or(0) as i64;
            (chunk, pos_ids, seq_offset)
        })
        .collect()
}

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
    /// Export raw logits to directory for correctness validation.
    export_logits_dir: Option<String>,
    /// Ring attention scheduling strategy (vanilla/striped/zigzag).
    ring_strategy: RingSchedulingStrategy,
    /// experimental (route-B 2c, test-only): inject one stationary
    /// continuation segment after the first legacy decode step.
    continuation_segment: Option<Vec<i64>>,
    /// experimental (route-B 2c, test-only): raw prompt token ids, bypassing
    /// the tokenizer, so the E2E can reproduce the golden scenario exactly.
    prompt_token_ids: Option<Vec<i64>>,
    /// experimental (route-B observability, test-only): align the frozen
    /// schedule phase with a local golden request.
    continuation_request_id: u64,
    /// experimental (route-B observability, test-only): reproduce a remote
    /// handshake's capacity-weighted frozen schedule in a local golden.
    continuation_capacity_tickets: Option<Vec<u64>>,
    /// Optional JSONL path for per-request structured traces (6c.0). Absent
    /// disables tracing entirely; tracing never changes the inference result.
    trace_jsonl: Option<String>,
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
    let mut export_logits_dir = None;
    let mut ring_strategy = RingSchedulingStrategy::Vanilla;
    let mut continuation_segment = None;
    let mut prompt_token_ids = None;
    let mut continuation_request_id = 1_u64;
    let mut continuation_capacity_tickets = None;
    let mut trace_jsonl = None;

    let mut args = std::env::args().skip(1); // skip binary name
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--distributed-role" => {
                let _ = args.next();
            } // consumed by main.rs, skip here
            "--model-dir" => model_dir = args.next().unwrap(),
            "--prompt" => prompt = args.next().unwrap(),
            "--max-tokens" => max_tokens = args.next().unwrap().parse().unwrap(),
            "--temperature" => temperature = args.next().unwrap().parse().unwrap(),
            "--top-p" => top_p = args.next().unwrap().parse().unwrap(),
            "--num-domains" => num_domains = args.next().unwrap().parse().unwrap(),
            "--worker-addrs" => {
                worker_addrs = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.to_string())
                    .collect();
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
            "--export-logits" => export_logits_dir = Some(args.next().unwrap()),
            "--ring-strategy" => {
                let s = args.next().unwrap();
                ring_strategy = RingSchedulingStrategy::from_str(&s)
                    .unwrap_or_else(|| panic!("unknown --ring-strategy: {s}"));
            }
            "--continuation-segment" => {
                let s = args.next().unwrap();
                continuation_segment = Some(s.split(',').map(|x| x.parse().unwrap()).collect());
            }
            "--prompt-token-ids" => {
                let s = args.next().unwrap();
                prompt_token_ids = Some(s.split(',').map(|x| x.parse().unwrap()).collect());
            }
            "--continuation-request-id" => {
                continuation_request_id = args.next().unwrap().parse().unwrap();
            }
            "--continuation-capacity-tickets" => {
                let s = args.next().unwrap();
                continuation_capacity_tickets =
                    Some(s.split(',').map(|x| x.parse().unwrap()).collect());
            }
            "--trace-jsonl" => trace_jsonl = Some(args.next().unwrap()),
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
        export_logits_dir,
        ring_strategy,
        continuation_segment,
        prompt_token_ids,
        continuation_request_id,
        continuation_capacity_tickets,
        trace_jsonl,
    }
}

/// Write collected logits chunks to a binary file for correctness comparison.
///
/// Format matches single-node export:
///   - Header: [vocab_size: u64 LE][num_chunks: u64 LE]
///   - Body:  contiguous vocab_size f32 LE per chunk
fn write_logits_file(
    path: &std::path::Path,
    vocab_size: usize,
    chunks: &[Vec<f32>],
) -> Result<(), String> {
    let mut file =
        std::fs::File::create(path).map_err(|e| format!("failed to create logits file: {e}"))?;
    let num_chunks = chunks.len() as u64;
    file.write_all(&vocab_size.to_le_bytes())
        .map_err(|e| format!("failed to write header: {e}"))?;
    file.write_all(&num_chunks.to_le_bytes())
        .map_err(|e| format!("failed to write header: {e}"))?;
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.len() != vocab_size {
            return Err(format!(
                "logits chunk {} size mismatch: expected {}, got {}",
                i,
                vocab_size,
                chunk.len()
            ));
        }
        for &f in chunk {
            file.write_all(&f.to_le_bytes())
                .map_err(|e| format!("failed to write logits: {e}"))?;
        }
    }
    Ok(())
}

/// Write a single logits vector as headerless little-endian f32 values.
/// This matches the route-B dump format consumed by compare_route_b_dumps.py.
fn write_raw_logits_file(path: &Path, logits: &[f32]) -> Result<(), String> {
    let mut file =
        std::fs::File::create(path).map_err(|e| format!("failed to create logits file: {e}"))?;
    for value in logits {
        file.write_all(&value.to_le_bytes())
            .map_err(|e| format!("failed to write logits: {e}"))?;
    }
    Ok(())
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
    export_logits_dir: Option<&str>,
    strategy: RingSchedulingStrategy,
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
                sizes.len(),
                num_domains
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
                i,
                prompt_ids.len(),
                num_domains
            ));
        }
    }

    // Apply ring scheduling strategy (vanilla/striped/zigzag).
    let domain_inputs = apply_ring_strategy(&prompt_ids, &chunk_sizes, strategy);

    // Prefill
    for (domain_id, (send, _recv)) in worker_streams.iter_mut().enumerate() {
        let (chunk, position_ids, seq_offset) = &domain_inputs[domain_id];
        let cmd = WorkerCommand::Prefill {
            request_id,
            chunk: chunk.clone(),
            seq_offset: *seq_offset,
            position_ids: Some(position_ids.clone()),
            layer_kv_capacities: None,
        };
        send_command_quic(send, &cmd, rt.handle())
            .map_err(|e| format!("send Prefill failed: {e}"))?;
    }

    let mut max_global_seq_len = 0usize;
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    for (domain_id, (_send, recv)) in worker_streams.iter_mut().enumerate() {
        let resp = recv_response_quic(recv, rt.handle())
            .map_err(|e| format!("recv PrefillDone failed: {e}"))?;
        match resp {
            WorkerResponse::PrefillDone {
                last_logits_bytes: bytes,
                global_seq_len,
                ..
            } => {
                if global_seq_len > max_global_seq_len {
                    max_global_seq_len = global_seq_len;
                    last_logits_bytes = bytes;
                }
            }
            WorkerResponse::Error { message, .. } => {
                return Err(format!("worker {domain_id} prefill error: {message}"));
            }
            _ => {
                return Err(format!(
                    "unexpected response from worker {domain_id}: {resp:?}"
                ))
            }
        }
    }

    // Sync global_seq_len
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::SyncGlobalSeqLen {
            request_id,
            len: max_global_seq_len,
        };
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
            vocab_size,
            logits_vec.len()
        ));
    }
    let mut next_token = match sample_from_logits_vec(&logits_vec, temperature, top_p) {
        Ok(t) => t,
        Err(e) => return Err(format!("sample_token failed: {e}")),
    };

    let mut generated_ids: Vec<u32> = Vec::new();
    let mut all_logits: Vec<Vec<f32>> = Vec::new();

    // Decode loop: match single-node structure where logits are pushed at the
    // START of each iteration (logits that produce THIS token), and decode
    // is only sent when another token will be generated.
    let mut logits_vec = logits_vec;
    let mut finish_reason = None;
    for step in 0..max_tokens {
        all_logits.push(logits_vec);

        let token = next_token as u32;
        generated_ids.push(token);

        if Some(token) == eos_token {
            finish_reason = Some("stop".to_string());
            break;
        }

        for (send, _recv) in worker_streams.iter_mut() {
            let cmd = WorkerCommand::Decode {
                request_id,
                token: next_token,
            };
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

        logits_vec = logits_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        next_token = match sample_from_logits_vec(&logits_vec, temperature, top_p) {
            Ok(t) => t,
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

    // Export logits if requested
    if let Some(dir) = export_logits_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("failed to create export dir: {e}"))?;
        let out_path = Path::new(dir).join(format!("logits_{}.bin", request_id));
        write_logits_file(&out_path, vocab_size, &all_logits)
            .map_err(|e| format!("logits export failed: {e}"))?;
        println!("[coordinator] exported logits to {:?}", out_path);
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

/// Per-request structured trace for the 6c.0 observability plane.
///
/// Records lifecycle timestamps (elapsed ms since coordinator start) and byte
/// accounting per request so a client result can be correlated by `request_id`.
/// Hop counts are derived from the known N/L formulas — prefill runs
/// `layers * (domains - 1)` hops, each decode step the same — so they stay
/// consistent with the ring invariants without per-hop instrumentation.
#[derive(Clone, Debug, serde::Serialize)]
struct RequestTrace {
    request_id: u64,
    enqueued_elapsed_ms: u64,
    prefill_accepted_elapsed_ms: u64,
    first_token_elapsed_ms: u64,
    completed_elapsed_ms: u64,
    reserved_bytes: Vec<u64>,
    released_bytes: Vec<u64>,
    decode_steps: usize,
    prompt_tokens: usize,
    max_tokens: usize,
    finish_reason: Option<String>,
    error: Option<String>,
    prefill_hops: usize,
    decode_hops: usize,
}

/// JSONL sink for per-request traces (6c.0). Disabled when no path is given;
/// enabled it only appends one JSON object per finished request and never
/// changes the inference result.
struct TraceSink {
    writer: Option<std::io::BufWriter<std::fs::File>>,
    start: std::time::Instant,
    in_flight: HashMap<u64, RequestTrace>,
    layers: usize,
    domains: usize,
}

impl TraceSink {
    fn new(path: Option<String>, layers: usize, domains: usize) -> Self {
        let writer = match path {
            Some(path) => match std::fs::File::create(&path) {
                Ok(file) => {
                    println!("[coordinator] tracing per-request traces to {path}");
                    Some(std::io::BufWriter::new(file))
                }
                Err(e) => {
                    eprintln!("[coordinator] cannot open trace file {path}: {e}; tracing disabled");
                    None
                }
            },
            None => None,
        };
        Self {
            writer,
            start: std::time::Instant::now(),
            in_flight: HashMap::new(),
            layers,
            domains,
        }
    }

    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    fn enqueue(&mut self, request_id: u64) {
        let now = self.elapsed_ms();
        self.in_flight
            .entry(request_id)
            .or_insert_with(|| RequestTrace {
                request_id,
                enqueued_elapsed_ms: now,
                prefill_accepted_elapsed_ms: 0,
                first_token_elapsed_ms: 0,
                completed_elapsed_ms: 0,
                reserved_bytes: Vec::new(),
                released_bytes: Vec::new(),
                decode_steps: 0,
                prompt_tokens: 0,
                max_tokens: 0,
                finish_reason: None,
                error: None,
                prefill_hops: 0,
                decode_hops: 0,
            })
            .enqueued_elapsed_ms = now;
    }

    fn prefill_accepted(
        &mut self,
        request_id: u64,
        reserved: Vec<u64>,
        prompt_tokens: usize,
        max_tokens: usize,
    ) {
        let now = self.elapsed_ms();
        if let Some(trace) = self.in_flight.get_mut(&request_id) {
            trace.prefill_accepted_elapsed_ms = now;
            trace.reserved_bytes = reserved;
            trace.prompt_tokens = prompt_tokens;
            trace.max_tokens = max_tokens;
            trace.prefill_hops = self.layers * self.domains.saturating_sub(1);
        }
    }

    fn decode_step(&mut self, request_ids: &[u64]) {
        if self.writer.is_none() {
            return;
        }
        let now = self.elapsed_ms();
        for &request_id in request_ids {
            if let Some(trace) = self.in_flight.get_mut(&request_id) {
                trace.decode_steps += 1;
                if trace.first_token_elapsed_ms == 0 {
                    trace.first_token_elapsed_ms = now;
                }
                trace.decode_hops =
                    trace.decode_steps * self.layers * self.domains.saturating_sub(1);
            }
        }
    }

    fn complete(
        &mut self,
        request_id: u64,
        finish_reason: Option<String>,
        released_bytes: Vec<u64>,
    ) {
        if let Some(mut trace) = self.in_flight.remove(&request_id) {
            trace.completed_elapsed_ms = self.elapsed_ms();
            trace.finish_reason = finish_reason;
            trace.released_bytes = released_bytes;
            trace.decode_hops = trace.decode_steps * self.layers * self.domains.saturating_sub(1);
            self.emit(&trace);
        }
    }

    fn fail(&mut self, request_id: u64, error: String) {
        if let Some(mut trace) = self.in_flight.remove(&request_id) {
            trace.completed_elapsed_ms = self.elapsed_ms();
            trace.error = Some(error);
            self.emit(&trace);
        }
    }

    fn emit(&mut self, trace: &RequestTrace) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        use std::io::Write;
        if let Ok(line) = serde_json::to_string(trace) {
            let _ = writeln!(writer, "{line}");
            let _ = writer.flush();
        }
    }
}

/// Per-request KV byte occupancy ledger for the HTTP service loop.
///
/// Tracks how many bytes each worker's KV budget is currently reserved by
/// active requests. A request is admitted only if, on every domain, the active
/// sum plus its own reservation stays within the worker budget. This is
/// correctness accounting only — no paging, preemption, priority, eviction,
/// or repair planning.
struct ActiveKvReservation {
    budgets: Vec<u64>,
    per_domain_used: Vec<u64>,
    by_request: HashMap<u64, Vec<u64>>,
}

impl ActiveKvReservation {
    fn new(budgets: Vec<u64>) -> Self {
        let domains = budgets.len();
        Self {
            budgets,
            per_domain_used: vec![0; domains],
            by_request: HashMap::new(),
        }
    }

    /// Atomically reserve `required` bytes across all domains for `request_id`.
    /// Succeeds only if every domain stays within budget; on failure nothing is
    /// recorded. A live request cannot reserve twice.
    fn try_reserve(&mut self, request_id: u64, required: &[u64]) -> Result<(), String> {
        if self.by_request.contains_key(&request_id) {
            return Err(format!("request {request_id} already has a KV reservation"));
        }
        if required.len() != self.budgets.len() {
            return Err(format!(
                "request {request_id} reservation has {} domains, ledger has {}",
                required.len(),
                self.budgets.len()
            ));
        }
        let mut used = self.per_domain_used.clone();
        for (domain, &bytes) in required.iter().enumerate() {
            used[domain] = used[domain].checked_add(bytes).ok_or_else(|| {
                format!("request {request_id} KV reservation overflow on domain {domain}")
            })?;
            if used[domain] > self.budgets[domain] {
                return Err(format!(
                    "request {request_id} KV reservation exceeds budget on domain {domain}: active+new={} budget={}",
                    used[domain], self.budgets[domain]
                ));
            }
        }
        self.per_domain_used = used;
        self.by_request.insert(request_id, required.to_vec());
        Ok(())
    }

    /// Release one request's reservation exactly once. Repeated release of the
    /// same id is a no-op; unknown ids are ignored, so accounting never goes
    /// negative.
    fn release(&mut self, request_id: u64) {
        let Some(required) = self.by_request.remove(&request_id) else {
            return;
        };
        for (domain, &bytes) in required.iter().enumerate() {
            self.per_domain_used[domain] = self.per_domain_used[domain].saturating_sub(bytes);
        }
    }

    fn reserved_bytes(&self, request_id: u64) -> Option<&[u64]> {
        self.by_request.get(&request_id).map(Vec::as_slice)
    }

    fn used_bytes(&self) -> &[u64] {
        &self.per_domain_used
    }
}

/// RAII guard: releases the reservation if prefill fails after reserving, and
/// keeps it once the request is admitted. Guarantees exactly-once release on
/// every error path without scattering manual cleanup.
struct ReservationGuard<'a> {
    ledger: &'a mut ActiveKvReservation,
    request_id: u64,
    committed: bool,
}

impl Drop for ReservationGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.ledger.release(self.request_id);
        }
    }
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
    kv_ledger: &mut ActiveKvReservation,
    rt: &tokio::runtime::Runtime,
    strategy: RingSchedulingStrategy,
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
                text: format!(
                    "[error: --chunk-sizes length ({}) must match num_domains ({})]",
                    sizes.len(),
                    num_domains
                ),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err("--chunk-sizes length must match num_domains".to_string());
        }
        let sum: usize = sizes.iter().sum();
        if sum != seq_len as usize {
            let _ = job.tx.send(InferenceResult {
                text: format!(
                    "[error: --chunk-sizes sum ({}) must equal prompt length ({})]",
                    sum, seq_len
                ),
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

    // Apply ring scheduling strategy (vanilla/striped/zigzag).
    let domain_inputs = apply_ring_strategy(&prompt_ids, &chunk_sizes, strategy);

    // Byte-level KV admission: freeze this request's per-domain per-layer
    // reservation and prove it fits before any worker sees a Prefill command.
    // Unknown capacity, unit overflow, and one-byte-short all fail closed here.
    let reservation = service_layer_capacities(
        &chunk_sizes,
        seq_len as usize,
        job.max_tokens,
        config.num_layers,
    );
    let budget_bytes = match capacity_mb_to_bytes(worker_capacities) {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = job.tx.send(InferenceResult {
                text: format!("[error: service KV byte admission failed: {error}]"),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err(format!("service KV byte admission failed: {error}"));
        }
    };
    let admission = match admit_reserved_kv_bytes(
        &reservation,
        &budget_bytes,
        config.num_kv_heads(),
        config.head_dim(),
        config.kv_element_size_bytes(),
    ) {
        Ok(admission) => admission,
        Err(error) => {
            let _ = job.tx.send(InferenceResult {
                text: format!("[error: service KV byte admission failed: {error}]"),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: Some("error".to_string()),
            });
            return Err(format!("service KV byte admission failed: {error}"));
        }
    };
    println!(
        "[coordinator] service request {} KV byte admission: required={:?} budget={budget_bytes:?} bytes_per_token_per_layer={} status=accepted",
        job.request_id, admission.required_bytes_per_domain, admission.bytes_per_token_per_layer
    );

    // Active-request accounting: atomically check active sum + this request
    // against every worker budget before dispatch. A guard releases the
    // reservation on any later prefill error; the caller commits it on success.
    if let Err(error) = kv_ledger.try_reserve(job.request_id, &admission.required_bytes_per_domain)
    {
        let _ = job.tx.send(InferenceResult {
            text: format!("[error: {error}]"),
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: Some("error".to_string()),
        });
        return Err(error);
    }
    let mut reservation_guard = ReservationGuard {
        ledger: kv_ledger,
        request_id: job.request_id,
        committed: false,
    };

    // Legacy chunk boundaries (only meaningful for vanilla; kept for ActiveRequest compatibility).
    let mut chunk_boundaries = vec![0usize];
    for size in &chunk_sizes {
        chunk_boundaries.push(chunk_boundaries.last().unwrap() + size);
    }

    // Prefill with the frozen per-domain reservations from admission.
    for (domain_id, (send, _recv)) in worker_streams.iter_mut().enumerate() {
        let (chunk, position_ids, seq_offset) = &domain_inputs[domain_id];
        let cmd = WorkerCommand::Prefill {
            request_id: job.request_id,
            chunk: chunk.clone(),
            seq_offset: *seq_offset,
            position_ids: Some(position_ids.clone()),
            layer_kv_capacities: Some(reservation[domain_id].clone()),
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
            WorkerResponse::PrefillDone {
                last_logits_bytes: bytes,
                global_seq_len,
                ..
            } => {
                if global_seq_len > max_global_seq_len {
                    max_global_seq_len = global_seq_len;
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
                return Err(format!(
                    "unexpected response from worker {domain_id}: {resp:?}"
                ));
            }
        }
    }

    // Sync global_seq_len
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::SyncGlobalSeqLen {
            request_id: job.request_id,
            len: max_global_seq_len,
        };
        let _ = send_command_quic(send, &cmd, rt.handle());
    }

    // Sample first token from last worker's logits
    let logits_vec: Vec<f32> = last_logits_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if logits_vec.len() != vocab_size {
        let _ = job.tx.send(InferenceResult {
            text: format!(
                "[error: logits size mismatch: expected {}, got {}]",
                vocab_size,
                logits_vec.len()
            ),
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: Some("error".to_string()),
        });
        return Err(format!(
            "logits size mismatch: expected {}, got {}",
            vocab_size,
            logits_vec.len()
        ));
    }
    let first_token = match sample_from_logits_vec(&logits_vec, job.temperature, job.top_p) {
        Ok(t) => t,
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

    // Prefill succeeded and the request is entering the active set: keep the
    // reservation. Drop of the guard then releases nothing.
    reservation_guard.committed = true;

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

/// Deterministic decode batch for all active requests.
///
/// Orders by `request_id` so the vector is stable across iterations and
/// platforms. The coordinator builds this exactly once per iteration and
/// broadcasts the same vector verbatim to every worker — that is the FIFO
/// contract the multi-request Q-ring relies on, because `RingPacket` carries
/// no request identifier and workers must decode in the same per-layer order.
fn batch_request_tokens(scheduler: &BatchScheduler) -> Vec<(u64, i64)> {
    let mut request_tokens: Vec<(u64, i64)> = scheduler
        .active_requests()
        .values()
        .map(|req| (req.request_id, req.next_token))
        .collect();
    request_tokens.sort_unstable_by_key(|&(request_id, _)| request_id);
    request_tokens
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

    // Collect next tokens from all active requests, exactly once. The same
    // vector is broadcast verbatim to every worker (FIFO decode contract).
    let request_tokens = batch_request_tokens(scheduler);

    if request_tokens.is_empty() {
        return Ok(Vec::new());
    }

    // Send DecodeBatch to all workers
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::DecodeBatch {
            request_tokens: request_tokens.clone(),
        };
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
        let next_token = match sample_from_logits_vec(&logits_vec, req.temperature, req.top_p) {
            Ok(t) => t as u32,
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

// ===== experimental route-B stationary continuation E2E (phase-2 node 2c) =====

/// Production legacy decode-ring KV growth rule (attention/ring.rs
/// `update_sharded` keep rule): position `p` is persisted only by domain
/// `p % domains`; every other domain drops it. Reserved-capacity planning and
/// tests share this single derivation.
fn legacy_decode_owner(position: i64, domains: usize) -> usize {
    (position as usize) % domains
}

/// Per-domain per-layer reserved capacities for the continuation E2E:
/// prefix split + this domain's owned legacy decode positions + this domain's
/// frozen continuation offsets (identical for every layer).
fn reserved_layer_capacities(
    prefix_splits: &[usize],
    decode_positions: &[i64],
    continuation_offsets: &[Vec<usize>],
    layers: usize,
    domains: usize,
) -> Vec<Vec<usize>> {
    (0..domains)
        .map(|domain| {
            let owned = decode_positions
                .iter()
                .filter(|&&p| legacy_decode_owner(p, domains) == domain)
                .count();
            let capacity = prefix_splits[domain] + owned + continuation_offsets[domain].len();
            vec![capacity; layers]
        })
        .collect()
}

/// Per-domain per-layer reserved KV capacities for a plain service prefill.
///
/// The request reserves its full decode horizon up front: the `prompt_len`
/// prefix split by `chunk_sizes` plus every decode position in
/// `[prompt_len, prompt_len + max_tokens)`, each owned by `position % domains`
/// (the same keep rule the ring decode applies at attention/ring.rs). Route-B
/// continuation offsets are deliberately absent — that path reserves separately.
fn service_layer_capacities(
    chunk_sizes: &[usize],
    prompt_len: usize,
    max_tokens: usize,
    layers: usize,
) -> Vec<Vec<usize>> {
    let domains = chunk_sizes.len();
    let decode_positions: Vec<i64> =
        (prompt_len as i64..(prompt_len + max_tokens) as i64).collect();
    let no_continuation: Vec<Vec<usize>> = vec![Vec::new(); domains];
    reserved_layer_capacities(
        chunk_sizes,
        &decode_positions,
        &no_continuation,
        layers,
        domains,
    )
}

/// Final finisher of a stationary continuation ring run: the finisher of the
/// last layer (starter rotated `layers` times by `domains - 1`).
fn continuation_final_finisher(
    starter: usize,
    layers: usize,
    domains: usize,
) -> Result<usize, String> {
    let starters = stationary_layer_starters(starter, layers, domains)?;
    let last_starter = *starters.last().expect("non-empty starters");
    Ok((last_starter + domains - 1) % domains)
}

/// Broadcast one StationaryContinuation command and collect the finisher's
/// logits; every non-finisher worker must acknowledge with `None`.
#[allow(clippy::too_many_arguments)]
fn stationary_continuation(
    request_id: u64,
    tokens: &[i64],
    position_ids: &[i64],
    capacity_tickets: &[u64],
    starter_domain: usize,
    layers: usize,
    worker_streams: &mut [(quinn::SendStream, quinn::RecvStream)],
    rt: &tokio::runtime::Runtime,
) -> Result<Vec<f32>, String> {
    let domains = worker_streams.len();
    validate_stationary_continuation(
        domains,
        tokens,
        position_ids,
        capacity_tickets,
        starter_domain,
    )?;
    let cmd = WorkerCommand::StationaryContinuation {
        request_id,
        tokens: tokens.to_vec(),
        position_ids: position_ids.to_vec(),
        capacity_tickets: capacity_tickets.to_vec(),
        starter_domain,
    };
    for (send, _recv) in worker_streams.iter_mut() {
        send_command_quic(send, &cmd, rt.handle())
            .map_err(|e| format!("send StationaryContinuation failed: {e}"))?;
    }
    let finisher = continuation_final_finisher(starter_domain, layers, domains)?;
    let mut logits = None;
    for (domain, (_send, recv)) in worker_streams.iter_mut().enumerate() {
        let resp = recv_response_quic(recv, rt.handle())
            .map_err(|e| format!("recv StationaryContinuationDone failed: {e}"))?;
        match resp {
            WorkerResponse::StationaryContinuationDone { logits_bytes, .. } => {
                if domain == finisher {
                    let bytes = logits_bytes.ok_or_else(|| {
                        format!("finisher domain {domain} returned no continuation logits")
                    })?;
                    logits = Some(
                        bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect::<Vec<f32>>(),
                    );
                } else if logits_bytes.is_some() {
                    return Err(format!(
                        "non-finisher domain {domain} returned continuation logits"
                    ));
                }
            }
            WorkerResponse::Error { message, .. } => {
                return Err(format!(
                    "worker {domain} stationary continuation error: {message}"
                ));
            }
            _ => {
                return Err(format!(
                    "unexpected response from worker {domain}: {resp:?}"
                ));
            }
        }
    }
    logits.ok_or_else(|| "finisher continuation logits missing".to_string())
}

/// experimental (route-B 2c, test-only) E2E: reserved prefill -> one legacy
/// decode step -> one injected StationaryContinuation -> remaining legacy
/// decode steps. Reproduces the route_b_cross_node_smoke golden scenario when
/// invoked with --prompt-token-ids 151644,9707,0,16 --chunk-sizes 1,3
/// --continuation-segment 11,13,17,19 --continuation-request-id 75. A local
/// production-path golden can also pass --continuation-capacity-tickets with
/// the remote workers' handshake values to reproduce the same frozen schedule.
#[allow(clippy::too_many_arguments)]
fn run_continuation_e2e(
    request_id: u64,
    prompt_ids: &[i64],
    continuation_tokens: &[i64],
    max_tokens: usize,
    chunk_sizes_override: &Option<Vec<usize>>,
    config: &ModelConfig,
    capacity_tickets: &[u64],
    worker_capacity_mb: &[u64],
    worker_streams: &mut [(quinn::SendStream, quinn::RecvStream)],
    rt: &tokio::runtime::Runtime,
    export_logits_dir: Option<&str>,
) -> Result<(), String> {
    let domains = worker_streams.len();
    let layers = config.num_layers;
    let prompt_len = prompt_ids.len();
    let m = continuation_tokens.len();
    if max_tokens < 2 {
        return Err("continuation E2E needs --max-tokens >= 2".to_string());
    }

    let chunk_sizes: Vec<usize> = if let Some(ref sizes) = chunk_sizes_override {
        if sizes.len() != domains || sizes.iter().sum::<usize>() != prompt_len {
            return Err(format!(
                "--chunk-sizes {sizes:?} must match domains {domains} and prompt length {prompt_len}"
            ));
        }
        sizes.clone()
    } else {
        let chunk_size = prompt_len.div_ceil(domains).max(1);
        let mut chunks = Vec::with_capacity(domains);
        let mut offset = 0usize;
        for i in 0..domains {
            let end = if i == domains - 1 {
                prompt_len
            } else {
                (offset + chunk_size).min(prompt_len)
            };
            chunks.push(end - offset);
            offset = end;
        }
        chunks
    };
    if chunk_sizes.contains(&0) {
        return Err("continuation E2E requires a non-zero chunk per domain".to_string());
    }
    let domain_inputs =
        apply_ring_strategy(prompt_ids, &chunk_sizes, RingSchedulingStrategy::Vanilla);

    // Tickets only select the frozen assignment. They may be overridden to
    // reproduce an experiment and are intentionally separate from budgets.
    let capacity_tickets = capacity_tickets.to_vec();
    let cont_schedule = FrozenKvAssigneeSchedule::new(&capacity_tickets, request_id, m)?;
    let mut continuation_offsets = vec![Vec::new(); domains];
    for offset in 0..m {
        let domain = cont_schedule.assignee_for(offset, 0, 1).unwrap();
        continuation_offsets[domain].push(offset);
    }
    let starter_domain = cont_schedule.assignee_for(m - 1, 0, 1).unwrap();

    // Legacy decode growth positions: step 0 sits at prompt_len; the
    // post-continuation steps resume after the continuation segment.
    let mut decode_positions = vec![prompt_len as i64];
    for step in 0..(max_tokens - 2) {
        decode_positions.push((prompt_len + 1 + m + step) as i64);
    }
    let capacities = reserved_layer_capacities(
        &chunk_sizes,
        &decode_positions,
        &continuation_offsets,
        layers,
        domains,
    );
    let budget_bytes = capacity_mb_to_bytes(worker_capacity_mb)
        .map_err(|error| format!("continuation KV byte admission failed: {error}"))?;
    let admission = admit_reserved_kv_bytes(
        &capacities,
        &budget_bytes,
        config.num_kv_heads(),
        config.head_dim(),
        config.kv_element_size_bytes(),
    )
    .map_err(|error| format!("continuation KV byte admission failed: {error}"))?;
    println!(
        "[coordinator] experimental continuation E2E: tickets={capacity_tickets:?} splits={chunk_sizes:?} starter={starter_domain} offsets={continuation_offsets:?} capacities={capacities:?}"
    );
    println!(
        "[coordinator] continuation KV byte admission: required={:?} budget={budget_bytes:?} bytes_per_token_per_layer={} status=accepted",
        admission.required_bytes_per_domain, admission.bytes_per_token_per_layer
    );

    // Prefill with per-domain reservations.
    for (domain_id, (send, _recv)) in worker_streams.iter_mut().enumerate() {
        let (chunk, position_ids, seq_offset) = &domain_inputs[domain_id];
        let cmd = WorkerCommand::Prefill {
            request_id,
            chunk: chunk.clone(),
            seq_offset: *seq_offset,
            position_ids: Some(position_ids.clone()),
            layer_kv_capacities: Some(capacities[domain_id].clone()),
        };
        send_command_quic(send, &cmd, rt.handle())
            .map_err(|e| format!("send Prefill failed: {e}"))?;
    }
    let mut max_global_seq_len = 0usize;
    let mut last_logits_bytes: Vec<u8> = Vec::new();
    for (domain_id, (_send, recv)) in worker_streams.iter_mut().enumerate() {
        let resp = recv_response_quic(recv, rt.handle())
            .map_err(|e| format!("recv PrefillDone failed: {e}"))?;
        match resp {
            WorkerResponse::PrefillDone {
                last_logits_bytes: bytes,
                global_seq_len,
                ..
            } => {
                if global_seq_len > max_global_seq_len {
                    max_global_seq_len = global_seq_len;
                    last_logits_bytes = bytes;
                }
            }
            WorkerResponse::Error { message, .. } => {
                return Err(format!("worker {domain_id} prefill error: {message}"));
            }
            _ => {
                return Err(format!(
                    "unexpected response from worker {domain_id}: {resp:?}"
                ))
            }
        }
    }
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::SyncGlobalSeqLen {
            request_id,
            len: max_global_seq_len,
        };
        let _ = send_command_quic(send, &cmd, rt.handle());
    }

    let prefill_logits: Vec<f32> = last_logits_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut next_token = sample_from_logits_vec(&prefill_logits, 0.0, 1.0)?;
    let decode_token = next_token;
    let mut generated_ids: Vec<i64> = vec![next_token];

    // One legacy decode step (decode-ring growth rule: position p persists on
    // domain p % domains).
    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::Decode {
            request_id,
            token: next_token,
        };
        let _ = send_command_quic(send, &cmd, rt.handle());
    }
    let decode_response = recv_response_quic(&mut worker_streams[0].1, rt.handle())
        .map_err(|e| format!("recv DecodeDone failed: {e}"))?;
    let decode_logits = match decode_response {
        WorkerResponse::DecodeDone { logits_bytes, .. } => logits_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<f32>>(),
        WorkerResponse::Error { message, .. } => {
            return Err(format!("worker 0 decode error: {message}"));
        }
        response => {
            return Err(format!("unexpected response from worker 0: {response:?}"));
        }
    };
    let (decode_argmax, _) = decode_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    for (_send, recv) in worker_streams.iter_mut().skip(1) {
        let _ = recv_response_quic(recv, rt.handle());
    }

    // Injected stationary continuation.
    let continuation_positions: Vec<i64> =
        (prompt_len as i64 + 1..prompt_len as i64 + 1 + m as i64).collect();
    let continuation_logits = stationary_continuation(
        request_id,
        continuation_tokens,
        &continuation_positions,
        &capacity_tickets,
        starter_domain,
        layers,
        worker_streams,
        rt,
    )?;
    let (continuation_argmax, _) = continuation_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    next_token = sample_from_logits_vec(&continuation_logits, 0.0, 1.0)?;
    generated_ids.push(next_token);
    println!(
        "[coordinator] experimental stationary continuation: decode_token={decode_token} continuation_argmax={continuation_argmax} sampled_next={next_token}"
    );

    // Remaining legacy decode steps.
    for _ in 0..(max_tokens - 2) {
        for (send, _recv) in worker_streams.iter_mut() {
            let cmd = WorkerCommand::Decode {
                request_id,
                token: next_token,
            };
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
        let logits_vec: Vec<f32> = logits_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        next_token = sample_from_logits_vec(&logits_vec, 0.0, 1.0)?;
        generated_ids.push(next_token);
    }

    for (send, _recv) in worker_streams.iter_mut() {
        let cmd = WorkerCommand::ReleaseRequest { request_id };
        let _ = send_command_quic(send, &cmd, rt.handle());
    }
    if let Some(dir) = export_logits_dir {
        let out_dir = Path::new(dir);
        std::fs::create_dir_all(out_dir)
            .map_err(|e| format!("failed to create continuation export dir: {e}"))?;
        write_raw_logits_file(&out_dir.join("prefill_last_logits.f32le"), &prefill_logits)?;
        write_raw_logits_file(&out_dir.join("decode_logits.f32le"), &decode_logits)?;
        write_raw_logits_file(
            &out_dir.join("continuation_last_logits.f32le"),
            &continuation_logits,
        )?;
        let meta = serde_json::json!({
            "mode": "production-quic",
            "device": "distributed-workers",
            "request_id": request_id,
            "domains": domains,
            "layers": layers,
            "tickets": capacity_tickets,
            "capacity_tickets": capacity_tickets,
            "prefix_splits": chunk_sizes,
            "starter_domain": starter_domain,
            "continuation_offsets_by_domain": &continuation_offsets,
            "layer_kv_capacities": &capacities,
            "decode_token": decode_token,
            "prefill_argmax": decode_token,
            "decode_argmax": decode_argmax,
            "continuation_argmax": continuation_argmax,
            "generated_ids": &generated_ids,
        });
        std::fs::write(
            out_dir.join("meta.json"),
            serde_json::to_vec_pretty(&meta)
                .map_err(|e| format!("continuation meta encode failed: {e}"))?,
        )
        .map_err(|e| format!("continuation meta write failed: {e}"))?;
        println!(
            "[coordinator] exported experimental continuation logits to {}",
            out_dir.display()
        );
    }
    println!("[coordinator] experimental continuation E2E generated ids: {generated_ids:?}");
    Ok(())
}

/// Coordinator 主入口。
pub fn run() {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let args = parse_args();
    println!(
        "[coordinator] starting, num_domains={}, workers={:?}, listen={}",
        args.num_domains, args.worker_addrs, args.listen_addr
    );

    // Load tokenizer and config
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config = ModelConfig::from_file(&config_path).expect("load config failed");
    let tokenizer_path = Path::new(&args.model_dir).join("tokenizer.json");
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).expect("load tokenizer failed");

    // Determine serving mode
    let has_cli_prompts =
        args.prompts_file.is_some() || args.prompt_file.is_some() || !args.prompt.is_empty();

    let cli_prompts: Vec<String> = if let Some(ref path) = args.prompts_file {
        let content = std::fs::read_to_string(path).expect("read prompts-file failed");
        content
            .lines()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else if let Some(ref path) = args.prompt_file {
        vec![std::fs::read_to_string(path).expect("read prompt-file failed")]
    } else {
        vec![args.prompt.clone()]
    };

    // Create QUIC endpoint and wait for workers
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime failed");
    let listen_addr: SocketAddr = args.listen_addr.parse().expect("invalid listen_addr");

    let endpoint = rt
        .block_on(async { crate::distributed::transport::quic::create_endpoint(listen_addr) })
        .expect("create_endpoint failed");
    println!(
        "[coordinator] QUIC endpoint listening on {}",
        args.listen_addr
    );

    let mut worker_handshakes: Vec<(usize, u64, quinn::SendStream, quinn::RecvStream)> =
        Vec::with_capacity(args.num_domains);
    for i in 0..args.num_domains {
        let (send, mut recv) = rt
            .block_on(async {
                let incoming = match tokio::time::timeout(
                    std::time::Duration::from_secs(
                        crate::distributed::protocol::default_quic_timeout_secs(),
                    ),
                    endpoint.accept(),
                )
                .await
                {
                    Ok(Some(incoming)) => incoming,
                    Ok(None) => return Err("endpoint closed".to_string()),
                    Err(_) => return Err("accept timeout after 600s".to_string()),
                };
                let conn = incoming
                    .await
                    .map_err(|e| format!("connection failed: {e}"))?;
                println!("[coordinator] worker connection established (accept order {i})");
                let (send, recv) = conn
                    .accept_bi()
                    .await
                    .map_err(|e| format!("accept_bi failed: {e}"))?;
                Ok::<_, String>((send, recv))
            })
            .unwrap_or_else(|e| panic!("accept worker {i} failed: {e}"));

        let handshake = crate::distributed::protocol::read_handshake_quic(&mut recv, rt.handle())
            .expect("handshake read failed");
        println!(
            "[coordinator] worker {} connected (accept order {i}), capacity={} MB",
            handshake.domain_id, handshake.capacity_mb
        );
        worker_handshakes.push((
            handshake.domain_id as usize,
            handshake.capacity_mb,
            send,
            recv,
        ));
    }
    worker_handshakes.sort_by_key(|(domain_id, _, _, _)| *domain_id);
    let worker_capacities: Vec<u64> = worker_handshakes
        .iter()
        .map(|(_, cap, _, _)| *cap)
        .collect();
    let worker_streams: Vec<(quinn::SendStream, quinn::RecvStream)> = worker_handshakes
        .into_iter()
        .map(|(_, _, send, recv)| (send, recv))
        .collect();

    // Wrap worker_streams in Arc<Mutex> for shared access between concurrent requests.
    let worker_streams = Arc::new(std::sync::Mutex::new(worker_streams));

    // experimental (route-B 2c, test-only): stationary continuation E2E entry.
    // Default off — without --continuation-segment the behavior is unchanged.
    if let Some(ref segment) = args.continuation_segment {
        if args.ring_strategy != RingSchedulingStrategy::Vanilla {
            eprintln!("[coordinator] --continuation-segment requires the vanilla ring strategy");
            std::process::exit(1);
        }
        let prompt_ids: Vec<i64> = if let Some(ref ids) = args.prompt_token_ids {
            ids.clone()
        } else {
            tokenizer
                .encode(args.prompt.as_str(), true)
                .expect("encode failed")
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect()
        };
        let result = {
            let mut guard = worker_streams.lock().unwrap_or_else(|e| e.into_inner());
            let capacity_tickets = args
                .continuation_capacity_tickets
                .as_deref()
                .unwrap_or(&worker_capacities);
            run_continuation_e2e(
                args.continuation_request_id,
                &prompt_ids,
                segment,
                args.max_tokens,
                &args.chunk_sizes,
                &config,
                capacity_tickets,
                &worker_capacities,
                &mut guard,
                &rt,
                args.export_logits_dir.as_deref(),
            )
        };
        let mut worker_streams = match Arc::try_unwrap(worker_streams) {
            Ok(mutex) => mutex.into_inner().unwrap_or_else(|e| e.into_inner()),
            Err(_) => {
                eprintln!(
                    "[coordinator] warning: worker_streams still shared, cannot shutdown cleanly"
                );
                return;
            }
        };
        shutdown_workers(&mut worker_streams, &endpoint, &rt);
        match result {
            Ok(()) => {
                println!("[coordinator] experimental continuation E2E done");
                return;
            }
            Err(e) => {
                eprintln!("[coordinator] experimental continuation E2E failed: {e}");
                std::process::exit(1);
            }
        }
    }

    if has_cli_prompts && !cli_prompts.is_empty() {
        // Batch mode: process CLI prompts then exit (serial, no concurrency needed)
        println!("[coordinator] loaded {} prompt(s)", cli_prompts.len());
        for (req_idx, prompt_text) in cli_prompts.iter().enumerate() {
            let request_id = (req_idx + 1) as u64;
            println!(
                "\n[coordinator] === Request {} / {} ===",
                request_id,
                cli_prompts.len()
            );
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
                args.export_logits_dir.as_deref(),
                args.ring_strategy,
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
                eprintln!(
                    "[coordinator] warning: worker_streams still shared, cannot shutdown cleanly"
                );
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

    // Active-request KV byte ledger for the whole HTTP service lifetime.
    let kv_budgets = capacity_mb_to_bytes(&worker_capacities)
        .expect("worker capacities must be convertible to byte budgets");
    let mut kv_ledger = ActiveKvReservation::new(kv_budgets);

    // 6c.0 observability: optional per-request JSONL trace sink. Disabled when
    // --trace-jsonl is absent; enabled it never changes the inference result.
    let mut trace_sink = TraceSink::new(
        args.trace_jsonl.clone(),
        config.num_layers,
        args.num_domains,
    );

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
                    trace_sink.enqueue(job.request_id);
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
                    let job_request_id = job.request_id;
                    match prefill_single_request(
                        job,
                        &tokenizer,
                        &config,
                        &mut guard,
                        &args.chunk_sizes,
                        args.capacity_aware,
                        &worker_capacities,
                        &mut kv_ledger,
                        &rt,
                        args.ring_strategy,
                    ) {
                        Ok(active_req) => {
                            active_counter.fetch_add(1, Ordering::SeqCst);
                            trace_sink.prefill_accepted(
                                active_req.request_id,
                                kv_ledger
                                    .reserved_bytes(active_req.request_id)
                                    .unwrap_or_default()
                                    .to_vec(),
                                active_req.prompt_tokens,
                                active_req.max_tokens,
                            );
                            scheduler.add_active(active_req);
                        }
                        Err(e) => {
                            eprintln!("[coordinator] prefill failed: {e}");
                            trace_sink.fail(job_request_id, e);
                            // Error result already sent via job.tx in prefill_single_request
                        }
                    }
                }
            }

            // 2b: Decode all active requests
            if !scheduler.active_is_empty() {
                // The batch is exactly the deterministic FIFO vector; every id
                // in it takes one decode step this iteration (observability).
                let batch_ids: Vec<u64> = batch_request_tokens(&scheduler)
                    .into_iter()
                    .map(|(request_id, _)| request_id)
                    .collect();
                match decode_iteration(&mut scheduler, &mut guard, eos_token, vocab_size, &rt) {
                    Ok(completed) => {
                        trace_sink.decode_step(&batch_ids);
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
                                let cmd = WorkerCommand::ReleaseRequest {
                                    request_id: *request_id,
                                };
                                let _ = send_command_quic(send, &cmd, rt.handle());
                            }
                        }
                        for request_id in completed {
                            if let Some(req) = scheduler.remove_active(request_id) {
                                active_counter.fetch_sub(1, Ordering::SeqCst);
                                let released_bytes = kv_ledger
                                    .reserved_bytes(request_id)
                                    .unwrap_or_default()
                                    .to_vec();
                                // Free this request's KV byte reservation.
                                kv_ledger.release(request_id);
                                trace_sink.complete(
                                    request_id,
                                    req.finish_reason.clone(),
                                    released_bytes,
                                );

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
                                // Free each failed request's KV byte reservation.
                                kv_ledger.release(request_id);
                                trace_sink.fail(request_id, format!("decode batch failed: {e}"));
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
                    trace_sink.enqueue(job.request_id);
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
            eprintln!(
                "[coordinator] warning: worker_streams still shared, using best-effort shutdown"
            );
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
            send,
            &WorkerCommand::Shutdown,
            rt.handle(),
            10,
        );
        let _ = send.finish();
    }
    endpoint.close(0u32.into(), b"coordinator shutdown");
    rt.block_on(async {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    });
    println!("[coordinator] shutdown complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_decode_owner_is_position_mod_domains() {
        assert_eq!(legacy_decode_owner(4, 2), 0);
        assert_eq!(legacy_decode_owner(5, 2), 1);
        assert_eq!(legacy_decode_owner(9, 2), 1);
        assert_eq!(legacy_decode_owner(4, 3), 1);
        assert_eq!(legacy_decode_owner(9, 3), 0);
    }

    #[test]
    fn reserved_layer_capacities_cover_golden_e2e_scenario() {
        // Golden E2E shape: prompt 4 tokens split [1,3], one pre-continuation
        // decode at position 4, continuation m=4, then 2 more decode steps at
        // positions 9 and 10 (max_tokens = 4), continuation offsets [0] / [1,2,3].
        let decode_positions = [4_i64, 9, 10];
        let continuation_offsets = vec![vec![0_usize], vec![1_usize, 2, 3]];
        let capacities =
            reserved_layer_capacities(&[1, 3], &decode_positions, &continuation_offsets, 24, 2);
        assert_eq!(capacities.len(), 2);
        // domain 0: 1 prefix + {4, 10} decode + 1 continuation offset
        assert!(capacities[0].iter().all(|&c| c == 4));
        // domain 1: 3 prefix + {9} decode + 3 continuation offsets
        assert!(capacities[1].iter().all(|&c| c == 7));
        assert_eq!(capacities[0].len(), 24);
    }

    #[test]
    fn continuation_final_finisher_matches_starter_rotation() {
        // N=2 golden: starter 1, 24 layers -> finisher 1.
        assert_eq!(continuation_final_finisher(1, 24, 2).unwrap(), 1);
        // N=3 smoke scenario: starter 2, 24 layers -> finisher 2.
        assert_eq!(continuation_final_finisher(2, 24, 3).unwrap(), 2);
        // Single layer: finisher is one hop before the starter.
        assert_eq!(continuation_final_finisher(0, 1, 3).unwrap(), 2);
        assert!(continuation_final_finisher(3, 24, 3).is_err());
    }

    fn active_request_for_test(
        scheduler: &mut BatchScheduler,
        request_id: u64,
        next_token: i64,
        max_tokens: usize,
    ) {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id,
            prompt: format!("prompt-{request_id}"),
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            prompt_ids: vec![1],
            prompt_tokens: 1,
            chunk_boundaries: vec![0, 1],
            generated_ids: vec![10],
            next_token,
            finish_reason: None,
            result_tx: tx,
            stream_tx: None,
        });
    }

    #[test]
    fn batch_request_tokens_is_sorted_by_request_id() {
        // Insert active requests in an order that would be non-deterministic if
        // taken from the backing HashMap; the batch contract must be sorted.
        let mut scheduler = BatchScheduler::new(4);
        active_request_for_test(&mut scheduler, 30, 7, 3);
        active_request_for_test(&mut scheduler, 10, 5, 3);
        active_request_for_test(&mut scheduler, 20, 6, 3);

        let tokens = batch_request_tokens(&scheduler);
        assert_eq!(tokens, vec![(10, 5), (20, 6), (30, 7)]);
    }

    #[test]
    fn batch_request_tokens_empty_when_no_active_requests() {
        let scheduler = BatchScheduler::new(4);
        assert!(batch_request_tokens(&scheduler).is_empty());
    }

    #[test]
    fn trace_sink_disabled_never_writes_and_emits_nothing() {
        // No --trace-jsonl: the sink keeps no writer, and lifecycle calls are
        // no-ops that must not panic or change any inference-visible state.
        let mut sink = TraceSink::new(None, 24, 2);
        sink.enqueue(1);
        sink.prefill_accepted(1, vec![100, 200], 4, 5);
        sink.decode_step(&[1]);
        sink.complete(1, Some("length".to_string()), vec![100, 200]);
        sink.fail(2, "boom".to_string());
        assert!(sink.writer.is_none());
        assert!(sink.in_flight.is_empty());
    }

    #[test]
    fn trace_sink_hop_counts_follow_n_l_formula() {
        // N=3, L=24: prefill hops = 24*2 = 48, each decode step = 48.
        // Trace a request through enqueue -> accepted -> 2 decode steps -> complete.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hcp_trace_test_{}.jsonl", std::process::id()));
        let mut sink = TraceSink::new(Some(path.to_string_lossy().to_string()), 24, 3);
        sink.enqueue(7);
        sink.prefill_accepted(7, vec![36864, 49152, 36864], 4, 5);
        sink.decode_step(&[7]);
        sink.decode_step(&[7]);
        sink.complete(7, Some("length".to_string()), vec![36864, 49152, 36864]);

        let line = std::fs::read_to_string(&path).unwrap();
        let record: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(record["request_id"], 7);
        assert_eq!(record["prompt_tokens"], 4);
        assert_eq!(record["max_tokens"], 5);
        assert_eq!(record["decode_steps"], 2);
        assert_eq!(record["prefill_hops"], 48);
        assert_eq!(record["decode_hops"], 96);
        assert_eq!(record["finish_reason"], "length");
        assert!(record["error"].is_null());
        assert_eq!(
            record["reserved_bytes"],
            serde_json::json!([36864, 49152, 36864])
        );
        assert_eq!(
            record["released_bytes"],
            serde_json::json!([36864, 49152, 36864])
        );
        assert!(record["enqueued_elapsed_ms"].as_u64().is_some());
        assert!(record["completed_elapsed_ms"].as_u64().is_some());
        // cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn kv_ledger_accepts_two_individually_fitting_requests() {
        let budgets = vec![100, 100];
        let mut ledger = ActiveKvReservation::new(budgets);
        ledger.try_reserve(1, &[40, 40]).unwrap();
        ledger.try_reserve(2, &[40, 40]).unwrap();
        assert_eq!(ledger.used_bytes(), &[80, 80]);
        assert_eq!(ledger.reserved_bytes(1), Some(&[40, 40][..]));
        assert_eq!(ledger.reserved_bytes(2), Some(&[40, 40][..]));
    }

    #[test]
    fn kv_ledger_rejects_second_request_over_joint_budget() {
        let budgets = vec![100, 100];
        let mut ledger = ActiveKvReservation::new(budgets);
        ledger.try_reserve(1, &[80, 80]).unwrap();
        // Second request fits alone but not jointly: must fail atomically and
        // leave the ledger untouched so the first request's reservation stands.
        let error = ledger.try_reserve(2, &[60, 60]).unwrap_err();
        assert!(error.contains("exceeds budget on domain 0"));
        assert!(ledger.reserved_bytes(2).is_none());
        assert_eq!(ledger.used_bytes(), &[80, 80]);
        // Budget restores after the first request completes.
        ledger.release(1);
        assert_eq!(ledger.used_bytes(), &[0, 0]);
        ledger.try_reserve(2, &[60, 60]).unwrap();
        assert_eq!(ledger.reserved_bytes(2), Some(&[60, 60][..]));
    }

    #[test]
    fn kv_ledger_duplicate_reserve_and_repeated_release_are_safe() {
        let budgets = vec![100];
        let mut ledger = ActiveKvReservation::new(budgets);
        ledger.try_reserve(1, &[30]).unwrap();
        assert!(ledger.try_reserve(1, &[30]).is_err());
        // Repeated release is a no-op, never double-refund or negative.
        ledger.release(1);
        ledger.release(1);
        ledger.release(999);
        assert_eq!(ledger.used_bytes(), &[0]);
        // A new request can take the freed budget.
        ledger.try_reserve(2, &[30]).unwrap();
        assert_eq!(ledger.used_bytes(), &[30]);
    }

    #[test]
    fn kv_ledger_rejects_overflow_and_shape_mismatch() {
        let budgets = vec![100];
        let mut ledger = ActiveKvReservation::new(budgets);
        // Overflow only triggers when the domain already has a live reservation.
        ledger.try_reserve(1, &[50]).unwrap();
        assert!(ledger
            .try_reserve(2, &[u64::MAX])
            .unwrap_err()
            .contains("overflow"));
        assert!(ledger
            .try_reserve(3, &[10, 10])
            .unwrap_err()
            .contains("domains"));
        // Failed attempts leave no accounting behind.
        assert_eq!(ledger.used_bytes(), &[50]);
        assert!(ledger.reserved_bytes(2).is_none());
        assert!(ledger.reserved_bytes(3).is_none());
    }

    #[test]
    fn service_layer_capacities_cover_full_decode_horizon() {
        // 2 domains, prompt 4 split [1,3], max_tokens=3 -> decode positions
        // 4,5,6 owned by p%2 = {0,1,0}. Per-layer capacities must match the
        // golden E2E shape minus the continuation offsets.
        let capacities = service_layer_capacities(&[1, 3], 4, 3, 24);
        assert_eq!(capacities.len(), 2);
        // domain 0: 1 prefix + {4,6} decode = 3; domain 1: 3 prefix + {5} = 4.
        assert!(capacities[0].iter().all(|&c| c == 3));
        assert!(capacities[1].iter().all(|&c| c == 4));
        assert_eq!(capacities[0].len(), 24);
        assert_eq!(capacities[1].len(), 24);
    }

    #[test]
    fn service_layer_capacities_with_zero_max_tokens_reserve_prefix_only() {
        // No decode steps -> reservation is just the prompt prefix split.
        let capacities = service_layer_capacities(&[1, 3], 4, 0, 24);
        assert!(capacities[0].iter().all(|&c| c == 1));
        assert!(capacities[1].iter().all(|&c| c == 3));
    }

    #[test]
    fn service_layer_capacities_three_domains_cover_each_owner() {
        // 3 domains, prompt 6 split [2,2,2], max_tokens=4 -> decode positions
        // 6,7,8,9 owned by {0,1,2,0}.
        let capacities = service_layer_capacities(&[2, 2, 2], 6, 4, 2);
        // domain 0: 2 + {6,9} = 4; domain 1: 2 + {7} = 3; domain 2: 2 + {8} = 3.
        assert!(capacities[0].iter().all(|&c| c == 4));
        assert!(capacities[1].iter().all(|&c| c == 3));
        assert!(capacities[2].iter().all(|&c| c == 3));
    }

    #[test]
    fn frozen_continuation_offsets_cover_segment() {
        // 97ca355 tickets [1,3], request 75, m=4 -> counts [1,3], full cover.
        let schedule = FrozenKvAssigneeSchedule::new(&[1, 3], 75, 4).unwrap();
        let mut offsets = vec![Vec::new(); 2];
        for offset in 0..4 {
            let domain = schedule.assignee_for(offset, 0, 1).unwrap();
            offsets[domain].push(offset);
        }
        assert_eq!(offsets[0].len(), 1);
        assert_eq!(offsets[1].len(), 3);
        let mut union = offsets.concat();
        union.sort_unstable();
        assert_eq!(union, vec![0, 1, 2, 3]);
        // starter = owner of the last continuation position.
        assert_eq!(schedule.assignee_for(3, 0, 1).unwrap(), 1);
    }
}
