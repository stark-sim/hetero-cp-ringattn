//! Experimental cross-node smoke for the route-B stationary continuation path.
//!
//! Replicates the verified single-process test
//! `real_qwen_two_worker_stationary_continuation_matches_reference`
//! (rust/src/worker_sdk/tch_backend.rs) as:
//! - `local`:  single-process golden with N in-process backends (mock ring),
//!   reproducing the test scenario and all of its assertions, then dumping
//!   logits + meta to `--out`.
//! - `server` / `client`: legacy N=2 pair mode (domain 1 / domain 0) over one
//!   full-duplex TCP connection per layer.
//! - `node`:  true N-domain ring topology. Each node only connects to its
//!   successor (outgoing dial to `--peer`) and its predecessor (incoming
//!   accept on `--bind`). Prefill KV blocks and stationary `LayerPacket`s
//!   both flow hop by hop around the ring. `--transport tcp|quic` selects
//!   per-layer TCP connections or per-layer QUIC bidirectional streams over
//!   one connection per neighbor (default tcp).
//!
//! Scenario constants (no coordinator): model Qwen2-0.5B BF16, request_id=75,
//! prompt=[151644,9707,0,16], one decode token at position 4, continuation
//! [11,13,17,19] at positions [5,6,7,8]. tickets / prefix_splits / domains
//! come from the CLI (defaults: N=2, tickets=[1,3], splits=[1,3], exactly the
//! original test); every node derives the full scenario deterministically.

use hcp_ringattn_rust::{
    create_endpoint, process_layer_packet_with_reserved_history,
    process_layer_packet_with_reserved_history_for_positions, project_final_logits,
    FrozenKvAssigneeSchedule, KvBlock, KvCacheImpl, KvTransport, LayerPacket, LayerStepOutcome,
    LinkedMockKvTransport, LlamaModel, ModelConfig, ModelWeights, QuicKvTransport, RingPacket,
    SelfDrivingPacket, TchWorkerBackend, TcpKvTransport, WorkerBackend,
};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tch::{Device, Kind, Tensor};

const REQUEST_ID: u64 = 75;
const PROMPT: [i64; 4] = [151644, 9707, 0, 16];
const CONTINUATION_PROMPT: [i64; 4] = [11, 13, 17, 19];
const EXPECTED_NUM_LAYERS: usize = 24;
const DEFAULT_TICKETS: [u64; 2] = [1, 3];
const DEFAULT_PREFIX_SPLITS: [usize; 2] = [1, 3];

/// Scenario constants derived deterministically on every node from the frozen
/// schedules, exactly like the single-process test.
struct Scenario {
    domains: usize,
    tickets: Vec<u64>,
    prefix_splits: Vec<usize>,
    /// (start, end) contiguous prompt slice per domain.
    prefix_ranges: Vec<(usize, usize)>,
    /// Owner of the last prefix position; produces prefill last logits and
    /// starts the decode ping-pong.
    decode_starter: usize,
    decode_position: i64,
    continuation_positions: Vec<i64>,
    decode_counts: Vec<usize>,
    decode_assignees: Vec<usize>,
    continuation_offsets_by_domain: Vec<Vec<usize>>,
    capacities: Vec<Vec<usize>>,
}

fn build_scenario(
    layers: usize,
    tickets: Vec<u64>,
    prefix_splits: Vec<usize>,
) -> Result<Scenario, String> {
    let domains = tickets.len();
    if domains < 2 {
        return Err(format!(
            "scenario requires at least 2 domains, got {domains}"
        ));
    }
    if prefix_splits.len() != domains {
        return Err(format!(
            "prefix_splits len {} != domains {domains}",
            prefix_splits.len()
        ));
    }
    if prefix_splits.iter().sum::<usize>() != PROMPT.len() {
        return Err(format!(
            "prefix_splits sum {} != prompt length {}",
            prefix_splits.iter().sum::<usize>(),
            PROMPT.len()
        ));
    }
    if prefix_splits.contains(&0) {
        return Err("prefix_splits must be non-zero for every domain".to_string());
    }
    if tickets.iter().all(|&t| t == 0) {
        return Err("tickets must not be all zero".to_string());
    }

    let mut prefix_ranges = Vec::with_capacity(domains);
    let mut cursor = 0_usize;
    for &split in &prefix_splits {
        prefix_ranges.push((cursor, cursor + split));
        cursor += split;
    }
    let decode_starter = domains - 1; // splits are non-zero, so the last domain owns the last position

    let decode_schedule = FrozenKvAssigneeSchedule::new(&tickets, REQUEST_ID, layers)?;
    let decode_counts = decode_schedule.counts().to_vec();
    let decode_assignees = (0..layers)
        .map(|layer_idx| decode_schedule.assignee_for(0, layer_idx, layers).unwrap())
        .collect::<Vec<_>>();

    let continuation_len = CONTINUATION_PROMPT.len();
    let continuation_schedule =
        FrozenKvAssigneeSchedule::new(&tickets, REQUEST_ID, continuation_len)?;
    let mut continuation_offsets_by_domain = vec![Vec::new(); domains];
    for offset in 0..continuation_len {
        let domain = continuation_schedule.assignee_for(offset, 0, 1).unwrap();
        continuation_offsets_by_domain[domain].push(offset);
    }
    let mut assigned_offsets = continuation_offsets_by_domain
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assigned_offsets.sort_unstable();
    if assigned_offsets != (0..continuation_len).collect::<Vec<_>>() {
        return Err("continuation offsets do not cover 0..continuation_len".to_string());
    }

    let decode_position = PROMPT.len() as i64;
    let continuation_positions =
        (decode_position + 1..=decode_position + continuation_len as i64).collect::<Vec<_>>();
    let capacities = (0..domains)
        .map(|domain| {
            (0..layers)
                .map(|layer_idx| {
                    prefix_splits[domain]
                        + usize::from(decode_assignees[layer_idx] == domain)
                        + continuation_offsets_by_domain[domain].len()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(Scenario {
        domains,
        tickets,
        prefix_splits,
        prefix_ranges,
        decode_starter,
        decode_position,
        continuation_positions,
        decode_counts,
        decode_assignees,
        continuation_offsets_by_domain,
        capacities,
    })
}

/// Expected committed positions of one domain's shard at one layer, in commit
/// order: prefix positions, then the decode position (if assignee), then the
/// domain's frozen continuation offsets.
fn expected_positions(scenario: &Scenario, domain: usize, layer_idx: usize) -> Vec<i64> {
    let (start, end) = scenario.prefix_ranges[domain];
    let mut positions: Vec<i64> = (start as i64..end as i64).collect();
    if scenario.decode_assignees[layer_idx] == domain {
        positions.push(scenario.decode_position);
    }
    for &offset in &scenario.continuation_offsets_by_domain[domain] {
        positions.push(scenario.decode_position + 1 + offset as i64);
    }
    positions
}

/// Expected 24-layer KV total per domain, derived from the scenario (never
/// hardcoded): prefix + decode assignee count + continuation offsets.
fn expected_domain_kv_total(scenario: &Scenario, domain: usize, layers: usize) -> usize {
    layers * scenario.prefix_splits[domain]
        + scenario.decode_counts[domain]
        + layers * scenario.continuation_offsets_by_domain[domain].len()
}

/// BF16 near-tie aware argmax equality. Cross-device bf16 rounding can
/// legitimately flip an exact tie (observed on ROCm HIP: tokens 17/198 both
/// 12.0625 in the prefill reference). Passes when both argmaxes agree, or when
/// each side's chosen token is within one bf16 ulp of that side's max logit.
fn assert_argmax_tie_aware(distributed: &Tensor, reference: &Tensor, label: &str) -> (i64, i64) {
    const TIE_EPS: f32 = 0.0625; // one bf16 ulp at |logit| in [8, 16)
    let d = distributed
        .to_kind(Kind::Float)
        .to_device(Device::Cpu)
        .squeeze();
    let r = reference
        .to_kind(Kind::Float)
        .to_device(Device::Cpu)
        .squeeze();
    let dv = Vec::<f32>::try_from(&d).unwrap();
    let rv = Vec::<f32>::try_from(&r).unwrap();
    let argmax_of = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let d_arg = argmax_of(&dv);
    let r_arg = argmax_of(&rv);
    if d_arg != r_arg {
        let d_gap = dv[d_arg] - dv[r_arg];
        let r_gap = rv[r_arg] - rv[d_arg];
        assert!(
            d_gap <= TIE_EPS && r_gap <= TIE_EPS,
            "{label}: argmax {d_arg} vs {r_arg} is not a near-tie (gaps {d_gap:.6}/{r_gap:.6})"
        );
        println!(
            "{label}: argmax {d_arg} vs {r_arg} accepted as bf16 near-tie (gaps {d_gap:.6}/{r_gap:.6})"
        );
    }
    (d_arg as i64, r_arg as i64)
}

/// Finisher of a ring ping-pong layer: the packet visits all N domains in
/// successor order and stops one hop before the starter.
fn layer_finisher(starter: usize, domains: usize) -> usize {
    (starter + domains - 1) % domains
}

fn parse_device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "mps" => Ok(Device::Mps),
        "cuda" => Ok(Device::Cuda(0)),
        _ => value
            .strip_prefix("cuda:")
            .filter(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
            .ok_or_else(|| format!("invalid --device {value}: expected cpu|mps|cuda:N"))
            .and_then(|index| {
                index
                    .parse::<usize>()
                    .map(Device::Cuda)
                    .map_err(|e| format!("invalid --device {value}: {e}"))
            }),
    }
}

fn parse_usize_list(value: &str, flag: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid {flag} value {value}: {e}"))
        })
        .collect()
}

fn parse_u64_list(value: &str, flag: &str) -> Result<Vec<u64>, String> {
    value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<u64>()
                .map_err(|e| format!("invalid {flag} value {value}: {e}"))
        })
        .collect()
}

fn default_model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HCP_ROUTE_B_MODEL_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
        .join("Qwen2-0.5B")
}

fn embed_tokens(model: &LlamaModel, tokens: &[i64], device: Device) -> Tensor {
    let input = Tensor::from_slice(tokens).unsqueeze(0).to_device(device);
    Tensor::embedding(&model.embedding, &input, -1, false, false)
}

fn write_tensor_f32le(path: &Path, tensor: &Tensor) -> Result<(), String> {
    let values: Vec<f32> = Vec::try_from(&tensor.to_kind(Kind::Float).contiguous())
        .map_err(|e| format!("tensor to f32 vec failed: {e}"))?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| format!("write {} failed: {e}", path.display()))
}

fn storage_snapshot(
    backend: &TchWorkerBackend,
    layers: usize,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let context = backend
        .request_contexts
        .get(&REQUEST_ID)
        .ok_or_else(|| format!("request {REQUEST_ID} not found"))?;
    (0..layers)
        .map(|layer_idx| {
            let Some(KvCacheImpl::ReservedPositioned(shard)) = &context.kv_caches[layer_idx] else {
                return Err(format!(
                    "layer {layer_idx} did not use reserved positioned KV"
                ));
            };
            Ok((
                shard.active_k().data_ptr() as usize,
                shard.active_v().data_ptr() as usize,
                shard.reserved_capacity(),
            ))
        })
        .collect()
}

/// Per-domain tail checks: stable storage, committed == reserved, positions
/// match the frozen offsets, and the scenario-derived 24-layer KV total.
fn verify_domain_kv_state(
    backend: &TchWorkerBackend,
    scenario: &Scenario,
    domain: usize,
    storage_before: &[(usize, usize, usize)],
) -> Result<usize, String> {
    let context = backend
        .request_contexts
        .get(&REQUEST_ID)
        .ok_or_else(|| format!("request {REQUEST_ID} not found"))?;
    let mut kv_total = 0_usize;
    for (layer_idx, &(k_ptr, v_ptr, reserved)) in storage_before.iter().enumerate() {
        let Some(KvCacheImpl::ReservedPositioned(shard)) = &context.kv_caches[layer_idx] else {
            return Err(format!(
                "domain {domain} layer {layer_idx} did not use reserved positioned KV"
            ));
        };
        if shard.active_k().data_ptr() as usize != k_ptr
            || shard.active_v().data_ptr() as usize != v_ptr
        {
            return Err(format!(
                "domain {domain} layer {layer_idx} KV storage moved during decode/continuation"
            ));
        }
        if shard.reserved_capacity() != reserved {
            return Err(format!(
                "domain {domain} layer {layer_idx} reserved capacity changed: {reserved} -> {}",
                shard.reserved_capacity()
            ));
        }
        if shard.committed_len() != shard.reserved_capacity() {
            return Err(format!(
                "domain {domain} layer {layer_idx} committed {} != reserved {}",
                shard.committed_len(),
                shard.reserved_capacity()
            ));
        }
        let expected = expected_positions(scenario, domain, layer_idx);
        if shard.positions() != expected.as_slice() {
            return Err(format!(
                "domain {domain} layer {layer_idx} positions {:?} != frozen {:?}",
                shard.positions(),
                expected
            ));
        }
        kv_total += shard.committed_len();
    }
    let expected_total = expected_domain_kv_total(scenario, domain, storage_before.len());
    if kv_total != expected_total {
        return Err(format!(
            "domain {domain} KV total {kv_total} != scenario-derived {expected_total}"
        ));
    }
    Ok(kv_total)
}

/// Full-duplex per-layer connection shared between the backend (prefill KV
/// ring) and this binary (self-driving packets after prefill).
#[derive(Clone)]
struct SharedTcpTransport(Arc<Mutex<TcpKvTransport>>);

impl SharedTcpTransport {
    fn lock(&self) -> std::sync::MutexGuard<'_, TcpKvTransport> {
        self.0.lock().expect("shared TCP transport mutex poisoned")
    }
}

impl KvTransport for SharedTcpTransport {
    fn submit_send(&mut self, block: &KvBlock) -> Result<(), String> {
        self.lock().submit_send(block)
    }

    fn poll_recv(&mut self) -> Result<Option<KvBlock>, String> {
        self.lock().poll_recv()
    }

    fn flush_send(&mut self) -> Result<(), String> {
        self.lock().flush_send()
    }

    fn supports_ring_packets(&self) -> bool {
        self.lock().supports_ring_packets()
    }

    fn submit_send_packet(&mut self, packet: &RingPacket) -> Result<(), String> {
        self.lock().submit_send_packet(packet)
    }

    fn poll_recv_packet(&mut self) -> Result<Option<RingPacket>, String> {
        self.lock().poll_recv_packet()
    }

    fn supports_self_driving_packets(&self) -> bool {
        self.lock().supports_self_driving_packets()
    }

    fn submit_send_self_driving_packet(
        &mut self,
        packet: &SelfDrivingPacket,
    ) -> Result<(), String> {
        self.lock().submit_send_self_driving_packet(packet)
    }

    fn poll_recv_self_driving_packet(&mut self) -> Result<Option<SelfDrivingPacket>, String> {
        self.lock().poll_recv_self_driving_packet()
    }
}

/// Transport-agnostic view of one per-layer ring link: KvTransport for the
/// backend's prefill KV exchange, plus bin-level self-driving packet I/O for
/// the decode/continuation ping-pong. Sends always go successor-wards.
trait RingTransportOps: KvTransport + Clone + 'static {
    fn send_packet(&self, packet: &SelfDrivingPacket) -> Result<(), String>;
    fn recv_packet(&self) -> Result<Option<SelfDrivingPacket>, String>;
}

/// True ring per-layer transport: sends go to the successor (outgoing dial),
/// receives come from the predecessor (incoming accept). There is no direct
/// connection to any other domain, so both prefill KV blocks and stationary
/// self-driving packets flow hop by hop around the ring.
#[derive(Clone)]
struct RingTcpTransport {
    outgoing: SharedTcpTransport,
    incoming: SharedTcpTransport,
}

impl RingTransportOps for RingTcpTransport {
    fn send_packet(&self, packet: &SelfDrivingPacket) -> Result<(), String> {
        self.outgoing
            .lock()
            .send_self_driving_packet(packet)
            .map(|_| ())
    }

    fn recv_packet(&self) -> Result<Option<SelfDrivingPacket>, String> {
        self.incoming.lock().recv_self_driving_packet()
    }
}

impl KvTransport for RingTcpTransport {
    fn submit_send(&mut self, block: &KvBlock) -> Result<(), String> {
        self.outgoing.lock().submit_send(block)
    }

    fn poll_recv(&mut self) -> Result<Option<KvBlock>, String> {
        self.incoming.lock().poll_recv()
    }

    fn flush_send(&mut self) -> Result<(), String> {
        self.outgoing.lock().flush_send()
    }

    fn supports_ring_packets(&self) -> bool {
        self.outgoing.lock().supports_ring_packets()
    }

    fn submit_send_packet(&mut self, packet: &RingPacket) -> Result<(), String> {
        self.outgoing.lock().submit_send_packet(packet)
    }

    fn poll_recv_packet(&mut self) -> Result<Option<RingPacket>, String> {
        self.incoming.lock().poll_recv_packet()
    }

    fn supports_self_driving_packets(&self) -> bool {
        self.outgoing.lock().supports_self_driving_packets()
    }

    fn submit_send_self_driving_packet(
        &mut self,
        packet: &SelfDrivingPacket,
    ) -> Result<(), String> {
        self.outgoing.lock().submit_send_self_driving_packet(packet)
    }

    fn poll_recv_self_driving_packet(&mut self) -> Result<Option<SelfDrivingPacket>, String> {
        self.incoming.lock().poll_recv_self_driving_packet()
    }
}

/// Shared QUIC ring link. `QuicKvTransport::new(send, recv, ...)` already
/// separates the two halves: constructed with the send half of the stream to
/// the successor and the recv half of the stream from the predecessor, it
/// natively satisfies the ring direction semantics. The Arc<Mutex> wrapper
/// solves the `setup_kv_transports` ownership hand-off, same as the TCP path.
#[derive(Clone)]
struct SharedQuicTransport(Arc<Mutex<QuicKvTransport>>);

impl SharedQuicTransport {
    fn lock(&self) -> std::sync::MutexGuard<'_, QuicKvTransport> {
        self.0.lock().expect("shared QUIC transport mutex poisoned")
    }
}

impl KvTransport for SharedQuicTransport {
    fn submit_send(&mut self, block: &KvBlock) -> Result<(), String> {
        self.lock().submit_send(block)
    }

    fn poll_recv(&mut self) -> Result<Option<KvBlock>, String> {
        self.lock().poll_recv()
    }

    fn flush_send(&mut self) -> Result<(), String> {
        self.lock().flush_send()
    }

    fn supports_ring_packets(&self) -> bool {
        self.lock().supports_ring_packets()
    }

    fn submit_send_packet(&mut self, packet: &RingPacket) -> Result<(), String> {
        self.lock().submit_send_packet(packet)
    }

    fn poll_recv_packet(&mut self) -> Result<Option<RingPacket>, String> {
        self.lock().poll_recv_packet()
    }

    fn supports_self_driving_packets(&self) -> bool {
        self.lock().supports_self_driving_packets()
    }

    fn submit_send_self_driving_packet(
        &mut self,
        packet: &SelfDrivingPacket,
    ) -> Result<(), String> {
        self.lock().submit_send_self_driving_packet(packet)
    }

    fn poll_recv_self_driving_packet(&mut self) -> Result<Option<SelfDrivingPacket>, String> {
        self.lock().poll_recv_self_driving_packet()
    }
}

impl RingTransportOps for SharedQuicTransport {
    fn send_packet(&self, packet: &SelfDrivingPacket) -> Result<(), String> {
        let mut transport = self.lock();
        transport.submit_send_self_driving_packet(packet)?;
        transport.flush_send()
    }

    fn recv_packet(&self) -> Result<Option<SelfDrivingPacket>, String> {
        self.lock().recv_self_driving_packet()
    }
}

/// Holds every QUIC object that must stay alive for the whole node run.
struct QuicRingLinks {
    runtime: tokio::runtime::Runtime,
    _endpoint: quinn::Endpoint,
    outgoing_conn: quinn::Connection,
    incoming_conn: quinn::Connection,
    /// Unused stream halves (outgoing recv / incoming send) kept so quinn does
    /// not signal stream resets to the peer.
    _unused_halves: Vec<(quinn::SendStream, quinn::RecvStream)>,
    links: Vec<SharedQuicTransport>,
}

impl QuicRingLinks {
    /// Graceful ring shutdown barrier. QUIC delivers nothing after connection
    /// close, so a node must not exit while its successor can still be missing
    /// data. Protocol (per node, after all stages):
    /// 1. send a done byte to the successor (new stream on the outgoing conn),
    /// 2. read the predecessor's done byte (its connection closing also
    ///    counts: this node's stages already prove the predecessor sent
    ///    everything this node needed),
    /// 3. ack the predecessor on the same stream,
    /// 4. wait for the successor's ack before closing both connections.
    ///
    /// The successor only acks after finishing its own stages, which required
    /// every packet this node ever sent it — so after step 4 no in-flight
    /// data can still be needed downstream.
    fn barrier(&self, domain: usize) -> Result<(), String> {
        self.runtime.block_on(async {
            let (mut done_send, mut ack_recv) = self
                .outgoing_conn
                .open_bi()
                .await
                .map_err(|e| format!("barrier open_bi failed: {e}"))?;
            done_send
                .write_all(b"\x01")
                .await
                .map_err(|e| format!("barrier done write failed: {e}"))?;
            done_send
                .finish()
                .map_err(|e| format!("barrier done finish failed: {e}"))?;

            let (mut ack_send, mut done_recv) = self
                .incoming_conn
                .accept_bi()
                .await
                .map_err(|e| format!("barrier accept_bi failed: {e}"))?;
            let mut byte = [0_u8; 1];
            // Any resolution (byte or connection close) ends the wait; see step 2.
            let _ = done_recv.read(&mut byte).await;
            ack_send
                .write_all(b"\x02")
                .await
                .map_err(|e| format!("barrier ack write failed: {e}"))?;
            ack_send
                .finish()
                .map_err(|e| format!("barrier ack finish failed: {e}"))?;

            // Any resolution (ack byte or connection close) ends the wait;
            // a peer that exited already proved downstream delivery (step 4).
            let _ = ack_recv.read(&mut byte).await;
            self.outgoing_conn.close(0_u32.into(), b"done");
            self.incoming_conn.close(0_u32.into(), b"done");
            Ok::<_, String>(())
        })?;
        println!("[route-b smoke] domain {domain}: quic barrier complete");
        Ok(())
    }
}

/// Establish the per-layer QUIC ring links: one connection to the successor
/// (24 `open_bi`, layer order) and one connection from the predecessor (24
/// `accept_bi`, matching the peer's open order). Every opened stream starts
/// with the 1-byte dummy that `QuicKvTransport`'s recv task skips.
fn establish_quic_ring_links(
    bind: &str,
    peer: &str,
    layers: usize,
    device: Device,
    domain: usize,
) -> Result<QuicRingLinks, String> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let runtime =
        tokio::runtime::Runtime::new().map_err(|e| format!("tokio runtime failed: {e}"))?;
    let bind_addr = bind
        .parse()
        .map_err(|e| format!("invalid --bind {bind}: {e}"))?;
    // `Endpoint::server` spawns the UDP driver task and therefore requires an
    // active tokio runtime context.
    let endpoint = runtime
        .block_on(async { create_endpoint(bind_addr) })
        .map_err(|e| format!("create quic endpoint on {bind} failed: {e}"))?;
    eprintln!("[route-b smoke] domain {domain}: quic endpoint bound on {bind}");
    let peer_addr = peer
        .parse()
        .map_err(|e| format!("invalid --peer {peer}: {e}"))?;

    // Drive the outgoing handshake and the incoming accept concurrently: an
    // endpoint whose peer also dials must progress its server side while its
    // own client handshake is in flight.
    let (outgoing_conn, incoming_conn) = runtime.block_on(async {
        let connect_future = async {
            let mut attempt = 0_u32;
            loop {
                let result = match endpoint.connect(peer_addr, "localhost") {
                    Ok(connecting) => connecting.await,
                    Err(e) => {
                        attempt += 1;
                        if attempt >= 240 {
                            return Err(format!(
                                "quic connect to successor {peer} failed: {e}"
                            ));
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        continue;
                    }
                };
                match result {
                    Ok(connection) => return Ok(connection),
                    Err(e) => {
                        attempt += 1;
                        if attempt <= 5 {
                            eprintln!(
                                "[route-b smoke] domain {domain}: quic handshake attempt {attempt} failed: {e}"
                            );
                        }
                        if attempt >= 240 {
                            return Err(format!(
                                "quic handshake with successor {peer} failed after {attempt} attempts: {e}"
                            ));
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    }
                }
            }
        };
        let accept_future = async {
            endpoint
                .accept()
                .await
                .ok_or_else(|| "quic endpoint closed before the predecessor connected".to_string())?
                .await
                .map_err(|e| format!("accept predecessor connection failed: {e}"))
        };
        tokio::try_join!(connect_future, accept_future)
    })?;
    println!("[route-b smoke] domain {domain}: quic ring connections up (successor {peer}, predecessor on {bind})");

    let mut outgoing_halves = Vec::with_capacity(layers);
    for layer_idx in 0..layers {
        let (mut send, recv) = runtime
            .block_on(outgoing_conn.open_bi())
            .map_err(|e| format!("open_bi layer {layer_idx} failed: {e}"))?;
        runtime
            .block_on(send.write_all(&[0_u8]))
            .map_err(|e| format!("dummy write layer {layer_idx} failed: {e}"))?;
        outgoing_halves.push((send, recv));
    }
    println!("[route-b smoke] domain {domain}: opened {layers} streams to successor {peer}");

    let mut incoming_halves = Vec::with_capacity(layers);
    for layer_idx in 0..layers {
        let (send, recv) = runtime
            .block_on(incoming_conn.accept_bi())
            .map_err(|e| format!("accept_bi layer {layer_idx} failed: {e}"))?;
        incoming_halves.push((send, recv));
    }
    println!("[route-b smoke] domain {domain}: accepted {layers} streams from predecessor");

    let mut unused_halves = Vec::with_capacity(layers);
    let links = outgoing_halves
        .into_iter()
        .zip(incoming_halves)
        .map(|((out_send, out_recv), (in_send, in_recv))| {
            unused_halves.push((in_send, out_recv));
            SharedQuicTransport(Arc::new(Mutex::new(QuicKvTransport::new(
                out_send,
                in_recv,
                runtime.handle().clone(),
                device,
            ))))
        })
        .collect::<Vec<_>>();
    Ok(QuicRingLinks {
        runtime,
        _endpoint: endpoint,
        outgoing_conn,
        incoming_conn,
        _unused_halves: unused_halves,
        links,
    })
}

/// One ring ping-pong phase (decode or continuation) for this node.
///
/// Per layer the packet starts at `phase_starter(layer)`, visits every domain
/// in successor order, and finishes one hop before the starter. This node is
/// exactly one of: starter (build + process + forward), middle (recv +
/// process + forward), or finisher (recv + process + keep hidden states).
/// Returns (sends, finisher-of-last-layer, hidden states).
#[allow(clippy::too_many_arguments)]
fn run_ring_phase<T: RingTransportOps>(
    phase: &str,
    backend: &mut TchWorkerBackend,
    ring: &[T],
    scenario: &Scenario,
    domain: usize,
    mut current_starter: usize,
    tokens: &[i64],
    position_ids: &Tensor,
    for_positions: bool,
    embed_first: bool,
    device: Device,
) -> Result<(usize, usize, Tensor), String> {
    let mut sends = 0_usize;
    let mut hidden_states: Option<Tensor> = None;
    for (layer_idx, transport) in ring.iter().enumerate() {
        let finisher = layer_finisher(current_starter, scenario.domains);
        if domain == current_starter {
            if hidden_states.is_none() {
                if !embed_first {
                    return Err(format!(
                        "{phase} layer {layer_idx}: starter has no hidden states"
                    ));
                }
                hidden_states = Some(embed_tokens(&backend.model, tokens, device));
            }
            let assignee = if for_positions {
                // Stationary continuation: every domain appends only its own
                // frozen offsets; the legacy scalar assignee is the starter.
                domain
            } else {
                scenario.decode_assignees[layer_idx]
            };
            let packet = LayerPacket::start(
                &mut backend.model.layers[layer_idx],
                hidden_states.as_ref().unwrap(),
                position_ids,
                domain,
                assignee,
                scenario.domains,
            )
            .map_err(|e| format!("{phase} layer {layer_idx} start failed: {e}"))?;
            let outcome = {
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    return Err(format!(
                        "domain {domain} layer {layer_idx} did not use reserved positioned KV"
                    ));
                };
                if for_positions {
                    process_layer_packet_with_reserved_history_for_positions(
                        &mut backend.model.layers[layer_idx],
                        packet,
                        shard,
                        &scenario.continuation_offsets_by_domain[domain],
                    )
                } else {
                    process_layer_packet_with_reserved_history(
                        &mut backend.model.layers[layer_idx],
                        packet,
                        shard,
                    )
                }
                .map_err(|e| format!("{phase} layer {layer_idx} starter step failed: {e}"))?
            };
            let LayerStepOutcome::Forward(next_packet) = outcome else {
                return Err(format!(
                    "{phase} layer {layer_idx} starter finished a {}-domain route",
                    scenario.domains
                ));
            };
            let wire = next_packet
                .into_self_driving_packet(layer_idx)
                .map_err(|e| format!("{phase} layer {layer_idx} wire encode failed: {e}"))?;
            transport
                .send_packet(&wire)
                .map_err(|e| format!("{phase} layer {layer_idx} forward failed: {e}"))?;
            sends += 1;
        } else {
            let wire = transport
                .recv_packet()
                .map_err(|e| format!("{phase} layer {layer_idx} recv failed: {e}"))?
                .ok_or_else(|| format!("{phase} layer {layer_idx} predecessor closed"))?;
            if wire.layer_idx != layer_idx {
                return Err(format!(
                    "{phase} layer {layer_idx} received packet for layer {}",
                    wire.layer_idx
                ));
            }
            let packet = LayerPacket::from_self_driving_packet(wire)
                .map_err(|e| format!("{phase} layer {layer_idx} wire decode failed: {e}"))?;
            let outcome = {
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    return Err(format!(
                        "domain {domain} layer {layer_idx} did not use reserved positioned KV"
                    ));
                };
                if for_positions {
                    process_layer_packet_with_reserved_history_for_positions(
                        &mut backend.model.layers[layer_idx],
                        packet,
                        shard,
                        &scenario.continuation_offsets_by_domain[domain],
                    )
                } else {
                    process_layer_packet_with_reserved_history(
                        &mut backend.model.layers[layer_idx],
                        packet,
                        shard,
                    )
                }
                .map_err(|e| format!("{phase} layer {layer_idx} ring step failed: {e}"))?
            };
            if domain == finisher {
                let LayerStepOutcome::Finished {
                    hidden_states: next_hidden,
                    ..
                } = outcome
                else {
                    return Err(format!(
                        "{phase} layer {layer_idx} finisher forwarded a {}-domain route",
                        scenario.domains
                    ));
                };
                hidden_states = Some(next_hidden);
            } else {
                let LayerStepOutcome::Forward(next_packet) = outcome else {
                    return Err(format!(
                        "{phase} layer {layer_idx} middle node finished a {}-domain route",
                        scenario.domains
                    ));
                };
                let wire = next_packet
                    .into_self_driving_packet(layer_idx)
                    .map_err(|e| format!("{phase} layer {layer_idx} wire encode failed: {e}"))?;
                transport
                    .send_packet(&wire)
                    .map_err(|e| format!("{phase} layer {layer_idx} forward failed: {e}"))?;
                sends += 1;
            }
        }
        current_starter = layer_finisher(current_starter, scenario.domains);
    }
    let hidden_states =
        hidden_states.ok_or_else(|| format!("{phase}: this node never finished a layer"))?;
    Ok((sends, current_starter, hidden_states))
}

/// Wire transport for `node` mode ring links.
#[derive(Clone, Copy)]
enum TransportKind {
    Tcp,
    Quic,
}

impl TransportKind {
    fn as_str(self) -> &'static str {
        match self {
            TransportKind::Tcp => "tcp",
            TransportKind::Quic => "quic",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "tcp" => Ok(TransportKind::Tcp),
            "quic" => Ok(TransportKind::Quic),
            _ => Err(format!("invalid --transport {value}: expected tcp|quic")),
        }
    }
}

/// Establish the per-layer TCP ring links: bind first so the successor can
/// dial, then dial the successor (with retry), then accept the predecessor —
/// both in layer order.
fn establish_tcp_ring_links(
    bind: &str,
    peer: &str,
    layers: usize,
    device: Device,
    domain: usize,
) -> Result<Vec<RingTcpTransport>, String> {
    let listener = TcpListener::bind(bind).map_err(|e| format!("bind {bind} failed: {e}"))?;
    let mut outgoing = Vec::with_capacity(layers);
    for layer_idx in 0..layers {
        let mut attempt = 0_u32;
        let stream = loop {
            match TcpStream::connect(peer) {
                Ok(stream) => break stream,
                Err(e) => {
                    attempt += 1;
                    if attempt >= 240 {
                        return Err(format!(
                            "connect layer {layer_idx} to successor {peer} failed after {attempt} attempts: {e}"
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        };
        outgoing.push(SharedTcpTransport(Arc::new(Mutex::new(
            TcpKvTransport::new(stream, device)?,
        ))));
    }
    println!("[route-b smoke] domain {domain}: dialed successor {peer} ({layers} connections)");
    let mut incoming = Vec::with_capacity(layers);
    for layer_idx in 0..layers {
        let (stream, addr) = listener
            .accept()
            .map_err(|e| format!("accept layer {layer_idx} failed: {e}"))?;
        if layer_idx == 0 {
            println!("[route-b smoke] domain {domain}: accepted predecessor {addr} on {bind}");
        }
        incoming.push(SharedTcpTransport(Arc::new(Mutex::new(
            TcpKvTransport::new(stream, device)?,
        ))));
    }
    Ok((0..layers)
        .map(|layer_idx| RingTcpTransport {
            outgoing: outgoing[layer_idx].clone(),
            incoming: incoming[layer_idx].clone(),
        })
        .collect::<Vec<_>>())
}

/// `node` mode: one domain of a true N-domain ring over real TCP or QUIC.
#[allow(clippy::too_many_arguments)]
fn run_ring_node(
    domain: usize,
    device: Device,
    bind: &str,
    peer: &str,
    tickets: Vec<u64>,
    prefix_splits: Vec<usize>,
    transport: TransportKind,
    model_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    let config = ModelConfig::from_file(model_dir.join("config.json"))
        .map_err(|e| format!("load config failed: {e}"))?;
    if config.num_layers != EXPECTED_NUM_LAYERS {
        return Err(format!(
            "num_layers {} != {EXPECTED_NUM_LAYERS}",
            config.num_layers
        ));
    }
    let layers = config.num_layers;
    let scenario = build_scenario(layers, tickets, prefix_splits)?;
    let domains = scenario.domains;
    if domain >= domains {
        return Err(format!(
            "domain {domain} out of range for {domains} domains"
        ));
    }
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("create {} failed: {e}", out_dir.display()))?;
    let weights = ModelWeights::from_dir(model_dir, device)
        .map_err(|e| format!("load weights failed: {e}"))?;
    let model = LlamaModel::from_weights(config, &weights, device, domains)
        .map_err(|e| format!("build model failed: {e}"))?;
    let mut backend = TchWorkerBackend::from_model(model, device, domain);

    match transport {
        TransportKind::Tcp => {
            let ring = establish_tcp_ring_links(bind, peer, layers, device, domain)?;
            run_ring_node_stages(
                &mut backend,
                &ring,
                &scenario,
                domain,
                device,
                transport,
                out_dir,
            )
        }
        TransportKind::Quic => {
            // `quic_links` must stay alive until every phase is done: it owns
            // the tokio runtime, the endpoint, and both connections.
            let quic_links = establish_quic_ring_links(bind, peer, layers, device, domain)?;
            run_ring_node_stages(
                &mut backend,
                &quic_links.links,
                &scenario,
                domain,
                device,
                transport,
                out_dir,
            )?;
            // QUIC drops unacked data when a connection dies with the process;
            // drain the ring deterministically before exiting.
            quic_links.barrier(domain)
        }
    }
}

/// Shared node stages once the per-layer ring links exist: prefill KV ring,
/// decode + stationary continuation ping-pong, tail checks, and the dump.
fn run_ring_node_stages<T: RingTransportOps>(
    backend: &mut TchWorkerBackend,
    ring: &[T],
    scenario: &Scenario,
    domain: usize,
    device: Device,
    transport: TransportKind,
    out_dir: &Path,
) -> Result<(), String> {
    let layers = ring.len();
    let transports = ring
        .iter()
        .map(|transport| Box::new(transport.clone()) as Box<dyn KvTransport>)
        .collect::<Vec<_>>();
    backend.setup_kv_transports(transports);

    // ===== Phase 1: prefill (KV ring exchange happens inside the backend,
    // flowing successor-wards hop by hop through the ring links). =====
    let (prefix_start, prefix_end) = scenario.prefix_ranges[domain];
    let chunk = &PROMPT[prefix_start..prefix_end];
    let positions = (prefix_start as i64..prefix_end as i64).collect::<Vec<_>>();
    let (logits_vec, global_len) = backend
        .prefill_request_with_reservation(
            REQUEST_ID,
            chunk,
            prefix_start,
            Some(&positions),
            Some(&scenario.capacities[domain]),
        )
        .map_err(|e| format!("prefill failed: {e}"))?;
    if global_len != prefix_end {
        return Err(format!(
            "domain {domain} global_seq_len {global_len} != prefix end {prefix_end}"
        ));
    }
    let prefill_logits = Tensor::from_slice(&logits_vec);
    let prefill_argmax = prefill_logits.argmax(-1, false).int64_value(&[]);
    let mut decode_token = None;
    if domain == scenario.decode_starter {
        decode_token = Some(prefill_argmax);
        write_tensor_f32le(&out_dir.join("prefill_last_logits.f32le"), &prefill_logits)?;
    }
    println!("[route-b smoke] domain {domain} prefill done: global_len={global_len}");

    let storage_before = storage_snapshot(backend, layers)?;

    // ===== Phase 2: decode (24-layer ring ping-pong). =====
    let decode_position_ids = Tensor::from_slice(&[scenario.decode_position])
        .unsqueeze(0)
        .to_device(device);
    let (decode_sends, decode_finisher, decode_hidden) = run_ring_phase(
        "decode",
        backend,
        ring,
        scenario,
        domain,
        scenario.decode_starter,
        &decode_token.map_or([0_i64; 1], |token| [token]),
        &decode_position_ids,
        false,
        domain == scenario.decode_starter,
        device,
    )?;
    backend
        .request_contexts
        .get_mut(&REQUEST_ID)
        .unwrap()
        .global_seq_len = scenario.decode_position as usize + 1;
    let mut decode_argmax = None;
    if decode_finisher == domain {
        let logits = project_final_logits(&backend.model, &decode_hidden).squeeze();
        decode_argmax = Some(logits.argmax(-1, false).int64_value(&[]));
        write_tensor_f32le(&out_dir.join("decode_logits.f32le"), &logits)?;
    }
    println!("[route-b smoke] domain {domain} decode done: finisher={decode_finisher}");

    // ===== Phase 3: stationary continuation (24-layer ring ping-pong, m=4). =====
    let continuation_position_ids = Tensor::from_slice(&scenario.continuation_positions)
        .unsqueeze(0)
        .to_device(device);
    let (continuation_sends, continuation_finisher, continuation_hidden) = run_ring_phase(
        "continuation",
        backend,
        ring,
        scenario,
        domain,
        decode_finisher,
        &CONTINUATION_PROMPT,
        &continuation_position_ids,
        true,
        domain == decode_finisher,
        device,
    )?;
    let mut continuation_argmax = None;
    if continuation_finisher == domain {
        let last = project_final_logits(&backend.model, &continuation_hidden)
            .select(1, CONTINUATION_PROMPT.len() as i64 - 1)
            .squeeze();
        continuation_argmax = Some(last.argmax(-1, false).int64_value(&[]));
        write_tensor_f32le(&out_dir.join("continuation_last_logits.f32le"), &last)?;
    }
    println!("[route-b smoke] domain {domain} continuation done: finisher={continuation_finisher}");

    // ===== Phase 4: per-domain tail checks. =====
    let kv_total = verify_domain_kv_state(backend, scenario, domain, &storage_before)?;
    let handoffs = decode_sends + continuation_sends;

    let meta = serde_json::json!({
        "mode": "node",
        "device": format!("{device:?}"),
        "transport": transport.as_str(),
        "domain": domain,
        "domains": scenario.domains,
        "tickets": scenario.tickets,
        "prefix_splits": scenario.prefix_splits,
        "decode_starter": scenario.decode_starter,
        "decode_finisher": decode_finisher,
        "continuation_finisher": continuation_finisher,
        "decode_token": decode_token,
        "prefill_argmax": if domain == scenario.decode_starter { Some(prefill_argmax) } else { None },
        "decode_argmax": decode_argmax,
        "continuation_argmax": continuation_argmax,
        "domain_kv_totals": { domain.to_string(): kv_total },
        "handoffs": handoffs,
        "decode_sends": decode_sends,
        "continuation_sends": continuation_sends,
        "checks": {
            "num_layers": layers == EXPECTED_NUM_LAYERS,
            "kv_total_matches_expected": kv_total == expected_domain_kv_total(scenario, domain, layers),
            "storage_stable": true,
            "committed_eq_reserved": true,
            "positions_match_frozen_offsets": true,
        },
    });
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write meta.json failed: {e}"))?;
    println!(
        "[route-b smoke] domain {domain} done: kv_total={kv_total} handoffs={handoffs} out={}",
        out_dir.display()
    );
    Ok(())
}

/// Legacy N=2 pair mode (`server` / `client`): one full-duplex TCP connection
/// per layer, exactly the originally validated two-process behavior.
fn run_pair_node(
    mode: &str,
    domain: usize,
    device: Device,
    bind: &str,
    peer: Option<&str>,
    model_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    let config = ModelConfig::from_file(model_dir.join("config.json"))
        .map_err(|e| format!("load config failed: {e}"))?;
    if config.num_layers != EXPECTED_NUM_LAYERS {
        return Err(format!(
            "num_layers {} != {EXPECTED_NUM_LAYERS}",
            config.num_layers
        ));
    }
    let layers = config.num_layers;
    let scenario = build_scenario(
        layers,
        DEFAULT_TICKETS.to_vec(),
        DEFAULT_PREFIX_SPLITS.to_vec(),
    )?;
    let domains = scenario.domains;
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("create {} failed: {e}", out_dir.display()))?;
    let weights = ModelWeights::from_dir(model_dir, device)
        .map_err(|e| format!("load weights failed: {e}"))?;
    let model = LlamaModel::from_weights(config, &weights, device, domains)
        .map_err(|e| format!("build model failed: {e}"))?;
    let mut backend = TchWorkerBackend::from_model(model, device, domain);

    // One full-duplex TCP connection per layer: the server accepts 24 dials,
    // the client dials 24 times, both in layer order.
    let mut shared = Vec::with_capacity(layers);
    match domain {
        1 => {
            let listener =
                TcpListener::bind(bind).map_err(|e| format!("bind {bind} failed: {e}"))?;
            println!(
                "[route-b smoke] domain 1 listening on {bind}, waiting for {layers} connections"
            );
            for layer_idx in 0..layers {
                let (stream, _) = listener
                    .accept()
                    .map_err(|e| format!("accept layer {layer_idx} failed: {e}"))?;
                shared.push(SharedTcpTransport(Arc::new(Mutex::new(
                    TcpKvTransport::new(stream, device)?,
                ))));
            }
        }
        0 => {
            let peer =
                peer.ok_or_else(|| "client mode requires --peer <host>:<port>".to_string())?;
            for layer_idx in 0..layers {
                let stream = TcpStream::connect(peer)
                    .map_err(|e| format!("connect layer {layer_idx} to {peer} failed: {e}"))?;
                shared.push(SharedTcpTransport(Arc::new(Mutex::new(
                    TcpKvTransport::new(stream, device)?,
                ))));
            }
            println!("[route-b smoke] domain 0 connected {layers} connections to {peer}");
        }
        _ => return Err(format!("invalid domain {domain}")),
    }
    let transports = shared
        .iter()
        .map(|transport| Box::new(transport.clone()) as Box<dyn KvTransport>)
        .collect::<Vec<_>>();
    backend.setup_kv_transports(transports);

    // ===== Phase 1: prefill (KV ring exchange happens inside the backend). =====
    let (prefix_start, prefix_end) = scenario.prefix_ranges[domain];
    let chunk = &PROMPT[prefix_start..prefix_end];
    let positions = (prefix_start as i64..prefix_end as i64).collect::<Vec<_>>();
    let (logits_vec, global_len) = backend
        .prefill_request_with_reservation(
            REQUEST_ID,
            chunk,
            prefix_start,
            Some(&positions),
            Some(&scenario.capacities[domain]),
        )
        .map_err(|e| format!("prefill failed: {e}"))?;
    if global_len != prefix_end {
        return Err(format!(
            "domain {domain} global_seq_len {global_len} != prefix end {prefix_end}"
        ));
    }
    let prefill_logits = Tensor::from_slice(&logits_vec);
    let prefill_argmax = prefill_logits.argmax(-1, false).int64_value(&[]);
    let mut decode_token = None;
    if domain == scenario.decode_starter {
        decode_token = Some(prefill_argmax);
        write_tensor_f32le(&out_dir.join("prefill_last_logits.f32le"), &prefill_logits)?;
    }
    println!("[route-b smoke] domain {domain} prefill done: global_len={global_len}");

    let storage_before = storage_snapshot(&backend, layers)?;

    // ===== Phase 2: decode (24-layer ping-pong, starter = 1). =====
    let mut handoffs = 0_usize;
    let mut current_starter = scenario.decode_starter;
    let decode_position_ids = Tensor::from_slice(&[scenario.decode_position])
        .unsqueeze(0)
        .to_device(device);
    let mut hidden_states: Option<Tensor> = None;
    for (layer_idx, shared_transport) in shared.iter().enumerate() {
        if current_starter == domain {
            if layer_idx == 0 {
                let token = decode_token
                    .ok_or_else(|| "decode starter requires the prefill argmax".to_string())?;
                hidden_states = Some(embed_tokens(&backend.model, &[token], device));
            }
            let packet = LayerPacket::start(
                &mut backend.model.layers[layer_idx],
                hidden_states.as_ref().unwrap(),
                &decode_position_ids,
                domain,
                scenario.decode_assignees[layer_idx],
                domains,
            )
            .map_err(|e| format!("decode layer {layer_idx} start failed: {e}"))?;
            let outcome = {
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    return Err(format!(
                        "domain {domain} layer {layer_idx} did not use reserved positioned KV"
                    ));
                };
                process_layer_packet_with_reserved_history(
                    &mut backend.model.layers[layer_idx],
                    packet,
                    shard,
                )
                .map_err(|e| format!("decode layer {layer_idx} local step failed: {e}"))?
            };
            let LayerStepOutcome::Forward(next_packet) = outcome else {
                return Err(format!(
                    "decode layer {layer_idx} starter finished a 2-domain route"
                ));
            };
            let wire = next_packet
                .into_self_driving_packet(layer_idx)
                .map_err(|e| format!("decode layer {layer_idx} wire encode failed: {e}"))?;
            shared_transport
                .lock()
                .send_self_driving_packet(&wire)
                .map_err(|e| format!("decode layer {layer_idx} send failed: {e}"))?;
            handoffs += 1;
        } else {
            let wire = shared_transport
                .lock()
                .recv_self_driving_packet()
                .map_err(|e| format!("decode layer {layer_idx} recv failed: {e}"))?
                .ok_or_else(|| format!("decode layer {layer_idx} peer closed the connection"))?;
            if wire.layer_idx != layer_idx {
                return Err(format!(
                    "decode layer {layer_idx} received packet for layer {}",
                    wire.layer_idx
                ));
            }
            let packet = LayerPacket::from_self_driving_packet(wire)
                .map_err(|e| format!("decode layer {layer_idx} wire decode failed: {e}"))?;
            let outcome = {
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    return Err(format!(
                        "domain {domain} layer {layer_idx} did not use reserved positioned KV"
                    ));
                };
                process_layer_packet_with_reserved_history(
                    &mut backend.model.layers[layer_idx],
                    packet,
                    shard,
                )
                .map_err(|e| format!("decode layer {layer_idx} remote step failed: {e}"))?
            };
            let LayerStepOutcome::Finished {
                hidden_states: next_hidden,
                ..
            } = outcome
            else {
                return Err(format!(
                    "decode layer {layer_idx} finisher forwarded a 2-domain route"
                ));
            };
            hidden_states = Some(next_hidden);
        }
        current_starter = layer_finisher(current_starter, domains);
    }
    let decode_finisher = current_starter;
    backend
        .request_contexts
        .get_mut(&REQUEST_ID)
        .unwrap()
        .global_seq_len = scenario.decode_position as usize + 1;
    let mut decode_argmax = None;
    if decode_finisher == domain {
        let logits = project_final_logits(&backend.model, hidden_states.as_ref().unwrap());
        let logits = logits.squeeze();
        decode_argmax = Some(logits.argmax(-1, false).int64_value(&[]));
        write_tensor_f32le(&out_dir.join("decode_logits.f32le"), &logits)?;
    }
    println!("[route-b smoke] domain {domain} decode done: finisher={decode_finisher}");

    // ===== Phase 3: stationary continuation (24-layer ping-pong, m=4). =====
    current_starter = decode_finisher;
    let continuation_position_ids = Tensor::from_slice(&scenario.continuation_positions)
        .unsqueeze(0)
        .to_device(device);
    hidden_states = if current_starter == domain {
        Some(embed_tokens(&backend.model, &CONTINUATION_PROMPT, device))
    } else {
        None
    };
    for (layer_idx, shared_transport) in shared.iter().enumerate() {
        if current_starter == domain {
            let packet = LayerPacket::start(
                &mut backend.model.layers[layer_idx],
                hidden_states.as_ref().unwrap(),
                &continuation_position_ids,
                domain,
                domain,
                domains,
            )
            .map_err(|e| format!("continuation layer {layer_idx} start failed: {e}"))?;
            let outcome = {
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    return Err(format!(
                        "domain {domain} layer {layer_idx} did not use reserved positioned KV"
                    ));
                };
                process_layer_packet_with_reserved_history_for_positions(
                    &mut backend.model.layers[layer_idx],
                    packet,
                    shard,
                    &scenario.continuation_offsets_by_domain[domain],
                )
                .map_err(|e| format!("continuation layer {layer_idx} local step failed: {e}"))?
            };
            let LayerStepOutcome::Forward(next_packet) = outcome else {
                return Err(format!(
                    "continuation layer {layer_idx} starter finished a 2-domain route"
                ));
            };
            let wire = next_packet
                .into_self_driving_packet(layer_idx)
                .map_err(|e| format!("continuation layer {layer_idx} wire encode failed: {e}"))?;
            shared_transport
                .lock()
                .send_self_driving_packet(&wire)
                .map_err(|e| format!("continuation layer {layer_idx} send failed: {e}"))?;
            handoffs += 1;
        } else {
            let wire = shared_transport
                .lock()
                .recv_self_driving_packet()
                .map_err(|e| format!("continuation layer {layer_idx} recv failed: {e}"))?
                .ok_or_else(|| {
                    format!("continuation layer {layer_idx} peer closed the connection")
                })?;
            if wire.layer_idx != layer_idx {
                return Err(format!(
                    "continuation layer {layer_idx} received packet for layer {}",
                    wire.layer_idx
                ));
            }
            let packet = LayerPacket::from_self_driving_packet(wire)
                .map_err(|e| format!("continuation layer {layer_idx} wire decode failed: {e}"))?;
            let outcome = {
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    return Err(format!(
                        "domain {domain} layer {layer_idx} did not use reserved positioned KV"
                    ));
                };
                process_layer_packet_with_reserved_history_for_positions(
                    &mut backend.model.layers[layer_idx],
                    packet,
                    shard,
                    &scenario.continuation_offsets_by_domain[domain],
                )
                .map_err(|e| format!("continuation layer {layer_idx} remote step failed: {e}"))?
            };
            let LayerStepOutcome::Finished {
                hidden_states: next_hidden,
                ..
            } = outcome
            else {
                return Err(format!(
                    "continuation layer {layer_idx} finisher forwarded a 2-domain route"
                ));
            };
            hidden_states = Some(next_hidden);
        }
        current_starter = layer_finisher(current_starter, domains);
    }
    let continuation_finisher = current_starter;
    let mut continuation_argmax = None;
    if continuation_finisher == domain {
        let logits = project_final_logits(&backend.model, hidden_states.as_ref().unwrap());
        let last = logits
            .select(1, CONTINUATION_PROMPT.len() as i64 - 1)
            .squeeze();
        continuation_argmax = Some(last.argmax(-1, false).int64_value(&[]));
        write_tensor_f32le(&out_dir.join("continuation_last_logits.f32le"), &last)?;
    }
    println!("[route-b smoke] domain {domain} continuation done: finisher={continuation_finisher}");

    // ===== Phase 4: per-domain tail checks. =====
    let kv_total = verify_domain_kv_state(&backend, &scenario, domain, &storage_before)?;

    let meta = serde_json::json!({
        "mode": mode,
        "device": format!("{device:?}"),
        "domains": [domain],
        "decode_token": decode_token,
        "prefill_argmax": if domain == scenario.decode_starter { Some(prefill_argmax) } else { None },
        "decode_argmax": decode_argmax,
        "continuation_argmax": continuation_argmax,
        "domain_kv_totals": { domain.to_string(): kv_total },
        "handoffs": handoffs,
        "checks": {
            "num_layers": layers == EXPECTED_NUM_LAYERS,
            "kv_total_matches_expected": kv_total == expected_domain_kv_total(&scenario, domain, layers),
            "storage_stable": true,
            "committed_eq_reserved": true,
            "positions_match_frozen_offsets": true,
        },
    });
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write meta.json failed: {e}"))?;
    println!(
        "[route-b smoke] domain {domain} done: kv_total={kv_total} handoffs={handoffs} out={}",
        out_dir.display()
    );
    Ok(())
}

/// In-process decode identical to the test helper
/// `run_two_backend_reserved_local_decode`, generalized to N domains and
/// counting packet handoffs so the golden meta matches the ring send totals.
fn local_decode(
    backends: &mut [TchWorkerBackend],
    scenario: &Scenario,
    decode_token: i64,
    starter: usize,
    device: Device,
) -> (Tensor, usize, usize) {
    let domains = scenario.domains;
    let input_ids = Tensor::from_slice(&[decode_token])
        .unsqueeze(0)
        .to_device(device);
    let mut hidden_states = Tensor::embedding(
        &backends[starter].model.embedding,
        &input_ids,
        -1,
        false,
        false,
    );
    let position_ids = Tensor::from_slice(&[scenario.decode_position])
        .unsqueeze(0)
        .to_device(device);
    let mut current_starter = starter;
    let mut handoffs = 0_usize;

    for (layer_idx, &assignee) in scenario.decode_assignees.iter().enumerate() {
        let mut packet = Some(
            LayerPacket::start(
                &mut backends[current_starter].model.layers[layer_idx],
                &hidden_states,
                &position_ids,
                current_starter,
                assignee,
                domains,
            )
            .unwrap(),
        );
        let mut next_hidden = None;
        for visit_index in 0..domains {
            let domain = (current_starter + visit_index) % domains;
            let outcome = {
                let backend = &mut backends[domain];
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                };
                process_layer_packet_with_reserved_history(
                    &mut backend.model.layers[layer_idx],
                    packet.take().unwrap(),
                    shard,
                )
                .unwrap()
            };
            match outcome {
                LayerStepOutcome::Forward(next_packet) => {
                    handoffs += 1;
                    packet = Some(next_packet);
                }
                LayerStepOutcome::Finished { hidden_states, .. } => {
                    assert_eq!(visit_index + 1, domains);
                    next_hidden = Some(hidden_states);
                }
            }
        }
        hidden_states = next_hidden.expect("the final worker must finish the layer");
        current_starter = layer_finisher(current_starter, domains);
    }

    let logits = project_final_logits(&backends[current_starter].model, &hidden_states);
    for backend in backends.iter_mut() {
        let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
        context.global_seq_len = scenario.decode_position as usize + 1;
    }
    (logits, current_starter, handoffs)
}

/// Single-process golden: exact replica of
/// `real_qwen_two_worker_stationary_continuation_matches_reference`
/// generalized to N in-process backends on a mock ring, including every
/// assertion, plus dump files.
fn run_local(
    device: Device,
    tickets: Vec<u64>,
    prefix_splits: Vec<usize>,
    model_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    let config = ModelConfig::from_file(model_dir.join("config.json"))
        .map_err(|e| format!("load config failed: {e}"))?;
    assert_eq!(config.num_layers, EXPECTED_NUM_LAYERS);
    let layers = config.num_layers;
    let scenario = build_scenario(layers, tickets, prefix_splits)?;
    let domains = scenario.domains;
    let weights = ModelWeights::from_dir(model_dir, device)
        .map_err(|e| format!("load weights failed: {e}"))?;

    let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
    let mut backends = (0..domains)
        .map(|domain| {
            TchWorkerBackend::from_model(
                LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap(),
                device,
                domain,
            )
        })
        .collect::<Vec<_>>();
    // Mock ring per layer: backend d sends into d+1's inbox, receives from its
    // own inbox (fed by d-1) — identical direction semantics to RingTcpTransport.
    let mut per_backend = (0..domains)
        .map(|_| Vec::with_capacity(layers))
        .collect::<Vec<_>>();
    for _ in 0..layers {
        for (domain, endpoint) in LinkedMockKvTransport::create_ring(domains)
            .into_iter()
            .enumerate()
        {
            per_backend[domain].push(Box::new(endpoint) as Box<dyn KvTransport>);
        }
    }
    for (domain, transports) in per_backend.into_iter().enumerate() {
        backends[domain].setup_kv_transports(transports);
    }

    let mut reference_caches = reference.create_kv_caches();
    let reference_prefill_logits = reference
        .forward(
            &Tensor::from_slice(&PROMPT).unsqueeze(0).to_device(device),
            &mut reference_caches,
        )
        .unwrap()
        .select(1, PROMPT.len() as i64 - 1)
        .squeeze();

    // Sequential per-backend prefill; mock inboxes buffer the ring traffic and
    // causality lets early domains complete without future peer KV.
    let mut distributed_prefill_logits = None;
    for (domain, backend) in backends.iter_mut().enumerate() {
        let (prefix_start, prefix_end) = scenario.prefix_ranges[domain];
        let positions = (prefix_start as i64..prefix_end as i64).collect::<Vec<_>>();
        let (logits, global_len) = backend
            .prefill_request_with_reservation(
                REQUEST_ID,
                &PROMPT[prefix_start..prefix_end],
                prefix_start,
                Some(&positions),
                Some(&scenario.capacities[domain]),
            )
            .unwrap();
        assert_eq!(global_len, prefix_end);
        if domain == scenario.decode_starter {
            distributed_prefill_logits = Some(logits);
        }
    }
    let distributed_prefill_logits =
        Tensor::from_slice(&distributed_prefill_logits.expect("decode starter produced logits"));
    let (decode_token, reference_prefill_token) = assert_argmax_tie_aware(
        &distributed_prefill_logits,
        &reference_prefill_logits,
        "prefill",
    );

    let storage_before = backends
        .iter()
        .map(|backend| storage_snapshot(backend, layers).unwrap())
        .collect::<Vec<_>>();

    let (distributed_decode_logits, decode_finisher, decode_handoffs) = local_decode(
        &mut backends,
        &scenario,
        decode_token,
        scenario.decode_starter,
        device,
    );
    assert_eq!(decode_handoffs, layers * (domains - 1));
    let reference_decode_logits = reference
        .forward(
            &Tensor::from_slice(&[decode_token])
                .unsqueeze(0)
                .to_device(device),
            &mut reference_caches,
        )
        .unwrap()
        .squeeze();
    let distributed_decode_last = distributed_decode_logits.squeeze();
    let decode_max_diff = (&distributed_decode_last - &reference_decode_logits)
        .abs()
        .max()
        .double_value(&[]);
    let (decode_argmax, reference_decode_token) =
        assert_argmax_tie_aware(&distributed_decode_last, &reference_decode_logits, "decode");
    println!("local pre-continuation decode: max_diff={decode_max_diff:.6}");

    // Route B stationary continuation: historical KV never enters the packet;
    // each worker projects and appends only its own position offsets.
    let mut current_starter = decode_finisher;
    let continuation_input = Tensor::from_slice(&CONTINUATION_PROMPT)
        .unsqueeze(0)
        .to_device(device);
    let mut hidden_states = Tensor::embedding(
        &backends[current_starter].model.embedding,
        &continuation_input,
        -1,
        false,
        false,
    );
    let position_ids = Tensor::from_slice(&scenario.continuation_positions)
        .unsqueeze(0)
        .to_device(device);
    let mut handoffs = 0_usize;
    for layer_idx in 0..layers {
        // Position ownership comes from the frozen offsets, not the legacy scalar.
        let mut packet = Some(
            LayerPacket::start(
                &mut backends[current_starter].model.layers[layer_idx],
                &hidden_states,
                &position_ids,
                current_starter,
                current_starter,
                domains,
            )
            .unwrap(),
        );
        let mut next_hidden = None;
        for visit_index in 0..domains {
            let domain = (current_starter + visit_index) % domains;
            let outcome = {
                let backend = &mut backends[domain];
                let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
                let Some(KvCacheImpl::ReservedPositioned(shard)) =
                    &mut context.kv_caches[layer_idx]
                else {
                    panic!("worker {domain} layer {layer_idx} did not use reserved KV");
                };
                process_layer_packet_with_reserved_history_for_positions(
                    &mut backend.model.layers[layer_idx],
                    packet.take().unwrap(),
                    shard,
                    &scenario.continuation_offsets_by_domain[domain],
                )
                .unwrap()
            };
            match outcome {
                LayerStepOutcome::Forward(next_packet) => {
                    handoffs += 1;
                    packet = Some(next_packet);
                }
                LayerStepOutcome::Finished { hidden_states, .. } => {
                    assert_eq!(visit_index + 1, domains);
                    next_hidden = Some(hidden_states);
                }
            }
        }
        hidden_states = next_hidden.expect("the final worker must finish the layer");
        current_starter = layer_finisher(current_starter, domains);
    }
    assert_eq!(handoffs, layers * (domains - 1));

    let continuation_logits =
        project_final_logits(&backends[current_starter].model, &hidden_states);
    let distributed_continuation_last = continuation_logits
        .select(1, CONTINUATION_PROMPT.len() as i64 - 1)
        .squeeze();

    reference.set_prefill_position_ids(&scenario.continuation_positions, device);
    let reference_continuation_logits = reference
        .forward(
            &Tensor::from_slice(&CONTINUATION_PROMPT)
                .unsqueeze(0)
                .to_device(device),
            &mut reference_caches,
        )
        .unwrap()
        .select(1, CONTINUATION_PROMPT.len() as i64 - 1)
        .squeeze();

    let max_diff = (&distributed_continuation_last - &reference_continuation_logits)
        .abs()
        .max()
        .double_value(&[]);
    let mean_diff = (&distributed_continuation_last - &reference_continuation_logits)
        .abs()
        .mean(Kind::Float)
        .double_value(&[]);
    let (continuation_argmax, reference_continuation_token) = assert_argmax_tie_aware(
        &distributed_continuation_last,
        &reference_continuation_logits,
        "continuation",
    );
    println!(
        "local stationary continuation: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, tokens={continuation_argmax}/{reference_continuation_token}"
    );
    assert!(
        mean_diff < 0.1,
        "continuation mean logits diff: {mean_diff}"
    );
    assert!(max_diff < 0.75, "continuation max logits diff: {max_diff}");

    let mut domain_totals = Vec::with_capacity(domains);
    for (domain, backend) in backends.iter().enumerate() {
        let total =
            verify_domain_kv_state(backend, &scenario, domain, &storage_before[domain]).unwrap();
        domain_totals.push(total);
    }

    let last_position = (PROMPT.len() + CONTINUATION_PROMPT.len()) as i64;
    for layer_idx in 0..layers {
        let mut positions = Vec::new();
        for backend in &backends {
            let context = backend.request_contexts.get(&REQUEST_ID).unwrap();
            let Some(KvCacheImpl::ReservedPositioned(shard)) = &context.kv_caches[layer_idx] else {
                panic!("layer {layer_idx} did not use reserved KV");
            };
            positions.extend_from_slice(shard.positions());
        }
        positions.sort_unstable();
        assert_eq!(positions, (0_i64..=last_position).collect::<Vec<_>>());
    }

    // ===== Dump. =====
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("create {} failed: {e}", out_dir.display()))?;
    write_tensor_f32le(
        &out_dir.join("prefill_last_logits.f32le"),
        &distributed_prefill_logits,
    )?;
    write_tensor_f32le(
        &out_dir.join("decode_logits.f32le"),
        &distributed_decode_last,
    )?;
    write_tensor_f32le(
        &out_dir.join("continuation_last_logits.f32le"),
        &distributed_continuation_last,
    )?;
    let domain_kv_totals = domain_totals
        .iter()
        .enumerate()
        .map(|(domain, total)| (domain.to_string(), serde_json::Value::from(*total)))
        .collect::<serde_json::Map<_, _>>();
    let meta = serde_json::json!({
        "mode": "local",
        "device": format!("{device:?}"),
        "domains": (0..domains).collect::<Vec<_>>(),
        "tickets": scenario.tickets,
        "prefix_splits": scenario.prefix_splits,
        "decode_starter": scenario.decode_starter,
        "decode_finisher": decode_finisher,
        "continuation_finisher": current_starter,
        "decode_token": decode_token,
        "prefill_argmax": decode_token,
        "decode_argmax": decode_argmax,
        "continuation_argmax": continuation_argmax,
        "reference_prefill_argmax": reference_prefill_token,
        "reference_decode_argmax": reference_decode_token,
        "reference_continuation_argmax": reference_continuation_token,
        "domain_kv_totals": domain_kv_totals,
        "handoffs": decode_handoffs + handoffs,
        "checks": {
            "num_layers": layers == EXPECTED_NUM_LAYERS,
            "prefill_argmax_exact": decode_token == reference_prefill_token,
            "decode_argmax_exact": decode_argmax == reference_decode_token,
            "continuation_argmax_exact": continuation_argmax == reference_continuation_token,
            "decode_max_diff": decode_max_diff,
            "continuation_mean_diff": mean_diff,
            "continuation_mean_diff_lt_0_1": mean_diff < 0.1,
            "continuation_max_diff": max_diff,
            "continuation_max_diff_lt_0_75": max_diff < 0.75,
            "domain_kv_totals_match_scenario": domain_totals
                .iter()
                .enumerate()
                .all(|(domain, &total)| total == expected_domain_kv_total(&scenario, domain, layers)),
            "position_union_covers_all": true,
            "storage_stable": true,
            "committed_eq_reserved": true,
        },
    });
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write meta.json failed: {e}"))?;
    println!(
        "[route-b smoke] local golden done: domains={domains} decode_token={decode_token} continuation_argmax={continuation_argmax} domain_totals={domain_totals:?} out={}",
        out_dir.display()
    );
    Ok(())
}

struct Cli {
    mode: String,
    device: Device,
    domain: Option<usize>,
    domains: Option<usize>,
    tickets: Option<Vec<u64>>,
    prefix_splits: Option<Vec<usize>>,
    transport: TransportKind,
    bind: Option<String>,
    peer: Option<String>,
    out: PathBuf,
    model_dir: PathBuf,
}

fn parse_cli() -> Result<Cli, String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(mode) = args.first().cloned() else {
        return Err(
            "usage: route_b_cross_node_smoke <local|server|client|node> [--domain D] [--domains N] [--tickets t0,t1,...] [--prefix-splits s0,s1,...] [--device cpu|mps|cuda:N] [--bind host:port] [--peer host:port] --out <dir> [--model-dir <dir>]"
                .to_string(),
        );
    };
    if !matches!(mode.as_str(), "local" | "server" | "client" | "node") {
        return Err(format!(
            "invalid mode {mode}: expected local|server|client|node"
        ));
    }
    let mut device = Device::Cpu;
    let mut domain = None;
    let mut domains = None;
    let mut tickets = None;
    let mut prefix_splits = None;
    let mut transport = TransportKind::Tcp;
    let mut bind = None;
    let mut peer = None;
    let mut out = None;
    let mut model_dir = default_model_dir();
    let mut index = 1;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("flag {flag} requires a value"))?;
        match flag {
            "--device" => device = parse_device(value)?,
            "--domain" => {
                domain = Some(
                    value
                        .parse::<usize>()
                        .map_err(|e| format!("invalid --domain {value}: {e}"))?,
                )
            }
            "--domains" => {
                domains = Some(
                    value
                        .parse::<usize>()
                        .map_err(|e| format!("invalid --domains {value}: {e}"))?,
                )
            }
            "--tickets" => tickets = Some(parse_u64_list(value, "--tickets")?),
            "--prefix-splits" => prefix_splits = Some(parse_usize_list(value, "--prefix-splits")?),
            "--transport" => transport = TransportKind::parse(value)?,
            "--bind" => bind = Some(value.clone()),
            "--peer" => peer = Some(value.clone()),
            "--out" => out = Some(PathBuf::from(value)),
            "--model-dir" => model_dir = PathBuf::from(value),
            _ => return Err(format!("unknown flag {flag}")),
        }
        index += 2;
    }
    let out = out.ok_or_else(|| "missing required --out <dir>".to_string())?;
    Ok(Cli {
        mode,
        device,
        domain,
        domains,
        tickets,
        prefix_splits,
        transport,
        bind,
        peer,
        out,
        model_dir,
    })
}

fn main() -> Result<(), String> {
    let cli = parse_cli()?;
    match cli.mode.as_str() {
        "local" => {
            let domains = cli.domains.unwrap_or(DEFAULT_TICKETS.len());
            let tickets = cli.tickets.unwrap_or_else(|| DEFAULT_TICKETS.to_vec());
            let prefix_splits = cli
                .prefix_splits
                .unwrap_or_else(|| DEFAULT_PREFIX_SPLITS.to_vec());
            if tickets.len() != domains || prefix_splits.len() != domains {
                return Err(format!(
                    "--tickets (len {}) and --prefix-splits (len {}) must match --domains {domains}",
                    tickets.len(),
                    prefix_splits.len()
                ));
            }
            run_local(cli.device, tickets, prefix_splits, &cli.model_dir, &cli.out)
        }
        "server" => run_pair_node(
            "server",
            1,
            cli.device,
            cli.bind.as_deref().unwrap_or("0.0.0.0:29511"),
            None,
            &cli.model_dir,
            &cli.out,
        ),
        "client" => run_pair_node(
            "client",
            0,
            cli.device,
            cli.bind.as_deref().unwrap_or("0.0.0.0:29510"),
            cli.peer.as_deref(),
            &cli.model_dir,
            &cli.out,
        ),
        "node" => {
            let domain = cli
                .domain
                .ok_or_else(|| "node mode requires --domain D".to_string())?;
            let peer = cli
                .peer
                .ok_or_else(|| "node mode requires --peer <host>:<port> (successor)".to_string())?;
            let bind = cli
                .bind
                .ok_or_else(|| "node mode requires --bind <host>:<port>".to_string())?;
            let domains = cli.domains.unwrap_or(DEFAULT_TICKETS.len());
            let tickets = cli.tickets.unwrap_or_else(|| DEFAULT_TICKETS.to_vec());
            let prefix_splits = cli
                .prefix_splits
                .unwrap_or_else(|| DEFAULT_PREFIX_SPLITS.to_vec());
            if tickets.len() != domains || prefix_splits.len() != domains {
                return Err(format!(
                    "--tickets (len {}) and --prefix-splits (len {}) must match --domains {domains}",
                    tickets.len(),
                    prefix_splits.len()
                ));
            }
            run_ring_node(
                domain,
                cli.device,
                &bind,
                &peer,
                tickets,
                prefix_splits,
                cli.transport,
                &cli.model_dir,
                &cli.out,
            )
        }
        _ => unreachable!("mode validated by parse_cli"),
    }
}
