//! Experimental cross-node smoke for the route-B stationary continuation path.
//!
//! Replicates the verified single-process test
//! `real_qwen_two_worker_stationary_continuation_matches_reference`
//! (rust/src/worker_sdk/tch_backend.rs) as:
//! - `local`:  single-process golden that reproduces the test scenario and all
//!   of its assertions, then dumps logits + meta to `--out`.
//! - `server`: domain-1 worker over real TCP (`--bind 0.0.0.0:29511`).
//! - `client`: domain-0 worker over real TCP (`--peer <host>:29511`).
//!
//! Both node modes hardcode the same scenario constants (no coordinator):
//! model Qwen2-0.5B BF16, request_id=75, capacity_tickets=[1,3],
//! prompt=[151644,9707,0,16] split [1,3], one decode token at position 4,
//! continuation [11,13,17,19] at positions [5,6,7,8].
//!
//! One full-duplex TCP connection per layer carries both the prefill KV ring
//! blocks and the decode/continuation self-driving `LayerPacket`s.

use hcp_ringattn_rust::{
    process_layer_packet_with_reserved_history,
    process_layer_packet_with_reserved_history_for_positions, project_final_logits,
    FrozenKvAssigneeSchedule, KvBlock, KvCacheImpl, KvTransport, LayerPacket, LayerStepOutcome,
    LinkedMockKvTransport, LlamaModel, ModelConfig, ModelWeights, RingPacket, SelfDrivingPacket,
    TchWorkerBackend, TcpKvTransport, WorkerBackend,
};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tch::{Device, Kind, Tensor};

const REQUEST_ID: u64 = 75;
const CAPACITY_TICKETS: [u64; 2] = [1, 3];
const DOMAINS: usize = 2;
const PROMPT: [i64; 4] = [151644, 9707, 0, 16];
const PREFIX_SPLITS: [usize; 2] = [1, 3];
const CONTINUATION_PROMPT: [i64; 4] = [11, 13, 17, 19];
const DECODE_STARTER: usize = 1;
const EXPECTED_NUM_LAYERS: usize = 24;
const EXPECTED_DOMAIN_KV_TOTALS: [usize; 2] = [54, 162];

/// Scenario constants derived deterministically on both ends from the frozen
/// schedules, exactly like the single-process test.
struct Scenario {
    decode_position: i64,
    continuation_positions: Vec<i64>,
    decode_assignees: Vec<usize>,
    continuation_offsets_by_domain: Vec<Vec<usize>>,
    capacities: Vec<Vec<usize>>,
}

fn build_scenario(layers: usize) -> Result<Scenario, String> {
    let decode_schedule = FrozenKvAssigneeSchedule::new(&CAPACITY_TICKETS, REQUEST_ID, layers)?;
    if decode_schedule.counts() != [6, 18] {
        return Err(format!(
            "decode schedule counts {:?} != [6, 18]",
            decode_schedule.counts()
        ));
    }
    let decode_assignees = (0..layers)
        .map(|layer_idx| decode_schedule.assignee_for(0, layer_idx, layers).unwrap())
        .collect::<Vec<_>>();

    let continuation_len = CONTINUATION_PROMPT.len();
    let continuation_schedule =
        FrozenKvAssigneeSchedule::new(&CAPACITY_TICKETS, REQUEST_ID, continuation_len)?;
    if continuation_schedule.counts() != PREFIX_SPLITS {
        return Err(format!(
            "continuation schedule counts {:?} != {:?}",
            continuation_schedule.counts(),
            PREFIX_SPLITS
        ));
    }
    let mut continuation_offsets_by_domain = vec![Vec::new(); DOMAINS];
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
    let capacities = (0..DOMAINS)
        .map(|domain| {
            (0..layers)
                .map(|layer_idx| {
                    PREFIX_SPLITS[domain]
                        + usize::from(decode_assignees[layer_idx] == domain)
                        + continuation_offsets_by_domain[domain].len()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(Scenario {
        decode_position,
        continuation_positions,
        decode_assignees,
        continuation_offsets_by_domain,
        capacities,
    })
}

/// Expected committed positions of one domain's shard at one layer, in commit
/// order: prefix positions, then the decode position (if assignee), then the
/// domain's frozen continuation offsets.
fn expected_positions(scenario: &Scenario, domain: usize, layer_idx: usize) -> Vec<i64> {
    let mut positions: Vec<i64> = if domain == 0 { vec![0] } else { vec![1, 2, 3] };
    if scenario.decode_assignees[layer_idx] == domain {
        positions.push(scenario.decode_position);
    }
    for &offset in &scenario.continuation_offsets_by_domain[domain] {
        positions.push(scenario.decode_position + 1 + offset as i64);
    }
    positions
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
/// match the frozen offsets, and the expected 24-layer KV total.
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
    if kv_total != EXPECTED_DOMAIN_KV_TOTALS[domain] {
        return Err(format!(
            "domain {domain} KV total {kv_total} != {}",
            EXPECTED_DOMAIN_KV_TOTALS[domain]
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

fn run_node(
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
    let scenario = build_scenario(layers)?;
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("create {} failed: {e}", out_dir.display()))?;
    let weights = ModelWeights::from_dir(model_dir, device)
        .map_err(|e| format!("load weights failed: {e}"))?;
    let model = LlamaModel::from_weights(config, &weights, device, DOMAINS)
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
    let (chunk, seq_offset, positions): (&[i64], usize, Vec<i64>) = if domain == 0 {
        (&PROMPT[..1], 0, vec![0])
    } else {
        (&PROMPT[1..], 1, vec![1, 2, 3])
    };
    let (logits_vec, global_len) = backend
        .prefill_request_with_reservation(
            REQUEST_ID,
            chunk,
            seq_offset,
            Some(&positions),
            Some(&scenario.capacities[domain]),
        )
        .map_err(|e| format!("prefill failed: {e}"))?;
    let expected_global_len = if domain == 1 { PROMPT.len() } else { 1 };
    if global_len != expected_global_len {
        return Err(format!(
            "domain {domain} global_seq_len {global_len} != {expected_global_len}"
        ));
    }
    let prefill_logits = Tensor::from_slice(&logits_vec);
    let prefill_argmax = prefill_logits.argmax(-1, false).int64_value(&[]);
    let mut decode_token = None;
    if domain == 1 {
        decode_token = Some(prefill_argmax);
        write_tensor_f32le(&out_dir.join("prefill_last_logits.f32le"), &prefill_logits)?;
    }
    println!("[route-b smoke] domain {domain} prefill done: global_len={global_len}");

    let storage_before = storage_snapshot(&backend, layers)?;

    // ===== Phase 2: decode (24-layer ping-pong, starter = 1). =====
    let mut handoffs = 0_usize;
    let mut current_starter = DECODE_STARTER;
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
                DOMAINS,
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
        current_starter = (current_starter + 1) % DOMAINS;
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
                DOMAINS,
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
        current_starter = (current_starter + 1) % DOMAINS;
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
        "prefill_argmax": if domain == 1 { Some(prefill_argmax) } else { None },
        "decode_argmax": decode_argmax,
        "continuation_argmax": continuation_argmax,
        "domain_kv_totals": { domain.to_string(): kv_total },
        "handoffs": handoffs,
        "checks": {
            "num_layers": layers == EXPECTED_NUM_LAYERS,
            "kv_total_matches_expected": kv_total == EXPECTED_DOMAIN_KV_TOTALS[domain],
            "storage_stable": true,
            "committed_eq_reserved": true,
            "positions_match_frozen_offsets": true,
        },
    });
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("create {} failed: {e}", out_dir.display()))?;
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
/// `run_two_backend_reserved_local_decode`, additionally counting packet
/// handoffs so the golden meta matches the two-process send totals.
fn local_decode(
    backends: &mut [TchWorkerBackend],
    scenario: &Scenario,
    decode_token: i64,
    starter: usize,
    device: Device,
) -> (Tensor, usize, usize) {
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
                DOMAINS,
            )
            .unwrap(),
        );
        let mut next_hidden = None;
        for visit_index in 0..DOMAINS {
            let domain = (current_starter + visit_index) % DOMAINS;
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
                    assert_eq!(visit_index + 1, DOMAINS);
                    next_hidden = Some(hidden_states);
                }
            }
        }
        hidden_states = next_hidden.expect("the final worker must finish the layer");
        current_starter = (current_starter + DOMAINS - 1) % DOMAINS;
    }

    let logits = project_final_logits(&backends[current_starter].model, &hidden_states);
    for backend in backends.iter_mut() {
        let context = backend.request_contexts.get_mut(&REQUEST_ID).unwrap();
        context.global_seq_len = scenario.decode_position as usize + 1;
    }
    (logits, current_starter, handoffs)
}

/// Single-process golden: exact replica of
/// `real_qwen_two_worker_stationary_continuation_matches_reference`,
/// including every assertion, plus dump files.
fn run_local(device: Device, model_dir: &Path, out_dir: &Path) -> Result<(), String> {
    let config = ModelConfig::from_file(model_dir.join("config.json"))
        .map_err(|e| format!("load config failed: {e}"))?;
    assert_eq!(config.num_layers, EXPECTED_NUM_LAYERS);
    let layers = config.num_layers;
    let scenario = build_scenario(layers)?;
    let weights = ModelWeights::from_dir(model_dir, device)
        .map_err(|e| format!("load weights failed: {e}"))?;

    let mut reference = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
    let mut backend0 = TchWorkerBackend::from_model(
        LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
        device,
        0,
    );
    let mut backend1 = TchWorkerBackend::from_model(
        LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap(),
        device,
        1,
    );
    let mut transports0: Vec<Box<dyn KvTransport>> = Vec::with_capacity(layers);
    let mut transports1: Vec<Box<dyn KvTransport>> = Vec::with_capacity(layers);
    for _ in 0..layers {
        let (transport0, transport1) = LinkedMockKvTransport::create_pair();
        transports0.push(Box::new(transport0));
        transports1.push(Box::new(transport1));
    }
    backend0.setup_kv_transports(transports0);
    backend1.setup_kv_transports(transports1);
    let mut backends = vec![backend0, backend1];

    let mut reference_caches = reference.create_kv_caches();
    let reference_prefill_logits = reference
        .forward(
            &Tensor::from_slice(&PROMPT).unsqueeze(0),
            &mut reference_caches,
        )
        .unwrap()
        .select(1, PROMPT.len() as i64 - 1)
        .squeeze();

    let (_, global_len0) = backends[0]
        .prefill_request_with_reservation(
            REQUEST_ID,
            &PROMPT[..1],
            0,
            Some(&[0]),
            Some(&scenario.capacities[0]),
        )
        .unwrap();
    let (distributed_prefill_logits, global_len1) = backends[1]
        .prefill_request_with_reservation(
            REQUEST_ID,
            &PROMPT[1..],
            1,
            Some(&[1, 2, 3]),
            Some(&scenario.capacities[1]),
        )
        .unwrap();
    assert_eq!((global_len0, global_len1), (1, PROMPT.len()));
    let distributed_prefill_logits = Tensor::from_slice(&distributed_prefill_logits);
    let decode_token = distributed_prefill_logits
        .argmax(-1, false)
        .int64_value(&[]);
    let reference_prefill_token = reference_prefill_logits.argmax(-1, false).int64_value(&[]);
    assert_eq!(decode_token, reference_prefill_token);

    let storage_before = backends
        .iter()
        .map(|backend| storage_snapshot(backend, layers).unwrap())
        .collect::<Vec<_>>();

    let (distributed_decode_logits, decode_finisher, decode_handoffs) = local_decode(
        &mut backends,
        &scenario,
        decode_token,
        DECODE_STARTER,
        device,
    );
    assert_eq!(decode_handoffs, layers * (DOMAINS - 1));
    let reference_decode_logits = reference
        .forward(
            &Tensor::from_slice(&[decode_token]).unsqueeze(0),
            &mut reference_caches,
        )
        .unwrap()
        .squeeze();
    let distributed_decode_last = distributed_decode_logits.squeeze();
    let decode_max_diff = (&distributed_decode_last - &reference_decode_logits)
        .abs()
        .max()
        .double_value(&[]);
    let decode_argmax = distributed_decode_last.argmax(-1, false).int64_value(&[]);
    let reference_decode_token = reference_decode_logits.argmax(-1, false).int64_value(&[]);
    println!("local pre-continuation decode: max_diff={decode_max_diff:.6}");
    assert_eq!(decode_argmax, reference_decode_token);

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
                DOMAINS,
            )
            .unwrap(),
        );
        let mut next_hidden = None;
        for visit_index in 0..DOMAINS {
            let domain = (current_starter + visit_index) % DOMAINS;
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
                    assert_eq!(visit_index + 1, DOMAINS);
                    next_hidden = Some(hidden_states);
                }
            }
        }
        hidden_states = next_hidden.expect("the final worker must finish the layer");
        current_starter = (current_starter + DOMAINS - 1) % DOMAINS;
    }
    assert_eq!(handoffs, layers * (DOMAINS - 1));

    let continuation_logits =
        project_final_logits(&backends[current_starter].model, &hidden_states);
    let distributed_continuation_last = continuation_logits
        .select(1, CONTINUATION_PROMPT.len() as i64 - 1)
        .squeeze();

    reference.set_prefill_position_ids(&scenario.continuation_positions, device);
    let reference_continuation_logits = reference
        .forward(
            &Tensor::from_slice(&CONTINUATION_PROMPT).unsqueeze(0),
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
    let reference_continuation_token = reference_continuation_logits
        .argmax(-1, false)
        .int64_value(&[]);
    let continuation_argmax = distributed_continuation_last
        .argmax(-1, false)
        .int64_value(&[]);
    println!(
        "local stationary continuation: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, tokens={continuation_argmax}/{reference_continuation_token}"
    );
    assert_eq!(reference_continuation_token, continuation_argmax);
    assert!(
        mean_diff < 0.1,
        "continuation mean logits diff: {mean_diff}"
    );
    assert!(max_diff < 0.75, "continuation max logits diff: {max_diff}");

    let mut domain_totals = Vec::with_capacity(DOMAINS);
    for (domain, backend) in backends.iter().enumerate() {
        let total =
            verify_domain_kv_state(backend, &scenario, domain, &storage_before[domain]).unwrap();
        domain_totals.push(total);
    }
    assert_eq!(domain_totals, EXPECTED_DOMAIN_KV_TOTALS);

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
        assert_eq!(positions, (0_i64..=8).collect::<Vec<_>>());
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
    let meta = serde_json::json!({
        "mode": "local",
        "device": format!("{device:?}"),
        "domains": [0, 1],
        "decode_token": decode_token,
        "prefill_argmax": decode_token,
        "decode_argmax": decode_argmax,
        "continuation_argmax": continuation_argmax,
        "reference_prefill_argmax": reference_prefill_token,
        "reference_decode_argmax": reference_decode_token,
        "reference_continuation_argmax": reference_continuation_token,
        "domain_kv_totals": { "0": domain_totals[0], "1": domain_totals[1] },
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
            "domain_kv_totals_eq_54_162": domain_totals == EXPECTED_DOMAIN_KV_TOTALS,
            "position_union_0_to_8": true,
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
        "[route-b smoke] local golden done: decode_token={decode_token} continuation_argmax={continuation_argmax} out={}",
        out_dir.display()
    );
    Ok(())
}

struct Cli {
    mode: String,
    device: Device,
    bind: Option<String>,
    peer: Option<String>,
    out: PathBuf,
    model_dir: PathBuf,
}

fn parse_cli() -> Result<Cli, String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(mode) = args.first().cloned() else {
        return Err(
            "usage: route_b_cross_node_smoke <local|server|client> [--device cpu|mps|cuda:N] [--bind host:port] [--peer host:port] --out <dir> [--model-dir <dir>]"
                .to_string(),
        );
    };
    if !matches!(mode.as_str(), "local" | "server" | "client") {
        return Err(format!("invalid mode {mode}: expected local|server|client"));
    }
    let mut device = Device::Cpu;
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
        bind,
        peer,
        out,
        model_dir,
    })
}

fn main() -> Result<(), String> {
    let cli = parse_cli()?;
    match cli.mode.as_str() {
        "local" => run_local(cli.device, &cli.model_dir, &cli.out),
        "server" => run_node(
            "server",
            1,
            cli.device,
            cli.bind.as_deref().unwrap_or("0.0.0.0:29511"),
            None,
            &cli.model_dir,
            &cli.out,
        ),
        "client" => run_node(
            "client",
            0,
            cli.device,
            cli.bind.as_deref().unwrap_or("0.0.0.0:29510"),
            cli.peer.as_deref(),
            &cli.model_dir,
            &cli.out,
        ),
        _ => unreachable!("mode validated by parse_cli"),
    }
}
