# Real-Qwen Route-B Stationary Continuation Plan

**Goal:** Prove that route B's owner-local stationary continuation holds with real Qwen2-0.5B BF16 weights: `prefill(0..3) -> decode(4) -> continuation(5..8)` on two workers with tickets `[1,3]`, historical KV never entering the packet.

**Boundary:** Route-B phase-1 feasibility item 5. mac-local, CPU BF16, single request, ignored oracle (requires local weights). No TCP/QUIC, no runtime/coordinator, no multi-request, no performance claims. Cross-device validation is phase-1 item 6, not this node.

**Composition probe:** This is test-first composition (precedent: `b523bc7`). All required primitives already exist and are individually verified. If the oracle passes without production changes, record that as the result; do not manufacture a RED.

## Task 1: Add the failing real-model oracle

**Files:**
- Modify: `rust/src/worker_sdk/tch_backend.rs` (test module only)

Add `real_qwen_two_worker_stationary_continuation_matches_reference`:

1. Load local Qwen2-0.5B; one reference `LlamaModel` + two `TchWorkerBackend` (pattern: `real_qwen_two_worker_reserved_prefill_matches_reference`).
2. Scenario: prompt `[151644, 9707, 0, 16]`; prefix split `[1,3]` (worker0 position `[0]`, worker1 `[1,2,3]`); decode at position 4; continuation tokens at positions `[5,6,7,8]` with owner offsets from `FrozenKvAssigneeSchedule::new(&[1,3], request_id, 4)` (worker0 `[0]`, worker1 `[1,2,3]`).
3. Reservations per layer = prefix `[1,3]` + continuation `[1,3]` + decode assignee from `FrozenKvAssigneeSchedule::new(&[1,3], request_id, 24)`.
4. Prefill both workers via `prefill_request_with_reservation`; run one in-process decode step (packet handoff without TCP, mirroring `run_two_backend_reserved_tcp_decode` minus wire).
5. Continuation: per layer, `LayerPacket::start` on the current starter with the four embedded continuation tokens and `position_ids=[5,6,7,8]`; visit both domains in ring order with `process_layer_packet_with_reserved_history_for_positions(layer, packet, shard, offsets[domain])`; finisher hidden feeds the next layer; starter rotates per layer.
6. Assert per layer: position union is exactly `0..=8`; continuation increments are `[1,3]`; storage `data_ptr` unchanged across all phases; `committed_len == reserved_capacity` at the end.
7. Numeric contract (from established BF16 envelope lessons): continuation last-position logits vs contiguous reference — argmax exact, `mean < 0.1`, `max < 0.75` guard; print all three values.

Run:

```bash
LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib \
  HCP_ENABLE_TORCH=1 cargo test --manifest-path rust/Cargo.toml --features tch-backend \
  real_qwen_two_worker_stationary_continuation_matches_reference -- --ignored --nocapture
```

## Task 2: Verify and checkpoint

1. File-scoped `rustfmt --edition 2021` on the touched file only; `git diff --check`; confirm `git diff --name-only` shows only intended files.
2. Focused oracle (above), all `model::self_driving::tests`, full `cargo test --features tch-backend`, `cargo clippy --features tch-backend --lib --tests`.
3. Commit the Rust checkpoint first on `codex/route-b-continuation-stationary-packet`, push.
4. Attach exact command results to the Graph task with the commit SHA, update phase-1 item status, export views, commit, push.
