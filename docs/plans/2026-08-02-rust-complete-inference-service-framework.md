# Rust Complete Inference Service Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build the minimum complete Rust inference service path for real models: initial prefill, self-driving decode, continuation prefill, further decode, isolated concurrent requests, release, and a repeatable network service loop.

**Architecture:** Keep the verified self-driving ring math, packet route, capacity-weighted ownership, and reserved positioned KV semantics stable. Add only the request-local state and service adapters needed to connect real model weights to those cores. A worker owns only its local KV shards; all-domain helpers remain correctness oracles and never become runtime ownership APIs.

**Tech Stack:** Rust, tch-rs/libtorch, safetensors, QUIC/P2P ring, SQLite Graph Memory.

---

## Scope Rules

- Implement one vertically testable node at a time and commit it separately.
- Record interface tension as Graph Memory risk/uncertainty and continue unless it causes numerical errors, KV capacity overflow, request contamination, or an unusable service path.
- Do not reopen verified ring math without contradictory correctness evidence.
- Do not add production admission control, dynamic placement, optimized continuous batching, observability, retries, or performance tuning in this phase.

## Task 1: Runtime-Dtype Reserved KV

**Files:**
- Modify: `rust/src/model/self_driving.rs`
- Test: `rust/src/model/self_driving.rs`

1. Add a failing test named `reserved_positioned_kv_accepts_explicit_runtime_dtype`.
2. Construct a BF16 slab, append BF16 K/V, and assert active tensors remain BF16.
3. Add `ReservedPositionedKvShard::new_with_kind(config, capacity, device, kind)`.
4. Keep `new(config, capacity, device)` as the existing Float-compatible entry.
5. Run the focused test, the self-driving suite, and the complete Rust suite.

## Task 2: Real-Model Initial Prefill Into Worker-Local Reserved KV

**Files:**
- Modify: `rust/src/model/self_driving.rs`
- Modify or create the smallest request-state module selected by the node audit.
- Test with: `models/Qwen2-0.5B/config.json` and `models/Qwen2-0.5B/model.safetensors`

Acceptance evidence:

- Load the real 24-layer Qwen2-0.5B model on the selected local correctness device.
- Each logical worker owns only its local per-layer reserved positioned shard.
- Initial prefill commits every prompt position exactly once per layer across workers.
- Last-token logits/token match a contiguous single-model reference within the established BF16 tolerance.
- No runtime API receives all workers' shards.

## Task 3: Single-Request Self-Driving Decode

**Files:**
- Modify: `rust/src/model/self_driving.rs`
- Modify the request-state/service adapter introduced by Task 2.
- Reuse: `rust/src/model/transport/tcp.rs` or the existing runtime transport boundary.

Acceptance evidence:

- Start from Task 2's real prefill state.
- Run at least two decode tokens through the self-driving ring.
- Only the scheduled assignee appends current-token K/V for each layer.
- Each layer uses `N-1` peer hops and the finisher starts the next layer.
- Tokens/logits and per-layer global position unions match the reference.

## Task 4: Continuation Prefill Then Decode

**Files:**
- Modify: `rust/src/worker_sdk/backend.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`
- Modify: `rust/src/distributed/protocol.rs` only if an explicit continuation command is required.

Acceptance evidence:

- The service distinguishes initial prefill from continuation prefill without resetting request state.
- Continuation tokens use explicit global positions after prior decode tokens.
- Existing prefill plus decode history remains readable without format conversion.
- A following decode matches the contiguous reference.

## Task 5: Request-Local Ring State

**Files:**
- Modify: `rust/src/model/attention/ring.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`

Acceptance evidence:

- Request state includes every request-sensitive layer field needed by distributed attention, including prefill length, phase, and sequence offset.
- Interleaving request A and B cannot reuse another request's ring phase or local KV state.
- Existing single-request behavior remains unchanged.

## Task 6: Multi-Request Lifecycle

**Files:**
- Modify: `rust/src/worker_sdk/backend.rs`
- Modify: `rust/src/worker_sdk/tch_backend.rs`
- Modify: `rust/src/worker_sdk/runtime.rs`
- Modify: `rust/src/distributed/protocol.rs`

Acceptance evidence:

- Two different prompts can be interleaved through prefill, decode, continuation prefill, and decode.
- Per-request tokens/logits match two isolated reference runs.
- `release_request` removes all KV and ring phase state.
- Reusing a released request ID starts from empty state.

## Task 7: Stable Network Service Loop

**Files:**
- Modify: `rust/src/distributed/coordinator.rs`
- Modify: `rust/src/worker_sdk/runtime.rs`
- Modify: `rust/src/api/server.rs` only for the minimum request flow required by the existing API.

Acceptance evidence:

- A real request crosses the existing coordinator/worker protocol and completes the full phase sequence.
- Multiple requests run independently; they may pipeline but never wait on one another's decode step by design.
- Every worker platform in a heterogeneous validation performs model computation.
- Repeated request/release cycles do not leave request state behind.

## Verification Per Node

Use the local libtorch correctness environment first:

```bash
LIBTORCH=/Users/stark_sim/libtorch \
DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib:/opt/homebrew/opt/libomp/lib \
HCP_ENABLE_TORCH=1 \
CARGO_NET_OFFLINE=true \
cargo test --manifest-path rust/Cargo.toml --features tch-backend <focused-test> -- --nocapture
```

Before each checkpoint, also run the affected suite, complete Rust tests, targeted rustfmt, clippy, and `git diff --check`. Hardware/performance claims require separate MPS or heterogeneous accelerator evidence and are not inferred from CPU correctness.
