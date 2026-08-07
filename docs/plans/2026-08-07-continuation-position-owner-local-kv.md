# Continuation Position-Local KV Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Prove that one continuation segment can generate and retain new KV by capacity-weighted position ownership while historical KV stays stationary.

**Architecture:** Keep ownership in the frozen request placement state, outside `LayerPacket`. Each domain receives only its local position offsets, projects K/V for that normalized activation subset, appends those absolute positions to its reserved shard, and then computes the unchanged full-query partial against the complete local shard.

**Tech Stack:** Rust 2021, tch-rs/libtorch, existing `FrozenKvAssigneeSchedule`, `LayerPacket`, and `ReservedPositionedKvShard` test infrastructure.

---

### Task 1: Add the failing capacity-weighted single-layer oracle

**Files:**
- Modify: `rust/src/model/self_driving.rs`

**Step 1: Write the failing test**

Add `multi_token_packet_generates_new_kv_by_capacity_weighted_position_owner` with `N=3`, `m=6`, and tickets `[1,3,2]`. Build owner-local offsets from `FrozenKvAssigneeSchedule`, reserve exact per-domain growth, traverse all three domains, and assert:

- owner counts are `[1,3,2]` and cover offsets `0..m` exactly once;
- new absolute positions appear only in their owner shard;
- reserved storage pointers stay stable;
- attention and final layer hidden state match the dense reference;
- the packet tensor payload remains the existing `m*(4H+h_q+1)` elements.

**Step 2: Run the test to verify RED**

Run:

```bash
LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib \
  cargo test --manifest-path rust/Cargo.toml --features tch-backend \
  multi_token_packet_generates_new_kv_by_capacity_weighted_position_owner -- --nocapture
```

Expected: compilation fails because the positioned subset processing API does not exist.

### Task 2: Implement owner-local position subset projection

**Files:**
- Modify: `rust/src/model/self_driving.rs`

**Step 1: Add the minimal internal API**

Add a `pub(crate)` positioned processing function taking `new_position_offsets: &[usize]`. Validate every offset is in range and unique, `index_select` the packet's normalized activations and absolute position IDs, project only that subset's compact K/V, append it to the local reserved shard, then execute the existing full-query positioned partial.

Keep `process_layer_packet_with_reserved_history` as the compatibility wrapper: the legacy single assignee maps to all offsets on its domain and an empty slice elsewhere. Do not modify `SelfDrivingPacket`, transport, runtime, or scheduler.

**Step 2: Run GREEN and focused regression**

Run the RED command again, followed by:

```bash
LIBTORCH=/Users/stark_sim/libtorch DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib \
  cargo test --manifest-path rust/Cargo.toml --features tch-backend \
  model::self_driving::tests -- --nocapture
```

Expected: the new oracle passes; existing self-driving tests remain green.

### Task 3: Verify and checkpoint

**Files:**
- Modify: `graph-memory/graph.db` and exported views only after the Rust commit.

**Step 1: Verify formatting and lint**

Run targeted `rustfmt --check`, `git diff --check`, and `cargo clippy --features tch-backend --lib --tests`. Existing unrelated warnings may remain, but the command must exit zero.

**Step 2: Commit and push the route implementation**

Commit only the plan and `rust/src/model/self_driving.rs` on `codex/route-b-continuation-stationary-packet`, then push without force.

**Step 3: Record Graph evidence**

On the same route branch, close `task-continuation-position-owner-local-kv-20260807`, attach exact command results and the implementation SHA, retain the route-B branch mapping, export Markdown views, verify SQLite integrity, commit, and push.
