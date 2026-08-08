# 24-Layer Mixed-History Stationary Continuation Plan

**Goal:** Prove that route B can carry one `m=6` continuation segment through 24 layers from positioned prefix-plus-decode history while permanent KV remains capacity-weighted and stationary.

**Boundary:** `N=3`, `L=24`, tickets `[1,3,2]`, in-process CPU synthetic correctness. This node does not run a decode token after the continuation and does not touch TCP/QUIC, runtime, multi-request scheduling, or performance measurement.

## Task 1: Add one failing 24-layer behavior test

Modify `rust/src/model/self_driving.rs` with one integration-style test that:

1. reserves each layer for an initial six-position `[1,3,2]` prefill, one capacity-scheduled decode append, and six continuation positions split `[1,3,2]`;
2. builds initial positioned history and appends one decode token;
3. invokes a not-yet-existing reserved-positioned continuation ring runner for positions `7..13`;
4. compares final hidden/logits with a contiguous dense reference;
5. checks every layer's new-position union is complete and unique, per-domain growth is `[1,3,2]`, storage pointers are unchanged, starters/finishers rotate, and total hops are `48`.

Run the focused test and retain the compiler failure caused by the missing runner as RED evidence.

## Task 2: Implement the smallest route-B model runner

Add a crate-internal experimental runner and trace type. The runner accepts `new_position_offsets_by_domain`, validates that the lists partition `0..m` before mutating any shard, then for every layer:

1. creates one `LayerPacket` on the current starter;
2. visits only successor domains in ring order;
3. passes each domain's local offsets to `process_layer_packet_with_reserved_history_for_positions`;
4. records route, hop count, starter, finisher, and per-domain appended-position counts;
5. hands the finisher's hidden state directly to the next layer and projects final logits once after layer 24.

The legacy scalar `assignee` field remains only for compatibility with `LayerPacket::start`; it is not used as continuation ownership and no owner vector is placed in the packet.

Run the focused test to GREEN, then run all `model::self_driving::tests`.

## Task 3: Verify and checkpoint

Run `rustfmt`, focused and full Rust tests, clippy, `git diff --check`, and Graph integrity checks. Commit and push the Rust checkpoint first. Then attach exact command results to the Graph task using the Rust commit SHA, close this task, register continuation-after-decode as the next separate node, export views, commit, and push the Graph checkpoint.
