# Post-Continuation Decode Plan

**Goal:** Prove that the same positioned request state can return from a 24-layer `m=6` stationary continuation to one normal `m=1` decode step.

**Boundary:** `N=3`, `L=24`, tickets `[1,3,2]`, in-process CPU synthetic correctness. No transport, runtime, multi-request behavior, or performance measurement.

## Task 1: Extend the existing 24-layer behavior oracle

Modify only `rust/src/model/self_driving.rs`:

1. expand the frozen decode horizon from 24 to 48 layer-KV units and derive assignees for both decode steps;
2. reserve initial prefill, both decode steps, and the continuation before any tensor append;
3. retain all existing assertions for `prefill(0..5) -> decode(6) -> continuation(7..12)`;
4. sample the continuation's last-position logits, embed that token, and call the existing reserved-positioned decode helper from the continuation finisher at position 13;
5. compare the second decode hidden/logits/token with the contiguous reference and verify position union `0..14`, second-decode exact-once append, total decode counts `[8,24,16]`, stable storage pointers, and 48 decode hops.

This is a test-first composition probe. If the existing primitives pass without production changes, record that as the result; do not create a wrapper or force an artificial RED.

## Task 2: Verify and checkpoint

Run file-scoped rustfmt, the focused oracle, all self-driving tests, the full Rust suite, clippy, and diff checks. Commit and push the Rust test checkpoint first. Then attach exact evidence to the Graph task using the implementation commit SHA, close the task, export views, commit, and push the memory checkpoint.
