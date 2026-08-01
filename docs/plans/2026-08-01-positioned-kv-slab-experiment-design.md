# Positioned KV Slab Experiment

## Goal

Replace `Tensor::cat` growth on the distributed side of the existing 24-layer
continuation experiment with exact, preallocated positioned KV slabs. This
node proves append mechanics only; it does not introduce a production cache or
allocator.

## Motivation

1. **Problem**: `Tensor::cat` allocates a new full local shard and copies its
   history on every append. Semantic KV reuse is already proven, but this
   temporary full-shard allocation is incompatible with a strict local memory
   bound.
2. **Current state**: the test-only `PositionedKvShard` uses `Tensor::cat` for
   prefill, while its decode adapter passes active tensors through the existing
   cat-based decode runner.
3. **End state**: both distributed prefill and decode append through an exact
   reservation and write cursor; the four-phase 24-layer result remains equal
   to the independent dense-GQA reference; overflow is rejected before any
   write.
4. **Prior art**: serving engines use paged KV caches, block tables, or reserved
   arenas to separate logical growth from physical allocation.
5. **Approach**: use a test-only `ReservedPositionedKvShard`. Derive every
   `(layer, domain)` capacity from two `[1,3,2]` prefill blocks plus that layer's
   two frozen decode assignees. Append with `narrow(...).copy_()` and expose only
   the committed prefix to attention.
6. **Why this approach**: the experiment has a known 14-position horizon, so an
   exact slab proves the memory property without importing page allocation,
   admission, retries, or runtime lifecycle.

`VERDICT: IMPLEMENT EXPERIMENT ONLY`.

## Alternatives

- Reserve the largest shard size on every domain: simpler, but violates the
  capacity-weighted memory goal by over-reserving small domains.
- Add a paged or block-table allocator now: generally useful, but too broad for
  the current core-first milestone.

## Sacrifice Analysis

- `Tensor::cat` exists because it supports unknown and dynamically changing
  sequence lengths with a simple owned contiguous result.
- Exact reservation sacrifices growth beyond the frozen horizon and runtime
  reassignment after admission.
- Dynamic growth is intrinsically useful for open-ended generation, changing
  batch membership, and allocator-driven scheduling.
- That flexibility is not required in this fixed in-process proof. Production
  allocation remains a later decision and cannot be inferred from this test.

## Verification

- A focused slab test proves committed-prefix correctness, stable reserved
  capacity, and atomic overflow rejection.
- The 24-layer test proves every slab capacity equals its planned final usage.
- Distributed final totals remain `[56,168,112]`.
- All four phases still match the dense-GQA reference and continuation prefill
  still projects exactly `144` new positions.
- Full Rust tests, clippy, rustfmt check for the touched test file, and
  `git diff --check` pass.

## Boundaries

No production cache trait changes, physical GPU-memory claim, page allocator,
admission policy, schedule smoothing, networking, runtime integration, or
multi-request behavior.
