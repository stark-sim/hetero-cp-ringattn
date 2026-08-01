# 24-Layer Prefill/Decode Continuation Experiment

## Goal

Prove that one request can reuse the same distributed KV history across:

```text
prefill_1 -> decode_1 -> continuation prefill_2 -> decode_2
```

The experiment uses 24 real decoder layers, three logical domains, capacity
tickets `[1, 3, 2]`, six tokens in each prefill block, and one token in each
decode step. Decode KV placement is described as layer-striped KV growth, not
pipeline parallelism.

## Data Contract

Each `(layer, domain)` owns a positioned local shard:

```text
PositionedKvShard { K, V, global_position_ids }
```

K and V use the existing compact GQA layout
`[batch, num_kv_heads, local_seq_len, head_dim]`. Positions identify each local
sequence slot. The union of all domains must contain every global position
exactly once for every layer.

## Phase Flow

1. `prefill_1` processes positions `0..6` through all 24 layers. Each layer's
   new KV is split `[1, 3, 2]` across the three domains.
2. `decode_1` consumes the first sampled token at position 6 through the
   existing self-driving layer path. Its 24 KV append events are assigned
   `[4, 12, 8]` in aggregate.
3. `prefill_2` processes six continuation tokens at positions `7..13`. It must
   read the positioned history produced by phases 1 and 2 and project K/V only
   for the six new tokens.
4. `decode_2` consumes the next sampled token at position 13 using the mixed
   history from all previous phases.

The reference path keeps a full ordered KV history on one logical domain. It
may recompute independently for comparison; the distributed path may not
recompute historical K/V during `prefill_2`.

## Verification

- Hidden states and logits match the full-history reference after every phase.
- Both decode argmax tokens match the reference.
- Every layer has complete, duplicate-free global position coverage.
- K/V and position lengths remain aligned on every local shard.
- `prefill_2` projects exactly six new K/V positions per layer and zero old
  positions.
- The two decode steps create 48 append events with aggregate counts
  `[8, 24, 16]`; each individual 24-layer decode step is `[4, 12, 8]`.

## Boundaries

This is an in-process CPU correctness experiment. It intentionally does not
change the production KV cache trait, eliminate `Tensor::cat`, reserve physical
memory, alter schedule smoothing, add networking/runtime integration, or add
multi-request behavior. Physical append mechanics are a later node after the
mixed-history semantic contract passes.
