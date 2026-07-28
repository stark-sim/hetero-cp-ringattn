# Self-Driving Ring Decode Plan Revision

Date: 2026-07-29

## Outcome

The self-driving Rust decode direction is feasible, but the frozen plan has two
incorrect intermediate steps:

1. The vLLM plugin successor-seeded optimization cannot reduce an owner-return
   unidirectional ring from `N` network hops to `N-1`.
2. A single-packet attention step cannot be integrated independently while all
   workers still execute a synchronous full-model `forward()` call.

Task E is therefore rejected. Rust tasks D1 and the minimum part of D2 must be
implemented as one vertical slice.

## Topology Proof: Why Plugin Task E Cannot Save One Hop

The plugin has three fixed constraints:

- the vLLM owner is the only process that produces the layer query;
- only successor links are available;
- the attention result must return to the same owner-side `forward()` stack.

For ring nodes `0..N-1`, with owner `0`, a packet that visits every peer and
returns to the owner follows:

```text
0 -> 1 -> 2 -> ... -> N-1 -> 0
```

This path has exactly `N` edges. Moving the owner's local partial from the seed
to the final merge changes compute order, but not the path. For three nodes the
physical route remains `owner -> A -> B -> owner`, which is three hops, not two.

The `N-1` lower bound applies only when the result may stop at the predecessor
finisher. The plugin cannot consume the result there without taking over the
model's layer-to-layer execution, which is outside its supported extension
surface. The current plugin Q-ring is already hop-minimal under its owner-return
contract.

## Feasibility Audit

| Item | Verdict | Reason |
|---|---|---|
| Plugin successor-seeded E | Reject | Owner-return topology still requires `N` edges. |
| Rust single-packet math | Feasible | Online-softmax merge is associative; existing `(Q,O,LSE)` packet math is reusable. |
| D1 as a standalone runtime checkpoint | Reject | Only the finisher obtains the layer output; other synchronous model forwards cannot advance. |
| Rust role-driven layer continuation | Feasible with protocol changes | Every worker has model weights and local per-request KV, but control must move above the current attention call. |
| Fully autonomous sampling in the first slice | Defer | It unnecessarily couples attention routing, model continuation, sampling, and request lifecycle in one change. |

## Revised Minimal Architecture

The first runnable slice remains coordinator-triggered. The coordinator sends
the same decode step to all workers so they enter a collective decode call, but
only one packet and one non-attention forward path exist inside the ring.

For each layer:

1. The starter owns `hidden_L` and computes norm and Q. If it is not the K/V
   assignee, it seeds attention from only its durable historical KV. If it is
   also the assignee, it instead computes one `history + current` seed, appends
   current K/V after forming that partial, and sets `kv_committed` before the
   first hop.
2. Each relay normally merges only its durable historical KV partial. The
   assignee relay alone computes current K/V from `hidden_L`, forms one
   `history + current` partial, appends after forming the partial, and sets
   `kv_committed`. The finisher performs the same branch when it is the
   assignee, then rejects the completed packet unless `kv_committed` is true.
3. After `N-1` peer hops, the finisher applies output projection, residual,
   post-attention norm, and MLP to produce `hidden_(L+1)`.
4. That finisher becomes the next layer's starter. Other workers wait for the
   next role-relevant packet inside the same collective decode call.
5. The last-layer finisher alone computes logits and returns them with an
   explicit finisher identity. Other workers return acknowledgements, not
   redundant logits.

This slice removes redundant attention and MLP/logits computation while keeping
coordinator token broadcast and sampling temporarily. It proves the difficult
model-continuation boundary before changing request ownership.

## Required Protocol Changes

The current `RingPacket` is insufficient. The self-driving packet needs at
least:

- `request_id` and global token position;
- `layer_idx` and starter/finisher routing metadata;
- hidden/residual state needed by the finisher;
- Q, O, and LSE accumulator state;
- current-token K/V assignee and commit metadata (the K/V tensors are computed
  at the assignee and do not travel in the packet);
- a packet phase/version discriminator.

`WorkerBackend::decode_request()` currently promises logits from every worker,
and `WorkerRuntime` sends `DecodeDone` unconditionally. The first slice must
replace that contract with a result such as `Finished { logits, finisher }` or
`Participated`, then teach the coordinator to sample only the unique finisher
result.

## Revised Execution Order

1. **R0: Protocol contract and mock state-machine test.** Define packet phases,
   deterministic role rotation, exact-once KV growth, hop accounting, and one
   finisher result. Cover the assignee overlapping the starter, a middle relay,
   and the finisher. No production path switch yet.
2. **R1: Single-request vertical slice.** Implement collective self-driving
   decode across model layers, coordinator-triggered, with one packet and one
   logits producer. Keep the current Q-ring behind a fallback flag.
3. **R2: Request isolation and batch scheduling.** Add per-request state and
   test two concurrent requests whose starters differ.
4. **R3: Autonomous token loop.** Move sampling and embedding to the last-layer
   finisher; coordinator becomes admission/release and receives token events.
5. **R4: Validation ladder.** Mock correctness and counters, local MPS two-node,
   then cross-node CUDA+HIP. Require token parity, `N-1` peer hops per layer,
   one non-attention layer continuation, one logits producer, and exact-once KV
   ownership.

## Sacrifice Verdict

The optimization still sacrifices all-node result replication and multi-packet
concurrency. That remains acceptable for the current inference PoC because
packets are latency-dominated and ring failure already aborts the request.
However, it must not sacrifice runnable checkpoint semantics: an isolated D1
that deadlocks the synchronous forward stack is not a valid checkpoint.

Verdict: reject plugin Task E; implement the Rust path using R0-R4 above.
