# Self-Driving Ring Decode: Theoretical Fit for HCP

Date: 2026-07-29

## Verdict

A self-driving ring is a strong fit for HCP's decode data plane when the primary
goals are sharded KV capacity, heterogeneous peers, P2P-only communication, and
linear topology cost. It is not automatically a low-latency design: one token's
layer dependency remains serial around the ring, and the slowest heterogeneous
node limits steady-state throughput.

The design is valid if it preserves the invariants below and distinguishes one
logical distributed forward from redundant full-model forwards.

## Core Invariants

For request `r`, layer `l`, and all token positions `P_r`, let `K_i(r,l)` be the
durable KV positions stored by node `i`.

```text
K_i intersect K_j = empty                 for i != j
union_i K_i = P_r
|K_i| in {floor(|P_r|/N), ceil(|P_r|/N)}
```

Thus each node stores at most `ceil(|P_r|/N)` positions per layer. Exact `1/N`
is impossible when the token count is not divisible by `N`; the unavoidable
imbalance is at most one token per request. No node may transiently materialize
another node's durable shard. The circulating packet is `O(model_width)` and
does not grow with context length.

Every layer has exactly one logical continuation:

- all `N` nodes compute one attention partial against their disjoint local KV;
- Q projection is computed once by the starter;
- current-token K/V is computed and stored once by its assignee;
- output projection, residual, MLP, and next-layer continuation run once at the
  finisher;
- final norm, logits, and sampling run once at the token finisher.

"Single forward" therefore means one distributed logical forward. It does not
mean only one node performs attention: all shards must contribute exactly once.

## Per-Layer Packet State Machine

At layer `l`, starter `s` holds residual hidden state `h_l`.

1. `s` computes `norm(h_l)` and Q. If `s != assignee(position)`, it creates the
   `(O,LSE)` seed from only its durable historical KV. If `s == assignee`, it
   computes current K/V and creates one seed over `history + current`; after
   forming that partial it appends current K/V and sets `kv_committed` before
   the first hop. It never computes a second historical partial.
2. The packet moves only to the successor. A normal relay evaluates Q against
   only its durable historical KV and online-merges that partial.
3. When the assignee is a relay (including the finisher), it alone computes
   current-token K/V from `h_l`. It forms one partial over its durable positions
   `< position` plus current K/V, then appends current K/V and sets
   `kv_committed`. Other nodes never materialize it.
4. After `N-1` edges, the predecessor of `s` has all `N` partials. It asserts
   `kv_committed`, applies output projection + residual + MLP, and owns
   `h_(l+1)`.
5. The finisher becomes the next layer's starter, so no return hop is needed.

Protocol tests must cover all three role overlaps: assignee equals starter,
assignee equals a middle relay, and assignee equals finisher.

Minimum packet fields are:

```text
protocol_version, request_id, token_position, layer_idx,
starter_id, hops_remaining, kv_assignee, kv_committed,
h_residual, q, o_acc, lse_acc
```

The packet may carry normalized hidden state or recompute the inexpensive norm
at the K/V assignee. It need not carry context-sized KV.

## Optimality of N-1 Hops

Under the HCP contract, all `N` nodes own a non-empty logical shard and must
participate even while a short request has not yet populated every shard.
Starting with one node's partial, an exact result must incorporate the other
`N-1` shard contributions. On a unidirectional ring, reaching each previously
unseen node requires one edge, so `N-1` is the contract-level lower bound. The
state machine attains it by consuming the result at the last node instead of
returning it to the starter.

For `L` layers, attention circulation costs `L * (N-1)` packet edges per token.
The graph has exactly `N` physical links and each worker has one predecessor and
one successor, so both connection count and per-token hop count scale linearly.
No broadcast, all-reduce, all-gather, or fully connected peer graph is needed.

## Token-Boundary Role Rotation

Within a token, roles rotate naturally:

```text
starter(t, l+1) = starter(t, l) - 1 mod N
```

If the last-layer finisher directly samples and starts the next token, then:

```text
starter(t+1, 0) = starter(t, 0) - L mod N
```

This has a resonance edge case. When `L mod N == 0` (for example 24 layers and
3 nodes), the layer work is balanced but the same node remains token starter,
logits producer, and sampler for that request. This does not change the KV
partition: `kv_assignee(position)` remains independent of the sampler, so no
generated-token KV collapses onto that node.

The fixed sampler adds only final norm, LM-head, sampling, and transient logits
work. Model and LM-head weights are already replicated on every worker. The
transient logits footprint is `O(batch * vocab)` (about 0.58 MiB per fp32 row
for a 152K vocabulary), not `O(context_length)`. It can be a compute/queue hot
spot for many concurrent requests, but it is not a durable KV-memory hot spot.

The default should therefore keep the zero-handoff token boundary and assign
each request an initial phase such as `hash(request_id) mod N`. With `L mod N ==
0`, each request keeps one sampler while different requests distribute sampler
load across nodes. On heterogeneous hardware, admission may deliberately place
more sampler roles on faster nodes without changing KV placement.

Only if measurements show sampler queueing should per-token phase shifting be
enabled. If the token boundary advances `k` successor hops, the phase delta is
`k-L (mod N)`. Visiting every sampler requires `gcd(k-L, N) = 1`; making the
next starter exactly the previous starter's successor requires
`k = (L+1) mod N`. Thus a fixed one-hop policy is not universal. For the target
`L=24, N=3`, `k=1` is the minimal full-rotation choice. With that optional
target-specific policy, the honest total is:

```text
L * (N-1) + 1 small token-ID handoff per token
```

For `L=24, N=3`, the default remains 48 attention edges. Strict per-token
rotation costs 49 total edges; the extra message is only a token ID/control
packet. This is a measured scheduling choice, not a correctness requirement.

## Heterogeneity

The architecture supports heterogeneous execution because every node runs the
same semantic operations locally and exchanges a device-neutral packet over
P2P links. It does not require homogeneous collectives or a shared runtime.

Heterogeneity introduces a real trade-off:

- strict equal KV placement gives the strongest `1/N` memory guarantee and
  limits usable KV capacity to what the smallest node can sustain;
- capacity- or speed-weighted KV placement improves utilization but necessarily
  gives some nodes more than `1/N` of KV;
- a single request's latency includes every node's local partial and every ring
  edge, so a slow node cannot be skipped without losing its KV contribution.

For the stated HCP contract, equal placement should be the correctness default.
Weighted placement can be an explicit alternative policy, not an invisible
optimization that weakens the memory guarantee.

Multiple requests can pipeline independent packets to recover throughput. This
requires request IDs, bounded per-link queues, backpressure, and deterministic
per-request ordering. It must not create extra durable KV replicas.

## Failure and Lifecycle Boundaries

The minimal design is fail-stop: one failed node breaks the ring and aborts all
requests using its unique KV shard. That is the same consequence as strict KV
sharding today; replication-based fault tolerance would violate the `1/N`
memory target.

Admission and release may remain on a coordinator control plane without making
it a decode owner. The data plane is ownerless when no coordinator participates
in layer computation, logits, sampling, or KV movement.

## Suitability Summary

Self-driving ring is suitable for HCP if the product priority is:

1. aggregate KV capacity with a per-node `ceil(T/N)` bound;
2. P2P-only operation on partially reachable networks;
3. no full-model decode redundancy;
4. linear connection and communication growth;
5. acceptable fail-stop semantics and serial single-token latency.

It is not the right default if minimum single-request decode latency, elastic
node removal, or weighted use of highly unequal accelerators is more important
than the strict KV split.
