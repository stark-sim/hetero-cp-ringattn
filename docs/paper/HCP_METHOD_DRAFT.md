# HCP: Heterogeneous Context Parallelism over a Linear P2P Ring

> Method draft, 2026-08-02. This document fixes the current problem model,
> algorithm, proof obligations, evidence boundary, and evaluation plan. It is
> not a complete paper. It intentionally omits the abstract, implementation
> results, performance numbers, and final conclusion.

## Claim Discipline

This draft uses four labels so that an algorithmic statement is not confused
with an empirical result.

- **Method claim**: a property required by the HCP design.
- **Proved invariant**: a property derived from the stated model and
  assumptions.
- **Prototype evidence**: a property exercised by the current Rust correctness
  prototype. It is not a physical-memory or production-runtime result.
- **Open empirical question**: a property that requires measurement on the
  current end-to-end heterogeneous implementation.

## 1. Problem Definition and Research Position

Modern distributed inference systems usually form a fine-grained parallel
group from devices with symmetric memory, compute capability, and collective
communication support. Heterogeneous resources are more commonly separated at
coarser boundaries, such as request routing, model placement, or prefill/decode
service disaggregation. These boundaries are useful, but they do not aggregate
the unequal memory capacities of different accelerators for the context of one
request.

Heterogeneous Context Parallelism (HCP) addresses a narrower question:

> Can devices with unequal memory capacity and potentially different execution
> backends jointly hold and evaluate one request's attention context using only
> neighbor-to-neighbor communication?

The target pressure is the key-value (KV) cache. For an autoregressive
transformer, durable KV state grows with the context length even when model
parameters remain fixed. HCP therefore treats the logical attention context,
rather than a particular tensor collective or inference phase, as the object
being parallelized.

**Method claim.** HCP extends context parallelism across the complete request
lifecycle: initial prefill, autoregressive decode, and continuation prefill all
operate on one distributed logical KV context.

**Method claim.** A worker communicates only with its predecessor and successor
in a logical ring. HCP requires point-to-point send/receive semantics, not a
collective library, globally shared device memory, or an all-to-all peer graph.

The logical contract is transport-agnostic. It may be carried by any fabric
that can implement the required P2P ordering and tensor representation. Faster
general-purpose accelerator interconnects expand the useful operating region,
but no particular interconnect is assumed to make HCP efficient by itself.

**Open empirical question.** Whether heterogeneous fine-grained collaboration
outperforms coarse request- or phase-level placement depends on link bandwidth,
latency, device imbalance, kernel efficiency, and workload concurrency. This
draft does not claim that advantage has already been measured.

## 2. System Model

### 2.1 Workers and topology

Let the worker set be

\[
\mathcal{W}=\{0,1,\ldots,N-1\}.
\]

Workers form a unidirectional logical ring with

\[
\operatorname{succ}(i)=(i+1)\bmod N,
\qquad
\operatorname{pred}(i)=(i-1)\bmod N.
\]

Each worker holds a complete replica of the model parameters. Workers may have
different usable KV capacity, attention throughput, accelerator type, and
local kernel implementation, but they execute the same transformer semantics.
The current HCP method aggregates context capacity; it does not partition a
model whose parameters cannot fit on one worker.

For worker \(i\), let \(B_i\) denote the bytes admitted for durable KV after
model weights, runtime state, and bounded communication/compute workspaces have
been accounted for. \(B_i\), not nominal device memory, is the placement hard
bound.

### 2.2 Position-indexed logical KV context

For layer \(\ell\) and global token position \(p\), define

\[
\mathcal{C}_{\ell}[p]=(K_{\ell,p},V_{\ell,p}).
\]

The ownership function \(a_{\ell}(p)\in\mathcal{W}\) induces the local shard

\[
S_{i,\ell}=\{p\mid a_{\ell}(p)=i\}.
\]

Every layer must satisfy

\[
S_{i,\ell}\cap S_{j,\ell}=\varnothing \quad (i\ne j),
\qquad
\bigcup_{i=0}^{N-1}S_{i,\ell}=\mathcal{P},
\]

where \(\mathcal{P}\) is the set of committed global positions for the request.
Each local entry stores its global position together with K and V. Physical
storage order is not part of the logical sequence order.

**Proved invariant.** If the shard sets are disjoint and complete, and causal
masking is evaluated from global positions, concatenating or sorting physical
shards is unnecessary for attention correctness.

### 2.3 Capacity-weighted ownership

Define the normalized capacity target

\[
\alpha_i=\frac{B_i}{\sum_j B_j}.
\]

For a finite admitted set \(E\) of equal-sized KV events, HCP targets integer
counts

\[
n_i\in\{\lfloor \alpha_i|E|\rfloor,
          \lceil \alpha_i|E|\rceil\},
\qquad
\sum_i n_i=|E|,
\]

using deterministic remainder assignment. If KV event sizes differ by layer,
the same contract is applied in bytes rather than raw event counts:

\[
M_i=\sum_{(\ell,p):a_\ell(p)=i} b_\ell \le B_i,
\]

where \(b_\ell\) is the K-plus-V storage cost of one position at layer
\(\ell\).

The assignment order is smoothed so that every prefix stays near its weighted
target rather than placing all events for one worker contiguously. A
request-derived phase rotates this deterministic sequence, distributing
initial roles across concurrent requests without changing any request's total
reservation.

**Method claim.** Capacity weighting is the default HCP memory policy. Equal
\(1/N\) placement is only the special case in which all usable KV budgets are
equal.

**Method claim.** Ownership is frozen for the admitted horizon. Existing KV is
never migrated merely because a later inference phase uses a different data
flow. Extending a horizon requires a new capacity check before new positions
are admitted.

**Open empirical question.** A production allocator for open-ended generation,
request churn, fragmentation, and re-admission is outside the current core.
The hard bound has only been exercised with an explicitly reserved finite
horizon.

## 3. Unified HCP Request Lifecycle

HCP changes the object circulating on the ring according to the amount of
query parallelism available, while leaving the logical KV ownership contract
unchanged.

```text
initial prefill
  local Q/activation chunks stay at their workers
  bounded KV micro-blocks circulate
            |
            v
autoregressive decode
  historical KV stays at its workers
  one activation + Q + softmax-accumulator packet circulates
            |
            v
continuation prefill
  new Q/activation chunks are distributed again
  old and new positioned KV shards participate in the same ring attention
```

No phase conversion reconstructs a dense, globally ordered KV tensor. The
common interface is \(\mathcal{C}_{\ell}[p]\) plus global positions.

## 4. Capacity-Weighted Ring Attention Prefill

### 4.1 Local sequence work

For a prefill block with positions \(P\), HCP partitions its query positions
into disjoint capacity-weighted sets \(P_i\). Worker \(i\) holds the activation
and query rows for \(P_i\), computes K/V for those positions, and permanently
commits them to its local shard. The same position partition is retained across
layers of that prefill block so that layer normalization, residual operations,
and the MLP execute locally on the corresponding token activations.

Thus prefill distributes both attention queries and non-attention token work.
It does not send every activation through all workers.

### 4.2 KV circulation and online softmax

At layer \(\ell\), each target worker keeps its local Q block while KV blocks
visit all workers in ring order. A large source shard is streamed as bounded
micro-blocks so the receiver does not materialize another worker's complete
durable shard.

For a query row and a KV subset \(A\), define an accumulator

\[
\mathcal{A}_A=(m_A,z_A,u_A),
\]

where

\[
m_A=\max_{p\in A}s_p,
\quad
z_A=\sum_{p\in A}\exp(s_p-m_A),
\quad
u_A=\sum_{p\in A}\exp(s_p-m_A)V_p.
\]

For disjoint subsets \(A\) and \(B\), their stable merge is

\[
m=\max(m_A,m_B),
\]

\[
z=\exp(m_A-m)z_A+\exp(m_B-m)z_B,
\]

\[
u=\exp(m_A-m)u_A+\exp(m_B-m)u_B.
\]

The exact attention output is \(u/z\). An equivalent wire representation uses
the normalized output \(O_A=u_A/z_A\) and
\(\operatorname{LSE}_A=m_A+\log z_A\):

\[
\operatorname{LSE}=\operatorname{logaddexp}
  (\operatorname{LSE}_A,\operatorname{LSE}_B),
\]

\[
O=\exp(\operatorname{LSE}_A-\operatorname{LSE})O_A+
  \exp(\operatorname{LSE}_B-\operatorname{LSE})O_B.
\]

Global query and key positions determine the causal mask. This is required
because a capacity-weighted shard, especially after decode growth, need not
contain a contiguous position interval.

**Proved invariant.** Repeatedly merging disjoint, complete KV subsets produces
the same attention result as evaluating the corresponding dense causal
attention row, up to floating-point evaluation order.

**Method claim.** Only transient, bounded KV micro-blocks may visit a non-owner.
The durable KV cache remains capacity-weighted throughout prefill.

## 5. Self-Driving Decode Ring

Decode has one query token per request, so the query sequence dimension can no
longer be divided among workers. HCP instead keeps historical KV stationary
and moves the state required to evaluate one logical transformer layer.

### 5.1 Layer packet

At layer \(\ell\), the temporary starter \(s_\ell\) receives hidden state
\(h_\ell\), computes

\[
x_\ell=\operatorname{Norm}_{in}(h_\ell),
\qquad
Q_\ell=W_Qx_\ell,
\]

and creates a layer packet containing:

```text
request and route state:
  request_id, global_position, layer_idx, current_worker,
  visited_workers, kv_assignee

tensor state:
  residual h_l, normalized x_l, Q_l,
  attention output accumulator O, LSE
```

The normalized hidden state is carried because the unique KV assignee may not
be the starter. The packet size depends on model width and query heads, not on
the historical context length.

### 5.2 Exact-once layer state machine

For decode position \(p\), a frozen capacity-weighted schedule selects exactly
one assignee \(a_\ell(p)\) for every layer.

Each worker performs the following actions when the packet arrives:

1. If it is the assignee, compute
   \((K_{\ell,p},V_{\ell,p})\) once from \(x_\ell\) and commit the pair to the
   local positioned shard.
2. Evaluate the query against this worker's complete local shard, including
   the just-committed current pair when applicable.
3. Merge the local partial into \((O,\operatorname{LSE})\).
4. Forward the packet to the successor unless all \(N\) shards have
   contributed.

The last visited worker is the finisher. It computes the non-attention tail
once:

\[
a_\ell=W_OO_\ell,
\]

\[
r_\ell=h_\ell+a_\ell,
\]

\[
h_{\ell+1}=r_\ell+
\operatorname{MLP}_\ell(\operatorname{Norm}_{post}(r_\ell)).
\]

The finisher immediately becomes the next layer's starter. No result is
returned to the previous starter.

**Proved invariant.** One layer performs exactly one Q projection, one
current-token K/V projection and durable append, \(N\) disjoint local attention
partials, one output projection, and one residual/norm/MLP continuation.

**Proved invariant.** Since the starter contributes locally before sending and
the finisher consumes the completed result, an exact layer traverses \(N-1\)
physical ring edges. Under the assumption that every worker owns a shard that
must contribute, \(N-1\) is also the lower bound on a unidirectional ring.

### 5.3 Layer and token role recurrence

With the successor direction defined above, the finisher is the starter's
predecessor:

\[
s_{\ell+1}=s_\ell-1\pmod N.
\]

If the last-layer finisher performs final normalization, the language-model
head, sampling, and embedding for the next token without an extra handoff,
then

\[
s_{t+1,0}=s_{t,0}-L\pmod N.
\]

When \(L\bmod N=0\), the same worker remains the final sampler for that request.
This does not create a KV hotspot because the layer-position assignee schedule
is independent of the sampler. Model and language-model-head parameters are
already replicated; only context-independent activation/logit workspace and
compute remain concentrated.

HCP therefore keeps zero-handoff token continuation as the core rule and uses
a request-derived initial phase to spread fixed samplers across different
requests. An optional token-boundary handoff can rotate the sampler if measured
queueing justifies the additional edge, but that is a scheduling extension,
not a correctness requirement.

**Prototype evidence.** The Rust tensor model exercises arbitrary \(N\),
arbitrary \(L\), non-zero starters, wrap-around edges, all assignee/finisher
overlaps, finisher-to-starter continuation, final logits, and localhost TCP
packet transport. These tests establish modular data-flow behavior, not current
cross-backend hardware performance.

## 6. Continuation Prefill and Mixed-History KV

A request may append a multi-token continuation after one or more decode
steps. HCP does not normalize the old cache into a new physical layout.

Suppose the committed history before a continuation is \(\mathcal{P}_{old}\)
and the new block positions are \(\mathcal{P}_{new}\). For every layer,

\[
\mathcal{P}_{old}\cap\mathcal{P}_{new}=\varnothing.
\]

New positions receive a capacity-weighted prefill partition. Their activations
and Q rows remain local across layers, and their K/V pairs are appended to the
corresponding positioned shards. The resulting shard may contain:

- initial-prefill positions assigned by a block position partition;
- decode positions assigned independently for each layer;
- continuation-prefill positions assigned by a later block partition.

Ring Attention for the continuation uses local Q rows for
\(\mathcal{P}_{new}\) and visits the union

\[
\mathcal{P}_{old}\cup\mathcal{P}_{new}
\]

through positioned KV micro-blocks. Global positions enforce the causal rule
that a new query at \(q\) may attend only to keys at \(p\le q\).

**Proved invariant.** Phase history is composable if, for every layer, K, V,
and position arrays stay aligned and the global-position union is complete and
duplicate-free. Prefill and decode do not need identical physical shard shapes
or layer-wise ownership functions.

**Prototype evidence.** A 24-layer Rust correctness experiment executes

```text
prefill -> decode -> continuation prefill -> decode
```

with unequal capacity tickets. It verifies dense-reference hidden states and
logits after every phase, exact-once projection of only the new continuation
positions, complete position coverage, frozen weighted decode assignment, and
stable storage addresses in exact preallocated slabs.

**Evidence boundary.** The slab is a finite-horizon experimental cache. It
proves that the mixed-history semantics can be implemented without repeated
full-shard concatenation; it does not prove a production allocator or physical
accelerator peak-memory result.

## 7. Correctness Argument

HCP correctness follows from four lemmas.

### Lemma 1: ownership completeness

For every committed \((\ell,p)\), admission chooses one and only one owner.
Therefore the distributed cache represents the same logical set of K/V pairs
as a dense cache, without durable replication.

### Lemma 2: blockwise attention equivalence

The online-softmax merge is an exact regrouping of the numerator and
denominator of softmax over disjoint key sets. Causal masks are computed from
global positions. Consequently the merged attention output equals dense causal
attention in real arithmetic.

### Lemma 3: decode exact-once semantics

The packet visits every worker exactly once. Its assignee predicate is true at
exactly one worker, so current K/V is projected and committed exactly once. By
Lemma 1, the local partials cover the prior history plus the current position
exactly once; by Lemma 2, their merge yields the dense decode attention output.

### Lemma 4: transformer continuation

The finisher applies the same output projection, residual additions,
normalizations, and MLP as the dense transformer layer. Thus equality of
\(h_\ell\) implies equality of \(h_{\ell+1}\). Induction over layers, then over
prefill/decode/continuation phases, gives the same logical hidden states and
logits, modulo backend floating-point order.

**Proved invariant.** The two physical data flows are not two cache formats.
They are two evaluation strategies over the same position-indexed logical
context.

## 8. Memory, Communication, and Compute Complexity

Let \(T\) be the committed context length, \(b_\ell\) the KV bytes per position
at layer \(\ell\), \(D\) the model-width-scale packet dimension, and \(L\) the
number of layers.

### 8.1 Durable memory

Total logical KV memory is

\[
M_{KV}(T)=T\sum_{\ell=0}^{L-1}b_\ell.
\]

Worker \(i\) stores only its assigned pairs:

\[
M_i=\sum_{\ell}\sum_{p\in S_{i,\ell}}b_\ell\le B_i.
\]

Ignoring integer rounding, \(M_i\) approaches
\(\alpha_i M_{KV}\). Durable KV does not concentrate at the starter, finisher,
or sampler. Prefill requires a bounded KV micro-block buffer; decode requires
an \(O(D)\) packet/activation workspace. Model weights remain fully replicated.

### 8.2 Prefill communication

If all KV for a layer is circulated once to every non-owner target, aggregate
ring traffic for that layer is approximately

\[
(N-1)T b_\ell.
\]

For a continuation block, \(T\) denotes the history visible to its new queries
after including newly committed positions. For fixed context size, aggregate
traffic and hop work grow linearly with \(N\); each worker still has only two
logical peers.

### 8.3 Decode communication

One layer sends one packet across \(N-1\) edges:

\[
C_{decode/token}=L(N-1)\,|P_{layer}|,
\qquad |P_{layer}|=O(D).
\]

The wire payload is independent of \(T\). Attention compute is not independent
of \(T\): the workers collectively scan \(T+1\) KV positions at each layer,
with worker \(i\)'s share determined by its shard.

### 8.4 Latency consequence

A single decode packet has a serial dependency through all participating
workers and edges. HCP removes redundant full-model forwards and context-sized
decode transfers, but it does not remove the slowest-worker or slowest-link
term from single-request latency. Independent request packets may overlap in a
future scheduler; that concurrency mechanism is not part of the current core.

## 9. Assumptions, Limitations, and Threats to Validity

1. **Replicated parameters.** Every worker must fit the full model weights and
   implement compatible layer semantics.
2. **Finite admission horizon.** The current hard KV bound assumes a declared
   horizon and reservation. Open-ended growth requires allocator and
   re-admission policy not defined here.
3. **Fail-stop ring.** Without KV replication, loss of one worker loses a
   unique shard and aborts affected requests. Elastic removal and fault
   tolerance require a different memory trade-off.
4. **Serial decode path.** One request visits all workers at every layer. A very
   slow worker or link may dominate latency even if its memory contribution is
   useful.
5. **Prefill traffic.** KV circulation grows with visible context and ring
   size. Bounded streaming controls transient memory but not total bytes.
6. **Backend compatibility.** Device-local kernels may evaluate floating-point
   operations in different orders. Wire dtype/layout conversion and numerical
   tolerances require end-to-end validation.
7. **No current production runtime claim.** Admission, request multiplexing,
   backpressure, retries, fragmentation, observability, and multi-request
   fairness remain outside this method draft.
8. **No current economic claim.** Lower cost, energy efficiency, and better
   utilization of mixed accelerator pools are hypotheses until measured
   against appropriate homogeneous and coarse-grained baselines.

## 10. Evaluation Design

The eventual evaluation should answer research questions rather than merely
show that the prototype runs.

### RQ1: Does HCP preserve model semantics?

- Compare every phase boundary against a single-worker dense/incremental
  reference.
- Cover initial prefill, multiple decode steps, continuation prefill, and a
  second decode sequence.
- Sweep \(N\), \(L\), non-zero starters, wrap-around edges, unequal capacity
  vectors, and all starter/assignee/finisher overlaps.
- Report hidden-state and logit error, sampled-token agreement, position-union
  completeness, duplicate positions, and exact-once operation counts.

### RQ2: Does capacity weighting enforce the local memory contract?

- Measure actual peak device memory after weights and fixed workspaces are
  loaded.
- Compare admitted bytes, reserved bytes, committed bytes, allocator overhead,
  and transient receive buffers per worker.
- Verify that no worker materializes a remote durable shard or exceeds its
  declared \(B_i\).
- Sweep heterogeneous capacity ratios, prompt/continuation composition, decode
  horizons, and fragmentation patterns.

### RQ3: Does the P2P data plane match the scaling model?

- Instrument bytes and messages on every directed edge.
- Verify \(N-1\) decode hops per layer and a context-independent decode packet
  size.
- Verify bounded prefill receive buffers while measuring total KV traffic.
- Sweep ring size, context length, KV micro-block size, bandwidth, and latency.

### RQ4: Where is heterogeneous collaboration beneficial?

Compare at least:

- the fastest single worker when the workload fits;
- a homogeneous ring with the same worker count;
- an equal-shard P2P ring;
- capacity-weighted HCP;
- coarse request-level or prefill/decode-stage placement on the same resource
  pool where a fair implementation is available.

Report time to first token, inter-token latency, throughput, device utilization,
link utilization, peak memory, energy, and cost only after the measurement
method and resource accounting are fixed.

### RQ5: Which design choices control bottlenecks?

Ablate:

- capacity-weighted versus equal ownership;
- \(N-1\)-hop finisher consumption versus an \(N\)-hop return-to-starter route;
- KV micro-block size;
- fixed per-request sampler phase versus optional token-boundary rotation;
- single-request execution versus independent multi-request packet overlap,
  once the latter exists.

### Required evidence progression

The evaluation should progress through distinct evidence levels:

1. deterministic tensor correctness;
2. in-process positioned-cache lifecycle;
3. real P2P packet transport;
4. physical-memory validation on each participating backend;
5. full cross-backend heterogeneous lifecycle;
6. performance, scalability, and cost comparison.

Passing an earlier level must not be reported as evidence for a later one.
