# vLLM Backend Feasibility Study (M12 Phase 2)

## Executive Summary

**vLLM as a drop-in `WorkerBackend` replacement: HIGH feasibility.**
**vLLM participating in HCP token-level KV ring: LOW feasibility (requires deep vLLM patching).**

The practical path is a **dual-backend architecture**: `TchWorkerBackend` handles distributed ring attention (current path), `VllmWorkerBackend` handles single-node high-throughput serving. The coordinator routes requests based on deployment topology.

---

## Current State

An MVP vLLM backend (`python/hcp_vllm_worker.py`) already exists:
- Uses vLLM `LLM` class (high-level API)
- Calls `generate()` for every prefill/decode step
- **Does NOT reuse KV cache across steps** (re-prefills every time)
- Returns one-hot logits (not real logits)
- `get_kv_block()` and `apply_peer_kv()` are no-ops

This MVP proves control-plane communication works but is not suitable for production.

---

## vLLM Architecture Deep Dive

```
LLM (user API)
  └── LLMEngine
        ├── Scheduler (continuous batching, block allocation)
        ├── BlockAllocator (PagedAttention block table)
        └── Worker(s)
              └── ModelRunner
                    ├── Model (transformer layers)
                    └── Attention (PagedAttention kernel)
```

### Key Components

| Component | Purpose | Externally Accessible? |
|-----------|---------|----------------------|
| `LLM.generate()` | End-to-end generation API | Yes (public) |
| `LLMEngine.step()` | Scheduler + execute one iteration | Semi-private |
| `Worker.execute_model()` | Run forward on a batch | Private |
| `ModelRunner.forward()` | Model forward with block table | Private |
| `PagedAttention.forward()` | Custom CUDA kernel | Private (C++ extension) |
| `CacheEngine.gpu_cache` | Physical KV blocks on GPU | Private |

### The KV Ring Integration Problem

HCP ring attention requires:
1. **Extract local KV blocks** after prefill → send to peer
2. **Receive peer KV blocks** → merge into attention compute
3. **Maintain per-request KV cache state** across decode steps

vLLM's design makes these operations extremely difficult:

**Problem 1: KV blocks are GPU-resident and opaque**
- Physical storage: `List[Tensor]` where each tensor is `[num_blocks, block_size, num_kv_heads, head_dim]`
- Access requires GPU synchronization and vLLM internal APIs
- Block table is managed by `CacheEngine` with copy-on-write logic

**Problem 2: PagedAttention kernel reads from block table internally**
- The kernel accepts `query`, `block_table`, `kv_cache` as inputs
- There is no API to "inject external KV into the block table mid-forward"
- To support peer KV, we would need to either:
  a. Patch the PagedAttention kernel to accept additional KV tensors
  b. Copy peer KV into the block table before forward, then remove after
  c. Replace vLLM's attention with HCP's `ring_attention()`

**Problem 3: vLLM's scheduler conflicts with HCP's scheduler**
- vLLM `LLMEngine` has its own continuous batching scheduler
- HCP coordinator also has `BatchScheduler`
- Two schedulers managing the same requests would conflict

---

## Feasibility Assessment

### Path A: vLLM as Single-Node Backend (No KV Ring)

**Concept:** Each vLLM worker handles complete requests independently. HCP coordinator shards at the **request level** (not token level).

**Architecture:**
```
HCP Coordinator ──QUIC──> VllmWorkerBackend
                              │
                              ▼
                         vLLM LLMEngine
                         (full request, no ring)
```

**Implementation:**
1. `VllmWorkerBackend` receives `Prefill { request_id, chunk, seq_offset }`
2. It stores the chunk in a request queue
3. On `DecodeBatch`, it calls `llm.generate()` with all active request tokens
4. Returns logits for each request

**Pros:**
- Leverages vLLM's full optimization stack (PagedAttention, FlashAttention, continuous batching)
- No need to modify vLLM internals
- High throughput on single node

**Cons:**
- Loses HCP's core value proposition: token-level distributed attention
- Each worker needs enough memory for full model (no model parallelism)
- Not suitable for large models that don't fit on one GPU

**Feasibility: HIGH**

### Path B: vLLM with HCP KV Ring (Token-Level Distribution)

**Concept:** Patch vLLM's worker to support extracting/inserting KV blocks during forward.

**Implementation options:**

**B1. Patch vLLM's model runner**
- After prefill, extract KV from `CacheEngine.gpu_cache` using block table
- Send KV to peer via HCP transport
- Before decode, copy peer KV into temporary blocks
- Modify `ModelRunner.forward()` to pass peer KV to attention

**B2. Replace vLLM attention with HCP ring attention**
- Keep vLLM's scheduler and block allocator
- Replace `PagedAttention` module with `HcpRingAttentionBackend`
- Use vLLM block table for memory management, HCP ring for distributed compute

**B3. Fork vLLM and add KV transport hooks**
- Add `submit_kv_block()` / `recv_kv_block()` to vLLM's worker
- Modify `CacheEngine` to support external block injection

**Pros:**
- Retains HCP's token-level distribution
- Combines vLLM's kernel optimization with HCP's heterogeneity

**Cons:**
- Requires deep vLLM internals knowledge
- PagedAttention kernel is C++/CUDA — patching needs custom kernel development
- Tight coupling to vLLM version (every vLLM upgrade may break patches)
- High maintenance burden

**Feasibility: LOW** (months of work, ongoing maintenance)

### Path C: Hybrid (vLLM for Single-Node, tch-rs for Distributed)

**Concept:** Coordinator selects backend per request based on deployment topology.

**Rules:**
- If `num_domains == 1`: route to `VllmWorkerBackend` (maximum throughput)
- If `num_domains > 1`: route to `TchWorkerBackend` (distributed ring attention)

**Pros:**
- Best of both worlds
- No vLLM patching required
- Clean separation of concerns

**Cons:**
- Two backend implementations to maintain
- Cannot combine vLLM throughput with multi-node distribution for the SAME request

**Feasibility: HIGH**

---

## Recommended Architecture

**Path C (Hybrid) with Path A (vLLM single-node) as Phase 1.**

### Rust Side: `VllmWorkerBackend` Stub

Add a new backend variant to `WorkerBackend` architecture:

```rust
/// vLLM subprocess backend.
///
/// Launches a Python vLLM worker as a subprocess and communicates via JSON/HTTP.
/// Used for single-node high-throughput serving.
pub struct VllmWorkerBackend {
    /// Subprocess handle
    child: std::process::Child,
    /// HTTP client for sending commands
    client: reqwest::Client,
    /// Base URL of the vLLM worker HTTP server
    base_url: String,
    domain_id: usize,
}

impl WorkerBackend for VllmWorkerBackend {
    fn setup_kv_transports(&mut self, _transports: Vec<Box<dyn KvTransport>>) {
        // vLLM single-node backend does not use KV ring
    }

    fn prefill_request(&mut self, request_id: u64, chunk: &[i64], seq_offset: usize) -> Result<(Vec<f32>, usize), String> {
        // HTTP POST to vLLM worker: /prefill
    }

    fn decode_batch(&mut self, request_tokens: &[(u64, i64)]) -> Result<Vec<(u64, Vec<f32>)>, String> {
        // HTTP POST to vLLM worker: /decode_batch
    }

    // ... other trait methods
}
```

### Python Side: vLLM Worker HTTP Server

Wrap vLLM's `LLM` class in a minimal HTTP server:

```python
from vllm import LLM, SamplingParams
from fastapi import FastAPI

app = FastAPI()
llm = LLM(...)

@app.post("/prefill")
def prefill(req: PrefillRequest):
    # Use llm.generate() or lower-level API
    return {"logits": logits.tolist()}

@app.post("/decode_batch")
def decode_batch(req: DecodeBatchRequest):
    # Batch decode using vLLM's scheduler
    return {"request_logits": [...]}
```

### Integration Point

Coordinator's worker selection logic:
```rust
if num_domains == 1 && config.use_vllm_backend {
    // Launch VllmWorkerBackend subprocess
} else {
    // Launch TchWorkerBackend (distributed ring attention)
}
```

---

## Next Steps (If Approved)

1. **Implement `VllmWorkerBackend` Rust stub** (1 day)
   - Subprocess spawning
   - HTTP client for prefill/decode_batch
   - Trait implementation

2. **Implement Python vLLM worker HTTP server** (1 day)
   - FastAPI server wrapping vLLM `LLM`
   - `/prefill` and `/decode_batch` endpoints
   - Per-request state isolation

3. **Integration test** (1 day)
   - Single-node vLLM backend E2E
   - Compare output with `TchWorkerBackend` for same prompt
   - Measure throughput difference

**Total estimate: 3 days for a working vLLM single-node backend prototype.**

---

## Conclusion

- **Do NOT attempt to integrate vLLM into HCP's token-level KV ring.** The engineering cost is too high and the maintenance burden is unsustainable.
- **DO implement vLLM as an alternative single-node backend.** This provides real value (high throughput) with reasonable effort and preserves HCP's distributed architecture for the tch-rs path.
- The `KvCache` trait abstraction (Phase 1) already enables this decoupling: `TchWorkerBackend` uses `ContiguousKvCache` with ring attention, while `VllmWorkerBackend` delegates KV management entirely to vLLM.
