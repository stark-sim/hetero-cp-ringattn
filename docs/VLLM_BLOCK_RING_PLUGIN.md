# vLLM Block-Aware Ring Attention — Plugin Route

**Status:** design + first PoC  
**Goal:** keep vLLM's PagedAttention kernel unchanged, but let HCP exchange KV cache blocks between heterogeneous workers.

## Why plugin route

From `docs/BLOCK_RING_FUSION.md`, the wrong path is to extract a contiguous KV tensor and then gather/scatter back into vLLM.  The right path is to make the **ring exchange granularity equal to vLLM's block size** (default 16 tokens).

A plugin route avoids modifying vLLM's attention kernel:
- vLLM still computes attention via its existing FlashAttention/cutlass path over a `block_table`.
- HCP only manipulates the **block table** and copies KV bytes into physical cache blocks.

## vLLM cache structure (v0.6.4)

Observed on Qwen2.5-3B:

```text
CacheEngine.block_size = 16
CacheEngine.num_gpu_blocks = 4540
CacheEngine.num_attention_layers = 36
CacheEngine.num_kv_heads = 2
CacheEngine.head_size = 128
gpu_cache[layer].shape = (2, 4540, 16, 2, 128)
                      [key/value, physical_blocks, block_size, kv_heads, head_dim]
```

For a sequence of 64 tokens, vLLM allocates 4 physical blocks:

```text
block_table = [0, 1, 2, 3]
```

Extracting block `b` from layer `l`:

```python
k = ce.gpu_cache[l][0, b]  # [block_size, num_kv_heads, head_dim]
v = ce.gpu_cache[l][1, b]  # [block_size, num_kv_heads, head_dim]
```

Writing a received block into an unused physical slot `b'`:

```python
ce.gpu_cache[l][0, b'].copy_(k)
ce.gpu_cache[l][1, b'].copy_(v)
```

`scripts/poc_vllm_block_extract.py` verifies this round-trip is byte-identical.

## Proposed plugin architecture

```
┌─────────────────────────────────────────────────────────────┐
│  HCP Coordinator                                            │
│  - assigns global logical block ranges to each domain       │
│  - tells each worker which physical blocks it may use       │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Worker 0 (CUDA)     Worker 1 (HIP)      Worker N (...)
   ─────────────────   ─────────────────   ───────────────
   vLLM LLMEngine      vLLM LLMEngine      vLLM LLMEngine
   local prefill       local prefill       local prefill
   local blocks B0..   local blocks Bk..   local blocks Bm..
        │                   │                   │
        └────────── KV block ring exchange ───┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   remote blocks          remote blocks         remote blocks
   written into free      written into free     written into free
   physical slots         physical slots        physical slots
        │                   │                   │
        ▼                   ▼                   ▼
   combined block table   combined block table  combined block table
   (local + remote)       (local + remote)      (local + remote)
        │                   │                   │
        └────────── decode via vLLM model runner (bypass scheduler)
```

### Key components

1. **Coordinator block allocator**
   - Input: `seq_len`, `block_size`, `num_domains`, capacities.
   - Output: each domain's logical block range and allowed physical block pool.
   - Physical block pools must be disjoint across domains so remote writes never collide.

2. **Worker `VllmBlockRingBackend`**
   - Wraps a vLLM `LLMEngine`.
   - `prefill(chunk)`: runs vLLM prefill for the local chunk.
   - `get_kv_blocks(layer, block_ids)`: extracts K/V for the listed physical blocks.
   - `put_kv_blocks(layer, block_ids, k_blocks, v_blocks)`: writes received bytes into free physical slots.
   - `set_block_table(seq_id, block_ids)`: updates the sequence's block table to include local + remote blocks.
   - `decode(token)`: invokes vLLM's model runner directly with the updated block table.

3. **Ring exchange**
   - Same `KvTransport` abstraction as today.
   - Exchange unit is one vLLM block (or a group of consecutive blocks for fewer messages).
   - For 2 domains: one swap round is enough; for N domains, N-1 rounds like today.

4. **Online softmax**
   - vLLM's PagedAttention kernel already computes full softmax over the whole block table.
   - After the block table contains **all** blocks, a normal vLLM decode forward gives the correct next-token logits.
   - This is not the memory-optimal Ring Attention online softmax, but it is correct and keeps vLLM kernel untouched.
   - Future optimization: implement block-wise online softmax in a custom kernel or wrapper.

## Decode invocation options

vLLM's high-level `LLMEngine` scheduler does not expect a block table modified externally.  Two ways to run decode:

### Option A: bypass scheduler, call model_executor directly

Build a `ModelInput` (input_tokens, positions, slot_mapping, block_tables) and call:

```python
engine.model_executor.execute_model(model_input, ...)
```

Pros: full control of block table, no scheduler surprises.  
Cons: must manually handle CUDA graph, sampling, and batching.

### Option B: use scheduler but allocate remote blocks through block_manager

After receiving remote blocks, allocate them via `scheduler.block_manager.allocate(...)` and append to the sequence's block table.  Then a normal `engine.step()` will use them.

Pros: keeps vLLM scheduler/batching intact.  
Cons: block_manager APIs are internal and may change; remote blocks look like local blocks, but their K/V content came from the network.

**Recommended first step:** Option A for a correctness PoC; Option B for production integration.

## Open questions

1. **GQA broadcast timing**  
   vLLM stores KV per `num_kv_heads` in the cache, so ring exchange should transfer the compressed KV format.  This matches HCP Rust design.

2. **Decode after remote blocks**  
   When all domains have the same combined block table, each can independently decode the next token.  But they all produce the same next token, so no additional all-reduce is needed for greedy/temperature sampling.  For sampling, each worker must use the same seed or only one worker samples.

3. **Memory reservation for remote blocks**  
   Each worker must reserve enough physical blocks for its own chunk + all peer chunks.  Coordinator needs to cap per-worker logical chunk size so that `local_blocks + remote_blocks <= num_gpu_blocks`.

4. **Heterogeneous dtype / block size**  
   Both workers must use the same vLLM `block_size`, `kv_cache_dtype`, and head dimensions.  This is already required by the model config.

## Next steps

1. Implement `VllmBlockRingBackend.prefill()` + block extraction.
2. Implement a 2-worker single-machine PoC that:
   - prefill chunk A on worker 0 and chunk B on worker 1,
   - exchanges blocks A and B,
   - builds combined block table on both workers,
   - runs one decode step and compares next-token logits to a single-node vLLM reference.
3. Integrate with HCP coordinator and `KvTransport` for cross-node deployment.
4. Measure cross-node bandwidth impact and compare to the current Rust ring path.
