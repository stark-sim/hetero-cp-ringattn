#!/usr/bin/env python3
"""
vLLM Block-Aware Ring Attention Plugin (vLLM >= 0.23, V1 engine)

Same design as ``hcp_vllm_block_ring_plugin.py`` (vLLM 0.6.4) but ported to the
V1 engine.  The V1 engine no longer exposes ``SequenceGroupMetadata`` /
``SequenceData`` / the legacy ``CacheEngine``; instead it keeps the KV cache in
``GPUModelRunner.kv_caches`` (one tensor per layer with the same layout as the
legacy engine: ``[2, num_gpu_blocks, block_size, num_kv_heads, head_dim]``) and
drives the model with ``SchedulerOutput`` / ``NewRequestData``.

Key points:
- The engine is created with ``enable_multiprocessing=False`` so the model
  executor, scheduler and KV cache live in this process and can be touched
  directly (this is what makes block exchange possible without a connector).
- Blocks are allocated straight from the scheduler's ``block_pool`` so the
  plugin owns them (the scheduler never frees them because we bypass it).
- Each prefill/decode is a single ``execute_model`` call with a hand-built
  ``SchedulerOutput``.  ``add_requests`` re-registers the request each time
  (``_remove_request`` + ``add_request``), so we always pass the full token list
  and the full block table with ``overwrite=True``; ``num_computed_tokens``
  tells the model runner which tokens are already in the cache.

Scope / limitations (same as the 0.6.4 PoC):
- Only single-sequence, greedy (temperature=0) decode.
- Logits returned are pseudo-logits reconstructed from the sampler's top-k
  logprobs (matching the HcpWorkerBackend contract), not full-vocab logits.
- The simple two-domain block-table merge requires chunk lengths that are
  multiples of ``block_size``.
"""

from typing import List, Optional, Tuple
import torch

from hcp_worker_sdk import HcpWorkerBackend, KvBlock


class VllmBlockRingPluginV1(HcpWorkerBackend):
    """HCP Worker backend on vLLM V1 PagedAttention with block KV exchange."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        dtype: Optional[str] = None,
        gpu_memory_utilization: float = 0.5,
        block_size: int = 16,
        max_model_len: int = 4096,
    ):
        import os
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine

        self.model_dir = model_dir
        self.device = device
        if dtype is None:
            dtype = "float32"
        self.dtype_str = dtype

        print(f"[vllm v1 block ring] loading model from {model_dir} on {device} ...")
        engine_args = EngineArgs(
            model=model_dir,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_num_seqs=1,
            block_size=block_size,
            max_model_len=max_model_len,
            disable_log_stats=True,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, enable_multiprocessing=False
        )
        self.model_executor = self.llm_engine.model_executor
        self.model_runner = self.model_executor.driver_worker.model_runner
        self.scheduler = self.llm_engine.engine_core.engine_core.scheduler
        self.block_pool = self.scheduler.kv_cache_manager.block_pool

        self.model_config = self.llm_engine.model_config
        self.cache_config = self.llm_engine.vllm_config.cache_config
        self.block_size = self.cache_config.block_size

        # One tensor per attention layer: [2, num_gpu_blocks, block, kv_heads, dim]
        self.kv_caches = self.model_runner.kv_caches
        self.num_gpu_blocks = int(self.kv_caches[0].shape[1])

        self.vocab_size = self.model_config.get_vocab_size()
        hf_config = getattr(self.model_config, "hf_config", None)
        self.rope_base = getattr(hf_config, "rope_theta", 10000.0)

        self._history: List[int] = []
        self._combined_block_table: Optional[List[int]] = None
        self._local_block_table: List[int] = []
        self._remote_block_table: List[int] = []
        self._query_block: Optional[int] = None
        self._req_id = "ring_attn_v1"
        # Global sequence length across all ring domains.  ``_history`` only
        # holds the tokens this worker actually prefilled/decoded, so for
        # cross-node CP we track the full length separately; decode/last_token
        # positions are derived from it (earlier tokens are never recomputed,
        # their KV comes from the combined block table).
        self._global_seq_len: int = 0
        # Global position of this worker's own chunk (seq_offset of its chunk).
        self._local_seq_offset: int = 0

        print(
            f"[vllm v1 block ring] loaded, vocab_size={self.vocab_size}, "
            f"block_size={self.block_size}, num_gpu_blocks={self.num_gpu_blocks}, "
            f"num_layers={self.num_layers}"
        )

    # ------------------------------------------------------------------
    # Model / cache descriptors
    # ------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self.kv_caches)

    @property
    def num_kv_heads(self) -> int:
        return int(self.kv_caches[0].shape[3])

    @property
    def num_heads(self) -> int:
        return self.model_config.get_num_attention_heads(
            self.llm_engine.parallel_config
        )

    @property
    def head_dim(self) -> int:
        return int(self.kv_caches[0].shape[4])

    # ------------------------------------------------------------------
    # Block helpers
    # ------------------------------------------------------------------
    def _zero_block(self, physical_block_id: int) -> None:
        for layer_idx in range(self.num_layers):
            cache = self.kv_caches[layer_idx]
            cache[0, physical_block_id] = 0
            cache[1, physical_block_id] = 0

    def _allocate_blocks(self, num_tokens: int) -> List[int]:
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks = self.block_pool.get_new_blocks(num_blocks)
        ids = [b.block_id for b in blocks]
        for bid in ids:
            self._zero_block(bid)
        return ids

    def extract_block(
        self, layer_idx: int, physical_block_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = self.kv_caches[layer_idx]
        k = cache[0, physical_block_id].clone()
        v = cache[1, physical_block_id].clone()
        return k, v

    def insert_block(
        self,
        layer_idx: int,
        physical_block_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        cache = self.kv_caches[layer_idx]
        cache[0, physical_block_id].copy_(k)
        cache[1, physical_block_id].copy_(v)

    def get_local_block_table(self) -> List[int]:
        return self._local_block_table

    # ------------------------------------------------------------------
    # Engine step
    # ------------------------------------------------------------------
    def _run_step(
        self,
        token_ids: List[int],
        num_computed_tokens: int,
        block_table: List[int],
        num_scheduled_tokens: int,
    ) -> torch.Tensor:
        from vllm import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=10)
        new_req = NewRequestData(
            req_id=self._req_id,
            prompt_token_ids=token_ids,
            prefill_token_ids=token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=(list(block_table),),
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
        )
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.num_scheduled_tokens = {self._req_id: num_scheduled_tokens}
        scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens

        output = self.model_executor.execute_model(scheduler_output)
        if output is None:
            output = self.model_executor.sample_tokens(None)
        return self._output_to_logits(output)

    def _output_to_logits(self, output) -> torch.Tensor:
        logits = torch.full((self.vocab_size,), -1e9, dtype=torch.float32)
        idx = output.req_id_to_index[self._req_id]
        token_ids = output.logprobs.logprob_token_ids[idx]
        values = output.logprobs.logprobs[idx]
        for tok, val in zip(token_ids, values):
            if int(tok) >= 0:
                logits[int(tok)] = float(val)
        return logits

    # ------------------------------------------------------------------
    # HcpWorkerBackend interface
    # ------------------------------------------------------------------
    def load_model(self, model_dir: str, device: str) -> None:
        pass

    def set_global_tokens(self, tokens: List[int]) -> None:
        self._history = list(tokens)
        self._global_seq_len = len(tokens)

    def set_global_seq_len(self, length: int) -> None:
        """Set the global sequence length when the full token ids are unknown.

        In cross-node CP a worker only holds its own chunk's tokens; decode
        positions must still use the *global* length, so the coordinator's
        SyncGlobalSeqLen drives this.
        """
        self._global_seq_len = length

    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        """Prefill the local chunk into freshly allocated blocks."""
        self._history = list(chunk)
        self._global_seq_len = seq_offset + len(chunk)
        self._local_seq_offset = seq_offset
        self._local_block_table = self._allocate_blocks(len(chunk))
        logits = self._run_step(
            token_ids=self._history,
            num_computed_tokens=0,
            block_table=self._local_block_table,
            num_scheduled_tokens=len(chunk),
        )
        return logits, seq_offset + len(self._history)

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        return self.get_kv_block_from_table(
            layer_idx, seq_start, seq_end, self.get_local_block_table(),
            table_seq_offset=self._local_seq_offset,
        )

    def get_kv_block_from_table(
        self,
        layer_idx: int,
        seq_start: int,
        seq_end: int,
        block_table: List[int],
        table_seq_offset: int = 0,
    ) -> KvBlock:
        block_size = self.block_size
        start_block = (seq_start - table_seq_offset) // block_size
        end_block = (seq_end - table_seq_offset + block_size - 1) // block_size
        physical_ids = block_table[start_block:end_block]

        k_blocks = []
        v_blocks = []
        for bid in physical_ids:
            k, v = self.extract_block(layer_idx, bid)
            k_blocks.append(k)
            v_blocks.append(v)
        if k_blocks:
            k = torch.stack(k_blocks, dim=0)
            v = torch.stack(v_blocks, dim=0)
        else:
            k = torch.empty(0)
            v = torch.empty(0)
        return KvBlock(layer_idx, seq_start, seq_end, k, v)

    def _copy_block_table(self, block_table: List[int]) -> List[int]:
        new_blocks = self._allocate_blocks(len(block_table) * self.block_size)
        for src_bid, dst_bid in zip(block_table, new_blocks):
            for layer_idx in range(self.num_layers):
                cache = self.kv_caches[layer_idx]
                cache[0, dst_bid].copy_(cache[0, src_bid])
                cache[1, dst_bid].copy_(cache[1, src_bid])
        return new_blocks

    def prefill_peer_chunk_with_context(
        self,
        chunk: List[int],
        seq_offset: int,
        context_tokens: List[int],
        context_block_table: List[int],
    ) -> List[int]:
        """Prefill a peer chunk with prior context already resident in the cache."""
        peer_block_table = self._allocate_blocks(len(chunk))
        combined = list(context_block_table) + peer_block_table
        token_ids = list(context_tokens) + list(chunk)
        self._run_step(
            token_ids=token_ids,
            num_computed_tokens=len(context_tokens),
            block_table=combined,
            num_scheduled_tokens=len(chunk),
        )
        return peer_block_table

    def prefill_with_context_kv(
        self,
        chunk: List[int],
        seq_offset: int,
        context_kv_blocks: List[KvBlock],
        context_len: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Insert prior-domain context KV, then prefill ``chunk`` against it.

        This is the context-passing CP step for the *later* domain: earlier
        domains ship their chunk's KV blocks, we write them into reserved
        physical slots (with RoPE delta-rotation to their global positions),
        and prefill our own chunk with those blocks as the block-table prefix.
        Only our chunk's tokens are computed, but every layer sees the full
        causal context, so the resulting K/V and last-token logits are correct.

        ``context_len`` is the number of real context tokens (the chunks may
        pad their last block).  Returns (last_token_logits, global_seq_len).
        """
        if not context_kv_blocks:
            # No context: degenerate to a plain prefill.
            return self.prefill(chunk, seq_offset)

        num_ctx_blocks = context_kv_blocks[0].k.shape[0]
        ctx_block_ids = self._allocate_blocks(num_ctx_blocks * self.block_size)
        for kv in context_kv_blocks:
            delta = kv.global_seq_start
            rot_k = self._rope_delta_rotate_keys(kv.k, delta)
            for i, bid in enumerate(ctx_block_ids):
                self.insert_block(kv.layer_idx, bid, rot_k[i], kv.v[i])

        peer_block_table = self._allocate_blocks(len(chunk))
        combined = ctx_block_ids + peer_block_table
        # Only the count of context tokens matters for positions; the actual
        # ids are not recomputed (their KV is read from ctx_block_ids).
        token_ids = [0] * context_len + list(chunk)
        logits = self._run_step(
            token_ids=token_ids,
            num_computed_tokens=context_len,
            block_table=combined,
            num_scheduled_tokens=len(chunk),
        )

        self._history = list(chunk)
        self._local_block_table = peer_block_table
        self._combined_block_table = combined
        self._global_seq_len = context_len + len(chunk)
        self._local_seq_offset = seq_offset
        return logits, self._global_seq_len

    def apply_peer_kv(
        self,
        layer_idx: int,
        peer_block: KvBlock,
        rotate_delta: Optional[int] = None,
    ) -> None:
        if peer_block.k.numel() == 0:
            return

        num_blocks = peer_block.k.shape[0]
        if layer_idx == 0:
            self._remote_block_table = self._allocate_blocks(
                num_blocks * self.block_size
            )
            self._query_block = self._allocate_blocks(self.block_size)[0]
            self._build_combined_block_table(peer_block.global_seq_start)
        else:
            assert len(self._remote_block_table) == num_blocks

        delta = peer_block.global_seq_start if rotate_delta is None else rotate_delta
        peer_k = self._rope_delta_rotate_keys(peer_block.k, delta)
        peer_v = peer_block.v
        total_peer_tokens = peer_block.global_seq_end - peer_block.global_seq_start
        for i, bid in enumerate(self._remote_block_table):
            k_blk = peer_k[i]
            v_blk = peer_v[i]
            valid = min(self.block_size, total_peer_tokens - i * self.block_size)
            if valid < self.block_size:
                k_blk = k_blk.clone()
                v_blk = v_blk.clone()
                k_blk[valid:] = 0.0
                v_blk[valid:] = 0.0
            self.insert_block(layer_idx, bid, k_blk, v_blk)

    def _rope_delta_rotate_keys(
        self, k_blocks: torch.Tensor, delta: int
    ) -> torch.Tensor:
        if delta == 0:
            return k_blocks
        orig_shape = k_blocks.shape
        k = k_blocks.reshape(-1, *orig_shape[2:])
        dtype = k.dtype
        device = k.device
        kf = k.to(torch.float32)

        head_dim = self.head_dim
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        angles = delta * inv_freq
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        k_rot = kf * cos + rotate_half(kf) * sin
        return k_rot.reshape(orig_shape).to(dtype)

    def _build_combined_block_table(self, peer_seq_start: int) -> None:
        local_table = self.get_local_block_table()
        peer_reserved = self._remote_block_table
        peer_start_block = (peer_seq_start + self.block_size - 1) // self.block_size
        combined = [0] * (peer_start_block + len(peer_reserved))
        combined[:peer_start_block] = local_table[:peer_start_block]
        combined[peer_start_block:] = peer_reserved
        self._combined_block_table = combined

    def decode(self, token: int) -> torch.Tensor:
        self._history.append(token)
        seq_len = self._global_seq_len + 1
        self._global_seq_len = seq_len
        block_table = list(self._combined_block_table or self.get_local_block_table())
        if self._query_block is None:
            self._query_block = self._allocate_blocks(self.block_size)[0]
        block_table.append(self._query_block)
        # Earlier token ids are placeholders (never recomputed); only the last
        # decode token is a real id.
        token_ids = [0] * (seq_len - 1) + [token]
        return self._run_step(
            token_ids=token_ids,
            num_computed_tokens=seq_len - 1,
            block_table=block_table,
            num_scheduled_tokens=1,
        )

    def last_token_logits(self) -> torch.Tensor:
        seq_len = self._global_seq_len
        block_table = list(self._combined_block_table or self.get_local_block_table())
        token_ids = [0] * (seq_len - 1) + [self._history[-1]]
        return self._run_step(
            token_ids=token_ids,
            num_computed_tokens=seq_len - 1,
            block_table=block_table,
            num_scheduled_tokens=1,
        )

    def last_token_logits_prefill(self) -> torch.Tensor:
        # In V1 the prefill/decode distinction is carried by
        # num_computed_tokens/num_scheduled_tokens, so this is identical to
        # last_token_logits().
        return self.last_token_logits()

    @property
    def capacity_mb(self) -> int:
        try:
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return int(free // (1024 * 1024))
        except Exception:
            pass
        return 4096

    def shutdown(self) -> None:
        print("[vllm v1 block ring] shutting down...")
        try:
            self.llm_engine.shutdown()
        except Exception:
            pass
        try:
            del self.llm_engine
            import gc
            gc.collect()
        except Exception as e:
            print(f"[vllm v1 block ring] shutdown warning: {e}")
