#!/usr/bin/env python3
"""
vLLM Block-Aware Ring Attention Plugin

A minimal external plugin that lets HCP exchange KV cache at vLLM's physical
block granularity, without modifying the vLLM attention kernel.

Design (vLLM 0.6.4):
- CacheEngine.gpu_cache[layer] has shape
  (2, num_gpu_blocks, block_size, num_kv_heads, head_dim).
- dim 0 = 0 for K, 1 for V.
- The plugin extracts/inserts whole physical blocks and builds a combined
  block table so that vLLM's PagedAttention kernel attends over local + peer
  KV.

Current scope (correctness PoC):
1. Allocate physical blocks for the local chunk and prefill by calling the
   model executor directly with a custom SequenceGroupMetadata.
2. Extract the local physical blocks for the chunk.
3. Receive peer blocks and write them into reserved physical slots.
4. Decode the next token by directly calling the model executor with a
   SequenceGroupMetadata whose block table contains local + remote blocks.
   This bypasses the scheduler, so no kernel or scheduler changes are needed.

Known limitations:
- decode() currently returns a one-hot logits tensor (sampled token = 0,
  others = -1e9) to stay compatible with HcpWorkerBackend.  Returning full
  logits requires either capturing sampler output or calling model.forward
  directly; left for a follow-up.
- Only supports single-sequence, greedy/temperature=0 decode.
"""

from typing import List, Optional, Tuple
import torch

from hcp_worker_sdk import HcpWorkerBackend, KvBlock


class VllmBlockRingPlugin(HcpWorkerBackend):
    """
    HCP Worker backend that uses vLLM's PagedAttention as the attention
    implementation and exchanges KV cache blocks via the ring.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        dtype: Optional[str] = None,
        gpu_memory_utilization: float = 0.5,
    ):
        from vllm import LLM, SamplingParams

        self.model_dir = model_dir
        self.device = device
        if dtype is None:
            dtype = "float16" if device == "npu" else "float32"
        self.dtype_str = dtype

        print(f"[vllm block ring] loading model from {model_dir} on {device} ...")
        self.llm = LLM(
            model=model_dir,
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_num_seqs=1,
        )
        engine = self.llm.llm_engine
        self.model_config = engine.model_config
        self.cache_config = engine.cache_config
        self.block_size = self.cache_config.block_size
        self.num_gpu_blocks = self.cache_config.num_gpu_blocks

        self.vocab_size = self.model_config.get_vocab_size()
        hf_config = getattr(self.model_config, "hf_config", None)
        self.rope_base = getattr(hf_config, "rope_theta", 10000.0)
        self._history: List[int] = []
        self._request_id = "ring_attn_request"
        self._seq_id = 0

        # Combined block table used for decode; set after peer KV exchange.
        self._combined_block_table: Optional[List[int]] = None
        self._local_block_table: List[int] = []
        # Physical blocks reserved for the current peer chunk.  Shared across
        # all layers because one block table is used for every layer in vLLM.
        self._remote_block_table: List[int] = []

        print(
            f"[vllm block ring] loaded, vocab_size={self.vocab_size}, "
            f"block_size={self.block_size}, num_gpu_blocks={self.num_gpu_blocks}"
        )

    # ------------------------------------------------------------------
    # Cache / block helpers
    # ------------------------------------------------------------------
    @property
    def _cache_engine(self):
        """Return the driver worker's CacheEngine (TP=1)."""
        return self.llm.llm_engine.model_executor.driver_worker.cache_engine[0]

    def _gpu_cache(self, layer_idx: int) -> torch.Tensor:
        """Return the full KV cache tensor for a layer."""
        return self._cache_engine.gpu_cache[layer_idx]

    @property
    def num_layers(self) -> int:
        return self.model_config.get_num_layers(self.llm.llm_engine.parallel_config)

    @property
    def num_heads(self) -> int:
        return self.model_config.get_num_attention_heads(
            self.llm.llm_engine.parallel_config
        )

    @property
    def head_dim(self) -> int:
        return self.model_config.get_head_size()

    @property
    def num_kv_heads(self) -> int:
        return self.model_config.get_num_kv_heads(
            self.llm.llm_engine.parallel_config
        )

    def _cache_block_is_flat(self, cache: torch.Tensor) -> bool:
        """Detect XFormers-style flat block layout."""
        expected = self.block_size * self.num_kv_heads * self.head_dim
        return cache.dim() == 3 and cache.shape[2] == expected

    def extract_block(
        self, layer_idx: int, physical_block_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract K/V for one physical block.

        Returns:
            k, v tensors of shape [block_size, num_kv_heads, head_dim].
        """
        cache = self._gpu_cache(layer_idx)
        k = cache[0, physical_block_id].clone()
        v = cache[1, physical_block_id].clone()
        if self._cache_block_is_flat(cache):
            shape = (self.block_size, self.num_kv_heads, self.head_dim)
            k = k.view(shape)
            v = v.view(shape)
        return k, v

    def insert_block(
        self,
        layer_idx: int,
        physical_block_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write K/V into a reserved physical block slot."""
        cache = self._gpu_cache(layer_idx)
        if self._cache_block_is_flat(cache):
            k = k.reshape(-1)
            v = v.reshape(-1)
        cache[0, physical_block_id].copy_(k)
        cache[1, physical_block_id].copy_(v)

    def get_local_block_table(self) -> List[int]:
        """
        Return the physical block ids for the local chunk.

        After prefill these are the blocks we allocated explicitly; before
        prefill they come from the scheduler.
        """
        if self._local_block_table:
            return self._local_block_table
        engine = self.llm.llm_engine
        bm = engine.scheduler[0].block_manager
        seq = self._find_sequence()
        if seq is None:
            return []
        return bm.get_block_table(seq)

    def _find_sequence(self):
        """Find the first running/waiting sequence in the scheduler."""
        engine = self.llm.llm_engine
        for sg in engine.scheduler[0].running + engine.scheduler[0].waiting:
            if sg.get_seqs():
                return sg.get_seqs()[0]
        return None

    # ------------------------------------------------------------------
    # HcpWorkerBackend interface
    # ------------------------------------------------------------------
    def load_model(self, model_dir: str, device: str) -> None:
        pass

    def set_global_tokens(self, tokens: List[int]) -> None:
        """Set the full global token sequence after KV exchange."""
        self._history = list(tokens)

    def _zero_block(self, physical_block_id: int) -> None:
        """Zero a physical block across all KV-cache layers."""
        for layer_idx in range(self.num_layers):
            cache = self._gpu_cache(layer_idx)
            cache[0, physical_block_id] = 0
            cache[1, physical_block_id] = 0

    def _allocate_local_blocks(self, num_tokens: int) -> List[int]:
        """Allocate physical blocks for a local token sequence."""
        from vllm.utils import Device

        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        bm = self.llm.llm_engine.scheduler[0].block_manager
        blocks: List[int] = []
        prev_block = None
        for _ in range(num_blocks):
            block = bm.block_allocator.allocate_mutable_block(
                prev_block=prev_block, device=Device.GPU
            )
            assert block.block_id is not None
            self._zero_block(block.block_id)
            blocks.append(block.block_id)
            prev_block = block
        return blocks

    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        """
        Prefill the local chunk by directly calling the model executor.

        We allocate physical blocks ourselves (so they stay resident) and
        pass them via SequenceGroupMetadata.  This avoids the high-level
        generate() path, which would finish the request and free the blocks.
        """
        from vllm import SamplingParams
        from vllm.sequence import (
            ExecuteModelRequest,
            SequenceData,
            SequenceGroupMetadata,
        )

        self._history = list(chunk)
        self._local_block_table = self._allocate_local_blocks(len(chunk))

        seq_data = SequenceData.from_seqs(self._history)
        sampling_params = SamplingParams(temperature=0, max_tokens=1)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=self._request_id,
            is_prompt=True,
            seq_data={self._seq_id: seq_data},
            sampling_params=sampling_params,
            block_tables={self._seq_id: self._local_block_table},
            do_sample=True,
        )
        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=[seq_group_metadata]
        )

        outputs = self.llm.llm_engine.model_executor.execute_model(execute_model_req)
        token_id = outputs[0].outputs[0].samples[0].output_token

        logits = torch.full((self.vocab_size,), -1e9, dtype=torch.float32)
        logits[token_id] = 0.0
        return logits, seq_offset + len(self._history)

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        """
        Extract the local KV blocks covering [seq_start, seq_end).

        For vLLM the exchange granularity is a physical block, so we return
        all blocks that intersect the requested range.  The peer side will
        write them into matching positions in its combined block table.
        """
        return self.get_kv_block_from_table(
            layer_idx, seq_start, seq_end, self.get_local_block_table()
        )

    def get_kv_block_from_table(
        self,
        layer_idx: int,
        seq_start: int,
        seq_end: int,
        block_table: List[int],
        table_seq_offset: int = 0,
    ) -> KvBlock:
        """Extract K/V for a range using an explicit physical block table.

        ``table_seq_offset`` is the global token position represented by
        ``block_table[0]``.  For a peer chunk that starts at global position
        ``G``, pass ``table_seq_offset=G`` so the physical block indices line
        up with the requested global ``seq_start..seq_end``.
        """
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
            k = torch.stack(k_blocks, dim=0)  # [num_blocks, block_size, kv_heads, head_dim]
            v = torch.stack(v_blocks, dim=0)
        else:
            k = torch.empty(0)
            v = torch.empty(0)

        return KvBlock(layer_idx, seq_start, seq_end, k, v)

    def prefill_peer_chunk(self, chunk: List[int], seq_offset: int) -> List[int]:
        """
        Prefill a peer chunk into a fresh set of physical blocks.

        This is useful for single-process sanity checks where both the local
        and peer chunks are prefilled by the same vLLM instance.  The returned
        block ids can be passed to get_kv_block_from_table() and then to
        apply_peer_kv().
        """
        from vllm.sequence import (
            ExecuteModelRequest,
            SequenceData,
            SequenceGroupMetadata,
        )

        peer_block_table = self._allocate_local_blocks(len(chunk))
        from vllm import SamplingParams

        seq_data = SequenceData.from_seqs(chunk)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"{self._request_id}_peer_{seq_offset}",
            is_prompt=True,
            seq_data={self._seq_id: seq_data},
            sampling_params=SamplingParams(temperature=0, max_tokens=1),
            block_tables={self._seq_id: peer_block_table},
            do_sample=False,
        )
        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=[seq_group_metadata]
        )
        self.llm.llm_engine.model_executor.execute_model(execute_model_req)
        return peer_block_table

    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        """
        Insert peer KV blocks into reserved physical slots.

        The peer_block.k/v tensors are expected to be stacked physical blocks
        in global token order.  The same physical blocks are reused for every
        layer because vLLM's block table is shared across layers.

        Peer workers prefill their chunk with local positions (0..chunk_len-1),
        so we rotate the peer keys by the global start offset so that RoPE
        positions match the global sequence during decode.
        """
        if peer_block.k.numel() == 0:
            return

        num_blocks = peer_block.k.shape[0]
        if layer_idx == 0:
            # Reserve once and reuse the same physical ids for all layers.
            self._remote_block_table = self._reserve_remote_blocks(num_blocks)
            self._build_combined_block_table(peer_block.global_seq_start)
        else:
            # Sanity check: every layer must ship the same number of blocks.
            assert len(self._remote_block_table) == num_blocks, (
                f"peer block count mismatch at layer {layer_idx}: "
                f"expected {len(self._remote_block_table)}, got {num_blocks}"
            )

        # Correct RoPE positions from local (0-based) to global.
        peer_k = self._rope_delta_rotate_keys(
            peer_block.k, peer_block.global_seq_start
        )
        peer_v = peer_block.v
        total_peer_tokens = peer_block.global_seq_end - peer_block.global_seq_start
        for i, bid in enumerate(self._remote_block_table):
            k_blk = peer_k[i]
            v_blk = peer_v[i]
            # Zero out slots that do not contain real peer tokens so that any
            # backend that reads the full physical block is not poisoned by
            # uninitialized cache memory.
            valid = min(self.block_size, total_peer_tokens - i * self.block_size)
            if valid < self.block_size:
                k_blk = k_blk.clone()
                v_blk = v_blk.clone()
                k_blk[valid:] = 0.0
                v_blk[valid:] = 0.0
            self.insert_block(layer_idx, bid, k_blk, v_blk)

    def _reserve_remote_blocks(self, num_blocks: int) -> List[int]:
        """Reserve free physical blocks for incoming peer KV."""
        engine = self.llm.llm_engine
        bm = engine.scheduler[0].block_manager
        from vllm.utils import Device

        reserved = []
        prev_block = None
        for _ in range(num_blocks):
            # Use the allocator directly to obtain a mutable GPU block.
            block = bm.block_allocator.allocate_mutable_block(
                prev_block=prev_block, device=Device.GPU
            )
            assert block.block_id is not None
            self._zero_block(block.block_id)
            reserved.append(block.block_id)
            prev_block = block
        return reserved

    def _rope_delta_rotate_keys(
        self, k_blocks: torch.Tensor, delta: int
    ) -> torch.Tensor:
        """
        Rotate cached key vectors by ``delta`` positions.

        Peer workers prefill with local positions (0..chunk_len-1).  Applying
        a delta rotation brings those keys to their global positions so that
        RoPE-aligned decode queries attend correctly.  Values are not
        position-dependent and are left unchanged.
        """
        if delta == 0:
            return k_blocks

        orig_shape = k_blocks.shape
        # [num_blocks * block_size, num_kv_heads, head_dim]
        k = k_blocks.reshape(-1, *orig_shape[2:])
        dtype = k.dtype
        device = k.device
        kf = k.to(torch.float32)

        # Reshape to complex pairs (Neox/GPT-NeoX style RoPE).
        head_dim = self.head_dim
        x = kf.reshape(*kf.shape[:-1], head_dim // 2, 2)
        x_complex = torch.view_as_complex(x)

        inv_freq = 1.0 / (
            self.rope_base
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
        )
        # Rotate by -delta: the peer prefilled with local positions 0..L-1,
        # but the key must appear at global positions delta..delta+L-1.
        # Standard RoPE applies key rotation exp(-i*m*theta); shifting by
        # +delta positions therefore multiplies by exp(-i*delta*theta).
        angles = -delta * inv_freq  # [head_dim // 2]
        rot = torch.exp(1j * angles)  # [head_dim // 2]

        x_rot = x_complex * rot[None, None, :]
        x_out = torch.view_as_real(x_rot)  # [L, H, head_dim//2, 2]
        return x_out.reshape(orig_shape).to(dtype)

    def _build_combined_block_table(self, peer_seq_start: int) -> None:
        """
        Merge local and remote physical block ids into a single block table
        in global token order.
        """
        local_table = self.get_local_block_table()
        peer_reserved = self._remote_block_table
        peer_start_block = (peer_seq_start + self.block_size - 1) // self.block_size
        # Simple two-domain merge: local covers [0, peer_start), remote covers
        # [peer_start, peer_start + len(peer_reserved)).
        combined = [0] * (peer_start_block + len(peer_reserved))
        combined[:peer_start_block] = local_table[:peer_start_block]
        combined[peer_start_block:] = peer_reserved
        self._combined_block_table = combined

    def decode(self, token: int) -> torch.Tensor:
        """
        Decode one token using the combined block table.

        We bypass the scheduler and call model_executor.execute_model with a
        manually constructed SequenceGroupMetadata so that vLLM's
        PagedAttention kernel attends over local + remote KV blocks.
        """
        from vllm import SamplingParams
        from vllm.sequence import (
            ExecuteModelRequest,
            SequenceData,
            SequenceGroupMetadata,
        )

        self._history.append(token)
        full_token_ids = self._history
        seq_len = len(full_token_ids)
        seq_id = self._seq_id

        # Build a SequenceData whose last token is the decode input and whose
        # earlier tokens are marked as computed.  is_prompt=False tells the
        # model runner to treat this as a decode step.
        seq_data = SequenceData.from_seqs(full_token_ids)
        seq_data.update_num_computed_tokens(seq_len - 1)

        block_tables = {seq_id: self._combined_block_table or self.get_local_block_table()}
        sampling_params = SamplingParams(temperature=0, max_tokens=1)

        seq_group_metadata = SequenceGroupMetadata(
            request_id=self._request_id,
            is_prompt=False,
            seq_data={seq_id: seq_data},
            sampling_params=sampling_params,
            block_tables=block_tables,
            do_sample=True,
        )
        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=[seq_group_metadata]
        )

        outputs = self.llm.llm_engine.model_executor.execute_model(execute_model_req)
        # outputs[0] is a SamplerOutput for the batch.
        token_id = outputs[0].outputs[0].samples[0].output_token

        logits = torch.full((self.vocab_size,), -1e9, dtype=torch.float32)
        logits[token_id] = 0.0
        return logits

    @property
    def capacity_mb(self) -> int:
        try:
            if self.device == "npu":
                import torch_npu
                free, _ = torch_npu.npu.mem_get_info()
                return int(free // (1024 * 1024))
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return int(free // (1024 * 1024))
        except Exception:
            pass
        return 4096

    def shutdown(self) -> None:
        print("[vllm block ring] shutting down...")
        try:
            del self.llm
            import gc
            gc.collect()
        except Exception as e:
            print(f"[vllm block ring] shutdown warning: {e}")
