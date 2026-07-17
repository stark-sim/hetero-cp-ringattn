"""
Context-passing CP worker server for vLLM block-ring backends.

The stock ``QuicWorkerServer`` runs *prefill-then-exchange*: every worker
prefills its own chunk in isolation and only afterwards swaps KV around the
ring.  For the transformers backend that is a tolerable approximation (the
last domain recomputes the last-token logits with ``recalculate_logits``),
but for vLLM PagedAttention it is *incorrect*: a later domain's K/V at layer
L depends on the earlier domains' K/V at layer L-1 (causal context), so a
context-free prefill produces wrong K/V at every layer >= 1.

This server implements *context-passing* prefill for 2 domains:

- domain 0: prefill own chunk -> send its KV to the peer -> receive the
  peer's KV -> apply it -> now holds the full KV for decode.
- domain 1: receive the prior domain's KV *first* -> prefill its own chunk
  WITH that context (``prefill_with_context_kv``) -> send its KV to the peer.

After prefill both domains hold the full KV, so decode proceeds identically
on either side and the coordinator can read decode logits from worker 0.
"""

import torch

from .quic_server import QuicWorkerServer
from .types import KvBlock


class CpVllmWorkerServer(QuicWorkerServer):
    """Context-passing variant of QuicWorkerServer for vLLM block-ring plugins."""

    async def _handle_prefill(self, cmd: dict) -> dict:
        request_id = cmd["request_id"]
        chunk = cmd["chunk"]
        self.seq_offset = cmd["seq_offset"]

        if self.domain_id == 0:
            logits, seq_len = self.backend.prefill(chunk, self.seq_offset)
            await self._send_own_kv()
            await self._recv_and_apply_peer_kv()
        else:
            context_kv = await self._recv_all_context_kv()
            logits, seq_len = self.backend.prefill_with_context_kv(
                chunk,
                self.seq_offset,
                context_kv,
                context_len=self.seq_offset,
            )
            await self._send_own_kv()

        # Report the seq_len this worker actually computed logits for.  The
        # coordinator picks the worker with the max, which must be the *last*
        # domain (its logits are the full-sequence next-token prediction).
        # Domain 0 must NOT report the post-exchange full length, or its
        # chunk-local logits would wrongly win.
        self.global_seq_len = seq_len
        logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()
        return {
            "kind": "PrefillDone",
            "request_id": request_id,
            "last_logits_bytes": logits_bytes,
            "global_seq_len": self.global_seq_len,
        }

    async def _handle_decode(self, cmd: dict) -> dict:
        request_id = cmd["request_id"]
        token = cmd["token"]
        logits = self.backend.decode(token)
        logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()
        return {
            "kind": "DecodeDone",
            "request_id": request_id,
            "logits_bytes": logits_bytes,
        }

    # ------------------------------------------------------------------
    # KV ring helpers
    # ------------------------------------------------------------------
    async def _send_own_kv(self) -> None:
        """Send this worker's own chunk KV (all layers) to the peer."""
        if self.num_domains <= 1 or self.kv_transport is None:
            return
        start = self.seq_offset
        end = self.backend._global_seq_len
        for layer_idx in range(self.backend.num_layers):
            block = self.backend.get_kv_block(layer_idx, start, end)
            await self.kv_transport._send_kv_block(block)
        print(f"[worker {self.domain_id}] sent KV for [{start},{end}) "
              f"({self.backend.num_layers} layers)")

    async def _recv_all_context_kv(self) -> list:
        """Receive the prior domain's KV (all layers) to use as context."""
        blocks = []
        if self.kv_transport is None:
            return blocks
        for _ in range(self.backend.num_layers):
            block = await self.kv_transport._recv_kv_block()
            if block is None:
                break
            blocks.append(block)
        print(f"[worker {self.domain_id}] received context KV "
              f"({len(blocks)} layers, "
              f"[{blocks[0].global_seq_start},{blocks[0].global_seq_end}) "
              f"if any)")
        return blocks

    async def _recv_and_apply_peer_kv(self) -> None:
        """Receive the peer's KV (all layers) and merge it into our cache."""
        if self.kv_transport is None:
            return
        for _ in range(self.backend.num_layers):
            block = await self.kv_transport._recv_kv_block()
            if block is None:
                break
            # Peer chunk was prefilled with global positions, so no rotation.
            self.backend.apply_peer_kv(block.layer_idx, block, rotate_delta=0)
        # The peer's chunk extends the global sequence; adopt the larger end.
        self.backend.set_global_seq_len(
            max(self.backend._global_seq_len,
                self._combined_end_from_block_table())
        )
        print(f"[worker {self.domain_id}] applied peer KV, "
              f"combined table={self.backend._combined_block_table}")

    def _combined_end_from_block_table(self) -> int:
        table = self.backend._combined_block_table or []
        return len(table) * self.backend.block_size
