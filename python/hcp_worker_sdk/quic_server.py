"""
HCP Worker Server — QUIC control-plane + QUIC KV transport version.

Replaces HcpWorkerServer (TCP/JSON) with QUIC/bincode for coordinator
and QUIC streams for peer KV ring exchange.
"""

import asyncio
import torch
from typing import Optional

from .backend import HcpWorkerBackend
from .bincode import encode_response, decode_command
from .quic_control import QuicControlClient
from .quic_transport import QuicKvTransport, create_quic_server, create_quic_client
from .types import KvBlock


class QuicWorkerServer:
    """
    HCP Worker server using QUIC for both control plane and data plane.

    Flow:
    1. Connect to coordinator (QUIC + bincode)
    2. Setup peer connection (QUIC KV transport)
    3. Command loop: Prefill → KV ring exchange → response
                     Decode  → response
                     Shutdown → exit
    """

    def __init__(
        self,
        backend: HcpWorkerBackend,
        domain_id: int,
        num_domains: int,
        device: torch.device,
    ):
        self.backend = backend
        self.domain_id = domain_id
        self.num_domains = num_domains
        self.device = device
        self.global_seq_len = 0
        self.seq_offset = 0
        self.control_client: Optional[QuicControlClient] = None
        self.kv_transport: Optional[QuicKvTransport] = None

    async def run(
        self,
        coordinator_host: str,
        coordinator_port: int,
        peer_listen_host: str,
        peer_listen_port: int,
        next_peer_host: str,
        next_peer_port: int,
    ) -> None:
        """Main worker event loop."""
        # 1. Setup peer connection (before coordinator, to avoid deadlock)
        await self._setup_peer_connection(
            peer_listen_host, peer_listen_port,
            next_peer_host, next_peer_port,
        )

        # 2. Connect to coordinator
        self.control_client = QuicControlClient()
        await self.control_client.connect(coordinator_host, coordinator_port)
        await self.control_client.send_handshake(
            domain_id=self.domain_id,
            capacity_mb=self.backend.capacity_mb,
        )
        print(f"[worker {self.domain_id}] handshake sent, capacity={self.backend.capacity_mb} MB")

        # 3. Command loop
        while True:
            cmd = await self.control_client.recv_command()
            kind = cmd["kind"]
            print(f"[worker {self.domain_id}] received: {kind}")

            if kind == "Prefill":
                resp = await self._handle_prefill(cmd)
                await self.control_client.send_response(**resp)

            elif kind == "SyncGlobalSeqLen":
                self.global_seq_len = cmd["global_seq_len"]
                print(f"[worker {self.domain_id}] synced global_seq_len = {self.global_seq_len}")

            elif kind == "Decode":
                resp = await self._handle_decode(cmd)
                await self.control_client.send_response(**resp)

            elif kind == "Shutdown":
                print(f"[worker {self.domain_id}] shutting down")
                break

        await self.control_client.close()

    async def _setup_peer_connection(
        self,
        listen_host: str,
        listen_port: int,
        next_host: str,
        next_port: int,
    ) -> None:
        """Setup bidirectional QUIC stream with next peer in the ring."""
        if self.num_domains <= 1:
            return

        if self.domain_id == 0:
            # Domain 0 connects to next_peer first
            print(f"[worker {self.domain_id}] connecting to peer {next_host}:{next_port}...")
            reader, writer, conn_mgr = await create_quic_client(next_host, next_port, send_dummy=True)
            self.kv_transport = QuicKvTransport(reader, writer, self.device, dummy_sent=True)
            # Store conn_mgr to keep connection alive
            self._peer_conn_mgr = conn_mgr
            print(f"[worker {self.domain_id}] peer connected")
        else:
            # Domain N listens first
            print(f"[worker {self.domain_id}] listening for peer on {listen_host}:{listen_port}...")
            connected_event, accepted_streams, server_task = await create_quic_server(listen_host, listen_port)
            await asyncio.wait_for(connected_event.wait(), timeout=30.0)
            reader, writer = accepted_streams[0]
            self.kv_transport = QuicKvTransport(reader, writer, self.device)
            self._peer_server_task = server_task
            print(f"[worker {self.domain_id}] peer accepted")

    async def _handle_prefill(self, cmd: dict) -> dict:
        """Run prefill, exchange KV ring, return PrefillDone."""
        chunk = cmd["chunk"]
        self.seq_offset = cmd["seq_offset"]

        logits, seq_len = self.backend.prefill(chunk, self.seq_offset)
        self.global_seq_len = seq_len

        # KV Ring exchange
        await self._exchange_kv_ring(prefill=True)

        logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()
        return {
            "kind": "PrefillDone",
            "last_logits_bytes": logits_bytes,
            "global_seq_len": self.global_seq_len,
        }

    async def _handle_decode(self, cmd: dict) -> dict:
        """Run decode, return DecodeDone."""
        token = cmd["token"]
        logits = self.backend.decode(token)

        # Decode phase typically skips KV exchange (all workers have same full KV)
        # await self._exchange_kv_ring(prefill=False)

        logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()
        return {
            "kind": "DecodeDone",
            "logits_bytes": logits_bytes,
        }

    async def _exchange_kv_ring(self, prefill: bool) -> None:
        """Exchange KV blocks through the ring."""
        if self.num_domains <= 1 or self.kv_transport is None:
            return

        if not prefill:
            return  # Skip decode phase KV exchange

        for layer_idx in range(self.backend.num_layers):
            seq_start = self.seq_offset
            seq_end = self.global_seq_len

            local_block = self.backend.get_kv_block(layer_idx, seq_start, seq_end)

            for _round in range(self.num_domains - 1):
                peer_block = await self.kv_transport._exchange_kv_block(local_block)
                if peer_block is None:
                    break
                self.backend.apply_peer_kv(layer_idx, peer_block)
                local_block = peer_block  # Forward to next peer
