"""
HCP Worker Server — QUIC control-plane + QUIC KV transport version.

Replaces HcpWorkerServer (TCP/JSON) with QUIC/bincode for coordinator
and QUIC streams for peer KV ring exchange.
"""

import asyncio
import torch
from typing import List, Optional, Tuple

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
        self.kv_transport_out: Optional[QuicKvTransport] = None  # to next peer
        self.kv_transport_in: Optional[QuicKvTransport] = None   # from prev peer

    async def run(
        self,
        coordinator_host: str,
        coordinator_port: int,
        peer_listen_host: str,
        peer_listen_port: int,
        next_peer_host: str,
        next_peer_port: int,
        shutdown_event: Optional[asyncio.Event] = None,
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
            if shutdown_event is not None and shutdown_event.is_set():
                print(f"[worker {self.domain_id}] shutdown event received, exiting command loop")
                break

            try:
                cmd = await self.control_client.recv_command()
            except ConnectionError:
                print(f"[worker {self.domain_id}] coordinator closed connection, exiting")
                break
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

    async def cleanup(self) -> None:
        """Cleanup QUIC connections and peer resources."""
        print(f"[worker {self.domain_id}] cleaning up connections...")
        if self.control_client is not None:
            try:
                await self.control_client.close()
            except Exception as e:
                print(f"[worker {self.domain_id}] control client close warning: {e}")
        if hasattr(self, "_peer_conn_mgr"):
            try:
                self._peer_conn_mgr.close()
            except Exception as e:
                print(f"[worker {self.domain_id}] peer conn mgr close warning: {e}")
        if hasattr(self, "_peer_server_task"):
            try:
                self._peer_server_task.cancel()
            except Exception as e:
                print(f"[worker {self.domain_id}] peer server task cancel warning: {e}")
        print(f"[worker {self.domain_id}] cleanup done")

    async def _setup_peer_connection(
        self,
        listen_host: str,
        listen_port: int,
        next_host: str,
        next_port: int,
    ) -> None:
        """Setup N-domain ring: every worker dials next AND accepts from prev.

        KV flows i -> i+1 over each worker's outbound (client) connection;
        inbound (server) connection carries KV from the prev peer.
        num_domains=2 is just the N=2 special case of the same topology.
        """
        if self.num_domains <= 1:
            return

        print(f"[worker {self.domain_id}] listening for peer on {listen_host}:{listen_port}...")
        connected_event, accepted_streams, server_task = await create_quic_server(listen_host, listen_port)
        self._peer_server_task = server_task

        async def dial_next():
            print(f"[worker {self.domain_id}] connecting to peer {next_host}:{next_port}...")
            for attempt in range(1, 31):
                try:
                    reader, writer, conn_mgr = await asyncio.wait_for(
                        create_quic_client(next_host, next_port, send_dummy=True),
                        timeout=30.0,
                    )
                    return reader, writer, conn_mgr
                except (ConnectionError, asyncio.TimeoutError, OSError) as e:
                    print(f"[worker {self.domain_id}] peer connect attempt {attempt}/30 failed: {e}")
                    await asyncio.sleep(2.0)
            raise ConnectionError(f"failed to connect to peer {next_host}:{next_port} after 30 attempts")

        (out_reader, out_writer, conn_mgr), _ = await asyncio.gather(
            dial_next(),
            asyncio.wait_for(connected_event.wait(), timeout=300.0),
        )
        self._peer_conn_mgr = conn_mgr
        self.kv_transport_out = QuicKvTransport(out_reader, out_writer, self.device, dummy_sent=True)
        print(f"[worker {self.domain_id}] peer connected (outbound)")
        in_reader, in_writer = accepted_streams[0]
        self.kv_transport_in = QuicKvTransport(in_reader, in_writer, self.device)
        print(f"[worker {self.domain_id}] peer accepted (inbound)")

    async def _handle_prefill(self, cmd: dict) -> dict:
        """Run prefill, exchange KV ring, return PrefillDone."""
        request_id = cmd["request_id"]
        chunk = cmd["chunk"]
        self.seq_offset = cmd["seq_offset"]
        position_ids = cmd.get("position_ids")
        if position_ids is not None:
            expected = list(range(self.seq_offset, self.seq_offset + len(chunk)))
            if list(position_ids) != expected:
                raise NotImplementedError(
                    f"non-contiguous position_ids (striped/zigzag ring strategy) "
                    f"are not supported by the Python worker"
                )

        if self.num_domains > 1 and hasattr(self.backend, "prefill_ring_exact"):
            # 精确 CP：逐层 KV ring 交换在 forward 过程中完成（与 Rust Q-ring
            # 同算法）；最后一个 domain 的最后位置 logits 直接精确，无需 recalc。
            logits, seq_len = await self._prefill_ring_exact(chunk)
        else:
            logits, seq_len = self.backend.prefill(chunk, self.seq_offset, position_ids=position_ids)

            # KV Ring exchange
            await self._exchange_kv_ring(prefill=True)

            # KV exchange 后，最后一个 domain 用完整 KV 重新计算 logits
            # （只有最后一个 domain 的 self._history[-1] 是 prompt 最后一个 token）
            if hasattr(self.backend, 'recalculate_logits') and self.domain_id == self.num_domains - 1:
                logits = self.backend.recalculate_logits()
        self.global_seq_len = seq_len

        logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()
        return {
            "kind": "PrefillDone",
            "request_id": request_id,
            "last_logits_bytes": logits_bytes,
            "global_seq_len": self.global_seq_len,
        }

    async def _prefill_ring_exact(self, chunk: List[int]) -> Tuple[torch.Tensor, int]:
        """精确 CP prefill：逐层调用 backend.prefill_ring_exact，每层做 N-1 轮
        ring 交换（发送本层本地块、接收并转发前序块），与 Rust worker 同构。"""
        from .types import KvBlock

        my_start = self.seq_offset
        my_len = len(chunk)

        async def exchange_layer(layer_idx: int, k: torch.Tensor, v: torch.Tensor):
            block = KvBlock(layer_idx, my_start, my_start + my_len, k, v)
            remote = []
            cur = block
            for _round in range(self.num_domains - 1):
                send_task = asyncio.create_task(self.kv_transport_out._send_kv_block(cur))
                recv_task = asyncio.create_task(self.kv_transport_in._recv_kv_block())
                await asyncio.gather(send_task, recv_task)
                blk = recv_task.result()
                if blk is None:
                    raise ConnectionError(f"ring peer closed during prefill layer {layer_idx}")
                remote.append(blk)
                cur = blk  # forward to next peer
            return remote

        return await self.backend.prefill_ring_exact(chunk, my_start, exchange_layer)

    async def _handle_decode(self, cmd: dict) -> dict:
        """Run decode, return DecodeDone."""
        request_id = cmd["request_id"]
        token = cmd["token"]
        logits = self.backend.decode(token)

        # Decode phase typically skips KV exchange (all workers have same full KV)
        # await self._exchange_kv_ring(prefill=False)

        logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()
        return {
            "kind": "DecodeDone",
            "request_id": request_id,
            "logits_bytes": logits_bytes,
        }

    async def _exchange_kv_ring(self, prefill: bool) -> None:
        """Exchange KV blocks through the ring (N-domain).

        Per layer: send local block to next peer while receiving from prev
        peer, forward the received block next round. After num_domains-1
        rounds every worker has seen every domain's chunk exactly once.
        """
        if self.num_domains <= 1 or self.kv_transport_out is None:
            return

        if not prefill:
            return  # Skip decode phase KV exchange

        for layer_idx in range(self.backend.num_layers):
            seq_start = self.seq_offset
            seq_end = self.global_seq_len

            local_block = self.backend.get_kv_block(layer_idx, seq_start, seq_end)

            for _round in range(self.num_domains - 1):
                send_task = asyncio.create_task(self.kv_transport_out._send_kv_block(local_block))
                recv_task = asyncio.create_task(self.kv_transport_in._recv_kv_block())
                await asyncio.gather(send_task, recv_task)
                peer_block = recv_task.result()
                if peer_block is None:
                    break
                self.backend.apply_peer_kv(layer_idx, peer_block)
                local_block = peer_block  # Forward to next peer
