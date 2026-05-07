#!/usr/bin/env python3
"""
Phase 3.2: Two Python workers exchange KV blocks via QUIC.

Tests bidirectional KV transport between two Python processes using aioquic.
"""

import asyncio
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hcp_worker_sdk.quic_transport import create_quic_server, create_quic_client, QuicKvTransport
from hcp_worker_sdk.types import KvBlock


PEER_HOST = "127.0.0.1"
PEER_PORT = 26099
DEVICE = torch.device("cpu")


def make_test_block(layer_idx=0, seq_start=0, seq_end=4) -> KvBlock:
    k = torch.arange(seq_end - seq_start, dtype=torch.float32, device=DEVICE).view(1, 1, seq_end - seq_start, 1)
    v = torch.arange(seq_end - seq_start, dtype=torch.float32, device=DEVICE).view(1, 1, seq_end - seq_start, 1) + 100.0
    return KvBlock(layer_idx, seq_start, seq_end, k, v)


async def worker0_client():
    """Worker 0: connect to worker 1, send KV block, receive modified block back."""
    print("[worker0] connecting to worker1...")
    reader, writer, conn_mgr = await create_quic_client(PEER_HOST, PEER_PORT, send_dummy=True)
    transport = QuicKvTransport(reader, writer, DEVICE, dummy_sent=True)
    print("[worker0] connected")

    block = make_test_block(layer_idx=0, seq_start=0, seq_end=4)
    print(f"[worker0] sending KV block: k_sum={block.k.sum().item():.1f}, v_sum={block.v.sum().item():.1f}")

    await transport._send_kv_block(block)
    peer_block = await transport._recv_kv_block()

    if peer_block is None:
        print("[worker0] ❌ received None")
        await conn_mgr.__aexit__(None, None, None)
        return False

    print(f"[worker0] received peer block: k_sum={peer_block.k.sum().item():.1f}, v_sum={peer_block.v.sum().item():.1f}")

    expected_k = block.k + 1.0
    expected_v = block.v + 1.0
    k_diff = (peer_block.k - expected_k).abs().max().item()
    v_diff = (peer_block.v - expected_v).abs().max().item()

    print(f"[worker0] k_diff={k_diff:.6f}, v_diff={v_diff:.6f}")

    await conn_mgr.__aexit__(None, None, None)
    return k_diff < 1e-5 and v_diff < 1e-5


async def worker1_server():
    """Worker 1: listen, receive KV block, add 1.0, send back."""
    print("[worker1] starting server...")
    connected_event, accepted_streams, server_task = await create_quic_server(PEER_HOST, PEER_PORT)

    await asyncio.wait_for(connected_event.wait(), timeout=10.0)
    print("[worker1] connection accepted")

    reader, writer = accepted_streams[0]
    transport = QuicKvTransport(reader, writer, DEVICE)

    peer_block = await transport._recv_kv_block()
    if peer_block is None:
        print("[worker1] ❌ received None")
        server_task.cancel()
        return False

    print(f"[worker1] received block: k_sum={peer_block.k.sum().item():.1f}, v_sum={peer_block.v.sum().item():.1f}")

    modified = KvBlock(
        layer_idx=peer_block.layer_idx,
        global_seq_start=peer_block.global_seq_start,
        global_seq_end=peer_block.global_seq_end,
        k=peer_block.k + 1.0,
        v=peer_block.v + 1.0,
    )
    print(f"[worker1] sending back modified block: k_sum={modified.k.sum().item():.1f}, v_sum={modified.v.sum().item():.1f}")

    await transport._send_kv_block(modified)

    server_task.cancel()
    return True


async def main():
    server_task = asyncio.create_task(worker1_server())
    await asyncio.sleep(0.5)
    client_task = asyncio.create_task(worker0_client())

    results = await asyncio.gather(server_task, client_task, return_exceptions=True)
    print(f"\nresults: {results}")

    if all(r is True for r in results):
        print("\n✅ Python↔Python QUIC KV ring test PASSED")
        return 0
    else:
        print("\n❌ test FAILED")
        for r in results:
            if isinstance(r, Exception):
                import traceback
                traceback.print_exception(type(r), r, r.__traceback__)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
