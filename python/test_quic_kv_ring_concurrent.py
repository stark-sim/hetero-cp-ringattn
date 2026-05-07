#!/usr/bin/env python3
"""
Phase 3.2: Concurrent bidirectional KV exchange between two Python workers.

Both workers simultaneously send and receive KV blocks via QUIC bidirectional
stream. This matches the real ring-attention exchange pattern where each peer
sends its local KV block while receiving the peer's block concurrently.
"""

import asyncio
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hcp_worker_sdk.quic_transport import create_quic_server, create_quic_client, QuicKvTransport
from hcp_worker_sdk.types import KvBlock


PEER_HOST = "127.0.0.1"
PEER_PORT = 26098
DEVICE = torch.device("cpu")


def make_test_block(layer_idx, seq_start, seq_end, seed) -> KvBlock:
    torch.manual_seed(seed)
    k = torch.randn(1, 2, seq_end - seq_start, 4, dtype=torch.float32, device=DEVICE)
    v = torch.randn(1, 2, seq_end - seq_start, 4, dtype=torch.float32, device=DEVICE)
    return KvBlock(layer_idx, seq_start, seq_end, k, v)


async def worker0_client():
    """Worker 0: connect, then concurrently send local block and receive peer block."""
    print("[worker0] connecting...")
    reader, writer, conn_mgr = await create_quic_client(PEER_HOST, PEER_PORT, send_dummy=True)
    transport = QuicKvTransport(reader, writer, DEVICE, dummy_sent=True)
    print("[worker0] connected")

    local_block = make_test_block(0, 0, 8, seed=42)
    print(f"[worker0] sending block: k_sum={local_block.k.sum().item():.3f}, v_sum={local_block.v.sum().item():.3f}")

    # Direct async call (avoid ThreadPoolExecutor loop mismatch)
    peer_block = await transport._exchange_kv_block(local_block)

    if peer_block is None:
        print("[worker0] ❌ received None")
        await conn_mgr.__aexit__(None, None, None)
        return False

    print(f"[worker0] received block: k_sum={peer_block.k.sum().item():.3f}, v_sum={peer_block.v.sum().item():.3f}")

    # Verify received block matches what worker1 sent
    expected = make_test_block(0, 0, 8, seed=123)
    k_diff = (peer_block.k - expected.k).abs().max().item()
    v_diff = (peer_block.v - expected.v).abs().max().item()
    print(f"[worker0] k_diff={k_diff:.6f}, v_diff={v_diff:.6f}")

    await conn_mgr.__aexit__(None, None, None)
    return k_diff < 1e-5 and v_diff < 1e-5


async def worker1_server():
    """Worker 1: listen, then concurrently send local block and receive peer block."""
    print("[worker1] starting server...")
    connected_event, accepted_streams, server_task = await create_quic_server(PEER_HOST, PEER_PORT)

    await asyncio.wait_for(connected_event.wait(), timeout=10.0)
    print("[worker1] connection accepted")

    reader, writer = accepted_streams[0]
    transport = QuicKvTransport(reader, writer, DEVICE)

    local_block = make_test_block(0, 0, 8, seed=123)
    print(f"[worker1] sending block: k_sum={local_block.k.sum().item():.3f}, v_sum={local_block.v.sum().item():.3f}")

    peer_block = await transport._exchange_kv_block(local_block)

    if peer_block is None:
        print("[worker1] ❌ received None")
        server_task.cancel()
        return False

    print(f"[worker1] received block: k_sum={peer_block.k.sum().item():.3f}, v_sum={peer_block.v.sum().item():.3f}")

    # Verify received block matches what worker0 sent
    expected = make_test_block(0, 0, 8, seed=42)
    k_diff = (peer_block.k - expected.k).abs().max().item()
    v_diff = (peer_block.v - expected.v).abs().max().item()
    print(f"[worker1] k_diff={k_diff:.6f}, v_diff={v_diff:.6f}")

    server_task.cancel()
    return k_diff < 1e-5 and v_diff < 1e-5


async def main():
    # Both start at the same time → concurrent send+recv
    server_task = asyncio.create_task(worker1_server())
    client_task = asyncio.create_task(worker0_client())

    results = await asyncio.gather(server_task, client_task, return_exceptions=True)
    print(f"\nresults: {results}")

    if all(r is True for r in results):
        print("\n✅ Concurrent Python↔Python QUIC KV exchange PASSED")
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
