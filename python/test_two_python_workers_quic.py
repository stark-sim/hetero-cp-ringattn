#!/usr/bin/env python3
"""
Phase 3.2: Two Python mock workers via QUIC control-plane + QUIC KV ring.
"""

import asyncio
import os
import subprocess
import sys
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hcp_worker_sdk.quic_control import QuicControlClient
from hcp_worker_sdk.quic_transport import create_quic_server, create_quic_client, QuicKvTransport
from hcp_worker_sdk.types import KvBlock
import torch


MODEL_DIR = "models/Qwen2-0.5B"
VOCAB_SIZE = 151936
COORDINATOR_ADDR = ("127.0.0.1", 26011)
PEER_PORTS = [26081, 26082]
DEVICE = torch.device("cpu")


def make_dummy_logits() -> bytes:
    return struct.pack(f"<{VOCAB_SIZE}f", *[0.0] * VOCAB_SIZE)


async def run_worker(domain_id: int, num_domains: int, server_ready: asyncio.Event = None):
    """Python mock worker with QUIC control-plane + QUIC KV ring."""
    my_peer_port = PEER_PORTS[domain_id]
    next_peer_port = PEER_PORTS[(domain_id + 1) % num_domains]

    kv_transport = None
    peer_conn = None

    if num_domains > 1:
        if domain_id == 0:
            # Wait for server to be ready before connecting
            if server_ready is not None:
                await asyncio.wait_for(server_ready.wait(), timeout=10.0)
                await asyncio.sleep(0.3)
            print(f"[worker {domain_id}] connecting to peer on port {next_peer_port}...")
            reader, writer, conn_mgr = await create_quic_client("127.0.0.1", next_peer_port, send_dummy=True)
            kv_transport = QuicKvTransport(reader, writer, DEVICE, dummy_sent=True)
            peer_conn = conn_mgr
            print(f"[worker {domain_id}] peer connected")
        else:
            print(f"[worker {domain_id}] listening for peer on port {my_peer_port}...")
            connected_event, accepted_streams, server_task = await create_quic_server("127.0.0.1", my_peer_port)
            # Signal that server is ready
            if server_ready is not None:
                server_ready.set()
            await asyncio.wait_for(connected_event.wait(), timeout=30.0)
            reader, writer = accepted_streams[0]
            kv_transport = QuicKvTransport(reader, writer, DEVICE)
            peer_conn = server_task
            print(f"[worker {domain_id}] peer accepted")

    # Connect to coordinator
    client = QuicControlClient()
    await client.connect(*COORDINATOR_ADDR)
    await client.send_handshake(domain_id=domain_id, capacity_mb=4096)
    print(f"[worker {domain_id}] handshake sent")

    global_seq_len = 0
    exchanged_blocks = []

    try:
        while True:
            cmd = await client.recv_command()
            kind = cmd["kind"]
            print(f"[worker {domain_id}] cmd: {kind}")

            if kind == "Prefill":
                chunk = cmd["chunk"]
                seq_offset = cmd["seq_offset"]
                global_seq_len = len(chunk) + seq_offset

                if kv_transport is not None:
                    dummy_k = torch.ones(1, 2, len(chunk), 4, dtype=torch.float32)
                    dummy_v = torch.ones(1, 2, len(chunk), 4, dtype=torch.float32) * 2.0
                    local_block = KvBlock(0, seq_offset, global_seq_len, dummy_k, dummy_v)
                    print(f"[worker {domain_id}] exchanging KV block...")
                    peer_block = await kv_transport._exchange_kv_block(local_block)
                    if peer_block is not None:
                        exchanged_blocks.append(peer_block)
                        print(f"[worker {domain_id}] KV exchange done")

                logits_bytes = make_dummy_logits()
                await client.send_response(
                    "PrefillDone",
                    last_logits_bytes=logits_bytes,
                    global_seq_len=global_seq_len,
                )

            elif kind == "SyncGlobalSeqLen":
                global_seq_len = cmd["global_seq_len"]
                print(f"[worker {domain_id}] SyncGlobalSeqLen = {global_seq_len}")

            elif kind == "Decode":
                token = cmd["token"]
                logits_bytes = make_dummy_logits()
                await client.send_response("DecodeDone", logits_bytes=logits_bytes)

            elif kind == "Shutdown":
                print(f"[worker {domain_id}] Shutdown")
                break

    except Exception as e:
        print(f"[worker {domain_id}] error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await client.close()
        if peer_conn is not None:
            if hasattr(peer_conn, '__aexit__'):
                await peer_conn.__aexit__(None, None, None)
            elif hasattr(peer_conn, 'cancel'):
                peer_conn.cancel()
        print(f"[worker {domain_id}] disconnected")

    return len(exchanged_blocks)


def start_coordinator(num_domains: int = 2):
    model_dir = os.path.abspath(MODEL_DIR)
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = "/Users/stark_sim/libtorch/lib"
    cmd = [
        "cargo", "run", "--features", "tch-backend", "--bin", "hcp-ringattn-rust",
        "--", "--distributed-role", "coordinator",
        "--model-dir", model_dir,
        "--prompt", "Hello world",
        "--max-tokens", "2",
        "--num-domains", str(num_domains),
        "--listen-addr", f"{COORDINATOR_ADDR[0]}:{COORDINATOR_ADDR[1]}",
    ]
    print(f"[test] starting coordinator: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd="rust", env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return proc


async def main():
    num_domains = 2
    proc = start_coordinator(num_domains)
    await asyncio.sleep(3)

    try:
        server_ready = asyncio.Event()
        tasks = [
            asyncio.create_task(run_worker(0, num_domains, server_ready)),
            asyncio.create_task(run_worker(1, num_domains, server_ready)),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"\n[test] worker results: {results}")

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"[test] worker {i} failed: {r}")
                raise r
            print(f"[test] worker {i} exchanged {r} KV block(s)")

    finally:
        try:
            stdout, _ = proc.communicate(timeout=30)
            print("[coordinator stdout]\n", stdout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            print("[coordinator stdout]\n", stdout)

    if proc.returncode == 0:
        print("\n✅ Two Python workers QUIC control-plane + KV ring test PASSED")
        return 0
    else:
        print(f"\n❌ coordinator exited with code {proc.returncode}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
