#!/usr/bin/env python3
"""
End-to-end test: Python mock worker connects to Rust coordinator via QUIC + bincode.

This validates the control-plane protocol without requiring vLLM or GPU.
"""

import asyncio
import sys
import os
import subprocess
import time
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hcp_worker_sdk.quic_control import QuicControlClient


MODEL_DIR = "models/Qwen2-0.5B"
VOCAB_SIZE = 151936  # Qwen2-0.5B vocab size


async def run_mock_worker(coordinator_host: str, coordinator_port: int, domain_id: int):
    """Python mock worker that connects to Rust coordinator."""
    client = QuicControlClient()
    try:
        await client.connect(coordinator_host, coordinator_port)
        await client.send_handshake(domain_id=domain_id, capacity_mb=4096)

        while True:
            cmd = await client.recv_command()
            kind = cmd["kind"]

            if kind == "Prefill":
                chunk = cmd["chunk"]
                seq_offset = cmd["seq_offset"]
                print(f"[mock worker] Prefill chunk len={len(chunk)}, seq_offset={seq_offset}")
                # Return dummy logits (all zeros) as f32 LE bytes
                # Use small payload for debugging
                logits_bytes = struct.pack(f"<{VOCAB_SIZE}f", *[0.0] * VOCAB_SIZE)
                print(f"[mock worker] logits_bytes len = {len(logits_bytes)}")
                global_seq_len = len(chunk) + seq_offset
                await client.send_response(
                    "PrefillDone",
                    last_logits_bytes=logits_bytes,
                    global_seq_len=global_seq_len,
                )

            elif kind == "SyncGlobalSeqLen":
                global_seq_len = cmd["global_seq_len"]
                print(f"[mock worker] SyncGlobalSeqLen = {global_seq_len}")
                # No response needed

            elif kind == "Decode":
                token = cmd["token"]
                print(f"[mock worker] Decode token={token}")
                logits_bytes = struct.pack(f"<{VOCAB_SIZE}f", *[0.0] * VOCAB_SIZE)
                await client.send_response("DecodeDone", logits_bytes=logits_bytes)

            elif kind == "Shutdown":
                print("[mock worker] Shutdown received")
                break

    except Exception as e:
        print(f"[mock worker] error: {e}")
        raise
    finally:
        await client.close()
        print("[mock worker] disconnected")


def start_rust_coordinator(listen_addr: str, num_domains: int = 1):
    """Start Rust coordinator as a subprocess."""
    host, port = listen_addr.rsplit(":", 1)
    model_dir = os.path.abspath(MODEL_DIR)
    
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = "/Users/stark_sim/libtorch/lib"
    
    cmd = [
        "cargo", "run", "--features", "tch-backend", "--bin", "hcp-ringattn-rust",
        "--", "--distributed-role", "coordinator",
        "--model-dir", model_dir,
        "--prompt", "Hello world",
        "--max-tokens", "3",
        "--num-domains", str(num_domains),
        "--listen-addr", listen_addr,
    ]
    
    print(f"[test] starting coordinator: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd="rust",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


async def main():
    listen_addr = "127.0.0.1:26001"
    coordinator_host, coordinator_port = listen_addr.rsplit(":", 1)
    coordinator_port = int(coordinator_port)

    # Start Rust coordinator
    proc = start_rust_coordinator(listen_addr, num_domains=1)

    # Give coordinator time to start
    await asyncio.sleep(3)

    try:
        # Start mock worker
        await run_mock_worker(coordinator_host, coordinator_port, domain_id=0)
    finally:
        # Wait for coordinator to finish
        try:
            stdout, _ = proc.communicate(timeout=30)
            print("[coordinator stdout]\n", stdout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            print("[coordinator stdout]\n", stdout)

        if proc.returncode == 0:
            print("\n✅ Python worker ↔ Rust coordinator QUIC control-plane test PASSED")
        else:
            print(f"\n❌ coordinator exited with code {proc.returncode}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
