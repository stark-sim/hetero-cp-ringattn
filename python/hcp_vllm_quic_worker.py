#!/usr/bin/env python3
"""
vLLM HCP Worker — QUIC control-plane version.

Connects to Rust coordinator via QUIC + bincode protocol.
Usage:
    python hcp_vllm_quic_worker.py \
        --model-dir models/Qwen2-0.5B \
        --coordinator-host 127.0.0.1 \
        --coordinator-port 26001 \
        --domain-id 0
"""

import argparse
import asyncio
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from hcp_worker_sdk.quic_control import QuicControlClient
from hcp_vllm_worker import VllmBackend


async def run_worker(model_dir: str, host: str, port: int, domain_id: int):
    # Load vLLM backend (must be done before asyncio event loop on some platforms)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[vllm worker] loading model from {model_dir} on {device} ...")
    backend = VllmBackend(model_dir, device=device)
    print(f"[vllm worker] loaded, vocab_size={backend.vocab_size}, capacity={backend.capacity_mb} MB")

    client = QuicControlClient()
    try:
        await client.connect(host, port)
        await client.send_handshake(domain_id=domain_id, capacity_mb=backend.capacity_mb)

        global_seq_len = 0
        while True:
            cmd = await client.recv_command()
            kind = cmd["kind"]

            if kind == "Prefill":
                chunk = cmd["chunk"]
                seq_offset = cmd["seq_offset"]
                print(f"[vllm worker] Prefill chunk len={len(chunk)}, seq_offset={seq_offset}")

                logits, seq_len = backend.prefill(chunk, seq_offset)
                global_seq_len = seq_len

                # Convert logits to f32 LE bytes
                logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()

                await client.send_response(
                    "PrefillDone",
                    last_logits_bytes=logits_bytes,
                    global_seq_len=global_seq_len,
                )

            elif kind == "SyncGlobalSeqLen":
                global_seq_len = cmd["global_seq_len"]
                print(f"[vllm worker] SyncGlobalSeqLen = {global_seq_len}")

            elif kind == "Decode":
                token = cmd["token"]
                print(f"[vllm worker] Decode token={token}")

                logits = backend.decode(token)
                logits_bytes = logits.detach().cpu().numpy().astype("float32").tobytes()

                await client.send_response("DecodeDone", logits_bytes=logits_bytes)

            elif kind == "Shutdown":
                print("[vllm worker] Shutdown received")
                break

    except Exception as e:
        print(f"[vllm worker] error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await client.close()
        print("[vllm worker] disconnected")


def main():
    parser = argparse.ArgumentParser(description="vLLM HCP Worker (QUIC)")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--coordinator-host", default="127.0.0.1", help="Coordinator host")
    parser.add_argument("--coordinator-port", type=int, default=26001, help="Coordinator port")
    parser.add_argument("--domain-id", type=int, default=0, help="Domain ID")
    args = parser.parse_args()

    asyncio.run(run_worker(args.model_dir, args.coordinator_host, args.coordinator_port, args.domain_id))


if __name__ == "__main__":
    main()
