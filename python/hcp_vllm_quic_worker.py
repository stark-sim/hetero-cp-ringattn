#!/usr/bin/env python3
"""
vLLM HCP Worker — QUIC control-plane + QUIC KV ring.

Usage:
    python hcp_vllm_quic_worker.py \
        --model-dir models/Qwen2-0.5B \
        --coordinator-host 127.0.0.1 \
        --coordinator-port 26001 \
        --domain-id 0 \
        --num-domains 2 \
        --peer-listen-host 0.0.0.0 \
        --peer-listen-port 26091 \
        --next-peer-host 127.0.0.1 \
        --next-peer-port 26092
"""

import argparse
import asyncio
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from hcp_worker_sdk.quic_server import QuicWorkerServer
from hcp_vllm_worker import VllmBackend


async def run_worker(
    model_dir: str,
    coordinator_host: str,
    coordinator_port: int,
    domain_id: int,
    num_domains: int,
    peer_listen_host: str,
    peer_listen_port: int,
    next_peer_host: str,
    next_peer_port: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[vllm worker] loading model from {model_dir} on {device} ...")
    backend = VllmBackend(model_dir, device=device)
    print(f"[vllm worker] loaded, vocab_size={backend.vocab_size}, capacity={backend.capacity_mb} MB")

    server = QuicWorkerServer(backend, domain_id, num_domains, torch.device(device))
    await server.run(
        coordinator_host, coordinator_port,
        peer_listen_host, peer_listen_port,
        next_peer_host, next_peer_port,
    )


def main():
    parser = argparse.ArgumentParser(description="vLLM HCP Worker (QUIC)")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--coordinator-host", default="127.0.0.1", help="Coordinator host")
    parser.add_argument("--coordinator-port", type=int, default=26001, help="Coordinator port")
    parser.add_argument("--domain-id", type=int, default=0, help="Domain ID")
    parser.add_argument("--num-domains", type=int, default=2, help="Number of domains")
    parser.add_argument("--peer-listen-host", default="0.0.0.0", help="Peer listen host")
    parser.add_argument("--peer-listen-port", type=int, default=26091, help="Peer listen port")
    parser.add_argument("--next-peer-host", default="127.0.0.1", help="Next peer host")
    parser.add_argument("--next-peer-port", type=int, default=26092, help="Next peer port")
    args = parser.parse_args()

    asyncio.run(run_worker(
        args.model_dir,
        args.coordinator_host, args.coordinator_port,
        args.domain_id, args.num_domains,
        args.peer_listen_host, args.peer_listen_port,
        args.next_peer_host, args.next_peer_port,
    ))


if __name__ == "__main__":
    main()
