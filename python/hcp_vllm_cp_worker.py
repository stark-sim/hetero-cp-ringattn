#!/usr/bin/env python3
"""
vLLM block-ring HCP Worker with context-passing CP (QUIC).

Hosts the vLLM block-ring plugin (auto-detects vLLM 0.6.x legacy engine vs
>=0.23 V1 engine) behind the context-passing CP server, so two vLLM workers on
different nodes/GPUs can cooperate on one sequence via physical-block KV
exchange.

Usage (domain 0, e.g. white / CUDA):
    python hcp_vllm_cp_worker.py \
        --model-dir /path/to/model --domain-id 0 --num-domains 2 \
        --coordinator-host 127.0.0.1 --coordinator-port 29500 \
        --peer-listen-host 0.0.0.0 --peer-listen-port 29501 \
        --next-peer-host <pearl> --next-peer-port 29502

Usage (domain 1, e.g. pearl / ROCm):
    python hcp_vllm_cp_worker.py \
        --model-dir /path/to/model --domain-id 1 --num-domains 2 \
        --coordinator-host <white> --coordinator-port 29500 \
        --peer-listen-host 0.0.0.0 --peer-listen-port 29502 \
        --next-peer-host <white> --next-peer-port 29501
"""

import argparse
import asyncio
import os
import signal
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from hcp_worker_sdk.cp_server import CpVllmWorkerServer


def make_backend(model_dir: str, device: str, block_size: int,
                 gpu_mem: float, max_model_len: int):
    """Instantiate the block-ring plugin matching the installed vLLM engine."""
    try:
        # vLLM >= 0.23 removed the legacy SequenceGroupMetadata engine.
        from vllm.v1.core.sched.output import SchedulerOutput  # noqa: F401
        from hcp_vllm_block_ring_plugin_v1 import VllmBlockRingPluginV1
        print("[cp worker] detected vLLM V1 engine, using VllmBlockRingPluginV1")
        return VllmBlockRingPluginV1(
            model_dir,
            device=device,
            gpu_memory_utilization=gpu_mem,
            block_size=block_size,
            max_model_len=max_model_len,
        )
    except ImportError:
        from hcp_vllm_block_ring_plugin import VllmBlockRingPlugin
        print("[cp worker] detected vLLM legacy engine, using VllmBlockRingPlugin")
        return VllmBlockRingPlugin(
            model_dir,
            device=device,
            gpu_memory_utilization=gpu_mem,
            block_size=block_size,
        )


async def run_worker(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[cp worker] loading model from {args.model_dir} on {device} ...")
    backend = make_backend(
        args.model_dir, device, args.block_size, args.gpu_mem, args.max_model_len
    )
    print(f"[cp worker] loaded, vocab={backend.vocab_size}, "
          f"capacity={backend.capacity_mb} MB, layers={backend.num_layers}")

    server = CpVllmWorkerServer(
        backend, args.domain_id, args.num_domains, torch.device(device)
    )

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler():
        print("[cp worker] received shutdown signal")
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    loop.add_signal_handler(signal.SIGINT, _signal_handler)

    try:
        await server.run(
            args.coordinator_host, args.coordinator_port,
            args.peer_listen_host, args.peer_listen_port,
            args.next_peer_host, args.next_peer_port,
            shutdown_event=shutdown_event,
        )
    finally:
        print("[cp worker] running cleanup...")
        await server.cleanup()
        backend.shutdown()
        try:
            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGINT)
        except Exception:
            pass
        print("[cp worker] graceful shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="vLLM block-ring CP worker (QUIC)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--coordinator-host", default="127.0.0.1")
    parser.add_argument("--coordinator-port", type=int, default=29500)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--peer-listen-host", default="0.0.0.0")
    parser.add_argument("--peer-listen-port", type=int, default=29501)
    parser.add_argument("--next-peer-host", default="127.0.0.1")
    parser.add_argument("--next-peer-port", type=int, default=29502)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    asyncio.run(run_worker(args))


if __name__ == "__main__":
    main()
