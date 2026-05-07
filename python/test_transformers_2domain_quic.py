#!/usr/bin/env python3
"""
Phase 3.3 prep: Local 2-domain TransformersBackend + QUIC KV ring end-to-end.

Validates hcp_transformers_quic_worker.py in a single-machine setup before
cross-machine deployment.
"""

import asyncio
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hcp_transformers_quic_worker import run_worker


MODEL_DIR = "models/Qwen2-0.5B"
COORDINATOR_ADDR = ("127.0.0.1", 26021)
PEER_PORTS = [26031, 26032]


def start_coordinator(num_domains: int = 2):
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
    await asyncio.sleep(5)  # Longer startup for model loading

    try:
        tasks = [
            asyncio.create_task(run_worker(
                MODEL_DIR,
                COORDINATOR_ADDR[0], COORDINATOR_ADDR[1],
                0, num_domains,
                "0.0.0.0", PEER_PORTS[0],
                "127.0.0.1", PEER_PORTS[1],
                "cpu",
            )),
            asyncio.create_task(run_worker(
                MODEL_DIR,
                COORDINATOR_ADDR[0], COORDINATOR_ADDR[1],
                1, num_domains,
                "0.0.0.0", PEER_PORTS[1],
                "127.0.0.1", PEER_PORTS[0],
                "cpu",
            )),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"\n[test] worker results: {results}")

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"[test] worker {i} failed: {r}")
                raise r

    finally:
        try:
            stdout, _ = proc.communicate(timeout=60)
            print("[coordinator stdout]\n", stdout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            print("[coordinator stdout]\n", stdout)

    if proc.returncode == 0:
        print("\n✅ Local 2-domain TransformersBackend QUIC distributed test PASSED")
        return 0
    else:
        print(f"\n❌ coordinator exited with code {proc.returncode}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
