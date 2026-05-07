#!/usr/bin/env python3
"""
Transformers HCP Worker — QUIC control-plane + QUIC KV ring.

Usage:
    python hcp_transformers_quic_worker.py \
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
from typing import List, Tuple

import torch

sys.path.insert(0, os.path.dirname(__file__))

from hcp_worker_sdk import HcpWorkerBackend, KvBlock
from hcp_worker_sdk.quic_server import QuicWorkerServer


class TransformersBackend(HcpWorkerBackend):
    """transformers backend for HCP Worker SDK."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        print(f"[transformers backend] loading model from {model_dir} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.model.eval()

        config = self.model.config
        self._num_layers = getattr(config, "num_hidden_layers", 24)
        self._num_heads = getattr(config, "num_attention_heads", 14)
        self._head_dim = getattr(config, "hidden_size", 896) // self._num_heads
        self._history: List[int] = []
        self._past_key_values = None
        self._layer_kv_start: List[int] = [0] * self._num_layers
        print(f"[transformers backend] loaded: {self._num_layers} layers")

    def load_model(self, model_dir: str, device: str) -> None:
        pass

    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        from transformers.cache_utils import DynamicCache
        self._history = list(chunk)
        self._layer_kv_start = [seq_offset] * self._num_layers
        input_ids = torch.tensor([self._history], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            logits = outputs.logits[0, -1]
            if outputs.past_key_values is not None:
                if isinstance(outputs.past_key_values, tuple):
                    self._past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else:
                    self._past_key_values = outputs.past_key_values
        return logits.to(torch.float32).cpu(), len(self._history) + seq_offset

    def decode(self, token: int) -> torch.Tensor:
        from transformers.cache_utils import DynamicCache
        self._history.append(token)
        input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[0, -1]
            if outputs.past_key_values is not None:
                if isinstance(outputs.past_key_values, tuple):
                    self._past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else:
                    self._past_key_values = outputs.past_key_values
        return logits.to(torch.float32).cpu()

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        if self._past_key_values is None:
            k = torch.empty(0)
            v = torch.empty(0)
            return KvBlock(layer_idx, seq_start, seq_end, k, v)
        k, v = self._past_key_values[layer_idx]
        layer_start = self._layer_kv_start[layer_idx]
        local_start = max(0, seq_start - layer_start)
        local_end = max(0, seq_end - layer_start)
        k_slice = k[:, :, local_start:local_end, :].clone()
        v_slice = v[:, :, local_start:local_end, :].clone()
        return KvBlock(layer_idx, seq_start, seq_end, k_slice, v_slice)

    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        if self._past_key_values is None or peer_block.k.numel() == 0:
            return
        k_local, v_local = self._past_key_values[layer_idx]
        layer_start = self._layer_kv_start[layer_idx]
        if peer_block.global_seq_start < layer_start:
            k_new = torch.cat([peer_block.k.to(k_local.device), k_local], dim=2)
            v_new = torch.cat([peer_block.v.to(v_local.device), v_local], dim=2)
            self._layer_kv_start[layer_idx] = peer_block.global_seq_start
        else:
            k_new = torch.cat([k_local, peer_block.k.to(k_local.device)], dim=2)
            v_new = torch.cat([v_local, peer_block.v.to(v_local.device)], dim=2)
        self._past_key_values.layers[layer_idx].keys = k_new
        self._past_key_values.layers[layer_idx].values = v_new

    @property
    def capacity_mb(self) -> int:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        return 4096

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim


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
    device: str,
):
    backend = TransformersBackend(model_dir, device=device)
    server = QuicWorkerServer(backend, domain_id, num_domains, torch.device(device))
    await server.run(
        coordinator_host, coordinator_port,
        peer_listen_host, peer_listen_port,
        next_peer_host, next_peer_port,
    )


def main():
    parser = argparse.ArgumentParser(description="Transformers HCP Worker (QUIC)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--coordinator-host", default="127.0.0.1")
    parser.add_argument("--coordinator-port", type=int, default=26001)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--peer-listen-host", default="0.0.0.0")
    parser.add_argument("--peer-listen-port", type=int, default=26091)
    parser.add_argument("--next-peer-host", default="127.0.0.1")
    parser.add_argument("--next-peer-port", type=int, default=26092)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    asyncio.run(run_worker(
        args.model_dir,
        args.coordinator_host, args.coordinator_port,
        args.domain_id, args.num_domains,
        args.peer_listen_host, args.peer_listen_port,
        args.next_peer_host, args.next_peer_port,
        args.device,
    ))


if __name__ == "__main__":
    main()
