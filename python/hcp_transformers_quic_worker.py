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

try:
    import torch_npu  # noqa: F401  (registers the "npu" device backend)
except ImportError:
    torch_npu = None

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
        self._num_kv_heads = getattr(config, "num_key_value_heads", self._num_heads)
        self._head_dim = getattr(config, "hidden_size", 896) // self._num_heads
        self._history: List[int] = []
        self._past_key_values = None
        self._layer_kv_start: List[int] = [0] * self._num_layers
        print(f"[transformers backend] loaded: {self._num_layers} layers")

    def load_model(self, model_dir: str, device: str) -> None:
        pass

    def prefill(self, chunk: List[int], seq_offset: int, position_ids=None) -> Tuple[torch.Tensor, int]:
        from transformers.cache_utils import DynamicCache
        self._history = list(chunk)
        self._layer_kv_start = [seq_offset] * self._num_layers
        input_ids = torch.tensor([self._history], dtype=torch.long, device=self.device)
        # RoPE must use GLOBAL positions: without explicit position_ids,
        # transformers would number this chunk 0..len-1, corrupting the
        # rotation phase of every non-zero domain's KV.
        if position_ids is None:
            position_ids = list(range(seq_offset, seq_offset + len(chunk)))
        position_ids_t = torch.tensor([position_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, position_ids=position_ids_t, use_cache=True)
            logits = outputs.logits[0, -1]
            if outputs.past_key_values is not None:
                if isinstance(outputs.past_key_values, tuple):
                    self._past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else:
                    self._past_key_values = outputs.past_key_values
        return logits.to(torch.float32).cpu(), len(self._history) + seq_offset

    async def prefill_ring_exact(self, chunk: List[int], seq_offset: int, exchange_layer) -> Tuple[torch.Tensor, int]:
        """精确 context-parallel prefill：逐层 KV ring 交换 + 全局因果注意力。

        与 Rust Q-ring 同一算法：每层先算本地 chunk 的 Q/K/V（RoPE 用全局位置），
        经 ring 拿到所有 domain 该层的 KV，再按全局位置做精确因果注意力
        （远端 KV 全可见、本地 chunk 内因果）。每个 worker 缓存该层全量 KV，
        decode 阶段在完整 cache 上是精确的。

        exchange_layer: async (layer_idx, k, v) -> List[KvBlock]，返回其余
        domain 该层的 KV 块（带全局 seq 范围）。
        """
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        device = self.device
        self._history = list(chunk)
        self._layer_kv_start = [0] * self._num_layers  # cache holds the full sequence

        core = self.model.model
        input_ids = torch.tensor([self._history], dtype=torch.long, device=device)
        my_start = seq_offset
        my_len = len(chunk)
        pos = torch.arange(my_start, my_start + my_len, device=device, dtype=torch.long).unsqueeze(0)

        hidden = core.embed_tokens(input_ids)
        cos, sin = core.rotary_emb(hidden, pos)

        full_cache = DynamicCache()
        with torch.no_grad():
            for li, layer in enumerate(core.layers):
                attn = layer.self_attn
                residual = hidden
                h = layer.input_layernorm(hidden)
                b, s, _ = h.shape
                q = attn.q_proj(h).view(b, s, self._num_heads, self._head_dim).transpose(1, 2)
                k = attn.k_proj(h).view(b, s, self._num_kv_heads, self._head_dim).transpose(1, 2)
                v = attn.v_proj(h).view(b, s, self._num_kv_heads, self._head_dim).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

                remote = await exchange_layer(li, k, v)

                # 按全局起始位置排序拼接全量 KV，key 位置 = 各块的全局区间
                blocks = [(my_start, k, v)] + [
                    (blk.global_seq_start, blk.k.to(device), blk.v.to(device)) for blk in remote
                ]
                blocks.sort(key=lambda t: t[0])
                k_all = torch.cat([t[1] for t in blocks], dim=2)
                v_all = torch.cat([t[2] for t in blocks], dim=2)
                key_pos = torch.cat([
                    torch.arange(t[0], t[0] + t[1].size(2), device=device, dtype=torch.long)
                    for t in blocks
                ])

                # 精确因果掩码：key_pos <= query_pos（GQA 手动 repeat，避开
                # sdpa enable_gqa 在部分后端（torch_npu）上的兼容性问题）
                mask = key_pos[None, :] <= pos[0, :, None]  # [s, total_kv]
                gqa_repeat = self._num_heads // self._num_kv_heads
                if gqa_repeat > 1:
                    k_att = k_all.repeat_interleave(gqa_repeat, dim=1)
                    v_att = v_all.repeat_interleave(gqa_repeat, dim=1)
                else:
                    k_att, v_att = k_all, v_all
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q, k_att, v_att, attn_mask=mask[None, None, :, :]
                )
                attn_out = attn_out.transpose(1, 2).reshape(b, s, self._num_heads * self._head_dim)
                hidden = residual + attn.o_proj(attn_out)

                residual = hidden
                h = layer.post_attention_layernorm(hidden)
                hidden = residual + layer.mlp(h)

                full_cache.update(k_all, v_all, li)

            hidden = core.norm(hidden)
            logits = self.model.lm_head(hidden[:, -1])

        self._past_key_values = full_cache
        return logits[0].to(torch.float32).cpu(), my_start + my_len

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

    def _get_kv_layer(self, layer_idx: int):
        """兼容 transformers 4.x/5.x 的 DynamicCache KV 访问。"""
        cache = self._past_key_values
        if hasattr(cache, 'layers'):
            layer = cache.layers[layer_idx]
            return layer.keys, layer.values
        else:
            return cache[layer_idx]

    def _set_kv_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """兼容 transformers 4.x/5.x 的 DynamicCache KV 设置。"""
        cache = self._past_key_values
        if hasattr(cache, 'layers'):
            cache.layers[layer_idx].keys = k
            cache.layers[layer_idx].values = v
        elif hasattr(cache, 'key_cache'):
            cache.key_cache[layer_idx] = k
            cache.value_cache[layer_idx] = v
        else:
            cache._key_cache[layer_idx] = k
            cache._value_cache[layer_idx] = v

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        if self._past_key_values is None:
            k = torch.empty(0)
            v = torch.empty(0)
            return KvBlock(layer_idx, seq_start, seq_end, k, v)
        k, v = self._get_kv_layer(layer_idx)
        layer_start = self._layer_kv_start[layer_idx]
        local_start = max(0, seq_start - layer_start)
        local_end = max(0, seq_end - layer_start)
        k_slice = k[:, :, local_start:local_end, :].clone()
        v_slice = v[:, :, local_start:local_end, :].clone()
        return KvBlock(layer_idx, seq_start, seq_end, k_slice, v_slice)

    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        if self._past_key_values is None or peer_block.k.numel() == 0:
            return
        k_local, v_local = self._get_kv_layer(layer_idx)
        layer_start = self._layer_kv_start[layer_idx]
        if peer_block.global_seq_start < layer_start:
            k_new = torch.cat([peer_block.k.to(k_local.device), k_local], dim=2)
            v_new = torch.cat([peer_block.v.to(v_local.device), v_local], dim=2)
            self._layer_kv_start[layer_idx] = peer_block.global_seq_start
        else:
            k_new = torch.cat([k_local, peer_block.k.to(k_local.device)], dim=2)
            v_new = torch.cat([v_local, peer_block.v.to(v_local.device)], dim=2)
        self._set_kv_layer(layer_idx, k_new, v_new)

    def _trim_last_token_from_cache(self):
        """从 KV cache 中移除最后一个 token 的 KV，用于 recalculate_logits。"""
        if self._past_key_values is None:
            return None
        from transformers.cache_utils import DynamicCache
        trimmed = DynamicCache()
        for layer_idx in range(self._num_layers):
            k, v = self._get_kv_layer(layer_idx)
            if k.size(2) > 0:
                trimmed.update(k[:, :, :-1, :], v[:, :, :-1, :], layer_idx)
        return trimmed

    def recalculate_logits(self) -> torch.Tensor:
        """KV 交换后，用完整的 past_key_values 重新计算最后位置的 logits。"""
        from transformers.cache_utils import DynamicCache
        if not self._history:
            return torch.zeros(self.model.config.vocab_size, dtype=torch.float32)
        trimmed_past = self._trim_last_token_from_cache()
        input_ids = torch.tensor([[self._history[-1]]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=trimmed_past,
                use_cache=True,
            )
            logits = outputs.logits[0, -1]
            if outputs.past_key_values is not None:
                if isinstance(outputs.past_key_values, tuple):
                    self._past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
                else:
                    self._past_key_values = outputs.past_key_values
        return logits.to(torch.float32).cpu()

    @property
    def capacity_mb(self) -> int:
        if self.device.type == "npu":
            if torch_npu is None:
                return 4096
            free, _ = torch_npu.npu.mem_get_info(self.device)
            return int(free // (1024 * 1024))
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
