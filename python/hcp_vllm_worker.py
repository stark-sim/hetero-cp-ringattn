#!/usr/bin/env python3
"""
vLLM-based HCP Worker — Phase 1 MVP

目标：单节点接入 Coordinator 控制面，验证 Prefill/Decode/Shutdown 命令循环。
Phase 1 暂不接入 KV ring（使用 NoOpKvTransport）。

用法：
    python hcp_vllm_worker.py \
        --model-dir ~/models/Qwen2-0.5B \
        --domain-id 0 \
        --num-domains 1 \
        --coordinator-addr 127.0.0.1:29500 \
        --listen-addr 0.0.0.0:29501 \
        --next-peer-addr 127.0.0.1:29502
"""

import argparse
import sys
import torch
from typing import List, Tuple

# HCP Worker SDK
from hcp_worker_sdk import (
    HcpWorkerBackend,
    KvBlock,
    HcpWorkerServer,
    NoOpKvTransport,
)


class VllmWorkerBackend(HcpWorkerBackend):
    """
    vLLM 实现的 HCP Worker 后端（Phase 1 简化版）。

    设计决策：
    - 使用 vLLM LLM 加载模型，通过 model_runner.model 直接获取 logits
    - Phase 1 每次 decode 重新 forward 完整序列（效率低但实现简单）
    - KV ring 相关方法为 stub，Phase 2 再接入 PagedAttention
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self._history: List[int] = []

        # 延迟导入 vllm，避免未安装时 import 失败
        try:
            from vllm import LLM
        except ImportError as e:
            print(f"ERROR: vllm not installed: {e}")
            print("Install: pip install vllm")
            sys.exit(1)

        print(f"[vllm backend] loading model from {model_dir} ...")
        self.llm = LLM(
            model=model_dir,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=True,
        )

        # 获取底层 transformers 模型用于直接 forward
        worker = self.llm.llm_engine.model_executor.driver_worker
        self.model = worker.model_runner.model
        self.model.eval()

        # 从模型 config 读取维度信息
        config = self.model.config
        self._num_layers = getattr(config, "num_hidden_layers", 24)
        self._num_heads = getattr(config, "num_attention_heads", 14)
        self._head_dim = getattr(config, "hidden_size", 896) // self._num_heads

        print(f"[vllm backend] loaded: {self._num_layers} layers, "
              f"{self._num_heads} heads, head_dim={self._head_dim}")

    def load_model(self, model_dir: str, device: str) -> None:
        """加载模型（已在 __init__ 中完成）。"""
        pass

    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        """
        执行 prefill forward。

        Phase 1 简化：直接对 chunk 做一次性 forward，取 last token logits。
        """
        self._history = list(chunk)
        logits = self._forward(self._history)
        global_seq_len = len(self._history) + seq_offset
        return logits, global_seq_len

    def decode(self, token: int) -> torch.Tensor:
        """
        执行单 token decode forward。

        Phase 1 简化：将 token 追加到 history，重新 forward 完整序列。
        效率低但实现简单，Phase 2 接入 vLLM PagedAttention 优化。
        """
        self._history.append(token)
        return self._forward(self._history)

    def _forward(self, token_ids: List[int]) -> torch.Tensor:
        """对完整 token 序列做 forward，返回 last token 的 logits。"""
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            # outputs.logits: [batch, seq_len, vocab_size]
            logits = outputs.logits[0, -1]  # last token
        return logits.to(torch.float32).cpu()

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        """Phase 1 stub：返回空 block。"""
        k = torch.empty(0)
        v = torch.empty(0)
        return KvBlock(layer_idx, seq_start, seq_end, k, v)

    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        """Phase 1 stub：不执行任何操作。"""
        pass

    @property
    def capacity_mb(self) -> int:
        """上报可用显存。"""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        return 8192  # fallback

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim


def main():
    parser = argparse.ArgumentParser(description="HCP vLLM Worker")
    parser.add_argument("--model-dir", required=True, help="模型目录（HF 格式）")
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--num-domains", type=int, default=1)
    parser.add_argument("--coordinator-addr", required=True, help="host:port")
    parser.add_argument("--listen-addr", default="0.0.0.0:29451", help="host:port")
    parser.add_argument("--next-peer-addr", default="127.0.0.1:29452", help="host:port")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    args = parser.parse_args()

    backend = VllmWorkerBackend(args.model_dir, args.device)
    transport = NoOpKvTransport()
    server = HcpWorkerServer(
        backend=backend,
        transport=transport,
        domain_id=args.domain_id,
        num_domains=args.num_domains,
    )

    print(f"[hcp-vllm-worker] starting domain={args.domain_id}/{args.num_domains}")
    server.run(
        coordinator_addr=args.coordinator_addr,
        listen_addr=args.listen_addr,
        next_peer_addr=args.next_peer_addr,
    )
    print("[hcp-vllm-worker] shutdown complete")


if __name__ == "__main__":
    main()
