#!/usr/bin/env python3
"""
vLLM Backend for HCP Worker SDK — Phase 1.5 MVP

集成要点：
- vLLM 的 LLM API 是为端到端生成设计的，不支持细粒度的 Prefill/Decode 分离。
- Phase 1.5 MVP 采用务实方案：每次 Prefill/Decode 都调用 generate()，
  Worker 内部自己采样，返回 one-hot logits 保持协议兼容。
- 性能非目标（每次 decode 都重新 prefill），先验证控制面通信可行。
"""

import torch
from typing import List, Tuple

from hcp_worker_sdk import HcpWorkerBackend, KvBlock


class VllmBackend(HcpWorkerBackend):
    """用 vLLM LLM 实现的 backend，用于验证 vLLM 接入 HCP Worker SDK。"""

    def __init__(self, model_dir: str, device: str = "cuda"):
        from vllm import LLM, SamplingParams

        self.model_dir = model_dir
        print(f"[vllm backend] loading model from {model_dir} ...")
        self.llm = LLM(
            model=model_dir,
            dtype="float32",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
        )
        self.vocab_size = self.llm.llm_engine.model_config.get_vocab_size()
        self._history: List[int] = []
        print(f"[vllm backend] loaded, vocab_size={self.vocab_size}")

    def load_model(self, model_dir: str, device: str) -> None:
        pass

    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        """Prefill：用 vLLM generate() 获取第一个 token，返回 one-hot logits。"""
        from vllm import SamplingParams

        self._history = list(chunk)
        outputs = self.llm.generate(
            prompt_token_ids=self._history,
            sampling_params=SamplingParams(max_tokens=1, temperature=0),
        )
        completion = outputs[0].outputs[0]
        token_id = completion.token_ids[0]

        # one-hot logits：生成的 token 为 0，其余为 -1e9
        logits = torch.full((self.vocab_size,), -1e9, dtype=torch.float32)
        logits[token_id] = 0.0
        return logits, len(self._history) + seq_offset

    def decode(self, token: int) -> torch.Tensor:
        """Decode：append token 后用 vLLM generate() 获取下一个 token。

        注意：vLLM LLM 不支持增量输入，每次都会重新 prefill。
        Phase 1.5 MVP 接受此性能代价。
        """
        from vllm import SamplingParams

        self._history.append(token)
        outputs = self.llm.generate(
            prompt_token_ids=self._history,
            sampling_params=SamplingParams(max_tokens=1, temperature=0),
        )
        completion = outputs[0].outputs[0]
        token_id = completion.token_ids[0]

        logits = torch.full((self.vocab_size,), -1e9, dtype=torch.float32)
        logits[token_id] = 0.0
        return logits

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        k = torch.empty(0)
        v = torch.empty(0)
        return KvBlock(layer_idx, seq_start, seq_end, k, v)

    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        pass

    @property
    def capacity_mb(self) -> int:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        return 4096

    @property
    def num_layers(self) -> int:
        # vLLM 的 model_config 有 num_hidden_layers
        return getattr(self.llm.llm_engine.model_config.hf_config, "num_hidden_layers", 24)

    @property
    def num_heads(self) -> int:
        return getattr(self.llm.llm_engine.model_config.hf_config, "num_attention_heads", 14)

    @property
    def head_dim(self) -> int:
        hidden = getattr(self.llm.llm_engine.model_config.hf_config, "hidden_size", 896)
        heads = getattr(self.llm.llm_engine.model_config.hf_config, "num_attention_heads", 14)
        return hidden // heads
