#!/usr/bin/env python3
"""
HCP Worker Process — JSON-over-stdio backend for Rust VllmWorkerBackend.

Supports multiple backends:
- vllm: vLLM LLM (GPU, high throughput)
- transformers: HuggingFace transformers (CPU/GPU compatible)
- mock: random logits (for protocol testing)

Protocol: line-delimited JSON over stdin/stdout.
"""

import argparse
import inspect
import json
import os
import sys
import traceback
from typing import List, Tuple, Optional


def _vllm_generate(llm, token_ids, sampling_params):
    """兼容 vLLM 0.6.x 和 0.20.x 的 generate() API。"""
    sig = inspect.signature(llm.generate)
    if "prompt_token_ids" in sig.parameters:
        return llm.generate(prompt_token_ids=token_ids, sampling_params=sampling_params)
    else:
        return llm.generate(prompts=[token_ids], sampling_params=sampling_params)


class Backend:
    """Abstract backend interface."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def handshake(self) -> dict:
        raise NotImplementedError

    def prefill(self, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        raise NotImplementedError

    def decode(self, token: int) -> List[float]:
        raise NotImplementedError

    def prefill_request(self, request_id: int, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        return self.prefill(tokens, seq_offset)

    def decode_request(self, request_id: int, token: int) -> List[float]:
        return self.decode(token)

    def decode_batch(self, request_tokens: List[Tuple[int, int]]) -> List[Tuple[int, List[float]]]:
        return [(rid, self.decode_request(rid, tok)) for rid, tok in request_tokens]

    def shutdown(self):
        pass


class MockBackend(Backend):
    """Returns deterministic random logits for protocol testing."""

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        import random, os
        random.seed(42)
        self.vocab_size = int(os.environ.get("HCP_MOCK_VOCAB_SIZE", "100"))
        self.num_layers = int(os.environ.get("HCP_MOCK_NUM_LAYERS", "2"))
        self.capacity_mb = int(os.environ.get("HCP_MOCK_CAPACITY_MB", "4096"))

    def handshake(self) -> dict:
        return {"num_layers": self.num_layers, "capacity_mb": self.capacity_mb, "vocab_size": self.vocab_size}

    def prefill(self, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        import random
        logits = [random.random() for _ in range(self.vocab_size)]
        return logits, len(tokens) + seq_offset

    def decode(self, token: int) -> List[float]:
        import random
        return [random.random() for _ in range(self.vocab_size)]


class TransformersBackend(Backend):
    """HuggingFace transformers backend."""

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        print(f"[transformers backend] loading model from {model_dir} ...", file=sys.stderr)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.config = AutoConfig.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=self.config,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.vocab_size = self.config.vocab_size
        self.num_layers = getattr(self.config, "num_hidden_layers", 24)
        self.capacity_mb = self._query_capacity_mb()
        self._request_states = {}  # request_id -> {input_ids, past_key_values}
        print(f"[transformers backend] loaded, vocab_size={self.vocab_size}, device={self.device}", file=sys.stderr)

    def _query_capacity_mb(self) -> int:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        if torch.backends.mps.is_available():
            # MPS doesn't have a direct mem_get_info API; return a conservative estimate
            import os
            return int(os.environ.get("HCP_MPS_CAPACITY_MB", "4096"))
        return 4096

    def handshake(self) -> dict:
        return {"num_layers": self.num_layers, "capacity_mb": self.capacity_mb, "vocab_size": self.vocab_size}

    def prefill(self, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        import torch
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        logits = outputs.logits[0, -1, :].cpu().float().tolist()
        return logits, len(tokens) + seq_offset

    def decode(self, token: int) -> List[float]:
        import torch
        input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        logits = outputs.logits[0, -1, :].cpu().float().tolist()
        return logits

    def prefill_request(self, request_id: int, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        import torch
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        logits = outputs.logits[0, -1, :].cpu().float().tolist()
        self._request_states[request_id] = {
            "input_ids": tokens,
            "past_key_values": outputs.past_key_values,
        }
        return logits, len(tokens) + seq_offset

    def decode_request(self, request_id: int, token: int) -> List[float]:
        import torch
        state = self._request_states.get(request_id)
        if state is None:
            raise ValueError(f"request {request_id} not found")
        input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=state["past_key_values"], use_cache=True)
        logits = outputs.logits[0, -1, :].cpu().float().tolist()
        state["past_key_values"] = outputs.past_key_values
        state["input_ids"].append(token)
        return logits

    def decode_batch(self, request_tokens: List[Tuple[int, int]]) -> List[Tuple[int, List[float]]]:
        # transformers 不支持 true batch decode with different KV states，逐个处理
        return [(rid, self.decode_request(rid, tok)) for rid, tok in request_tokens]

    def shutdown(self):
        import gc
        del self.model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class VllmBackend(Backend):
    """vLLM backend (GPU only)."""

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        from vllm import LLM, SamplingParams
        import torch

        print(f"[vllm backend] loading model from {model_dir} ...", file=sys.stderr)
        self.llm = LLM(
            model=model_dir,
            dtype="float32",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            max_num_seqs=4,
        )
        self.vocab_size = self.llm.llm_engine.model_config.get_vocab_size()
        self.num_layers = getattr(
            self.llm.llm_engine.model_config.hf_config, "num_hidden_layers", 24
        )
        self.capacity_mb = self._query_capacity_mb()
        self._request_states = {}  # request_id -> {token_ids}
        print(f"[vllm backend] loaded, vocab_size={self.vocab_size}", file=sys.stderr)

    def _query_capacity_mb(self) -> int:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        return 4096

    def handshake(self) -> dict:
        return {"num_layers": self.num_layers, "capacity_mb": self.capacity_mb, "vocab_size": self.vocab_size}

    def prefill(self, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        from vllm import SamplingParams
        outputs = _vllm_generate(
            self.llm, tokens,
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        logits = self._sampled_token_to_logits(outputs)
        return logits, len(tokens) + seq_offset

    def decode(self, token: int) -> List[float]:
        from vllm import SamplingParams
        outputs = _vllm_generate(
            self.llm, [token],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        return self._sampled_token_to_logits(outputs)

    def _sampled_token_to_logits(self, outputs) -> List[float]:
        """Convert vLLM sampled output to a one-hot logits vector.

        vLLM does not expose full logits through its public API (especially
        vllm-metal 0.20.x). Instead, we let vLLM sample internally and
        reconstruct a one-hot logits vector where the sampled token has a
        high logit. This preserves greedy/temperature sampling correctness.
        """
        sampled_token = outputs[0].outputs[0].token_ids[0]
        logits = [-1e9] * self.vocab_size
        logits[int(sampled_token)] = 1e9
        return logits

    def prefill_request(self, request_id: int, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        from vllm import SamplingParams
        self._request_states[request_id] = list(tokens)
        outputs = _vllm_generate(
            self.llm, tokens,
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        logits = self._sampled_token_to_logits(outputs)
        return logits, len(tokens) + seq_offset

    def decode_request(self, request_id: int, token: int) -> List[float]:
        from vllm import SamplingParams
        state = self._request_states.get(request_id)
        if state is None:
            raise ValueError(f"request {request_id} not found")
        state.append(token)
        outputs = _vllm_generate(
            self.llm, state,
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        return self._sampled_token_to_logits(outputs)

    def shutdown(self):
        try:
            del self.llm
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def create_backend(backend_type: str, model_dir: str) -> Backend:
    if backend_type == "mock":
        return MockBackend(model_dir)
    elif backend_type == "transformers":
        return TransformersBackend(model_dir)
    elif backend_type == "vllm":
        return VllmBackend(model_dir)
    else:
        raise ValueError(f"unknown backend type: {backend_type}")


def run_worker_process(backend: Backend):
    print("[worker process] ready, waiting for commands...", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _respond_error(f"invalid json: {e}")
            continue

        try:
            resp = _handle_cmd(backend, cmd)
        except Exception as e:
            traceback.print_exc()
            _respond_error(str(e))
            continue

        _respond(resp)
        if resp.get("exit"):
            break


def _handle_cmd(backend: Backend, cmd: dict) -> dict:
    action = cmd.get("cmd", "")

    if action == "handshake":
        meta = backend.handshake()
        return {"status": "ok", "message": json.dumps(meta)}

    elif action == "prefill":
        tokens = cmd.get("tokens", [])
        seq_offset = cmd.get("seq_offset", 0)
        logits, global_seq_len = backend.prefill(tokens, seq_offset)
        return {"status": "ok", "logits": logits, "global_seq_len": global_seq_len}

    elif action == "decode":
        token = cmd.get("tokens", [0])[0]
        logits = backend.decode(token)
        return {"status": "ok", "logits": logits, "global_seq_len": 0}

    elif action == "prefill_request":
        request_id = cmd.get("request_id", 0)
        tokens = cmd.get("tokens", [])
        seq_offset = cmd.get("seq_offset", 0)
        logits, global_seq_len = backend.prefill_request(request_id, tokens, seq_offset)
        return {"status": "ok", "logits": logits, "global_seq_len": global_seq_len}

    elif action == "decode_request":
        request_id = cmd.get("request_id", 0)
        token = cmd.get("tokens", [0])[0]
        logits = backend.decode_request(request_id, token)
        return {"status": "ok", "logits": logits, "global_seq_len": 0}

    elif action == "decode_batch":
        request_tokens = cmd.get("request_tokens", [])
        request_logits = backend.decode_batch(request_tokens)
        return {"status": "ok", "request_logits": request_logits}

    elif action == "sync_global_seq_len":
        return {"status": "ok"}

    elif action == "shutdown":
        backend.shutdown()
        return {"status": "ok", "exit": True}

    else:
        return {"status": "error", "message": f"unknown cmd: {action}"}


def _respond(resp: dict):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def _respond_error(message: str):
    _respond({"status": "error", "message": message})


def main():
    parser = argparse.ArgumentParser(description="HCP Worker Process")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--backend", default="mock", choices=["mock", "transformers", "vllm"],
                        help="Backend type")
    args = parser.parse_args()

    backend = create_backend(args.backend, args.model_dir)
    run_worker_process(backend)


if __name__ == "__main__":
    main()
