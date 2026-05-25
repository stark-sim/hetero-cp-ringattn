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
    """兼容 vLLM 0.6.x 和 0.20.x 的 generate() API。
    
    使用 lru_cache 风格的单例检测避免每次调用都反射 inspect。
    """
    cache_key = id(llm)
    if not hasattr(_vllm_generate, "_cache"):
        _vllm_generate._cache = {}
    if cache_key not in _vllm_generate._cache:
        sig = inspect.signature(llm.generate)
        _vllm_generate._cache[cache_key] = "prompt_token_ids" in sig.parameters
    if _vllm_generate._cache[cache_key]:
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

    def release_request(self, request_id: int):
        """Release per-request state. Default: no-op."""
        pass

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
        # Save past_key_values for implicit-state decode() reuse.
        self._last_past_key_values = outputs.past_key_values
        return logits, len(tokens) + seq_offset

    def decode(self, token: int) -> List[float]:
        import torch
        input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            past_kv = getattr(self, '_last_past_key_values', None)
            if past_kv is not None:
                outputs = self.model(input_ids, past_key_values=past_kv, use_cache=True)
            else:
                outputs = self.model(input_ids, use_cache=True)
        logits = outputs.logits[0, -1, :].cpu().float().tolist()
        self._last_past_key_values = outputs.past_key_values
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

    def release_request(self, request_id: int):
        if request_id in self._request_states:
            del self._request_states[request_id]
            print(f"[transformers backend] released request {request_id}", file=sys.stderr)

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
    """vLLM backend (GPU only).

    Logits extraction strategy:
    - vLLM's public API does not expose full logits vectors.
    - vLLM 0.20.x (vllm-metal): logprobs max=20, prompt_logprobs returns None.
    - We use the best available extraction:
      * decode: logprobs=20 (top-20 relative probabilities) + one-hot fallback
      * prefill: one-hot (prompt_logprobs unavailable)
    - For exact logits, see docs/VLLM_INTEGRATION.md#logits-extraction.
    """

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
        # Detect max supported logprobs (version-dependent)
        self._max_logprobs = self._detect_max_logprobs()
        print(f"[vllm backend] loaded, vocab_size={self.vocab_size}, max_logprobs={self._max_logprobs}", file=sys.stderr)

    def _detect_max_logprobs(self) -> int:
        """Detect the maximum logprobs supported by this vLLM version."""
        from vllm import SamplingParams
        for k in [20, 10, 5, 0]:
            if k == 0:
                return 0
            try:
                sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=k)
                # Try a dummy generation to validate
                self.llm.generate(prompts=[[1]], sampling_params=sp)
                return k
            except Exception:
                continue
        return 0

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
        logits = self._extract_logits(outputs, use_logprobs=False)
        return logits, len(tokens) + seq_offset

    def decode(self, token: int) -> List[float]:
        from vllm import SamplingParams
        outputs = _vllm_generate(
            self.llm, [token],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        return self._extract_logits(outputs, use_logprobs=True)

    def _extract_logits(self, outputs, use_logprobs: bool = True) -> List[float]:
        """Extract logits from vLLM output.

        Best-effort extraction:
        1. If logprobs available and use_logprobs=True, fill top-K logprobs.
        2. Always set the sampled token to highest logit (1e9).
        3. Unobserved tokens get -1e9.
        """
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty output")
        comp = outputs[0].outputs[0]
        token_ids = comp.token_ids
        if not token_ids:
            raise RuntimeError("vLLM returned empty token_ids")
        sampled_token = int(token_ids[0])
        if sampled_token < 0 or sampled_token >= self.vocab_size:
            raise RuntimeError(f"vLLM sampled out-of-bounds token {sampled_token} (vocab={self.vocab_size})")

        logits = [-1e9] * self.vocab_size

        # Try to fill logprobs if available
        if use_logprobs and self._max_logprobs > 0 and comp.logprobs:
            lp_dict = comp.logprobs[0]  # First generated token's logprobs
            for tok_id, lp in lp_dict.items():
                logprob_val = float(getattr(lp, "logprob", lp))
                logits[int(tok_id)] = logprob_val

        # Ensure sampled token has the highest logit
        logits[sampled_token] = 1e9
        return logits

    def prefill_request(self, request_id: int, tokens: List[int], seq_offset: int) -> Tuple[List[float], int]:
        from vllm import SamplingParams
        self._request_states[request_id] = list(tokens)
        outputs = _vllm_generate(
            self.llm, tokens,
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        logits = self._extract_logits(outputs, use_logprobs=False)
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
        return self._extract_logits(outputs, use_logprobs=True)

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
        if not isinstance(tokens, list):
            return {"status": "error", "message": "tokens must be a list"}
        logits, global_seq_len = backend.prefill(tokens, seq_offset)
        return {"status": "ok", "logits": logits, "global_seq_len": global_seq_len}

    elif action == "decode":
        tokens = cmd.get("tokens", [0])
        if not isinstance(tokens, list) or len(tokens) == 0:
            return {"status": "error", "message": "tokens must be a non-empty list"}
        logits = backend.decode(tokens[0])
        return {"status": "ok", "logits": logits, "global_seq_len": 0}

    elif action == "prefill_request":
        request_id = cmd.get("request_id", 0)
        tokens = cmd.get("tokens", [])
        seq_offset = cmd.get("seq_offset", 0)
        if not isinstance(tokens, list):
            return {"status": "error", "message": "tokens must be a list"}
        logits, global_seq_len = backend.prefill_request(request_id, tokens, seq_offset)
        return {"status": "ok", "logits": logits, "global_seq_len": global_seq_len}

    elif action == "decode_request":
        request_id = cmd.get("request_id", 0)
        tokens = cmd.get("tokens", [0])
        if not isinstance(tokens, list) or len(tokens) == 0:
            return {"status": "error", "message": "tokens must be a non-empty list"}
        logits = backend.decode_request(request_id, tokens[0])
        return {"status": "ok", "logits": logits, "global_seq_len": 0}

    elif action == "decode_batch":
        request_tokens = cmd.get("request_tokens", [])
        if not isinstance(request_tokens, list):
            return {"status": "error", "message": "request_tokens must be a list"}
        for item in request_tokens:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return {"status": "error", "message": "each request_tokens item must be [request_id, token]"}
        request_logits = backend.decode_batch(request_tokens)
        return {"status": "ok", "request_logits": request_logits}

    elif action == "release_request":
        request_id = cmd.get("request_id", 0)
        backend.release_request(request_id)
        return {"status": "ok"}

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
