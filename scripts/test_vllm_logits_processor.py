#!/usr/bin/env python3
"""
Test vLLM logits extraction via custom LogitsProcessor.

vLLM 0.20.x (vllm-metal) allows registering custom LogitsProcessor classes
via the LLM(logits_processors=[...]) constructor. The processor's apply()
method receives the full batch logits tensor before sampling.

For single-request inference (max_num_seqs=1), the batch has one row,
so we can directly capture the logits vector.
"""

import os
import sys

MODEL_DIR = os.path.expanduser("~/VSCodeProjects/hetero-cp-ringattn/models/Qwen2-0.5B")
TEST_TOKENS = [101, 102, 103, 104, 105]


def load_transformers_reference(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.float32
    ).to(device)

    @torch.no_grad()
    def get_logits(token_ids: list):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        outputs = model(input_ids, use_cache=True)
        return outputs.logits[0, -1, :].cpu().float().tolist()

    return get_logits


def compare_logits(name: str, v_logits, ref_logits):
    import numpy as np
    v = np.array(v_logits, dtype=np.float64)
    r = np.array(ref_logits, dtype=np.float64)
    diff = np.abs(v - r)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    v_argmax = int(np.argmax(v))
    r_argmax = int(np.argmax(r))
    match = v_argmax == r_argmax

    print(f"\n=== {name} ===")
    print(f"  max_abs_diff: {max_diff:.6e}")
    print(f"  mean_abs_diff: {mean_diff:.6e}")
    print(f"  vllm_argmax: {v_argmax}, ref_argmax: {r_argmax}, match: {match}")
    return match


def main():
    from vllm import LLM, SamplingParams
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor
    import torch

    # Define a custom logits processor that captures logits
    class CaptureLogitsProcessor(LogitsProcessor):
        _captured = {}  # request_id -> logits (class-level storage)

        def __init__(self, vllm_config, device, is_pin_memory):
            self.device = device

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            """Capture logits before sampling. logits shape: [batch_size, vocab_size]"""
            # For single-request inference, batch_size == 1
            CaptureLogitsProcessor._captured["last"] = logits.clone().cpu()
            return logits  # Don't modify

        def is_argmax_invariant(self) -> bool:
            return True

        def update_state(self, batch_update):
            pass

    print("Loading vLLM model with CaptureLogitsProcessor...")
    llm = LLM(
        model=MODEL_DIR,
        dtype="float32",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        max_num_seqs=1,
        logits_processors=[CaptureLogitsProcessor],
    )
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    print(f"Model loaded. vocab_size={vocab_size}")

    # Load transformers reference
    print("Loading transformers reference...")
    ref_get_logits = load_transformers_reference(MODEL_DIR)
    ref_logits = ref_get_logits(TEST_TOKENS)
    print("Reference loaded.")

    # ---- Test 1: prefill + 1 decode token ----
    print("\n-- Test 1: prefill + 1 decode token --")
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp)
    sampled_token = outputs[0].outputs[0].token_ids[0]
    print(f"Sampled token: {sampled_token}")

    captured = CaptureLogitsProcessor._captured.get("last")
    if captured is not None:
        print(f"Captured logits shape: {captured.shape}")
        v_logits = captured[0].float().tolist()  # First (and only) batch row
        compare_logits("LogitsProcessor capture (decode)", v_logits, ref_logits)
    else:
        print("FAILED: No logits captured!")

    # ---- Test 2: verify argmax matches sampled token ----
    if captured is not None:
        argmax_token = int(captured[0].argmax().item())
        print(f"Captured argmax: {argmax_token}, sampled: {sampled_token}, match: {argmax_token == sampled_token}")

    print("\nDone.")


if __name__ == "__main__":
    main()
