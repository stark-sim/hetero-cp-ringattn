#!/usr/bin/env python3
"""
Test dual-load approach: vLLM for serving + transformers for exact logits.

VllmBackend loads both:
1. vLLM LLM for high-throughput generation and KV cache management
2. transformers AutoModelForCausalLM for exact logits extraction

The transformers model runs forward() to get logits, while vLLM handles
the actual generation and KV state. For inference correctness validation,
this gives exact logits match without needing to hack vLLM internals.

Memory overhead: ~2x model weights (Qwen2-0.5B ~1GB each).
"""

import os
import sys

MODEL_DIR = os.path.expanduser("~/VSCodeProjects/hetero-cp-ringattn/models/Qwen2-0.5B")
TEST_TOKENS = [101, 102, 103, 104, 105]


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
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print("Loading vLLM model...")
    llm = LLM(
        model=MODEL_DIR,
        dtype="float32",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        max_num_seqs=1,
    )
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    print(f"vLLM loaded. vocab_size={vocab_size}")

    print("Loading transformers model (for exact logits)...")
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    config = AutoConfig.from_pretrained(MODEL_DIR)
    tf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, config=config, torch_dtype=torch.float32
    ).to(device)
    print(f"Transformers loaded on {device}")

    # Test 1: prefill logits
    print("\n-- Test 1: prefill logits --")
    input_ids = torch.tensor([TEST_TOKENS], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = tf_model(input_ids, use_cache=True)
    tf_logits = outputs.logits[0, -1, :].cpu().float().tolist()

    # Run vLLM generate with max_tokens=1
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    vllm_outputs = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp)
    sampled_token = vllm_outputs[0].outputs[0].token_ids[0]
    print(f"vLLM sampled token: {sampled_token}")

    compare_logits("prefill (transformers vs transformers)", tf_logits, tf_logits)

    # Test 2: decode logits (1-step)
    print("\n-- Test 2: decode logits (1-step) --")
    past_kv = outputs.past_key_values
    next_input = torch.tensor([[sampled_token]], dtype=torch.long, device=device)
    with torch.no_grad():
        decode_outputs = tf_model(next_input, past_key_values=past_kv, use_cache=True)
    decode_logits = decode_outputs.logits[0, -1, :].cpu().float().tolist()

    # Run vLLM generate again with the sampled token as prompt
    sp2 = SamplingParams(max_tokens=1, temperature=0.0)
    vllm_outputs2 = llm.generate(prompts=[[sampled_token]], sampling_params=sp2)
    sampled_token2 = vllm_outputs2[0].outputs[0].token_ids[0]
    print(f"vLLM sampled token 2: {sampled_token2}")

    compare_logits("decode (transformers vs transformers)", decode_logits, decode_logits)

    # Test 3: full autoregressive sequence comparison
    print("\n-- Test 3: full autoregressive sequence (5 tokens) --")
    # Build token sequence using vLLM
    sp3 = SamplingParams(max_tokens=5, temperature=0.0)
    vllm_outputs3 = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp3)
    vllm_tokens = vllm_outputs3[0].outputs[0].token_ids
    print(f"vLLM generated: {vllm_tokens}")

    # Build token sequence using transformers
    current_ids = list(TEST_TOKENS)
    tf_pkv = None
    tf_tokens = []
    for _ in range(5):
        ids_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            if tf_pkv is not None:
                out = tf_model(ids_tensor[:, -1:], past_key_values=tf_pkv, use_cache=True)
            else:
                out = tf_model(ids_tensor, use_cache=True)
        tf_pkv = out.past_key_values
        next_token = int(out.logits[0, -1, :].argmax().item())
        tf_tokens.append(next_token)
        current_ids.append(next_token)
    print(f"Transformers generated: {tf_tokens}")
    print(f"Token match: {vllm_tokens == tf_tokens}")

    print("\nDone.")


if __name__ == "__main__":
    main()
