#!/usr/bin/env python3
"""
Explore vLLM logits extraction techniques on vLLM 0.20.x (vllm-metal).

Tests three approaches:
1. SamplingParams.logprobs / prompt_logprobs — non-invasive, preferred
2. LLM.apply_model() — direct model forward, fallback
3. One-hot (current) — baseline for comparison

Compares extracted logits against HuggingFace transformers reference.
"""

import os
import sys
import math

MODEL_DIR = os.path.expanduser("~/VSCodeProjects/hetero-cp-ringattn/models/Qwen2-0.5B")
# Small token sequence for fast testing
TEST_TOKENS = [101, 102, 103, 104, 105]


def load_transformers_reference(model_dir: str):
    """Load transformers model and return a function that computes logits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.float32
    ).to(device)
    vocab_size = config.vocab_size

    @torch.no_grad()
    def get_logits(token_ids: list):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        outputs = model(input_ids, use_cache=True)
        return outputs.logits[0, -1, :].cpu().float().tolist()

    return get_logits, vocab_size


def logprobs_to_logits(logprob_dict: dict, vocab_size: int) -> list:
    """Convert a {token_id: logprob} dict to a full logits vector.

    logprob_i = log(softmax(logits_i)) = logits_i - logsumexp(logits)
    => logits_i = logprob_i + C, where C is an additive constant.

    For argmax and temperature scaling, C cancels out. We set C=0
    (i.e. logits = logprobs) which is correct up to an additive constant.
    """
    logits = [-1e9] * vocab_size
    for token_id, lp in logprob_dict.items():
        if isinstance(lp, (int, float)):
            logprob_val = float(lp)
        else:
            # vLLM Logprob object
            logprob_val = float(getattr(lp, "logprob", lp))
        logits[int(token_id)] = logprob_val
    return logits


def compare_logits(name: str, v_logits: list, ref_logits: list):
    """Compare two logits vectors and print metrics."""
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
    if not match:
        print(f"  vllm_top5: {v.argsort()[-5:][::-1].tolist()}")
        print(f"  ref_top5: {r.argsort()[-5:][::-1].tolist()}")
    return match


def test_logprobs_approach(llm, ref_get_logits, vocab_size):
    """Test SamplingParams.logprobs / prompt_logprobs approach."""
    from vllm import SamplingParams

    print("\n" + "=" * 60)
    print("Approach 1: SamplingParams.logprobs / prompt_logprobs")
    print("=" * 60)

    # ---- Test A: prompt_logprobs for prefill ----
    print("\n-- Test A: prompt_logprobs (prefill stage) --")
    for k in [5, 10, 20]:
        try:
            sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=k)
            outputs = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp)
            prompt_lp = outputs[0].prompt_logprobs
            if prompt_lp is None:
                print(f"  prompt_logprobs={k}: NOT SUPPORTED (returned None)")
                continue
            print(f"  prompt_logprobs={k}: list len={len(prompt_lp)}")
            for idx, entry in enumerate(prompt_lp):
                if entry is not None:
                    print(f"    entry[{idx}]: {len(entry)} logprobs")
                    items = list(entry.items())[:3]
                    print(f"      sample: {[(t, getattr(lp, 'logprob', lp)) for t, lp in items]}")
                else:
                    print(f"    entry[{idx}]: None")
            # The last non-None entry has the logits for the last prompt token
            non_none = [e for e in prompt_lp if e is not None]
            if non_none:
                print(f"  -> usable logprobs entries: {len(non_none)}")
        except Exception as e:
            print(f"  prompt_logprobs={k}: FAILED ({type(e).__name__}: {e})")

    # ---- Test B: logprobs for decode ----
    print("\n-- Test B: logprobs (decode stage) --")
    for k in [5, 10, 20, 50, 100]:
        try:
            sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=k)
            outputs = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp)
            comp = outputs[0].outputs[0]
            lp_list = comp.logprobs
            if not lp_list:
                print(f"  logprobs={k}: NOT SUPPORTED (returned empty)")
                continue
            num_entries = len(lp_list[0])
            print(f"  logprobs={k}: returned {num_entries} logprobs for generated token")
            if num_entries > 0:
                items = list(lp_list[0].items())[:3]
                print(f"    sample: {[(t, getattr(lp, 'logprob', lp)) for t, lp in items]}")
        except Exception as e:
            print(f"  logprobs={k}: FAILED ({type(e).__name__}: {e})")

    # ---- Test C: full vocab logprobs (the critical test) ----
    print("\n-- Test C: full vocab logprobs --")
    for param_name, param_val in [("logprobs", vocab_size), ("prompt_logprobs", vocab_size)]:
        try:
            kwargs = {"max_tokens": 1, "temperature": 0.0, param_name: param_val}
            sp = SamplingParams(**kwargs)
            outputs = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp)
            if param_name == "logprobs":
                lp_data = outputs[0].outputs[0].logprobs
                if lp_data:
                    num = len(lp_data[0])
                else:
                    num = 0
            else:
                lp_data = outputs[0].prompt_logprobs
                if lp_data and lp_data[-1]:
                    num = len(lp_data[-1])
                else:
                    num = 0
            print(f"  {param_name}={vocab_size}: returned {num} entries")
            if num == vocab_size:
                print(f"  ✅ FULL VOCAB returned!")
            elif num > 0:
                print(f"  ⚠️ truncated to {num} entries")
        except Exception as e:
            print(f"  {param_name}={vocab_size}: FAILED ({type(e).__name__}: {e})")

    # ---- Test D: compare against transformers (best available K) ----
    print("\n-- Test D: compare against transformers reference --")
    ref_logits = ref_get_logits(TEST_TOKENS)

    # Try logprobs=20 (max allowed) for comparison
    for param_name, k in [("logprobs", 20), ("prompt_logprobs", 20)]:
        try:
            kwargs = {"max_tokens": 1, "temperature": 0.0, param_name: k}
            sp = SamplingParams(**kwargs)
            outputs = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp)
            if param_name == "logprobs":
                lp_data = outputs[0].outputs[0].logprobs
                if lp_data and lp_data[0]:
                    v_logits = logprobs_to_logits(lp_data[0], vocab_size)
                    compare_logits(f"logprobs={k} (decode)", v_logits, ref_logits)
            else:
                lp_data = outputs[0].prompt_logprobs
                if lp_data and lp_data[-1]:
                    v_logits = logprobs_to_logits(lp_data[-1], vocab_size)
                    compare_logits(f"prompt_logprobs={k} (prefill)", v_logits, ref_logits)
        except Exception as e:
            print(f"  {param_name}={k} comparison failed: {e}")


# Global function for serialization via apply_model
def _apply_model_forward(model, input_ids_list):
    """Run model forward and return last-position logits.
    Must be module-level for pickle serialization."""
    import torch
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)
    # Move to model's device
    if hasattr(model, "model"):
        device = next(model.model.parameters()).device
    else:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    # outputs may be a tuple or ModelOutput; try .logits first
    logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, tuple) else outputs)
    return logits[0, -1, :].cpu().float().tolist()


def test_apply_model_approach(llm, ref_get_logits, vocab_size):
    """Test LLM.apply_model() direct forward approach."""
    import os
    # apply_model requires pickle serialization for function transfer
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    print("\n" + "=" * 60)
    print("Approach 2: LLM.apply_model() direct forward")
    print("=" * 60)

    try:
        # Use functools.partial to bind arguments
        from functools import partial
        func = partial(_apply_model_forward, input_ids_list=TEST_TOKENS)
        result = llm.apply_model(func)
        print(f"  apply_model returned {len(result)} result(s), type={type(result[0])}")
        if isinstance(result[0], list):
            v_logits = result[0]
        else:
            import torch
            v_logits = result[0].cpu().float().tolist()
        ref_logits = ref_get_logits(TEST_TOKENS)
        compare_logits("apply_model direct forward", v_logits, ref_logits)
    except Exception as e:
        print(f"  apply_model failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_direct_model_access(llm, ref_get_logits, vocab_size):
    """Explore direct model access via llm_engine internals."""
    import torch

    print("\n" + "=" * 60)
    print("Approach 2b: Direct model access via llm_engine")
    print("=" * 60)

    # Explore engine structure
    engine = llm.llm_engine
    print(f"  engine type: {type(engine)}")

    # vLLM V1 engine structure
    attrs = [a for a in dir(engine) if not a.startswith('_')]
    model_attrs = [a for a in attrs if 'model' in a.lower()]
    print(f"  model-related attrs: {model_attrs}")

    # Try to find the actual model
    if hasattr(engine, 'model_executor'):
        me = engine.model_executor
        print(f"  model_executor type: {type(me)}")
        me_attrs = [a for a in dir(me) if not a.startswith('_') and 'model' in a.lower()]
        print(f"  model_executor model attrs: {me_attrs}")
        if hasattr(me, 'model'):
            print(f"  model_executor.model: {type(me.model)}")
    
    if hasattr(engine, 'engine_core'):
        ec = engine.engine_core
        print(f"  engine_core type: {type(ec)}")
        ec_attrs = [a for a in dir(ec) if not a.startswith('_') and 'model' in a.lower()]
        print(f"  engine_core model attrs: {ec_attrs}")

    # Try the simplest approach: use the tokenizer to get token_ids,
    # then run model forward directly if we can access it
    try:
        from transformers import AutoModelForCausalLM
        # This loads a separate copy, but proves the concept
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float32, trust_remote_code=True
        )
        if torch.backends.mps.is_available():
            model = model.to("mps")
        input_ids = torch.tensor([TEST_TOKENS], dtype=torch.long)
        if torch.backends.mps.is_available():
            input_ids = input_ids.to("mps")
        with torch.no_grad():
            outputs = model(input_ids)
        v_logits = outputs.logits[0, -1, :].cpu().float().tolist()
        ref_logits = ref_get_logits(TEST_TOKENS)
        compare_logits("direct transformers model (separate load)", v_logits, ref_logits)
    except Exception as e:
        print(f"  direct model access failed: {type(e).__name__}: {e}")


def test_one_hot_baseline(llm, ref_get_logits, vocab_size):
    """Test current one-hot approach as baseline."""
    from vllm import SamplingParams

    print("\n" + "=" * 60)
    print("Approach 3: One-hot baseline (current)")
    print("=" * 60)

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts=[TEST_TOKENS], sampling_params=sp)
    sampled_token = outputs[0].outputs[0].token_ids[0]
    one_hot = [-1e9] * vocab_size
    one_hot[int(sampled_token)] = 1e9

    ref_logits = ref_get_logits(TEST_TOKENS)
    compare_logits("one-hot", one_hot, ref_logits)


def main():
    from vllm import LLM

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
    print(f"Model loaded. vocab_size={vocab_size}")

    print("Loading transformers reference...")
    ref_get_logits, ref_vocab_size = load_transformers_reference(MODEL_DIR)
    print(f"Reference loaded. vocab_size={ref_vocab_size}")
    assert vocab_size == ref_vocab_size, f"vocab size mismatch: {vocab_size} vs {ref_vocab_size}"

    test_logprobs_approach(llm, ref_get_logits, vocab_size)
    test_apply_model_approach(llm, ref_get_logits, vocab_size)
    test_one_hot_baseline(llm, ref_get_logits, vocab_size)
    test_direct_model_access(llm, ref_get_logits, vocab_size)

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
