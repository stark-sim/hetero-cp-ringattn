#!/usr/bin/env python3
"""
Single-process PoC for the vLLM V1 block-ring plugin (vLLM >= 0.23).

Mirrors scripts/poc_vllm_block_ring_2worker.py but uses the V1 engine plugin
(python/hcp_vllm_block_ring_plugin_v1.py).  It simulates two HCP domains on
one GPU by:
1. Prefilling prompt chunk A and writing its KV blocks into local physical slots.
2. "Exchanging" those blocks to a second set of physical slots (peer domain).
3. Building a combined block table and decoding the next token.
4. Comparing the decoded token / logits to a single-node vLLM reference on the
   full prompt A+B.

Usage:
    python scripts/poc_vllm_block_ring_v1.py \
        --model-dir /home/stark/models/Qwen2-0.5B-1M \
        --prompt "The capital of France is" \
        --chunk-len 16 --block-size 16 --decode
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from hcp_vllm_block_ring_plugin_v1 import VllmBlockRingPluginV1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to HF model")
    parser.add_argument("--prompt", default=(
        "The quick brown fox jumps over the lazy dog while the cat sleeps "
        "on the warm sunny afternoon near the old oak tree."
    ), help="Input prompt text")
    parser.add_argument("--chunk-len", type=int, default=16, help="Tokens in the local chunk")
    parser.add_argument("--block-size", type=int, default=16, help="vLLM KV-cache block size")
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    parser.add_argument("--baseline", action="store_true",
                        help="Prefill the full prompt directly via the plugin and decode (no peer exchange)")
    parser.add_argument("--decode", action="store_true",
                        help="Also run one autoregressive decode step and compare the second token")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"prompt tokens ({len(token_ids)}): {token_ids}")
    assert args.chunk_len < len(token_ids), "chunk-len must be smaller than prompt length"

    chunk_a = token_ids[:args.chunk_len]
    chunk_b = token_ids[args.chunk_len:]

    if not args.baseline:
        if len(chunk_a) % args.block_size != 0 or len(chunk_b) % args.block_size != 0:
            print(
                f"ERROR: peer mode currently requires both chunk lengths to be "
                f"multiples of --block-size ({args.block_size}). Got chunk_a="
                f"{len(chunk_a)}, chunk_b={len(chunk_b)}."
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Reference: full prefill + one decode with normal vLLM.
    # ------------------------------------------------------------------
    print("\n[ref] running single-node vLLM on full prompt ...")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    ref_llm = LLM(
        model=args.model_dir,
        dtype="float32",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        max_num_seqs=1,
        max_model_len=4096,
    )
    ref_max_tokens = 2 if args.decode else 1
    ref_out = ref_llm.generate(
        [TokensPrompt(prompt_token_ids=token_ids)],
        sampling_params=SamplingParams(max_tokens=ref_max_tokens, temperature=0, logprobs=10),
    )
    ref_token = ref_out[0].outputs[0].token_ids[0]
    ref_top5 = sorted(ref_out[0].outputs[0].logprobs[0].items(), key=lambda kv: -kv[1].logprob)[:5]
    print(f"[ref] next token: {ref_token} ('{tokenizer.decode([ref_token])}')  top5: " +
          " | ".join(f"{tok}({tokenizer.decode([tok])!r}):{lp.logprob:.3f}" for tok, lp in ref_top5))
    if args.decode:
        ref_token2 = ref_out[0].outputs[0].token_ids[1]
        ref_top5_2 = sorted(ref_out[0].outputs[0].logprobs[1].items(), key=lambda kv: -kv[1].logprob)[:5]
        print(f"[ref] second token: {ref_token2} ('{tokenizer.decode([ref_token2])}')  top5: " +
              " | ".join(f"{tok}({tokenizer.decode([tok])!r}):{lp.logprob:.3f}" for tok, lp in ref_top5_2))
    try:
        ref_llm.llm_engine.shutdown()
    except Exception:
        pass
    del ref_llm
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Distributed simulation with the V1 block-ring plugin.
    # ------------------------------------------------------------------
    print("\n[dist] loading V1 plugin backend as domain 0 ...")
    plugin = VllmBlockRingPluginV1(
        args.model_dir,
        dtype="float32",
        gpu_memory_utilization=args.gpu_mem,
        block_size=args.block_size,
        max_model_len=4096,
    )

    def top5_text(logits):
        vals, idxs = torch.topk(logits, 5)
        return " | ".join(f"{int(idxs[i])}({tokenizer.decode([int(idxs[i])])!r}):{vals[i]:.3f}" for i in range(5))

    if args.baseline:
        print(f"[dist] baseline prefill full prompt ({len(token_ids)} tokens)")
        dist_logits, _ = plugin.prefill(token_ids, seq_offset=0)
        dist_token = int(dist_logits.argmax())
        print(f"[dist] baseline next token: {dist_token} ('{tokenizer.decode([dist_token])}')  top5: {top5_text(dist_logits)}")
        match = dist_token == ref_token
        print(f"\n[result] tokens match: {match}")
        plugin.shutdown()
        sys.exit(0 if match else 1)

    # Domain 0 prefills chunk A.
    print(f"[dist] prefill chunk A: {chunk_a}")
    plugin.prefill(chunk_a, seq_offset=0)
    local_btable = plugin.get_local_block_table()
    print(f"[dist] local block table: {local_btable}")

    # Simulate the real ring-attention flow: domain 1 receives domain 0's KV,
    # prefills chunk B with that prior context, then domain 0 receives the
    # chunk-B KV and inserts it.
    print(f"[dist] copy local KV as prior context for peer prefill")
    context_btable = plugin._copy_block_table(local_btable)
    print(f"[dist] context block table: {context_btable}")

    print(f"[dist] prefill peer chunk B with context: {chunk_b}")
    peer_btable = plugin.prefill_peer_chunk_with_context(
        chunk_b,
        seq_offset=len(chunk_a),
        context_tokens=chunk_a,
        context_block_table=context_btable,
    )
    print(f"[dist] peer block table: {peer_btable}")

    for layer_idx in range(plugin.num_layers):
        peer_kv = plugin.get_kv_block_from_table(
            layer_idx,
            seq_start=len(chunk_a),
            seq_end=len(chunk_a) + len(chunk_b),
            block_table=peer_btable,
            table_seq_offset=len(chunk_a),
        )
        # Peer KV was already computed with global positions, so no rotation.
        plugin.apply_peer_kv(layer_idx, peer_kv, rotate_delta=0)

    print(f"[dist] combined block table: {plugin._combined_block_table}")

    plugin.set_global_tokens(token_ids)
    dist_logits = plugin.last_token_logits()
    dist_token = int(dist_logits.argmax())
    print(f"[dist] last-position next token: {dist_token} ('{tokenizer.decode([dist_token])}')  top5: {top5_text(dist_logits)}")

    match = dist_token == ref_token
    print(f"\n[result] first token match: {match}")
    if not match:
        print("WARNING: distributed last-position logits disagree with reference.")
        plugin.shutdown()
        sys.exit(1)

    if args.decode:
        print("[dist] running one autoregressive decode step ...")
        dist_logits2 = plugin.decode(ref_token)
        dist_token2 = int(dist_logits2.argmax())
        print(f"[dist] second token: {dist_token2} ('{tokenizer.decode([dist_token2])}')  top5: {top5_text(dist_logits2)}")
        match2 = dist_token2 == ref_token2
        print(f"[result] second token match: {match2}")
        if not match2:
            print("WARNING: distributed decode step disagrees with reference.")
            plugin.shutdown()
            sys.exit(1)

    plugin.shutdown()


if __name__ == "__main__":
    main()
