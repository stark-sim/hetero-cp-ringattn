#!/usr/bin/env python3
"""
Single-process PoC for the vLLM block-ring plugin.

It simulates two HCP domains on one GPU by:
1. Prefilling prompt chunk A and writing its KV blocks into local physical slots.
2. "Exchanging" those blocks to a second set of physical slots (peer domain).
3. Building a combined block table and decoding the next token.
4. Comparing the decoded token / logits to a single-node vLLM reference on the
   full prompt A+B.

Usage:
    python scripts/poc_vllm_block_ring_2worker.py \
        --model-dir models/Qwen2-0.5B \
        --prompt "The capital of France is" \
        --chunk-len 4
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from hcp_vllm_block_ring_plugin import VllmBlockRingPlugin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to HF model")
    parser.add_argument("--prompt", default=(
        "The quick brown fox jumps over the lazy dog while the cat sleeps "
        "on the warm sunny afternoon near the old oak tree."
    ), help="Input prompt text")
    parser.add_argument("--chunk-len", type=int, default=16, help="Tokens in the local chunk (multiple of vLLM block_size)")
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    parser.add_argument("--baseline", action="store_true",
                        help="Prefill the full prompt directly via the plugin and decode (no peer exchange)")
    args = parser.parse_args()

    # Load tokenizer to split prompt deterministically.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"prompt tokens ({len(token_ids)}): {token_ids}")
    assert args.chunk_len < len(token_ids), "chunk-len must be smaller than prompt length"

    chunk_a = token_ids[:args.chunk_len]
    chunk_b = token_ids[args.chunk_len:]

    # ------------------------------------------------------------------
    # Reference: full prefill + one decode with normal vLLM.
    # ------------------------------------------------------------------
    print("\n[ref] running single-node vLLM on full prompt ...")
    from vllm import LLM, SamplingParams
    ref_llm = LLM(
        model=args.model_dir,
        dtype="float32",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        max_num_seqs=1,
    )
    ref_out = ref_llm.generate(
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=1, temperature=0),
    )
    ref_token = ref_out[0].outputs[0].token_ids[0]
    print(f"[ref] next token: {ref_token} ('{tokenizer.decode([ref_token])}')")
    del ref_llm
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    # Distributed simulation with the block-ring plugin.
    # ------------------------------------------------------------------
    print("\n[dist] loading plugin backend as domain 0 ...")
    plugin = VllmBlockRingPlugin(
        args.model_dir,
        dtype="float32",
        gpu_memory_utilization=args.gpu_mem,
    )
    assert args.chunk_len % plugin.block_size == 0, (
        f"chunk-len ({args.chunk_len}) must be a multiple of "
        f"vLLM block_size ({plugin.block_size}) so peer KV starts on a block boundary"
    )

    # ------------------------------------------------------------------
    # Optional baseline: prefill the whole prompt directly, no peer exchange.
    # ------------------------------------------------------------------
    if args.baseline:
        print(f"[dist] baseline prefill full prompt ({len(token_ids)} tokens)")
        plugin.prefill(token_ids, seq_offset=0)
        plugin.set_global_tokens(token_ids)
        dist_logits = plugin.decode(token_ids[-1])
        dist_token = int(dist_logits.argmax())
        print(f"[dist] baseline next token: {dist_token} ('{tokenizer.decode([dist_token])}')")
        match = dist_token == ref_token
        print(f"\n[result] tokens match: {match}")
        if not match:
            print("WARNING: baseline decode disagrees with reference.")
            sys.exit(1)
        plugin.shutdown()
        return

    # Domain 0 prefills chunk A.
    print(f"[dist] prefill chunk A: {chunk_a}")
    plugin.prefill(chunk_a, seq_offset=0)
    local_btable = plugin.get_local_block_table()
    print(f"[dist] local block table: {local_btable}")

    # Prefill chunk B into its own physical blocks (simulating domain 1),
    # then extract its KV and apply it as peer KV on domain 0.
    print(f"[dist] prefill peer chunk B: {chunk_b}")
    peer_btable = plugin.prefill_peer_chunk(chunk_b, seq_offset=len(chunk_a))
    print(f"[dist] peer block table: {peer_btable}")

    # TEMP: inspect which slots in the peer block were written.
    peer_k0, peer_v0 = plugin.extract_block(0, peer_btable[0])
    print(f"[TEMP] peer block k shape {tuple(peer_k0.shape)}, token norms: "
          f"{peer_k0.reshape(plugin.block_size, -1).norm(dim=1).tolist()}")

    for layer_idx in range(plugin.num_layers):
        peer_kv = plugin.get_kv_block_from_table(
            layer_idx,
            seq_start=len(chunk_a),
            seq_end=len(chunk_a) + len(chunk_b),
            block_table=peer_btable,
            table_seq_offset=len(chunk_a),
        )
        plugin.apply_peer_kv(layer_idx, peer_kv)

    print(f"[dist] combined block table: {plugin._combined_block_table}")

    # Set the full global sequence so the decode step sees the complete prompt.
    plugin.set_global_tokens(token_ids)

    # Decode: the next token is generated from the last prompt token.
    decode_input = token_ids[-1]
    print(f"[dist] decode input token (last prompt token): {decode_input}")
    dist_logits = plugin.decode(decode_input)
    dist_token = int(dist_logits.argmax())
    print(f"[dist] next token: {dist_token} ('{tokenizer.decode([dist_token])}')")

    match = dist_token == ref_token
    print(f"\n[result] tokens match: {match}")
    if not match:
        print("WARNING: distributed decode disagrees with reference.")
        sys.exit(1)

    plugin.shutdown()


if __name__ == "__main__":
    main()
