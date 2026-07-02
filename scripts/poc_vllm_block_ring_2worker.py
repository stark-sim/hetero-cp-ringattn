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
    parser.add_argument("--prompt", default="The capital of France is", help="Input prompt text")
    parser.add_argument("--chunk-len", type=int, default=4, help="Tokens in the local chunk")
    parser.add_argument("--gpu-mem", type=float, default=0.5)
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

    # Domain 0 prefills chunk A.
    print(f"[dist] prefill chunk A: {chunk_a}")
    plugin.prefill(chunk_a, seq_offset=0)
    local_btable = plugin.get_local_block_table()
    print(f"[dist] local block table: {local_btable}")

    # Simulate receiving domain 1's chunk B by allocating remote slots and
    # copying A's blocks there, then pretending B's blocks are in A's slots.
    # For a real cross-node run, the tensors would be serialized and sent.
    num_remote_blocks = (len(chunk_b) + plugin.block_size - 1) // plugin.block_size
    remote_bids = plugin._reserve_remote_blocks(num_remote_blocks)
    print(f"[dist] reserved remote slots: {remote_bids}")

    # In the simulation we copy A's KV into the remote slots (pretend they are
    # B's KV).  The combined block table is then local + remote in token order.
    for layer_idx in range(plugin.num_layers):
        for i, rbid in enumerate(remote_bids):
            # Just copy the i-th local block into the remote slot.
            local_bid = local_btable[i % len(local_btable)]
            k, v = plugin.extract_block(layer_idx, local_bid)
            plugin.insert_block(layer_idx, rbid, k, v)

    # Build combined block table: local blocks cover chunk A, remote blocks
    # cover the rest.
    plugin._build_combined_block_table(len(chunk_a), remote_bids)
    print(f"[dist] combined block table: {plugin._combined_block_table}")

    # Decode: append the first token of chunk B as the decode input.
    decode_input = chunk_b[0]
    print(f"[dist] decode input token: {decode_input}")
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
