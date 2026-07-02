#!/usr/bin/env python3
"""Compare peer KV (prefill local + RoPE rotate) to a full-prefill reference."""
import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from hcp_vllm_block_ring_plugin import VllmBlockRingPlugin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default=(
        "The quick brown fox jumps over the lazy dog while the cat sleeps "
        "on the warm sunny afternoon near the old oak tree."
    ))
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"prompt tokens ({len(token_ids)}): {token_ids}")
    assert args.chunk_len < len(token_ids)
    assert args.chunk_len % 16 == 0
    chunk_a = token_ids[:args.chunk_len]
    chunk_b = token_ids[args.chunk_len:]
    delta = len(chunk_a)

    # Reference: full prefill -> extract chunk B KV -> CPU -> free GPU.
    print("\n[ref] full prefill plugin")
    ref_plugin = VllmBlockRingPlugin(args.model_dir, dtype="float32", gpu_memory_utilization=args.gpu_mem)
    ref_plugin.prefill(token_ids, seq_offset=0)
    ref_btable = ref_plugin.get_local_block_table()
    ref_kv_blocks = []
    for layer in range(ref_plugin.num_layers):
        kv = ref_plugin.get_kv_block_from_table(
            layer, seq_start=delta, seq_end=len(token_ids), block_table=ref_btable
        )
        ref_kv_blocks.append((kv.k.cpu().clone(), kv.v.cpu().clone()))
    ref_plugin.shutdown()
    del ref_plugin
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Peer: chunk A + chunk B prefilled locally, then chunk B extracted and rotated.
    print("\n[peer] chunk A + peer chunk B plugin")
    peer_plugin = VllmBlockRingPlugin(args.model_dir, dtype="float32", gpu_memory_utilization=args.gpu_mem)
    peer_plugin.prefill(chunk_a, seq_offset=0)
    peer_btable = peer_plugin.prefill_peer_chunk(chunk_b, seq_offset=delta)

    for layer in range(peer_plugin.num_layers):
        ref_k, ref_v = ref_kv_blocks[layer]
        peer_kv_raw = peer_plugin.get_kv_block_from_table(
            layer, seq_start=0, seq_end=len(chunk_b), block_table=peer_btable
        )
        peer_k = peer_kv_raw.k.cpu()
        peer_v = peer_kv_raw.v.cpu()
        # Try both signs and report which is closer.
        for sign_name, sign in [("+delta", 1), ("-delta", -1)]:
            peer_k_rot = peer_plugin._rope_delta_rotate_keys(peer_k, sign * delta)
            k_diff = (peer_k_rot - ref_k).abs().max().item()
            v_diff = (peer_v - ref_v).abs().max().item()
            print(f"layer={layer} sign={sign_name} k_max_diff={k_diff:.6f} v_max_diff={v_diff:.6f}")

    peer_plugin.shutdown()


if __name__ == "__main__":
    main()
