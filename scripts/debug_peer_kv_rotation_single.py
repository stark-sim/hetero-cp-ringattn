#!/usr/bin/env python3
"""Compare peer KV (local prefill + RoPE) to full-prefill reference using one model load."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from hcp_vllm_block_ring_plugin import VllmBlockRingPlugin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default=(
        "The quick brown fox jumps over the lazy dog while the cat sleeps "
        "on the warm sunny afternoon near the old oak tree by the river in the forest."
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

    plugin = VllmBlockRingPlugin(args.model_dir, dtype="float32", gpu_memory_utilization=args.gpu_mem)

    # Reference full prefill -> extract chunk B KV to CPU.
    plugin.prefill(token_ids, seq_offset=0)
    ref_btable = plugin.get_local_block_table()
    ref_kv_blocks = []
    for layer in range(plugin.num_layers):
        kv = plugin.get_kv_block_from_table(
            layer, seq_start=delta, seq_end=len(token_ids), block_table=ref_btable
        )
        ref_kv_blocks.append((kv.k.cpu().clone(), kv.v.cpu().clone()))

    # Now prefill chunk A + peer chunk B in the same instance.
    plugin.prefill(chunk_a, seq_offset=0)
    peer_btable = plugin.prefill_peer_chunk(chunk_b, seq_offset=delta)

    for layer in range(plugin.num_layers):
        ref_k, ref_v = ref_kv_blocks[layer]
        peer_kv = plugin.get_kv_block_from_table(
            layer, seq_start=0, seq_end=len(chunk_b), block_table=peer_btable
        )
        peer_k = peer_kv.k.cpu()
        peer_v = peer_kv.v.cpu()
        peer_k_rot = plugin._rope_delta_rotate_keys(peer_k, delta)
        k_diff = (peer_k_rot - ref_k).abs().max().item()
        v_diff = (peer_v - ref_v).abs().max().item()
        print(f"layer={layer} k_max_diff={k_diff:.6f} v_max_diff={v_diff:.6f}")

    plugin.shutdown()


if __name__ == "__main__":
    main()
