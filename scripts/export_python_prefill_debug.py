#!/usr/bin/env python3
"""
Export per-layer attention intermediates from Python transformers during prefill.

This mirrors the Rust `forward_prefill_debug_layer_0` export:
- embedding_output.bin
- layer_0_input_norm.bin
- q_proj_layer_0.bin, k_proj_layer_0.bin, v_proj_layer_0.bin
- q_rope_layer_0.bin, k_rope_layer_0.bin
- k_cache_layer_0.bin, v_cache_layer_0.bin
- attn_final_layer_0.bin
- layer_0_post_attn.bin, layer_0_post_mlp.bin
- prefill_k_layer_{i}.bin, prefill_v_layer_{i}.bin (all layers)

Usage:
    python scripts/export_python_prefill_debug.py \
        --model-dir ~/models/Qwen2.5-3B-Instruct \
        --prompt "Your prompt here" \
        --output-dir /tmp/python_prefill_debug
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import torch
import numpy as np


def write_tensor(path, tensor):
    """Write a tensor in Rust-compatible binary format."""
    arr = tensor.detach().cpu().to(torch.float32).numpy().astype(np.float32)
    shape = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(shape)))
        for dim in shape:
            f.write(struct.pack("<Q", dim))
        f.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_dir
    device = args.device

    print(f"[python-prefill-debug] loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if device == "cpu":
        input_ids = input_ids.to("cpu")
    else:
        input_ids = input_ids.to(model.device)

    print(f"[python-prefill-debug] prompt tokens: {input_ids.shape[1]}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    # Capture embedding output by running just the embedding layer
    with torch.no_grad():
        embed_output = model.model.embed_tokens(input_ids)
    write_tensor(out_dir / "embedding_output.bin", embed_output)

    # Now we need to capture layer 0 intermediates.
    # The new transformers architecture passes position_embeddings to each layer.
    # We'll manually run through layer 0 step by step.

    layer0 = model.model.layers[0]
    attn = layer0.self_attn

    # Step 1: Input norm
    with torch.no_grad():
        hidden_states = embed_output
        residual_attn = hidden_states.clone()  # Save original input for residual
        normed = layer0.input_layernorm(hidden_states)
    write_tensor(out_dir / "layer_0_input_norm.bin", normed)

    # Step 2: Q/K/V projections (replicate attention forward)
    input_shape = normed.shape[:-1]  # [batch, seq_len]
    hidden_shape = (*input_shape, -1, attn.head_dim)  # [batch, seq_len, num_heads, head_dim]

    with torch.no_grad():
        q_proj_out = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        k_proj_out = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        v_proj_out = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

    write_tensor(out_dir / "q_proj_layer_0.bin", q_proj_out)
    write_tensor(out_dir / "k_proj_layer_0.bin", k_proj_out)
    write_tensor(out_dir / "v_proj_layer_0.bin", v_proj_out)

    # Step 3: RoPE — need to get cos/sin from the model's rotary_emb
    # For prefill, position_ids = 0..seq_len-1
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    # Get cos/sin from the model's rotary_emb
    # In newer transformers, rotary_emb is on the model level, not per-layer
    rotary_emb = model.model.rotary_emb
    with torch.no_grad():
        # Qwen2RotaryEmbedding.forward(x, position_ids) returns (cos, sin)
        cos, sin = rotary_emb(q_proj_out, position_ids)

    # Apply rotary pos emb
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    with torch.no_grad():
        q_rot, k_rot = apply_rotary_pos_emb(q_proj_out, k_proj_out, cos, sin)

    write_tensor(out_dir / "q_rope_layer_0.bin", q_rot)
    write_tensor(out_dir / "k_rope_layer_0.bin", k_rot)

    # Step 4: KV cache update — use the model's cache
    # We need to create a cache and update it
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()

    with torch.no_grad():
        # The cache update needs cache_position
        cache_position = torch.arange(seq_len, device=input_ids.device)
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k_cached, v_cached = cache.update(k_rot, v_proj_out, 0, cache_kwargs)

    write_tensor(out_dir / "k_cache_layer_0.bin", k_cached)
    write_tensor(out_dir / "v_cache_layer_0.bin", v_cached)

    # Step 5: Attention computation (eager mode)
    with torch.no_grad():
        # Replicate eager_attention_forward or standard attention
        # For Qwen2 with eager implementation:
        scale = attn.scaling if hasattr(attn, 'scaling') else (1.0 / (attn.head_dim ** 0.5))

        # GQA: repeat k/v heads
        num_key_value_groups = attn.num_key_value_groups
        k_repeat = k_cached.repeat_interleave(num_key_value_groups, dim=1)
        v_repeat = v_cached.repeat_interleave(num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(q_rot, k_repeat.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_rot.dtype)
        attn_output = torch.matmul(attn_weights, v_repeat)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_final = attn.o_proj(attn_output)

    write_tensor(out_dir / "attn_final_layer_0.bin", attn_final)

    # Step 6: Post-attention residual
    with torch.no_grad():
        post_attn = attn_final + residual_attn  # Add to ORIGINAL input, not normed input
    write_tensor(out_dir / "layer_0_post_attn.bin", post_attn)

    # Step 7: MLP
    with torch.no_grad():
        residual_mlp = post_attn.clone()  # Save for residual
        mlp_normed = layer0.post_attention_layernorm(post_attn)
        mlp_out = layer0.mlp(mlp_normed)
        post_mlp = mlp_out + residual_mlp  # Add to original post_attn
    write_tensor(out_dir / "layer_0_post_mlp.bin", post_mlp)

    # Now run the full model to get KV cache for ALL layers
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    past_key_values = outputs.past_key_values

    # Export KV cache for all layers
    for i, (k, v) in enumerate(past_key_values):
        write_tensor(out_dir / f"prefill_k_layer_{i}.bin", k)
        write_tensor(out_dir / f"prefill_v_layer_{i}.bin", v)

    print(f"[python-prefill-debug] exported all intermediates to {args.output_dir}")


if __name__ == "__main__":
    main()
