#!/usr/bin/env python3
"""
Export per-layer hidden states from Python transformers during decode step 1.

Usage:
    python scripts/export_python_hidden_states.py \
        --model-dir ~/models/Qwen2.5-3B-Instruct \
        --prompt "Your prompt here" \
        --output-dir /tmp/python_hs

This script:
1. Loads the model with transformers
2. Runs prefill on the prompt
3. Samples the first token (greedy, temperature=0)
4. Runs decode step 1 with forward hooks to capture hidden states after each layer
5. Saves hidden states in binary format compatible with Rust export:
   [ndims: u64 LE][dim0: u64 LE]...[dimN: u64 LE][f32 data...]
"""

import argparse
import json
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

    print(f"[python-hs] loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if device == "cpu":
        input_ids = input_ids.to("cpu")
    else:
        input_ids = input_ids.to(model.device)

    print(f"[python-hs] prompt tokens: {input_ids.shape[1]}")

    # Storage for hidden states during decode step 1
    hidden_states_by_layer = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Qwen2DecoderLayer returns (hidden_states, attn_weights, past_kv)
            # or just hidden_states depending on flags
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            hidden_states_by_layer[layer_idx] = hs.detach().clone()
        return hook

    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Also capture final norm output
    final_norm_hs = {}

    def final_norm_hook(module, input, output):
        final_norm_hs["output"] = output.detach().clone()

    h = model.model.norm.register_forward_hook(final_norm_hook)
    hooks.append(h)

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits

        # Sample first token (greedy)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        print(f"[python-hs] first decode token id: {next_token_id}")

        # Decode step 1
        next_input = torch.tensor([[next_token_id]], device=input_ids.device)
        outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
        decode_logits = outputs.logits

    # Remove hooks
    for h in hooks:
        h.remove()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save per-layer hidden states
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        path = os.path.join(args.output_dir, f"layer_{i}.bin")
        write_tensor(path, hidden_states_by_layer[i])

    # Save final norm output
    path = os.path.join(args.output_dir, "final_norm.bin")
    write_tensor(path, final_norm_hs["output"])

    # Save logits
    vocab_size = decode_logits.shape[-1]
    logits_path = os.path.join(args.output_dir, "logits.bin")
    with open(logits_path, "wb") as f:
        f.write(struct.pack("<Q", vocab_size))
        f.write(struct.pack("<Q", 1))  # 1 chunk
        arr = decode_logits.squeeze().cpu().to(torch.float32).numpy().astype(np.float32)
        f.write(arr.tobytes())

    print(f"[python-hs] saved hidden states and logits to {args.output_dir}")


if __name__ == "__main__":
    main()
