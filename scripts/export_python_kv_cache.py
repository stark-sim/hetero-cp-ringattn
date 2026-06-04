#!/usr/bin/env python3
"""
Export KV cache after prefill from Python transformers.

Usage:
    python scripts/export_python_kv_cache.py \
        --model-dir ~/models/Qwen2.5-3B-Instruct \
        --prompt "Your prompt here" \
        --output-dir /tmp/python_kv
"""

import argparse
import os
import struct

import torch
import numpy as np


def write_tensor(path, tensor):
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

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if device == "cpu":
        input_ids = input_ids.to("cpu")
    else:
        input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

    os.makedirs(args.output_dir, exist_ok=True)

    # Handle DynamicCache (newer transformers) - iterate over layers
    num_layers = 0
    for i, item in enumerate(past_key_values):
        if isinstance(item, tuple):
            k, v = item[0], item[1]
        else:
            raise ValueError(f"Unexpected cache item type: {type(item)}")
        write_tensor(os.path.join(args.output_dir, f"k_layer_{i}.bin"), k)
        write_tensor(os.path.join(args.output_dir, f"v_layer_{i}.bin"), v)
        num_layers += 1

    print(f"[python-kv] exported {num_layers} layers to {args.output_dir}")


if __name__ == "__main__":
    main()
