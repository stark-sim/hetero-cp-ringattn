#!/usr/bin/env python3
"""Extract prefill logits from Python transformers for comparison with Rust."""

import argparse
import struct
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[python] loading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)

    print(f"[python] tokenizing: {args.prompt!r}")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)
    print(f"[python] tokens: {input_ids[0].tolist()}")
    print(f"[python] vocab_size: {model.config.vocab_size}")

    print("[python] forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    print(f"[python] logits shape: {logits.shape}")
    print(f"[python] logits min={logits.min():.4f} max={logits.max():.4f}")
    print(f"[python] argmax={int(np.argmax(logits))} logit={logits[np.argmax(logits)]:.4f}")

    # Save in same format as Rust: [vocab_size: u64 LE][num_chunks: u64 LE][f32 LE...]
    vocab_size = model.config.vocab_size
    num_chunks = 1
    with open(args.output, "wb") as f:
        f.write(struct.pack("<Q", vocab_size))
        f.write(struct.pack("<Q", num_chunks))
        f.write(logits.astype(np.float32).tobytes())

    # Also save hidden states from each layer for deep debugging
    hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors
    print(f"[python] hidden states: {len(hidden_states)} layers")
    for i, h in enumerate(hidden_states):
        h_last = h[0, -1, :].float().cpu().numpy()
        print(f"  layer {i}: shape={h.shape} min={h_last.min():.4f} max={h_last.max():.4f} mean={h_last.mean():.4f}")

    print(f"[python] saved to {args.output}")


if __name__ == "__main__":
    main()
