#!/usr/bin/env python3
"""Verify specific weight values match between Python and Rust expectations."""

import torch
from transformers import AutoModelForCausalLM

MODEL_DIR = "/Users/stark_sim/models/qwen2-0.5b"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")

layer0 = model.model.layers[0].self_attn

print("=== o_proj weight ===")
print(f"shape: {layer0.o_proj.weight.shape}")
print(f"row 0 first 10: {layer0.o_proj.weight[0, :10].tolist()}")
print(f"row 1 first 10: {layer0.o_proj.weight[1, :10].tolist()}")

print("\n=== q_proj bias ===")
print(f"shape: {layer0.q_proj.bias.shape}")
print(f"first 10: {layer0.q_proj.bias[:10].tolist()}")

print("\n=== k_proj bias ===")
print(f"shape: {layer0.k_proj.bias.shape}")
print(f"first 10: {layer0.k_proj.bias[:10].tolist()}")

print("\n=== v_proj bias ===")
print(f"shape: {layer0.v_proj.bias.shape}")
print(f"first 10: {layer0.v_proj.bias[:10].tolist()}")
