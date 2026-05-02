#!/usr/bin/env python3
"""
Debug script: compare Qwen2-0.5B forward pass between Python (transformers) and Rust.

This script loads the model with float32 weights, runs a fixed prompt through forward,
and prints/saves key intermediate values for layer-by-layer comparison.
"""

import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/Users/stark_sim/models/qwen2-0.5b"
PROMPT = "Hello, how are you?"

print(f"Loading model from {MODEL_DIR}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

print(f"Tokenizing: '{PROMPT}'")
inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]
print(f"input_ids shape: {input_ids.shape}")
print(f"input_ids: {input_ids[0].tolist()}")

# Print embedding for first token
with torch.no_grad():
    embed = model.model.embed_tokens(input_ids)
    print(f"\n=== Embedding (first token, first 10 dims) ===")
    print(f"{embed[0, 0, :10].tolist()}")
    print(f"Embedding shape: {embed.shape}")

# Run forward with hidden_states output
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    
hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors
logits = outputs.logits

print(f"\nNumber of hidden state tensors: {len(hidden_states)} (embed + {len(hidden_states)-1} layers)")

for i, h in enumerate(hidden_states):
    label = "embed" if i == 0 else f"layer_{i-1}_out"
    print(f"\n=== {label} ===")
    print(f"  shape: {h.shape}")
    print(f"  first token first 10 dims: {h[0, 0, :10].tolist()}")
    print(f"  last token first 10 dims:  {h[0, -1, :10].tolist()}")
    print(f"  mean: {h.mean().item():.6f}, std: {h.std().item():.6f}")

# Final logits
print(f"\n=== Final Logits (last position, top 20) ===")
last_logits = logits[0, -1, :]
topk = torch.topk(last_logits, 20)
for i in range(20):
    tok_id = topk.indices[i].item()
    tok_str = tokenizer.decode([tok_id])
    print(f"  rank {i}: id={tok_id:>6} prob={torch.softmax(last_logits, -1)[tok_id].item():.6f} token='{tok_str}'")

# Save to file for comparison
out_dir = "reports/debug_qwen_python"
os.makedirs(out_dir, exist_ok=True)

data = {
    "input_ids": input_ids[0].tolist(),
    "embedding_first_token": embed[0, 0, :].tolist(),
    "layers": [],
    "logits_top20": {
        "indices": topk.indices.tolist(),
        "values": topk.values.tolist(),
    },
}

for i, h in enumerate(hidden_states):
    data["layers"].append({
        "first_token": h[0, 0, :].tolist(),
        "last_token": h[0, -1, :].tolist(),
    })

with open(f"{out_dir}/forward_trace.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved to {out_dir}/forward_trace.json")
