#!/usr/bin/env python3
"""Compare greedy generation between Python transformers and Rust."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/Users/stark_sim/models/qwen2-0.5b"
PROMPT = "Hello, how are you?"

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Python: '{generated}'")
