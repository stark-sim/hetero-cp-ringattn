#!/usr/bin/env python3
"""
Compare Rust vs Python BF16 inference logits and generated tokens.

Usage:
    python scripts/compare_rust_python_bf16.py \
        --model-dir models/Qwen2-0.5B \
        --prompt "Hello, world!"
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_rust_logits(path: Path) -> np.ndarray:
    """Read logits from Rust binary format: [vocab_size u64][num_chunks u64][f32 data...]"""
    with open(path, "rb") as f:
        vocab_size = struct.unpack("<Q", f.read(8))[0]
        num_chunks = struct.unpack("<Q", f.read(8))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(num_chunks, vocab_size)


def export_python_logits(model_dir: str, prompt: str):
    """Run Python transformers inference and return logits and generated tokens."""
    print("[Python] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids, output_hidden_states=False)
        prefill_logits = outputs.logits.to(torch.float32).numpy()
        
        # Get last token logits for first decode step
        last_token_logits = prefill_logits[0, -1, :]
        
        # Greedy decode first token
        first_token_id = int(np.argmax(last_token_logits))
        first_token_text = tokenizer.decode([first_token_id], skip_special_tokens=True)
        
    print(f"[Python] First decode token: {first_token_id} -> '{first_token_text}'")
    return last_token_logits, first_token_id, first_token_text


def export_rust_logits(model_dir: str, prompt: str, export_dir: str):
    """Run Rust inference and export logits."""
    rust_bin = Path(__file__).parent.parent / "rust" / "target" / "release" / "hcp-ringattn-rust"

    env = os.environ.copy()
    env["HCP_ENABLE_TORCH"] = "1"
    env["HCP_TORCH_DEVICE"] = "cpu"
    if "LIBTORCH" in env:
        env["DYLD_LIBRARY_PATH"] = f"{env['LIBTORCH']}/lib:{env.get('DYLD_LIBRARY_PATH', '')}"
    elif Path("/Users/stark_sim/libtorch").exists():
        env["DYLD_LIBRARY_PATH"] = f"/Users/stark_sim/libtorch/lib:{env.get('DYLD_LIBRARY_PATH', '')}"

    cmd = [
        str(rust_bin),
        "--infer-model-dir", model_dir,
        "--infer-prompt", prompt,
        "--infer-max-tokens", "1",
        "--export-logits", export_dir,
    ]

    print(f"[Rust] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[Rust] STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Rust inference failed with code {result.returncode}")

    # Read Rust logits
    logits_path = Path(export_dir) / "logits.bin"
    rust_logits = read_rust_logits(logits_path)
    
    # Rust exports prefill last-token logits as first chunk, then decode step logits
    # For max_tokens=1, we have 2 chunks: prefill + decode
    print(f"[Rust] Logits shape: {rust_logits.shape}")
    
    # First decode token (from prefill last-token logits)
    first_token_id = int(np.argmax(rust_logits[0]))
    
    return rust_logits, first_token_id


def compare_logits(py_logits: np.ndarray, rust_logits: np.ndarray, atol: float = 1e-3):
    """Compare logits and print statistics."""
    diff = np.abs(py_logits - rust_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    rmse = float(np.sqrt(np.mean((py_logits - rust_logits) ** 2)))
    
    # Top-1 agreement
    py_top1 = int(np.argmax(py_logits))
    rust_top1 = int(np.argmax(rust_logits))
    
    # Top-5 agreement
    py_top5 = set(np.argpartition(py_logits, -5)[-5:])
    rust_top5 = set(np.argpartition(rust_logits, -5)[-5:])
    
    status = "PASS" if max_diff < atol else "FAIL"
    print(f"  [{status}] Logits comparison:")
    print(f"         max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, RMSE={rmse:.6e}")
    print(f"         Top-1: Python={py_top1}, Rust={rust_top1}, match={py_top1 == rust_top1}")
    print(f"         Top-5 overlap: {len(py_top5 & rust_top5)}/5")
    
    if max_diff >= atol:
        max_idx = int(np.argmax(diff))
        print(f"         max diff at token {max_idx}: Python={py_logits[max_idx]:.8e}, Rust={rust_logits[max_idx]:.8e}")
    
    return max_diff, py_top1 == rust_top1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/Qwen2-0.5B")
    parser.add_argument("--prompt", default="Hello, world!")
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 70)
    print("BF16 Rust vs Python Logit Comparison")
    print("=" * 70)
    print(f"Model: {args.model_dir}")
    print(f"Prompt: '{args.prompt}'")
    print()

    # Run Python
    print("--- Python BF16 Inference ---")
    py_logits, py_token_id, py_token_text = export_python_logits(args.model_dir, args.prompt)
    print()

    # Run Rust
    print("--- Rust BF16 Inference ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        rust_logits, rust_token_id = export_rust_logits(args.model_dir, args.prompt, tmpdir)
    print()

    # Compare
    print("--- Logits Comparison ---")
    
    # Python logits are for prefill last token
    # Rust logits[0] should be the prefill last-token logits
    if rust_logits.shape[0] >= 1:
        rust_prefill_logits = rust_logits[0]
        print("Comparing prefill last-token logits:")
        max_diff, top1_match = compare_logits(py_logits, rust_prefill_logits, args.atol)
    else:
        print("Rust logits empty!")
        return

    print()
    print(f"Python generated token: {py_token_id} -> '{py_token_text}'")
    print(f"Rust generated token:   {rust_token_id}")
    
    print()
    print("=" * 70)
    if max_diff < args.atol and py_token_id == rust_token_id:
        print("RESULT: PASS - Rust BF16 matches Python BF16")
    else:
        print("RESULT: FAIL - Differences detected")
    print("=" * 70)


if __name__ == "__main__":
    main()
