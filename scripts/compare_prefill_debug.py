#!/usr/bin/env python3
"""
Compare Rust vs Python prefill debug exports step-by-step.

Usage:
    python scripts/compare_prefill_debug.py \
        --rust-dir /tmp/rust_prefill_debug \
        --python-dir /tmp/python_prefill_debug
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def read_tensor(path):
    """Read a tensor from Rust binary format."""
    with open(path, "rb") as f:
        ndims = struct.unpack("<Q", f.read(8))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack("<Q", f.read(8))[0])
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def compare_tensors(name, rust_path, python_path, atol=1e-5):
    """Compare two tensors and print statistics."""
    if not rust_path.exists():
        print(f"  [SKIP] {name}: Rust file missing: {rust_path}")
        return None
    if not python_path.exists():
        print(f"  [SKIP] {name}: Python file missing: {python_path}")
        return None

    r = read_tensor(rust_path)
    p = read_tensor(python_path)

    if r.shape != p.shape:
        print(f"  [FAIL] {name}: Shape mismatch: Rust {r.shape} vs Python {p.shape}")
        return None

    diff = np.abs(r - p)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    rmse = float(np.sqrt(np.mean((r - p) ** 2)))

    status = "PASS" if max_diff < atol else "FAIL"
    print(f"  [{status}] {name}: shape={r.shape}, max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, RMSE={rmse:.6e}")

    # Find location of max diff
    if max_diff >= atol:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"         max diff at index {max_idx}: Rust={r[max_idx]:.8e}, Python={p[max_idx]:.8e}")

    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-dir", required=True)
    parser.add_argument("--python-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    rust_dir = Path(args.rust_dir)
    python_dir = Path(args.python_dir)

    print("=" * 70)
    print("Prefill Debug Comparison: Rust vs Python")
    print("=" * 70)

    files_to_compare = [
        ("embedding_output", "embedding_output.bin", "embedding_output.bin"),
        ("layer_0_input_norm", "layer_0_input_norm.bin", "layer_0_input_norm.bin"),
        ("q_proj_layer_0", "q_proj_layer_0.bin", "q_proj_layer_0.bin"),
        ("k_proj_layer_0", "k_proj_layer_0.bin", "k_proj_layer_0.bin"),
        ("v_proj_layer_0", "v_proj_layer_0.bin", "v_proj_layer_0.bin"),
        ("q_rope_layer_0", "q_rope_layer_0.bin", "q_rope_layer_0.bin"),
        ("k_rope_layer_0", "k_rope_layer_0.bin", "k_rope_layer_0.bin"),
        ("k_cache_layer_0", "k_cache_layer_0.bin", "prefill_k_layer_0.bin"),
        ("v_cache_layer_0", "v_cache_layer_0.bin", "prefill_v_layer_0.bin"),
        ("attn_out_layer_0", "attn_out_layer_0.bin", None),
        ("attn_final_layer_0", "attn_final_layer_0.bin", "attn_final_layer_0.bin"),
        ("layer_0_post_attn", "layer_0_post_attn.bin", "layer_0_post_attn.bin"),
        ("layer_0_post_mlp", "layer_0_post_mlp.bin", "layer_0_post_mlp.bin"),
    ]

    print("\n--- Layer 0 Attention Intermediates ---")
    for name, rust_file, python_file in files_to_compare:
        if python_file is None:
            print(f"  [SKIP] {name}: No Python equivalent exported")
            continue
        compare_tensors(name, rust_dir / rust_file, python_dir / python_file, args.atol)

    # Compare KV cache for all layers
    print("\n--- KV Cache (all layers) ---")
    for i in range(36):  # Qwen2.5-3B has 36 layers
        k_rust = rust_dir / f"prefill_k_layer_{i}.bin"
        k_python = python_dir / f"prefill_k_layer_{i}.bin"
        v_rust = rust_dir / f"prefill_v_layer_{i}.bin"
        v_python = python_dir / f"prefill_v_layer_{i}.bin"

        if not k_rust.exists() or not k_python.exists():
            break

        k_diff = compare_tensors(f"K_layer_{i}", k_rust, k_python, args.atol)
        v_diff = compare_tensors(f"V_layer_{i}", v_rust, v_python, args.atol)

    print("\n" + "=" * 70)
    print("Comparison complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
