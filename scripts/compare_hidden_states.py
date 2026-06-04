#!/usr/bin/env python3
"""
Compare per-layer hidden states between Rust and Python transformers exports.

Usage:
    python scripts/compare_hidden_states.py \
        --rust-dir /tmp/rust_hs \
        --python-dir /tmp/python_hs \
        --atol 1e-5

Reports per-layer max_abs_diff and RMSE.
"""

import argparse
import os
import struct
import sys

import numpy as np


def read_tensor(path):
    """Read a tensor in Rust binary format."""
    with open(path, "rb") as f:
        ndims = struct.unpack("<Q", f.read(8))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack("<Q", f.read(8))[0])
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def read_logits(path):
    """Read logits in the format used by compare_logits.py."""
    with open(path, "rb") as f:
        vocab_size = struct.unpack("<Q", f.read(8))[0]
        num_chunks = struct.unpack("<Q", f.read(8))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(num_chunks, vocab_size)


def compare_tensors(name, rust_t, py_t, atol=1e-5, rtol=1e-5):
    if rust_t.shape != py_t.shape:
        print(f"  {name}: SHAPE MISMATCH rust={rust_t.shape} py={py_t.shape}")
        return False

    diff = np.abs(rust_t - py_t)
    max_diff = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_diff = float(np.mean(diff))

    match = max_diff <= atol
    status = "PASS" if match else "FAIL"
    print(f"  {name}: {status} max_diff={max_diff:.6e} rmse={rmse:.6e} mean_diff={mean_diff:.6e} shape={rust_t.shape}")
    return match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-dir", required=True)
    parser.add_argument("--python-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    rust_files = set(os.listdir(args.rust_dir))
    py_files = set(os.listdir(args.python_dir))

    # Find common layer files
    layer_files = sorted([f for f in rust_files if f.startswith("layer_") and f in py_files],
                         key=lambda x: int(x.split("_")[1].split(".")[0]))

    all_pass = True

    print(f"\n=== Comparing hidden states ===")
    print(f"Rust dir: {args.rust_dir}")
    print(f"Python dir: {args.python_dir}")
    print(f"atol={args.atol}, rtol={args.rtol}\n")

    for fname in layer_files:
        rust_t = read_tensor(os.path.join(args.rust_dir, fname))
        py_t = read_tensor(os.path.join(args.python_dir, fname))
        match = compare_tensors(fname, rust_t, py_t, args.atol, args.rtol)
        if not match:
            all_pass = False

    # Compare final norm
    if "final_norm.bin" in rust_files and "final_norm.bin" in py_files:
        rust_t = read_tensor(os.path.join(args.rust_dir, "final_norm.bin"))
        py_t = read_tensor(os.path.join(args.python_dir, "final_norm.bin"))
        match = compare_tensors("final_norm.bin", rust_t, py_t, args.atol, args.rtol)
        if not match:
            all_pass = False
    else:
        print("  final_norm.bin: MISSING")
        all_pass = False

    # Compare logits
    if "logits.bin" in rust_files and "logits.bin" in py_files:
        rust_logits = read_logits(os.path.join(args.rust_dir, "logits.bin"))
        py_logits = read_logits(os.path.join(args.python_dir, "logits.bin"))
        match = compare_tensors("logits.bin", rust_logits, py_logits, args.atol, args.rtol)
        if not match:
            all_pass = False

        # Also show argmax
        rust_argmax = int(np.argmax(rust_logits[0]))
        py_argmax = int(np.argmax(py_logits[0]))
        print(f"  argmax: rust={rust_argmax} py={py_argmax} {'MATCH' if rust_argmax == py_argmax else 'MISMATCH'}")
    else:
        print("  logits.bin: MISSING")
        all_pass = False

    print(f"\n{'='*40}")
    if all_pass:
        print("OVERALL: ALL PASS")
    else:
        print("OVERALL: FAIL — divergence detected")
    print(f"{'='*40}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
