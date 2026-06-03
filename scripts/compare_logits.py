#!/usr/bin/env python3
"""Compare exported logits for distributed correctness validation.

Reads two binary logits files (reference and distributed) in the HCP format:
  - Header: [vocab_size: uint64 LE][num_chunks: uint64 LE]
  - Body:  contiguous vocab_size float32 LE per chunk

Reports per-step and aggregate statistics including max absolute difference,
RMSE, and top-k disagreeing tokens.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_logits(path: Path) -> tuple[int, np.ndarray]:
    """Load logits file and return (vocab_size, chunks_array)."""
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"{path}: file too small for header")
    vocab_size = struct.unpack("<Q", data[:8])[0]
    num_chunks = struct.unpack("<Q", data[8:16])[0]
    expected_bytes = 16 + num_chunks * vocab_size * 4
    if len(data) != expected_bytes:
        raise ValueError(
            f"{path}: size mismatch: expected {expected_bytes}, got {len(data)}"
        )
    arr = np.frombuffer(data[16:], dtype=np.float32)
    chunks = arr.reshape(num_chunks, vocab_size)
    return vocab_size, chunks


def topk_disagreements(ref_logits: np.ndarray, dist_logits: np.ndarray, k: int = 5) -> list[tuple[int, float, float, float]]:
    """Return top-k tokens with largest absolute difference."""
    diff = np.abs(ref_logits - dist_logits)
    topk_idx = np.argpartition(diff, -k)[-k:]
    topk_idx = topk_idx[np.argsort(-diff[topk_idx])]
    return [
        (int(idx), float(ref_logits[idx]), float(dist_logits[idx]), float(diff[idx]))
        for idx in topk_idx
    ]


def compare_logits(
    ref_path: Path,
    dist_path: Path,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    topk: int = 5,
    verbose: bool = False,
) -> bool:
    ref_vocab, ref_chunks = load_logits(ref_path)
    dist_vocab, dist_chunks = load_logits(dist_path)

    if ref_vocab != dist_vocab:
        print(f"FAIL: vocab size mismatch: ref={ref_vocab}, dist={dist_vocab}")
        return False

    if ref_chunks.shape[0] != dist_chunks.shape[0]:
        print(
            f"FAIL: chunk count mismatch: ref={ref_chunks.shape[0]}, dist={dist_chunks.shape[0]}"
        )
        return False

    all_ok = True
    max_diff_overall = 0.0
    rmse_total = 0.0
    total_tokens = 0

    for step in range(ref_chunks.shape[0]):
        ref_logits = ref_chunks[step]
        dist_logits = dist_chunks[step]

        diff = np.abs(ref_logits - dist_logits)
        max_diff = float(np.max(diff))
        rmse = float(np.sqrt(np.mean((ref_logits - dist_logits) ** 2)))

        max_diff_overall = max(max_diff_overall, max_diff)
        rmse_total += rmse * rmse
        total_tokens += 1

        step_ok = bool(np.allclose(ref_logits, dist_logits, atol=atol, rtol=rtol))
        if not step_ok:
            all_ok = False

        if verbose or not step_ok:
            status = "OK" if step_ok else "MISMATCH"
            print(f"  Step {step}: max_diff={max_diff:.6e} RMSE={rmse:.6e} [{status}]")
            if not step_ok and topk > 0:
                for idx, ref_v, dist_v, d in topk_disagreements(ref_logits, dist_logits, topk):
                    print(f"    token {idx}: ref={ref_v:.6e} dist={dist_v:.6e} diff={d:.6e}")

    overall_rmse = np.sqrt(rmse_total / total_tokens) if total_tokens > 0 else 0.0

    print(f"\nAggregate: max_diff={max_diff_overall:.6e} RMSE={overall_rmse:.6e}")
    if all_ok:
        print(f"PASS: all {total_tokens} steps within atol={atol}, rtol={rtol}")
    else:
        print(f"FAIL: some steps exceed atol={atol}, rtol={rtol}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare exported logits files")
    parser.add_argument("ref", type=Path, help="Reference logits file (single-node)")
    parser.add_argument("dist", type=Path, help="Distributed logits file")
    parser.add_argument(
        "--atol", type=float, default=1e-3, help="Absolute tolerance (default: 1e-3)"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3, help="Relative tolerance (default: 1e-3)"
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Show top-K disagreeing tokens per step"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-step stats even when matching")
    args = parser.parse_args()

    ok = compare_logits(args.ref, args.dist, args.atol, args.rtol, args.topk, args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
