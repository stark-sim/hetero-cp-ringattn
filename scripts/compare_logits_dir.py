#!/usr/bin/env python3
"""Compare two coordinator logits exports (logits_<request_id>.bin).

File format (rust distributed coordinator write_logits_file):
    u64le vocab_size, u64le num_chunks, then num_chunks * vocab_size f32le.

Per decode step reports: max|Δ|, mean|Δ|, argmax equality, and the top1-top2
margin of run A (the gate reference) as a noise-floor indicator.

Usage: compare_logits_dir.py <dir_a> <dir_b> [--json]
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_logits(path: Path) -> np.ndarray:
    data = path.read_bytes()
    vocab, chunks = struct.unpack_from("<QQ", data, 0)
    arr = np.frombuffer(data, dtype=np.float32, offset=16)
    expected = vocab * chunks
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {arr.size}")
    return arr.reshape(chunks, vocab)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files_a = sorted(Path(args.dir_a).glob("logits_*.bin"))
    files_b = sorted(Path(args.dir_b).glob("logits_*.bin"))
    if not files_a or not files_b:
        print("ERROR: no logits_*.bin in one of the dirs", file=sys.stderr)
        return 2

    results = []
    for fa, fb in zip(files_a, files_b):
        a = read_logits(fa)
        b = read_logits(fb)
        if a.shape != b.shape:
            results.append({"pair": [fa.name, fb.name], "error": f"shape mismatch {a.shape} vs {b.shape}"})
            continue
        steps = []
        for i in range(a.shape[0]):
            la, lb = a[i], b[i]
            diff = np.abs(la - lb)
            top2 = np.partition(la, -2)[-2:]
            steps.append({
                "step": i,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "argmax_a": int(la.argmax()),
                "argmax_b": int(lb.argmax()),
                "argmax_equal": bool(la.argmax() == lb.argmax()),
                "top1_top2_margin_a": float(top2[1] - top2[0]),
            })
        results.append({
            "pair": [fa.name, fb.name],
            "steps": steps,
            "all_argmax_equal": all(s["argmax_equal"] for s in steps),
            "worst_max_abs_diff": max(s["max_abs_diff"] for s in steps),
            "min_margin_a": min(s["top1_top2_margin_a"] for s in steps),
            "mean_margin_a": float(np.mean([s["top1_top2_margin_a"] for s in steps])),
        })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"{r['pair']}: {r['error']}")
                continue
            noise_ratio = (r["worst_max_abs_diff"] / r["min_margin_a"]
                           if r["min_margin_a"] > 0 else float("inf"))
            print(f"{r['pair'][0]} vs {r['pair'][1]}: all_argmax_equal={r['all_argmax_equal']} "
                  f"worst_max|Δ|={r['worst_max_abs_diff']:.6g} "
                  f"margin(min/mean)={r['min_margin_a']:.6g}/{r['mean_margin_a']:.6g} "
                  f"noise/margin={noise_ratio:.4g}")
            for s in r["steps"]:
                flag = "" if s["argmax_equal"] else "  <-- ARGMAX DIVERGES"
                print(f"  step {s['step']:3d}: max|Δ|={s['max_abs_diff']:.6g} "
                      f"mean|Δ|={s['mean_abs_diff']:.6g} margin={s['top1_top2_margin_a']:.6g}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
