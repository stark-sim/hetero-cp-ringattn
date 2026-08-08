#!/usr/bin/env python3
"""Compare two route_b_cross_node_smoke dump directories.

Each dump dir contains meta.json plus optional f32 little-endian logits
files (prefill_last_logits / decode_logits / continuation_last_logits).
For every artifact present in both dirs this prints argmax equality and
mean/max absolute diff, then exits non-zero if any argmax differs or any
diff exceeds the given tolerances.

Usage:
    compare_route_b_dumps.py DIR_A DIR_B [--mean-tol 0.1] [--max-tol 0.75]
"""

import argparse
import json
import sys
from pathlib import Path

import struct

ARTIFACTS = [
    "prefill_last_logits",
    "decode_logits",
    "continuation_last_logits",
]


def load_f32le(path: Path):
    data = path.read_bytes()
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


def argmax(values):
    best_idx = 0
    best_val = values[0]
    for i, v in enumerate(values):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


# One bf16 ulp at |logit| in [8, 16); cross-device bf16 rounding can
# legitimately flip an exact tie (observed on ROCm HIP: 17/198 both 12.0625).
TIE_EPS = 0.0625


def argmax_verdict(va, vb):
    """Returns (equal_or_tie, description)."""
    aa = argmax(va)
    ab = argmax(vb)
    if aa == ab:
        return True, f"argmax {aa} == {ab}"
    gap_a = va[aa] - va[ab]
    gap_b = vb[ab] - vb[aa]
    if gap_a <= TIE_EPS and gap_b <= TIE_EPS:
        return True, f"argmax {aa} vs {ab} NEAR-TIE (gaps {gap_a:.6f}/{gap_b:.6f})"
    return False, f"argmax {aa} vs {ab} DIFFER (gaps {gap_a:.6f}/{gap_b:.6f})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a", type=Path)
    parser.add_argument("dir_b", type=Path)
    parser.add_argument("--mean-tol", type=float, default=0.1)
    parser.add_argument("--max-tol", type=float, default=0.75)
    args = parser.parse_args()

    meta_a = json.loads((args.dir_a / "meta.json").read_text())
    meta_b = json.loads((args.dir_b / "meta.json").read_text())
    print(f"A: {args.dir_a} (mode={meta_a.get('mode')}, device={meta_a.get('device')})")
    print(f"B: {args.dir_b} (mode={meta_b.get('mode')}, device={meta_b.get('device')})")

    ok = True
    compared = 0
    for name in ARTIFACTS:
        path_a = args.dir_a / f"{name}.f32le"
        path_b = args.dir_b / f"{name}.f32le"
        if not (path_a.exists() and path_b.exists()):
            print(f"{name}: skipped (present in only one dump)")
            continue
        va = load_f32le(path_a)
        vb = load_f32le(path_b)
        if len(va) != len(vb):
            print(f"{name}: LENGTH MISMATCH {len(va)} vs {len(vb)}")
            ok = False
            continue
        argmax_ok, argmax_desc = argmax_verdict(va, vb)
        diffs = [abs(x - y) for x, y in zip(va, vb)]
        mean_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        status = (
            "OK"
            if argmax_ok and mean_diff <= args.mean_tol and max_diff <= args.max_tol
            else "FAIL"
        )
        if status == "FAIL":
            ok = False
        compared += 1
        print(
            f"{name}: {argmax_desc}, "
            f"mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f} -> {status}"
        )

    for key in ("decode_token", "prefill_argmax", "decode_argmax", "continuation_argmax"):
        va, vb = meta_a.get(key), meta_b.get(key)
        if va is not None and vb is not None:
            # Informational: the artifact-level tie-aware check above is the
            # authoritative gate; meta argmaxes may legitimately differ on a
            # bf16 near-tie.
            match = "equal" if va == vb else "differ (see artifact verdict)"
            print(f"meta.{key}: {va} vs {vb} ({match})")

    if compared == 0:
        print("no shared artifacts to compare")
        return 2
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
