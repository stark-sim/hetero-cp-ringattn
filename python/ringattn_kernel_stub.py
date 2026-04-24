#!/usr/bin/env python3
"""
Ring Attention correctness model.

This module intentionally stays NumPy-only. It models the low-boundary Ring
Attention pattern used by HCP: each domain owns a local Q chunk, while K/V
blocks are visited in ring order and folded into an online softmax state.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    seq_chunk_len: int
    block_size: int
    seq_offset: int


def _attention_scale(head_dim: int, use_scale: bool) -> float:
    if not use_scale:
        return 1.0
    return 1.0 / math.sqrt(float(head_dim))


def online_softmax_update(
    o_prev: np.ndarray,
    l_prev: np.ndarray,
    m_prev: np.ndarray,
    q_chunk: np.ndarray,
    k_block: np.ndarray,
    v_block: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fold one K/V block into the online softmax state for one local Q chunk.

    Shapes:
        o_prev:  [chunk_len, num_heads, head_dim]
        l_prev:  [chunk_len, num_heads]
        m_prev:  [chunk_len, num_heads]
        q_chunk: [chunk_len, num_heads, head_dim]
        k_block: [block_len, num_heads, head_dim]
        v_block: [block_len, num_heads, head_dim]

    Returns:
        (o_new, l_new, m_new)
    """
    if k_block.shape[0] == 0:
        return o_prev, l_prev, m_prev

    scores = np.einsum("qhd,khd->qhk", q_chunk, k_block) * scale
    m_local = np.max(scores, axis=-1)
    p_local = np.exp(scores - m_local[..., None])
    l_local = np.sum(p_local, axis=-1)
    pv_local = np.einsum("qhk,khd->qhd", p_local, v_block)

    m_new = np.maximum(m_prev, m_local)
    exp_prev = np.exp(m_prev - m_new)
    exp_local = np.exp(m_local - m_new)
    l_new = exp_prev * l_prev + exp_local * l_local

    numerator = (
        exp_prev[..., None] * l_prev[..., None] * o_prev
        + exp_local[..., None] * pv_local
    )
    o_new = numerator / l_new[..., None]
    return o_new, l_new, m_new


def ring_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    use_scale: bool = True,
) -> np.ndarray:
    """Standard full attention reference for correctness comparison."""
    scale = _attention_scale(q.shape[-1], use_scale)
    scores = np.einsum("qhd,khd->qhk", q, k) * scale
    p = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    p = p / np.sum(p, axis=-1, keepdims=True)
    return np.einsum("qhk,khd->qhd", p, v)


def build_domain_specs(config: dict[str, Any]) -> list[DomainSpec]:
    specs = []
    offset = 0
    for index, domain in enumerate(config["domains"]):
        seq_chunk_len = int(domain["seq_chunk_len"])
        block_size = int(domain["block_size"])
        if seq_chunk_len <= 0:
            raise ValueError(f"domain {index} has non-positive seq_chunk_len")
        if block_size <= 0:
            raise ValueError(f"domain {index} has non-positive block_size")
        specs.append(
            DomainSpec(
                domain_id=str(domain.get("domain_id", f"domain-{index}")),
                seq_chunk_len=seq_chunk_len,
                block_size=block_size,
                seq_offset=offset,
            )
        )
        offset += seq_chunk_len

    global_seq_len = int(config["global_seq_len"])
    if offset != global_seq_len:
        raise ValueError(
            f"sum(seq_chunk_len)={offset} does not match global_seq_len={global_seq_len}"
        )
    return specs


def _ring_source_order(target_index: int, domain_count: int) -> list[int]:
    return [(target_index + step) % domain_count for step in range(domain_count)]


def _iter_source_blocks(spec: DomainSpec):
    start = spec.seq_offset
    stop = spec.seq_offset + spec.seq_chunk_len
    block_start = start
    while block_start < stop:
        block_stop = min(block_start + spec.block_size, stop)
        yield block_start, block_stop
        block_start = block_stop


def ring_attention_domain_output(
    q_chunk: np.ndarray,
    k_global: np.ndarray,
    v_global: np.ndarray,
    specs: list[DomainSpec],
    target_index: int,
    use_scale: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute one domain output by visiting K/V blocks in ring order."""
    chunk_len, num_heads, head_dim = q_chunk.shape
    output = np.zeros((chunk_len, num_heads, head_dim), dtype=np.float64)
    running_sum = np.zeros((chunk_len, num_heads), dtype=np.float64)
    running_max = np.full((chunk_len, num_heads), -np.inf, dtype=np.float64)
    scale = _attention_scale(head_dim, use_scale)
    trace = []

    for source_index in _ring_source_order(target_index, len(specs)):
        source_spec = specs[source_index]
        for block_start, block_stop in _iter_source_blocks(source_spec):
            output, running_sum, running_max = online_softmax_update(
                output,
                running_sum,
                running_max,
                q_chunk,
                k_global[block_start:block_stop],
                v_global[block_start:block_stop],
                scale=scale,
            )
            trace.append(
                {
                    "source_domain": source_spec.domain_id,
                    "block_start": block_start,
                    "block_stop": block_stop,
                    "block_len": block_stop - block_start,
                }
            )

    return output, trace


def ring_attention_model(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    config: dict[str, Any],
    use_scale: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute all domain outputs and concatenate them in sequence order."""
    specs = build_domain_specs(config)
    outputs = []
    traces = []

    for target_index, spec in enumerate(specs):
        q_chunk = q[spec.seq_offset : spec.seq_offset + spec.seq_chunk_len]
        out, trace = ring_attention_domain_output(
            q_chunk,
            k,
            v,
            specs,
            target_index,
            use_scale=use_scale,
        )
        outputs.append(out)
        traces.append(
            {
                "domain_id": spec.domain_id,
                "seq_offset": spec.seq_offset,
                "seq_chunk_len": spec.seq_chunk_len,
                "block_visits": len(trace),
                "first_blocks": trace[: min(4, len(trace))],
            }
        )

    return np.concatenate(outputs, axis=0), traces


def _case_config(
    global_seq_len: int,
    num_heads: int,
    head_dim: int,
    chunks: list[int],
    block_sizes: list[int],
) -> dict[str, Any]:
    if len(chunks) != len(block_sizes):
        raise ValueError("chunks and block_sizes must have the same length")
    return {
        "global_seq_len": global_seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype": "float32",
        "domains": [
            {
                "domain_id": f"domain-{index}",
                "host": "127.0.0.1",
                "port": 26001 + index,
                "seq_chunk_len": chunk,
                "block_size": block_size,
                "device": "model",
            }
            for index, (chunk, block_size) in enumerate(zip(chunks, block_sizes))
        ],
    }


def default_correctness_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "2domain_uneven_chunks",
            "config": _case_config(128, 4, 16, [80, 48], [16, 12]),
        },
        {
            "name": "3domain_uneven_blocks",
            "config": _case_config(160, 3, 24, [64, 40, 56], [32, 10, 14]),
        },
        {
            "name": "4domain_small_tail_blocks",
            "config": _case_config(192, 2, 32, [32, 64, 48, 48], [7, 16, 11, 13]),
        },
    ]


def run_correctness_case(
    name: str,
    config: dict[str, Any],
    seed: int,
    atol: float,
    mean_atol: float,
    use_scale: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    seq_len = int(config["global_seq_len"])
    num_heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])

    q = rng.standard_normal((seq_len, num_heads, head_dim), dtype=np.float32).astype(np.float64)
    k = rng.standard_normal((seq_len, num_heads, head_dim), dtype=np.float32).astype(np.float64)
    v = rng.standard_normal((seq_len, num_heads, head_dim), dtype=np.float32).astype(np.float64)

    reference = ring_attention_reference(q, k, v, use_scale=use_scale)
    modeled, traces = ring_attention_model(q, k, v, config, use_scale=use_scale)
    diff = np.abs(reference - modeled)
    max_abs_err = float(np.max(diff))
    mean_abs_err = float(np.mean(diff))

    return {
        "name": name,
        "status": "pass" if max_abs_err <= atol and mean_abs_err <= mean_atol else "fail",
        "seed": seed,
        "tolerance": {
            "max_abs_err": atol,
            "mean_abs_err": mean_atol,
        },
        "metrics": {
            "max_abs_err": max_abs_err,
            "mean_abs_err": mean_abs_err,
        },
        "config": config,
        "ring_trace_summary": traces,
    }


def run_correctness_suite(
    seed: int,
    atol: float,
    mean_atol: float,
    use_scale: bool = True,
) -> dict[str, Any]:
    cases = [
        run_correctness_case(
            name=case["name"],
            config=case["config"],
            seed=seed + index,
            atol=atol,
            mean_atol=mean_atol,
            use_scale=use_scale,
        )
        for index, case in enumerate(default_correctness_cases())
    ]
    return {
        "status": "pass" if all(case["status"] == "pass" for case in cases) else "fail",
        "summary": {
            "cases": len(cases),
            "passed": sum(1 for case in cases if case["status"] == "pass"),
            "failed": sum(1 for case in cases if case["status"] == "fail"),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--mean-atol", type=float, default=1e-12)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--report-path", default="reports/ringattn_correctness.json")
    args = parser.parse_args()

    report = run_correctness_suite(
        seed=args.seed,
        atol=args.atol,
        mean_atol=args.mean_atol,
        use_scale=not args.no_scale,
    )
    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(
        "[ringattn-correctness] "
        f"status={report['status']} "
        f"passed={report['summary']['passed']}/{report['summary']['cases']} "
        f"report={args.report_path}"
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
