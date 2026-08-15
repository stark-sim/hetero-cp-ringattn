#!/usr/bin/env python3
"""K10 KV-transport quantitative ledger (bytes-on-wire).

Aggregate HCP_PERF_LOG JSONL files (one per worker/domain) into a per-request /
per-token KV-transport byte ledger, and emit the same-caliber comparison
formulas for vLLM PD (NIXL full-KV transfer) and TP (per-layer all-reduce).

Usage:
  python3 scripts/kv_transport_ledger.py --perf perf-w0.jsonl perf-w1.jsonl \
      [--layers 24 --kv-heads 8 --head-dim 128 --hidden 896 --seq-len 8192 --elem 2]

The script reports *measured* wire bytes for HCP (sum of wire_sent_bytes across
all domains, which equals sum of wire_recv_bytes in a lossless ring) and
*derived* reference bytes for PD/TP from the model geometry, so the three are
on the same "actual bytes moved" caliber.

HCP event kinds understood:
  ring_attention        prefill KV ring (per layer)     -> prefill KV bytes
  ring_decode           legacy Q-ring decode (per layer)
  stationary_decode     mainline self-driving decode (per token)
  stationary_continuation  multi-token continuation segment
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(paths):
    events = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"warn: skipping bad JSON line in {p}: {exc}", file=sys.stderr)
    return events


def _get(event, key, default=0):
    v = event.get(key, default)
    if v is None:
        return default
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return default
    return v


def aggregate(events):
    """Return (per_request, per_token, totals) wire-byte ledger."""
    # per (request_id) -> {"prefill_sent": .., "decode_sent": .., ...}
    per_request = defaultdict(lambda: defaultdict(int))
    per_token = defaultdict(lambda: defaultdict(int))  # (request_id, token)
    totals = defaultdict(int)
    layer_events = defaultdict(list)  # event kind -> list of events (for debug)

    for e in events:
        kind = e.get("event")
        sent = _get(e, "wire_sent_bytes")
        recv = _get(e, "wire_recv_bytes")
        domain = _get(e, "domain")
        layer = _get(e, "layer")
        req = e.get("request_id")
        token = e.get("token")

        if kind == "ring_attention":
            totals["prefill_sent"] += sent
            totals["prefill_recv"] += recv
            if req is not None:
                per_request[req]["prefill_sent"] += sent
                per_request[req]["prefill_recv"] += recv
        elif kind == "ring_decode":
            totals["legacy_decode_sent"] += sent
            totals["legacy_decode_recv"] += recv
            if req is not None:
                per_request[req]["decode_sent"] += sent
                per_request[req]["decode_recv"] += recv
        elif kind == "stationary_decode":
            totals["decode_sent"] += sent
            totals["decode_recv"] += recv
            if req is not None:
                per_request[req]["decode_sent"] += sent
                per_request[req]["decode_recv"] += recv
            if token is not None:
                key = (req, token)
                per_token[key]["sent"] += sent
                per_token[key]["recv"] += recv
        elif kind == "stationary_continuation":
            totals["continuation_sent"] += sent
            totals["continuation_recv"] += recv
            if req is not None:
                per_request[req]["continuation_sent"] += sent
                per_request[req]["continuation_recv"] += recv

        layer_events[kind].append(e)

    return per_request, per_token, totals, layer_events


def _mb(n):
    return f"{n / (1024 * 1024):.2f} MiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", nargs="+", required=True, help="HCP_PERF_LOG JSONL files")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=896)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--elem", type=int, default=2, help="bytes per element (bf16/fp16=2, fp32=4)")
    args = ap.parse_args()

    events = load_events(args.perf)
    if not events:
        print("No perf events found.")
        sys.exit(1)
    per_request, per_token, totals, layer_events = aggregate(events)

    # --- HCP measured wire bytes ---
    print("== HCP measured wire bytes (sum of per-domain wire_sent_bytes) ==")
    print(f"  prefill KV ring sent:      {totals['prefill_sent']} bytes  ({_mb(totals['prefill_sent'])})")
    print(f"  prefill KV ring recv:      {totals['prefill_recv']} bytes  ({_mb(totals['prefill_recv'])})")
    print(f"  mainline SD decode sent:   {totals['decode_sent']} bytes  ({_mb(totals['decode_sent'])})")
    print(f"  mainline SD decode recv:   {totals['decode_recv']} bytes  ({_mb(totals['decode_recv'])})")
    print(f"  continuation sent:         {totals['continuation_sent']} bytes")
    print(f"  legacy Q-ring decode sent: {totals['legacy_decode_sent']} bytes")
    hcp_total = (
        totals["prefill_sent"]
        + totals["decode_sent"]
        + totals["continuation_sent"]
        + totals["legacy_decode_sent"]
    )
    print(f"  TOTAL HCP wire bytes:      {hcp_total} bytes  ({_mb(hcp_total)})")

    # per-token decode ledger
    if per_token:
        sent_values = [v["sent"] for v in per_token.values()]
        print()
        print("== decode per-token wire bytes (mainline SD) ==")
        print(f"  tokens observed: {len(per_token)}")
        if sent_values:
            print(
                f"  sent per token: min={min(sent_values)} avg={sum(sent_values)/len(sent_values):.0f} "
                f"max={max(sent_values)} bytes"
            )

    # --- reference formulas (same caliber: actual bytes moved) ---
    # vLLM PD: full KV cache (2 tensors K,V) transferred once prefill->decode.
    #   KV bytes = layers * 2 * kv_heads * head_dim * seq_len * elem
    pd_kv = args.layers * 2 * args.kv_heads * args.head_dim * args.seq_len * args.elem
    # TP: per-layer all-reduce of activations [batch=1, seq, hidden].
    #   ring all-reduce moves 2*(N-1)/N * tensor_size per collective; here we
    #   report the per-layer activation size and the N=2 upper bound for context.
    tp_act_per_layer = args.seq_len * args.hidden * args.elem
    print()
    print("== reference (derived from model geometry, same 'actual bytes moved' caliber) ==")
    print(f"  vLLM PD full-KV transfer (one prefill->decode move): {pd_kv} bytes  ({_mb(pd_kv)})")
    print(f"  TP activation all-reduce per layer [seq x hidden]: {tp_act_per_layer} bytes  ({_mb(tp_act_per_layer)})")
    print(
        f"  TP all layers, N=2 ring all-reduce (2*(N-1)/N=1.0x): "
        f"{args.layers * tp_act_per_layer} bytes  ({_mb(args.layers * tp_act_per_layer)})"
    )
    if hcp_total and pd_kv:
        print(f"  HCP total / PD full-KV = {hcp_total / pd_kv:.3f}x")

    # per-request summary
    if per_request:
        print()
        print(f"== per-request wire-byte summary ({len(per_request)} requests) ==")
        for req, r in sorted(per_request.items()):
            if req is None:
                continue
            req_total = r["prefill_sent"] + r["decode_sent"] + r["continuation_sent"]
            print(
                f"  request {req}: prefill={_mb(r['prefill_sent'])} decode={_mb(r['decode_sent'])} "
                f"total={_mb(req_total)}"
            )


if __name__ == "__main__":
    main()
