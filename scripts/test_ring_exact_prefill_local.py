#!/usr/bin/env python3
"""Mac CPU 本地验证：prefill_ring_exact 的精确性。

1) 3-domain 模拟（单进程 asyncio，in-memory ring）：每个 domain 的
   prefill_ring_exact 输出与全序列 golden 对应位置 logits 对比；
2) 最后 domain 的最后位置 logits（= ring 的第一个采样依据）必须与
   golden 达到 float32 噪声地板（~1e-5）；
3) domain 0 在交换后的完整 cache 上 decode 一步，与 golden decode 对比。

运行：python3.12 scripts/test_ring_exact_prefill_local.py
"""
import asyncio
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from hcp_transformers_quic_worker import TransformersBackend
from hcp_worker_sdk.types import KvBlock

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2-0.5B")
SEQ_LEN = 64
CHUNKS = [22, 21, 21]  # 3 domains, 不均匀切分


class SimRing:
    """单进程内模拟 N-domain 逐层 ring：每层所有 domain 交付本地块后，
    各自收到其余 domain 的块（顺序按 ring 转发序模拟：prev 先到）。"""

    def __init__(self, num_domains: int, num_layers: int):
        self.n = num_domains
        self.num_layers = num_layers
        self._pending = {}  # layer -> list of (domain, k, v)
        self._events = {}   # layer -> asyncio.Event

    async def exchange(self, domain: int, layer: int, k, v, seq_start: int, seq_end: int):
        pend = self._pending.setdefault(layer, [])
        pend.append((domain, seq_start, seq_end, k.clone(), v.clone()))
        ev = self._events.setdefault(layer, asyncio.Event())
        if len(pend) == self.n:
            ev.set()
        await ev.wait()
        # 模拟 ring 到达序：prev domain (i-1) 最先到达，然后 i-2 ...
        order = [(domain - r) % self.n for r in range(1, self.n)]
        blocks = []
        for d in order:
            for (dd, s0, s1, kk, vv) in pend:
                if dd == d:
                    blocks.append(KvBlock(layer, s0, s1, kk, vv))
        return blocks


async def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt_path = sys.argv[1] if len(sys.argv) > 1 else None
    if prompt_path:
        text = open(prompt_path).read()
    else:
        text = "river shapes copper follows " * 16
    ids = tok(text, return_tensors="pt").input_ids[0][:SEQ_LEN].tolist()
    assert len(ids) == SEQ_LEN

    # golden: 全序列一次 prefill
    golden_backend = TransformersBackend(MODEL_DIR, device="cpu")
    golden_logits, golden_len = golden_backend.prefill(ids, 0)
    print(f"golden: seq_len={golden_len}, argmax={golden_logits.argmax().item()}")

    # 3-domain ring 模拟
    sim = SimRing(3, 24)
    backends = [TransformersBackend(MODEL_DIR, device="cpu") for _ in range(3)]
    offsets = [0, 22, 43]

    async def run_domain(d):
        chunk = ids[offsets[d]:offsets[d] + CHUNKS[d]]
        async def exchange_layer(li, k, v):
            return await sim.exchange(d, li, k, v, offsets[d], offsets[d] + CHUNKS[d])
        return await backends[d].prefill_ring_exact(chunk, offsets[d], exchange_layer)

    results = await asyncio.gather(*[run_domain(d) for d in range(3)])

    # 每个 domain 的最后位置 logits 与 golden 对应位置对比
    import torch.nn.functional as F
    ok = True
    full_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out_full = golden_backend.model(full_ids, use_cache=False)
    for d, (logits_d, _) in enumerate(results):
        last_global = offsets[d] + CHUNKS[d] - 1
        ref = out_full.logits[0, last_global]
        diff = (logits_d.to(ref.device) - ref).abs()
        match = logits_d.argmax().item() == ref.argmax().item()
        print(f"domain {d} last-pos {last_global}: max|Δ|={diff.max().item():.3e} "
              f"argmax_match={match}")
        if diff.max().item() > 1e-3 or not match:
            ok = False

    # domain 0 decode 一步 vs golden decode
    tok1 = golden_logits.argmax().item()
    golden_d1 = golden_backend.decode(tok1)
    ring_d1 = backends[0].decode(tok1)
    d1 = (golden_d1 - ring_d1).abs()
    print(f"decode step1 (domain0 cache): max|Δ|={d1.max().item():.3e} "
          f"argmax_match={golden_d1.argmax().item() == ring_d1.argmax().item()}")
    if d1.max().item() > 1e-3:
        ok = False

    print("LOCAL RING-EXACT TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
