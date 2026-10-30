#!/usr/bin/env python3
"""HCP 侧 KV 导出（PD 闭环 producer）：prefill + golden decode + KV dump。

产物 layout（hcp-kv-v1）：
    <out_dir>/meta.json:        format/prompt_token_ids/seq_len/num_layers/
                                num_kv_heads/head_dim/dtype/golden_token_ids/
                                golden_text
    <out_dir>/layer_<i>.safetensors: {"kv": [seq_len, 2, num_kv_heads, head_dim]}
                                    （[:,0]=K, [:,1]=V，post-RoPE，模型 dtype）

用法：
    python hcp_pd_export.py --model-dir <7B> --prompt-file prompt.txt \
        --out-dir /tmp/hcp_pd_kv --max-new-tokens 16 --device cuda --dtype bfloat16
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hcp_transformers_quic_worker import TransformersBackend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-file", required=True, help="纯文本 prompt 文件")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    backend = TransformersBackend(args.model_dir, device=args.device, dtype=args.dtype)
    with open(args.prompt_file) as f:
        prompt = f.read().strip()
    token_ids = backend.tokenizer(prompt)["input_ids"]
    seq_len = len(token_ids)
    print(f"[pd-export] prompt tokens: {seq_len}", flush=True)

    # 1-domain prefill：_past_key_values 缓存全量 KV（HCP 精确 CP 性质）
    logits, total = backend.prefill(token_ids, 0)
    assert total == seq_len

    # 先导出 KV（decode 会往 cache 追加，必须在 golden decode 之前）；
    # post-RoPE，与 vLLM paged cache 语义一致
    from safetensors.torch import save_file

    cfg = backend.model.config
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    for li in range(backend.num_layers):
        k, v = backend._get_kv_layer(li)  # [1, Hkv, S, D]
        kv = torch.stack(
            [k[0].permute(1, 0, 2), v[0].permute(1, 0, 2)], dim=1
        ).contiguous()  # [S, 2, Hkv, D]
        assert kv.shape == (seq_len, 2, num_kv_heads, head_dim)
        save_file({"kv": kv.cpu()}, os.path.join(args.out_dir, f"layer_{li}.safetensors"))

    # golden decode（greedy）
    golden_ids = []
    for _ in range(args.max_new_tokens):
        nxt = int(torch.argmax(logits).item())
        golden_ids.append(nxt)
        logits = backend.decode(nxt)
    golden_text = backend.tokenizer.decode(golden_ids)
    print(f"[pd-export] golden: {golden_text!r}", flush=True)

    meta = {
        "format": "hcp-kv-v1",
        "model_dir": os.path.abspath(args.model_dir),
        "prompt_token_ids": token_ids,
        "seq_len": seq_len,
        "num_layers": backend.num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "golden_token_ids": golden_ids,
        "golden_text": golden_text,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    size_mb = sum(
        os.path.getsize(os.path.join(args.out_dir, fn))
        for fn in os.listdir(args.out_dir)
    ) / 1e6
    print(f"[pd-export] wrote {backend.num_layers} layers to {args.out_dir} "
          f"({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
