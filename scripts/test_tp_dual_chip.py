#!/usr/bin/env python3
"""TP=2 正确性回归：TensorParallelQwen2 vs 同机单设备 HF 参考前向。

用法：
    python scripts/test_tp_dual_chip.py \
        --model-dir models/Qwen2-0.5B --backend gloo --device cpu \
        [--prompt-file config/test_prompts.txt | --seq-len 128] \
        [--master-port 29511]

硬门槛：全位置 argmax 100% 一致（不一致则非零退出）；max|Δlogit| 数值报告。
"""

import argparse
import os
import sys

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _worker(rank: int, args: argparse.Namespace, input_ids: list):
    from hcp_tp_engine import TensorParallelQwen2

    engine = TensorParallelQwen2(
        args.model_dir,
        device=args.device,
        rank=rank,
        world_size=2,
        backend=args.backend,
        init_method=f"tcp://127.0.0.1:{args.master_port}",
    )
    ids = torch.tensor([input_ids], dtype=torch.long, device=engine.device)
    logits = engine.forward_logits(ids)

    if rank == 0:
        from transformers import AutoModelForCausalLM

        ref = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.float32, trust_remote_code=True
        ).to(engine.device)
        ref.eval()
        with torch.no_grad():
            ref_logits = ref(ids).logits.to(torch.float32)
        tp = logits.to(torch.float32)
        am_tp = tp.argmax(-1)
        am_ref = ref_logits.argmax(-1)
        total = am_ref.numel()
        agree = int((am_tp == am_ref).sum().item())
        maxdiff = float((tp - ref_logits).abs().max().item())
        print(
            f"[RESULT] backend={args.backend} device={args.device} "
            f"seq_len={len(input_ids)} argmax_agree={agree}/{total} "
            f"max_abs_diff={maxdiff:.6e}",
            flush=True,
        )
        engine.close()
        if agree != total:
            raise SystemExit(f"argmax mismatch: {agree}/{total}")
        if args.max_diff_tol > 0 and maxdiff > args.max_diff_tol:
            raise SystemExit(f"max_abs_diff {maxdiff:.3e} > tol {args.max_diff_tol:.3e}")
    else:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description="TP=2 forward correctness regression")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--backend", default="gloo", choices=["gloo", "hccl"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    parser.add_argument("--prompt-file", default=None,
                        help="文本文件，取第一个非空行做 prompt")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="无 prompt-file 时生成确定性 token 序列的长度；"
                             "有 prompt-file 时作为截断上限")
    parser.add_argument("--master-port", type=int, default=29511)
    parser.add_argument("--max-diff-tol", type=float, default=0.0,
                        help=">0 时作为 max_abs_diff 的硬门槛")
    args = parser.parse_args()

    if args.prompt_file:
        from transformers import AutoTokenizer

        with open(args.prompt_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        prompt = lines[0] if lines else "Hello"
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        input_ids = tok(prompt)["input_ids"]
        if len(input_ids) > args.seq_len:
            input_ids = input_ids[: args.seq_len]
    else:
        # 确定性合成序列（合法 token id 范围内）
        input_ids = [(i * 7919 + 13) % 30000 + 1000 for i in range(args.seq_len)]

    mp.spawn(_worker, args=(args, input_ids), nprocs=2, join=True)


if __name__ == "__main__":
    main()
