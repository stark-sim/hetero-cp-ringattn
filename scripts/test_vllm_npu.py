#!/usr/bin/env python3
"""
Phase 0: 验证 vllm-ascend 单节点在 NPU 上可用。
"""

import os
import sys
import time

import torch
import torch_npu
from vllm import LLM, SamplingParams


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2-0.5B"

    print(f"[test] torch version: {torch.__version__}")
    print(f"[test] torch_npu available: {torch_npu.npu.is_available()}")
    print(f"[test] npu count: {torch_npu.npu.device_count()}")
    print(f"[test] model dir: {model_dir}")

    print("[test] loading model with vLLM ...")
    start = time.time()
    llm = LLM(
        model=model_dir,
        dtype="float16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        # 当前 NPU 只剩约 7GB 可用显存，保守设置
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=1,
    )
    print(f"[test] model loaded in {time.time() - start:.1f}s")

    prompts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
    ]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    print("[test] running generation ...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"[test] generation done in {time.time() - start:.1f}s")

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"[test] prompt {i}: {prompt!r}")
        print(f"[test] generated {i}: {generated!r}")

    print("[test] PASS")


if __name__ == "__main__":
    main()
