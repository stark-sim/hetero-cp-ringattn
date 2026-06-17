#!/usr/bin/env python3
"""验证 VllmBackend 在 NPU 上的 prefill/decode 和 capacity 上报。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from hcp_vllm_worker import VllmBackend

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2-0.5B")


def main():
    backend = VllmBackend(model_dir=MODEL_DIR)
    print(f"device={backend.device}, capacity_mb={backend.capacity_mb}")
    print(f"num_layers={backend.num_layers}, num_heads={backend.num_heads}, head_dim={backend.head_dim}")

    # 用模型自带 tokenizer 编码一个 prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    chunk = tokenizer.encode("The future of artificial intelligence is", add_special_tokens=True)
    print(f"prompt tokens: {chunk}")

    logits, seq_offset = backend.prefill(chunk, seq_offset=0)
    print(f"prefill logits shape={logits.shape}, dtype={logits.dtype}, seq_offset={seq_offset}")
    next_token = int(logits.argmax(dim=-1).item())
    print(f"prefill next_token={next_token} -> '{tokenizer.decode([next_token])}'")

    logits2 = backend.decode(next_token)
    next_token2 = int(logits2.argmax(dim=-1).item())
    print(f"decode next_token={next_token2} -> '{tokenizer.decode([next_token2])}'")

    del backend
    import gc
    gc.collect()
    print("VllmBackend NPU test passed")


if __name__ == "__main__":
    main()
