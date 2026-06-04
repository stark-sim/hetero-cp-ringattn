#!/usr/bin/env python3
"""Export Python transformers logits for comparison with Rust.

Usage:
    python export_python_logits.py --model-dir MODEL_DIR --prompt "PROMPT" --output OUTPUT_DIR
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_logits(model_dir: str, prompt: str, max_tokens: int, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[python] device: {device}")

    # Load model in BF16 (native dtype from config)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    all_logits = []
    generated_ids = []

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # Last token logits
        last_logits = logits[0, -1, :]

        for step in range(max_tokens):
            all_logits.append(last_logits.float().cpu().numpy())

            next_token_id = int(torch.argmax(last_logits))
            generated_ids.append(next_token_id)

            if next_token_id == model.config.eos_token_id:
                break

            # Decode step
            outputs = model(
                torch.tensor([[next_token_id]], device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            last_logits = logits[0, -1, :]

    # Write binary format: [vocab_size: u64 LE][num_chunks: u64 LE][f32 data...]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "logits.bin"

    vocab_size = all_logits[0].shape[0]
    num_chunks = len(all_logits)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", vocab_size))
        f.write(struct.pack("<Q", num_chunks))
        for chunk in all_logits:
            f.write(chunk.astype("float32").tobytes())

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(
        input_ids[0].tolist() + generated_ids, skip_special_tokens=True
    )

    print(f"[python] generated {len(generated_ids)} tokens")
    print(f"[python] generated text: '{generated_text}'")
    print(f"[python] full text: '{full_text}'")
    print(f"[python] logits exported to {out_path}")

    return generated_text, full_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    export_logits(args.model_dir, args.prompt, args.max_tokens, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
