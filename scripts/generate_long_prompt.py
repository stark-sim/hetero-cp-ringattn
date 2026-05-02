#!/usr/bin/env python3
"""Generate a prompt with exactly N tokens for long-context validation.

Uses the model's tokenizer to precisely control token count.
Example:
    python3 generate_long_prompt.py /Users/stark_sim/models/qwen2-0.5b 4096 /tmp/prompt_4k.txt
"""

import argparse
import sys


def generate_prompt(model_dir: str, target_tokens: int, output_file: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Use a repeating English sentence that tokenizes efficiently.
    unit = "The quick brown fox jumps over the lazy dog. "
    unit_ids = tokenizer.encode(unit, add_special_tokens=False)
    unit_len = len(unit_ids)
    print(f"Token unit '{unit.strip()}' -> {unit_len} tokens")

    # Iteratively build text until re-encoded length >= target.
    # Decoding can merge tokens, so we must verify by re-encoding.
    n_repeat = target_tokens // unit_len + 2
    while True:
        text = unit * n_repeat
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens:
            break
        n_repeat += 1

    # Truncate to exact target and decode
    ids = ids[:target_tokens]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)

    # Verify
    verify_ids = tokenizer.encode(prompt, add_special_tokens=False)
    actual_len = len(verify_ids)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Generated {actual_len} tokens -> {output_file} ({len(prompt)} chars)")
    return actual_len


def main():
    parser = argparse.ArgumentParser(description="Generate a precise-N-token prompt")
    parser.add_argument("model_dir", help="Path to HF model dir (contains tokenizer.json)")
    parser.add_argument("target_tokens", type=int, help="Exact number of tokens desired")
    parser.add_argument("output_file", help="Output text file path")
    args = parser.parse_args()

    actual = generate_prompt(args.model_dir, args.target_tokens, args.output_file)
    if actual != args.target_tokens:
        print(f"WARNING: requested {args.target_tokens} tokens but got {actual}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
