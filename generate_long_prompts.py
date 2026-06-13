#!/usr/bin/env python3
"""Generate long prompts by repeating a seed text. Token counts are approximate."""
import argparse
from tokenizers import Tokenizer

SEED = "The quick brown fox jumps over the lazy dog. "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lengths", default="4096,8192,16384,32768")
    parser.add_argument("--output", default="/tmp/long_prompts.txt")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    lengths = [int(x) for x in args.lengths.split(",")]
    seed_ids = tokenizer.encode(SEED).ids
    seed_len = len(seed_ids)

    with open(args.output, "w", encoding="utf-8") as f:
        for length in lengths:
            repeats = (length + seed_len - 1) // seed_len
            ids = (seed_ids * repeats)[:length]
            text = tokenizer.decode(ids, skip_special_tokens=False).replace("\n", " ")
            actual = len(tokenizer.encode(text).ids)
            f.write(text + "\n")
            print(f"target={length:>6} actual={actual:>6} diff={actual-length:>6}")


if __name__ == "__main__":
    main()
