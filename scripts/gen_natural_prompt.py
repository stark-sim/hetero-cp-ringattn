#!/usr/bin/env python3
"""Generate a deterministic NON-periodic prompt of exactly N tokens.

Unlike rust/bin/gen_prompt (which repeats one sentence and produces a
strongly periodic token stream), this builds a pseudo-random word sequence
from a fixed seed. Non-periodicity is the point: periodic prompts can mask
position-encoding bugs (RoPE offsets) because argmax margins stay huge.

Usage: gen_natural_prompt.py <tokenizer.json> <num_tokens> <seed> <out_text_file>
Output: prompt text (re-tokenization by the coordinator is shared by all
phases of a run, so gate consistency comes from the coordinator side).
Prints the actual encoded token count (may be >= num_tokens; the text is
truncated word-wise to stay close).
"""
import random
import sys

from tokenizers import Tokenizer

WORDS = (
    "science river market cloud engine stone forest harbor window signal "
    "copper meadow theory lantern circuit valley copper orbit fabric beacon "
    "harvest crystal tunnel pigment glacier trumpet ledger compass mosaic "
    "anchor velvet piston quarrel nectar siren turbine canvas ember fulcrum"
).split()

CONNECTORS = (
    "meets shapes follows crosses bends carries questions builds measures "
    "shatters gathers weaves drifts beyond beneath toward against through"
).split()


def main() -> None:
    tokenizer_path, num_tokens, seed, out_path = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
    )
    tok = Tokenizer.from_file(tokenizer_path)

    rng = random.Random(seed)
    text_parts = []
    ids: list[int] = []
    # Grow the text until the encoding reaches the target length.
    while True:
        text_parts.append(rng.choice(WORDS))
        text_parts.append(rng.choice(CONNECTORS))
        text = " ".join(text_parts)
        ids = tok.encode(text, add_special_tokens=True).ids
        if len(ids) >= num_tokens:
            break

    # Periodicity self-check on the encoded ids: no period < len/4 may
    # reproduce the stream (a periodic prompt can mask RoPE position bugs).
    n = len(ids)
    for period in range(1, max(2, n // 4)):
        if all(ids[i] == ids[i % period] for i in range(n)):
            raise SystemExit(f"prompt is periodic with period {period}; pick another seed")

    with open(out_path, "w") as f:
        f.write(text)
    print(f"generated non-periodic prompt: {n} tokens (target>={num_tokens}, seed={seed}) -> {out_path}")


if __name__ == "__main__":
    main()
