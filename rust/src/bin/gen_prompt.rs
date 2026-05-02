//! Generate a prompt with exactly N tokens for long-context validation.
//!
//! Replaces scripts/generate_long_prompt.py with a pure-Rust implementation
//! using the `tokenizers` crate (already a project dependency).
//!
//! Optimized to avoid O(n) re-encoding loops: measures the per-unit token
//! delta when units are concatenated, then computes the exact repeat count
//! in one shot.
//!
//! Usage:
//!     cargo run --bin gen_prompt -- \
//!         /path/to/tokenizer.json 4096 /tmp/prompt_4k.txt
//!
//! The unit text defaults to "The quick brown fox jumps over the lazy dog. ".
//! Use --unit "your text " to override.

use std::env;

const DEFAULT_UNIT: &str = "The quick brown fox jumps over the lazy dog. ";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("Usage: gen_prompt <tokenizer.json> <target_tokens> <output_file> [--unit \"text\"]");
        std::process::exit(1);
    }

    let tokenizer_path = &args[0];
    let target_tokens: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("ERROR: target_tokens must be a positive integer");
        std::process::exit(1);
    });
    let output_file = &args[2];

    let unit = args.windows(2)
        .find(|w| w[0] == "--unit")
        .map(|w| w[1].as_str())
        .unwrap_or(DEFAULT_UNIT);

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: failed to load tokenizer from {}: {}", tokenizer_path, e);
            std::process::exit(1);
        });

    // Measure unit length and cross-unit merge delta
    let unit_enc = tokenizer.encode(unit, false)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: failed to encode unit text: {}", e);
            std::process::exit(1);
        });
    let unit_ids = unit_enc.get_ids();
    let unit_len = unit_ids.len();
    if unit_len == 0 {
        eprintln!("ERROR: unit text tokenizes to 0 tokens");
        std::process::exit(1);
    }

    let two_units = unit.repeat(2);
    let two_enc = tokenizer.encode(&*two_units, false)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: failed to encode 2-unit text: {}", e);
            std::process::exit(1);
        });
    let two_len = two_enc.get_ids().len();
    let delta = two_len.saturating_sub(unit_len); // tokens added per extra unit
    if delta == 0 {
        eprintln!("ERROR: token delta between units is 0; cannot compute repeat count");
        std::process::exit(1);
    }

    eprintln!("Token unit '{}' -> {} tokens (delta per repeat: +{})",
              unit.trim(), unit_len, delta);

    // Compute repeat count: 1 unit + (n-1) extra units
    // unit_len + (n-1) * delta >= target  =>  n >= 1 + ceil((target - unit_len) / delta)
    let n_repeat = if target_tokens <= unit_len {
        1
    } else {
        1 + (target_tokens - unit_len + delta - 1) / delta
    };

    // Build large text and encode once
    let big_text = unit.repeat(n_repeat);
    let big_enc = tokenizer.encode(&*big_text, false)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: failed to encode repeated text: {}", e);
            std::process::exit(1);
        });
    let big_ids = big_enc.get_ids();

    if big_ids.len() < target_tokens {
        eprintln!("ERROR: computed repeat count insufficient: got {} tokens, need {}",
                  big_ids.len(), target_tokens);
        std::process::exit(1);
    }

    // Truncate to exact target and decode
    let truncated = &big_ids[..target_tokens];
    let prompt = tokenizer.decode(truncated, true)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: failed to decode tokens: {}", e);
            std::process::exit(1);
        });

    // Verify by re-encoding the decoded text
    let verify_enc = tokenizer.encode(&*prompt, false)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: failed to verify re-encode: {}", e);
            std::process::exit(1);
        });
    let actual_len = verify_enc.get_ids().len();

    // Write output
    std::fs::write(output_file, &prompt)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: failed to write {}: {}", output_file, e);
            std::process::exit(1);
        });

    eprintln!(
        "Generated {} tokens -> {} ({} chars)",
        actual_len, output_file, prompt.len()
    );

    if actual_len != target_tokens {
        eprintln!(
            "WARNING: requested {} tokens but got {}",
            target_tokens, actual_len
        );
        std::process::exit(1);
    }
}
