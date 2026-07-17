#!/usr/bin/env python3
"""
CP plugin consumer: loads the producer's chunk KV over HTTP as external
prefix and computes its own chunk; compares its output to a single-node
reference.  Runs on the node holding the *later* chunk.

Usage:
    python cp_consumer.py --model-dir M --full-file ids.txt --prefix-len 32 \
        --peer-url http://WHITE:8899 --shared-path /tmp/hcp_cp_x --run-id r1
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--full-file", required=True, help="text file with space-separated token ids for the FULL prompt")
    p.add_argument("--prefix-len", type=int, required=True)
    p.add_argument("--prefix-chunk-ids", default="chunk0")
    p.add_argument("--peer-url", required=True)
    p.add_argument("--shared-path", default="/tmp/hcp_cp")
    p.add_argument("--run-id", default="run")
    p.add_argument("--chunk-id", default="chunk1")
    p.add_argument("--gpu-mem", type=float, default=0.4)
    p.add_argument("--decode", type=int, default=1)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    ids = [int(x) for x in open(args.full_file).read().split()]
    print(f"[consumer] full prompt {len(ids)} tokens, prefix {args.prefix_len}, peer={args.peer_url}")

    tok = AutoTokenizer.from_pretrained(args.model_dir)

    # Reference: single-node full prefill in a plain LLM.
    ref = LLM(model=args.model_dir, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=1, max_model_len=4096)
    ref_out = ref.generate([TokensPrompt(prompt_token_ids=ids)],
                           SamplingParams(temperature=0, max_tokens=args.decode))
    ref_ids = list(ref_out[0].outputs[0].token_ids)
    print(f"[ref] tokens: {ref_ids} text={ref_out[0].outputs[0].text!r}")
    try:
        ref.llm_engine.shutdown()
    except Exception:
        pass
    del ref
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg = {
        "kv_connector": "HcpCpConnector",
        "kv_role": "kv_consumer",
        "kv_connector_module_path": "hcp_vllm_plugin.connector",
        "kv_connector_extra_config": {
            "cp_role": "consumer",
            "cp_chunk_id": args.chunk_id,
            "cp_prefix_chunk_ids": args.prefix_chunk_ids,
            "cp_prefix_len": args.prefix_len,
            "cp_shared_path": args.shared_path,
            "cp_run_id": args.run_id,
            "cp_peer_url": args.peer_url,
        },
    }
    cons = LLM(model=args.model_dir, dtype="float16", enforce_eager=True,
               gpu_memory_utilization=args.gpu_mem, max_num_seqs=1, max_model_len=4096,
               disable_hybrid_kv_cache_manager=True,
               kv_transfer_config=cfg)
    out = cons.generate([TokensPrompt(prompt_token_ids=ids)],
                        SamplingParams(temperature=0, max_tokens=args.decode))
    cons_ids = list(out[0].outputs[0].token_ids)
    print(f"[consumer] tokens: {cons_ids} text={out[0].outputs[0].text!r}")

    match = cons_ids == ref_ids
    print(f"\n[result] consumer tokens match reference: {match} (ref={ref_ids} cons={cons_ids})")
    raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    main()
