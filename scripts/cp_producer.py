#!/usr/bin/env python3
"""
CP plugin producer: computes its chunk and serves its KV over HTTP for the
consumer instance to pull.  Runs on the node holding the *earlier* chunk.

Usage:
    python cp_producer.py --model-dir M --chunk-file ids.txt \
        --serve-port 8899 --shared-path /tmp/hcp_cp_x --run-id r1 \
        --hold-secs 600
"""
import argparse
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--chunk-file", required=True, help="text file with space-separated token ids for THIS chunk")
    p.add_argument("--serve-port", type=int, default=8899)
    p.add_argument("--shared-path", default="/tmp/hcp_cp")
    p.add_argument("--run-id", default="run")
    p.add_argument("--chunk-id", default="chunk0")
    p.add_argument("--gpu-mem", type=float, default=0.4)
    p.add_argument("--hold-secs", type=int, default=600)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = [int(x) for x in open(args.chunk_file).read().split()]
    print(f"[producer] chunk has {len(ids)} tokens, serving on :{args.serve_port}")

    cfg = {
        "kv_connector": "HcpCpConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "hcp_vllm_plugin.connector",
        "kv_connector_extra_config": {
            "cp_role": "producer",
            "cp_chunk_id": args.chunk_id,
            "cp_shared_path": args.shared_path,
            "cp_run_id": args.run_id,
            "cp_serve_port": args.serve_port,
        },
    }
    llm = LLM(
        model=args.model_dir, dtype="float16", enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem, max_num_seqs=1, max_model_len=4096,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config=cfg,
    )
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0, max_tokens=1))
    print(f"[producer] prefilled chunk, sample token {out[0].outputs[0].token_ids[0]}; KV served on :{args.serve_port}")
    print(f"[producer] holding for {args.hold_secs}s ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < args.hold_secs:
        time.sleep(5)


if __name__ == "__main__":
    main()
