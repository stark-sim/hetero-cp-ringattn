#!/usr/bin/env python3
"""
Single-machine 2-instance PoC for the HcpCpConnector (vLLM V1 KV connector).

Runs three vLLM instances sequentially on one GPU:
1. reference: plain vLLM on the full prompt -> reference next token.
2. producer: HcpCpConnector (cp_role=producer) prefills chunk 0 and stores KV.
3. consumer: HcpCpConnector (cp_role=consumer) prefills chunk0+chunk1, loading
   the chunk-0 KV as external prefix, and only computes chunk 1.

The consumer's first generated token must match the reference next token.

Usage:
    python scripts/poc_hcp_cp_connector.py --model-dir /path/to/model
"""

import argparse
import gc
import os
import sys

import torch


def make_llm(model_dir, gpu_mem, kv_transfer_config=None):
    from vllm import LLM
    llm = LLM(
        model=model_dir,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=1,
        max_model_len=4096,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config=kv_transfer_config,
    )
    return llm


def free_llm(llm):
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--chunk-a", type=int, default=32)
    parser.add_argument("--gpu-mem", type=float, default=0.3)
    parser.add_argument("--run-id", default="poc")
    parser.add_argument("--http-port", type=int, default=0,
                        help="if >0, producer serves KV over HTTP on this port and consumer pulls via http://127.0.0.1:port")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    text = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
            "lima mike november oscar papa quebec romeo sierra tango uniform "
            "victor whiskey xray yankee zulu apple banana cherry dragon eagle "
            "falcon grape hotel igloo jungle koala lemon mango ninja orange panda qu")
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 64, f"expected 64 tokens, got {len(ids)}"
    chunk_a = ids[: args.chunk_a]
    chunk_b = ids[args.chunk_a:]
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(temperature=0, max_tokens=1)

    # 1. reference
    print("[ref] single-node full prefill ...")
    ref = make_llm(args.model_dir, args.gpu_mem)
    out = ref.generate([TokensPrompt(prompt_token_ids=ids)], sp)
    ref_token = out[0].outputs[0].token_ids[0]
    print(f"[ref] next token: {ref_token} ({tok.decode([ref_token])!r})")
    free_llm(ref)

    shared_path = f"/tmp/hcp_cp_conn_{args.run_id}"
    import shutil
    shutil.rmtree(shared_path, ignore_errors=True)

    # 2. producer: prefill chunk A, store KV
    print("[producer] prefill chunk A and store KV ...")
    prod_cfg = {
        "kv_connector": "HcpCpConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "hcp_vllm_plugin.connector",
        "kv_connector_extra_config": {
            "cp_role": "producer",
            "cp_chunk_id": "chunk0",
            "cp_shared_path": shared_path,
            "cp_run_id": args.run_id,
            "cp_serve_port": args.http_port,
        },
    }
    prod = make_llm(args.model_dir, args.gpu_mem, kv_transfer_config=prod_cfg)
    out = prod.generate([TokensPrompt(prompt_token_ids=chunk_a)], sp)
    prod_token = out[0].outputs[0].token_ids[0]
    print(f"[producer] chunk A next token (unused): {prod_token}")
    # Keep producer alive so its HTTP KV server stays up for the consumer.

    # 3. consumer: load chunk A KV, prefill chunk B with context
    print("[consumer] load chunk A KV, prefill chunk B with context ...")
    peer_url = f"http://127.0.0.1:{args.http_port}" if args.http_port else ""
    cons_cfg = {
        "kv_connector": "HcpCpConnector",
        "kv_role": "kv_consumer",
        "kv_connector_module_path": "hcp_vllm_plugin.connector",
        "kv_connector_extra_config": {
            "cp_role": "consumer",
            "cp_chunk_id": "chunk1",
            "cp_prefix_chunk_ids": "chunk0",
            "cp_prefix_len": len(chunk_a),
            "cp_shared_path": shared_path,
            "cp_run_id": args.run_id,
            "cp_peer_url": peer_url,
        },
    }
    cons = make_llm(args.model_dir, args.gpu_mem, kv_transfer_config=cons_cfg)
    out = cons.generate([TokensPrompt(prompt_token_ids=ids)], sp)
    cons_token = out[0].outputs[0].token_ids[0]
    print(f"[consumer] next token: {cons_token} ({tok.decode([cons_token])!r})")
    free_llm(cons)
    free_llm(prod)

    match = cons_token == ref_token
    print(f"\n[result] consumer token matches reference: {match}")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
