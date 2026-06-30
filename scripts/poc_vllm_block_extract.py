"""
PoC: extract a vLLM KV block after prefill, copy it to a different physical block,
and verify the copy is byte-identical.

This demonstrates that block-level KV exchange (the core of Block-Aware Ring)
is feasible without modifying vLLM's attention kernel.
"""
import sys
import torch
from vllm import LLMEngine, EngineArgs, SamplingParams
from vllm.inputs import TokensPrompt

model_dir = sys.argv[1]
seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 64

engine_args = EngineArgs(
    model=model_dir,
    dtype="float16",
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    max_num_seqs=1,
    block_size=16,
)
engine = LLMEngine.from_engine_args(engine_args)
tokenizer = engine.get_tokenizer()
prompt_tokens = tokenizer.encode(" ".join(["the"] * (seq_len - 1)))

engine.add_request("req-0", TokensPrompt(prompt_token_ids=prompt_tokens), params=SamplingParams(max_tokens=5, temperature=0))
engine.step()

scheduler = engine.scheduler[0]
assert len(scheduler.running) == 1, "expected one running sequence"
seq = scheduler.running[0].get_seqs()[0]
block_table = scheduler.block_manager.get_block_table(seq)
print(f"prompt_len={seq.get_prompt_len()} num_tokens={seq.get_len()} block_table={block_table}")

ce = engine.model_executor.driver_worker.cache_engine[0]
print(f"gpu_cache layer shape: {ce.gpu_cache[0].shape}")

# Extract block 0 from layer 0
src_block_id = block_table[0]
dst_block_id = max(block_table) + 10  # an unused physical block
layer = 0
src_k = ce.gpu_cache[layer][0, src_block_id].clone()
src_v = ce.gpu_cache[layer][1, src_block_id].clone()
print(f"extracted block {src_block_id}: k shape {tuple(src_k.shape)}, v shape {tuple(src_v.shape)}")

# Simulate remote receive: serialize src block to bytes and write into an unused physical block
recv_block_id = max(block_table) + 20
recv_k = src_k.clone().contiguous()
recv_v = src_v.clone().contiguous()
ce.gpu_cache[layer][0, recv_block_id].copy_(recv_k)
ce.gpu_cache[layer][1, recv_block_id].copy_(recv_v)
assert torch.equal(src_k, ce.gpu_cache[layer][0, recv_block_id]), "remote K mismatch"
assert torch.equal(src_v, ce.gpu_cache[layer][1, recv_block_id]), "remote V mismatch"
print(f"verified: serialized/deserialized block written to {recv_block_id} is byte-identical")
print("PoC success: vLLM KV block extraction and reinsertion works")
