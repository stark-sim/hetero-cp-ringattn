import sys
from vllm import LLMEngine, EngineArgs, SamplingParams

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
    device="cuda",
)
engine = LLMEngine.from_engine_args(engine_args)

tokenizer = engine.get_tokenizer()
prompt_tokens = tokenizer.encode(" ".join(["the"] * (seq_len - 1)))
print("prompt len:", len(prompt_tokens))

from vllm.inputs import TokensPrompt
engine.add_request("req-0", TokensPrompt(prompt_token_ids=prompt_tokens), params=SamplingParams(max_tokens=5, temperature=0))

# step once to run prefill
engine.step()

# inspect running sequence
for i, scheduler in enumerate(engine.scheduler):
    print(f"scheduler {i}: running={len(scheduler.running)} waiting={len(scheduler.waiting)} swapped={len(scheduler.swapped)}")
    for sg in scheduler.running:
        for seq in sg.get_seqs():
            print("seq id:", seq.seq_id)
            print("block table:", scheduler.block_manager.get_block_table(seq))
            print("num tokens:", seq.get_len())

# inspect cache engine
worker = engine.model_executor.driver_worker
ce = worker.cache_engine[0]
print("block_size:", ce.block_size)
print("num_gpu_blocks:", ce.num_gpu_blocks)
print("num_attention_layers:", ce.num_attention_layers)
print("num_kv_heads:", ce.num_kv_heads)
print("head_size:", ce.head_size)
print("gpu_cache layer shape:", ce.gpu_cache[0].shape)
