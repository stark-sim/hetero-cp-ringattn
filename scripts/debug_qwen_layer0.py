#!/usr/bin/env python3
"""
Debug script: compare Qwen2-0.5B layer 0 internals between Python and Rust.
"""

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/Users/stark_sim/models/qwen2-0.5b"
PROMPT = "Hello, how are you?"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]

layer0 = model.model.layers[0]

with torch.no_grad():
    hidden = model.model.embed_tokens(input_ids)
    print(f"embed first 10: {hidden[0, 0, :10].tolist()}")
    
    normed = layer0.input_layernorm(hidden)
    print(f"attn_norm first 10: {normed[0, 0, :10].tolist()}")
    
    q = layer0.self_attn.q_proj(normed)
    k = layer0.self_attn.k_proj(normed)
    v = layer0.self_attn.v_proj(normed)
    print(f"q_proj first 10: {q[0, 0, :10].tolist()}")
    print(f"k_proj first 10: {k[0, 0, :10].tolist()}")
    print(f"v_proj first 10: {v[0, 0, :10].tolist()}")
    
    # Reshape
    batch, seq_len, _ = q.shape
    num_heads = layer0.self_attn.config.num_attention_heads
    num_kv_heads = layer0.self_attn.config.num_key_value_heads
    head_dim = layer0.self_attn.head_dim
    
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # RoPE
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(v, position_ids)
    # apply_rotary_pos_emb
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos_pos = cos  # already [1, 1, seq_len, dim]
    sin_pos = sin
    
    q_rot = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_rot = (k * cos_pos) + (rotate_half(k) * sin_pos)
    
    print(f"q_rope first 10: {q_rot[0, 0, 0, :10].tolist()}")
    print(f"k_rope first 10: {k_rot[0, 0, 0, :10].tolist()}")
    
    # Repeat KV
    def repeat_kv(hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    k_rep = repeat_kv(k_rot, num_heads // num_kv_heads)
    v_rep = repeat_kv(v, num_heads // num_kv_heads)
    
    # Scores
    scores = torch.matmul(q_rot, k_rep.transpose(2, 3)) / math.sqrt(head_dim)
    print(f"scores first 10: {scores[0, 0, 0, :].tolist()}")
    
    # Causal mask
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    attention_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)
    attention_mask.masked_fill_(causal_mask, torch.finfo(torch.float32).min)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    
    scores_masked = scores + attention_mask
    print(f"scores_masked first 10: {scores_masked[0, 0, 0, :].tolist()}")
    
    attn_weights = torch.nn.functional.softmax(scores_masked, dim=-1, dtype=torch.float32)
    print(f"attn_w first 10: {attn_weights[0, 0, 0, :].tolist()}")
    
    attn_output = torch.matmul(attn_weights, v_rep)
    print(f"attn_o first 10: {attn_output[0, 0, 0, :10].tolist()}")
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
    attn_out = layer0.self_attn.o_proj(attn_output)
    print(f"attn_out (after o_proj) first 10: {attn_out[0, 0, :10].tolist()}")
    
    hidden_states = attn_out + hidden
    print(f"attn_residual first 10: {hidden_states[0, 0, :10].tolist()}")
    
    mlp_normed = layer0.post_attention_layernorm(hidden_states)
    print(f"mlp_norm first 10: {mlp_normed[0, 0, :10].tolist()}")
    
    gate = layer0.mlp.gate_proj(mlp_normed)
    up = layer0.mlp.up_proj(mlp_normed)
    activated = torch.nn.functional.silu(gate) * up
    mlp_out = layer0.mlp.down_proj(activated)
    print(f"mlp_out first 10: {mlp_out[0, 0, :10].tolist()}")
    
    layer_out = mlp_out + hidden_states
    print(f"layer_0_out first 10: {layer_out[0, 0, :10].tolist()}")
