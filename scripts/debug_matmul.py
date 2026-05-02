#!/usr/bin/env python3
"""Verify matmul behavior matches Rust expectations."""

import torch
from transformers import AutoModelForCausalLM

MODEL_DIR = "/Users/stark_sim/models/qwen2-0.5b"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
model.eval()

layer0 = model.model.layers[0]

# Use actual values from debug output
attn_o_vals = [-0.004695537034422159, 0.011012090370059013, 0.00816428568214178, -0.01145984698086977, -0.004324141889810562, 0.010643900372087955, 0.0007532398449257016, -0.004628523252904415, 0.02267366647720337, 0.01907443255186081]

with torch.no_grad():
    # Get actual attn_o for first token
    hidden = model.model.embed_tokens(torch.tensor([[9707, 11, 1246, 525, 498, 30]]))
    normed = layer0.input_layernorm(hidden)
    q = layer0.self_attn.q_proj(normed)
    k = layer0.self_attn.k_proj(normed)
    v = layer0.self_attn.v_proj(normed)
    
    batch, seq_len, _ = q.shape
    num_heads = layer0.self_attn.config.num_attention_heads
    num_kv_heads = layer0.self_attn.config.num_key_value_heads
    head_dim = layer0.self_attn.head_dim
    
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = model.model.rotary_emb(v, position_ids)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    def repeat_kv(hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    k_rep = repeat_kv(k_rot, num_heads // num_kv_heads)
    v_rep = repeat_kv(v, num_heads // num_kv_heads)
    
    scores = torch.matmul(q_rot, k_rep.transpose(2, 3)) / (head_dim ** 0.5)
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    attention_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)
    attention_mask.masked_fill_(causal_mask, torch.finfo(torch.float32).min)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    scores_masked = scores + attention_mask
    attn_weights = torch.nn.functional.softmax(scores_masked, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, v_rep)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
    
    # Manual matmul
    o_proj_w = layer0.self_attn.o_proj.weight  # [896, 896]
    
    # Method 1: using nn.Linear
    out_linear = layer0.self_attn.o_proj(attn_output)
    
    # Method 2: manual matmul with transpose
    out_manual = attn_output @ o_proj_w.T
    
    # Method 3: manual matmul with transpose, per token
    out_per_token = attn_output[0, 0, :] @ o_proj_w.T
    
    print(f"attn_output[0,0,:10]: {attn_output[0, 0, :10].tolist()}")
    print(f"out_linear[0,0,:10]:  {out_linear[0, 0, :10].tolist()}")
    print(f"out_manual[0,0,:10]:  {out_manual[0, 0, :10].tolist()}")
    print(f"out_per_token[:10]:   {out_per_token[:10].tolist()}")
    print(f"o_proj_w[0,:10]:      {o_proj_w[0, :10].tolist()}")
    print(f"o_proj_w.T[0,:10]:    {o_proj_w.T[0, :10].tolist()}")
    
    # Check if linear equals manual
    print(f"\nlinear == manual: {torch.allclose(out_linear, out_manual, atol=1e-5)}")
