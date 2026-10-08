#!/usr/bin/env python3
"""HCP 抽象 worker N1：head 并行 Tensor Parallelism 引擎核心（Qwen2）。

Megatron 式切分（tp_size 整除 num_heads / num_kv_heads / intermediate_size）：
- q_proj 按 Q head 切行，k_proj/v_proj 按 KV head 切行（column-parallel）；
- o_proj 按 Q head 切列、down_proj 对半切列（row-parallel），输出 all_reduce(SUM)；
- gate_proj/up_proj 对半切行；embed / 两个 layernorm / final norm 复制；
- lm_head 仅 rank 0 保留（只有 rank 0 出 logits）。
每层 2 次 all_reduce，hidden 在 rank 间保持复制语义；attention 完全本地
（rank 内 local_q_heads / local_kv_heads 的 GQA repeat）。

N2 预留：`_attention` 是每层注意力的可覆写点，ring 集成时由子类注入外部 KV。
backend 参数化（"gloo" 本地 / "hccl" NPU），未来 NCCL 只换字符串。
"""

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import torch_npu  # noqa: F401  (registers the "npu" device backend)
except ImportError:
    torch_npu = None


class TensorParallelQwen2:
    """Head-parallel TP forward engine for Qwen2 (fp32)."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        rank: int = 0,
        world_size: int = 1,
        backend: str = "gloo",
        init_method: str = "tcp://127.0.0.1:29511",
        timeout_s: Optional[float] = None,
        local_rank: Optional[int] = None,
        dtype: str = "float32",
    ):
        # local_rank = 本进程在单机内的设备序号；默认等于全局 rank（单机多卡
        # TP 的常规情形）。跨机 TP（每机单卡）必须显式传 local_rank=0，
        # 否则 rank1 会落到不存在的 cuda:1。
        if local_rank is None:
            local_rank = rank
        if backend == "hccl" and device != "npu":
            raise ValueError("hccl backend requires device='npu'")
        if device == "npu":
            if torch_npu is None:
                raise RuntimeError("torch_npu is not importable on this host")
            torch.npu.set_device(rank)
            self.device = torch.device(f"npu:{rank}")
            try:
                torch.npu.matmul.allow_hf32 = False  # 严格 fp32 数值
            except Exception:
                pass
        elif device == "cuda":
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(device)

        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        if world_size > 1:
            pg_kwargs = {}
            if timeout_s is not None:
                from datetime import timedelta

                pg_kwargs["timeout"] = timedelta(seconds=timeout_s)
            dist.init_process_group(
                backend, rank=rank, world_size=world_size, init_method=init_method,
                **pg_kwargs,
            )

        from transformers import AutoModelForCausalLM

        _DTYPES = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype}")
        self.compute_dtype = _DTYPES[dtype]
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=self.compute_dtype, trust_remote_code=True
        )
        model.eval()
        config = model.config
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.num_heads % world_size != 0:
            raise ValueError(f"num_heads={self.num_heads} not divisible by tp={world_size}")
        if self.num_kv_heads % world_size != 0:
            raise ValueError(f"num_kv_heads={self.num_kv_heads} not divisible by tp={world_size}")
        if config.intermediate_size % world_size != 0:
            raise ValueError(
                f"intermediate={config.intermediate_size} not divisible by tp={world_size}"
            )
        self.local_q_heads = self.num_heads // world_size
        self.local_kv_heads = self.num_kv_heads // world_size
        self.gqa_repeat = self.local_q_heads // self.local_kv_heads
        self.local_q_dim = self.local_q_heads * self.head_dim
        self.local_kv_dim = self.local_kv_heads * self.head_dim
        self.local_inter = config.intermediate_size // world_size

        core = model.model
        self.embed_tokens = core.embed_tokens  # 复制
        self.rotary_emb = core.rotary_emb
        self.final_norm = core.norm  # 复制

        qs = slice(rank * self.local_q_dim, (rank + 1) * self.local_q_dim)
        kvs = slice(rank * self.local_kv_dim, (rank + 1) * self.local_kv_dim)
        ms = slice(rank * self.local_inter, (rank + 1) * self.local_inter)

        self.layers: List[nn.Module] = []
        for layer in core.layers:
            attn = layer.self_attn
            # column-parallel：按 head 切行
            attn.q_proj.weight = nn.Parameter(attn.q_proj.weight.data[qs].contiguous())
            attn.k_proj.weight = nn.Parameter(attn.k_proj.weight.data[kvs].contiguous())
            attn.v_proj.weight = nn.Parameter(attn.v_proj.weight.data[kvs].contiguous())
            if attn.q_proj.bias is not None:
                attn.q_proj.bias = nn.Parameter(attn.q_proj.bias.data[qs].contiguous())
                attn.k_proj.bias = nn.Parameter(attn.k_proj.bias.data[kvs].contiguous())
                attn.v_proj.bias = nn.Parameter(attn.v_proj.bias.data[kvs].contiguous())
            # row-parallel：切列，all_reduce 后由 rank 0 加 bias（Qwen2 o_proj 无 bias）
            attn.o_proj.weight = nn.Parameter(attn.o_proj.weight.data[:, qs].contiguous())
            if attn.o_proj.bias is not None and rank != 0:
                attn.o_proj.bias = None
            mlp = layer.mlp
            mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.data[ms].contiguous())
            mlp.up_proj.weight = nn.Parameter(mlp.up_proj.weight.data[ms].contiguous())
            mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.data[:, ms].contiguous())
            if mlp.down_proj.bias is not None and rank != 0:
                mlp.down_proj.bias = None
            self.layers.append(layer)

        self.lm_head = model.lm_head if rank == 0 else None
        model.lm_head = None  # 释放非 rank0 的 lm_head 引用（tie 到 embed 的除外）

        self.embed_tokens.to(self.device)
        self.rotary_emb.to(self.device)
        self.final_norm.to(self.device)
        for layer in self.layers:
            layer.to(self.device)
        if self.lm_head is not None:
            self.lm_head.to(self.device)
        del model, core

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def _attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """单层本地注意力，返回 [b, s, local_q_dim]。

        N2 可覆写：先缓存/注入外部 KV，再做（全局）因果注意力。
        q: [b, local_q_heads, s, head_dim]，k/v: [b, local_kv_heads, s, head_dim]
        positions: [b, s] 全局位置。
        """
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        b, _, s, _ = q.shape
        key_pos = positions[0]
        mask = key_pos[None, :] <= positions[0, :, None]  # [s, s] 因果
        if self.gqa_repeat > 1:
            k = k.repeat_interleave(self.gqa_repeat, dim=1)
            v = v.repeat_interleave(self.gqa_repeat, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[None, None, :, :]
        )
        return out.transpose(1, 2).reshape(b, s, self.local_q_dim)

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """全序列前向。rank 0 返回 [b, s, vocab] fp32 logits，其余 rank 返回 None。"""
        input_ids = input_ids.to(self.device)
        b, s = input_ids.shape
        positions = torch.arange(s, device=self.device, dtype=torch.long).unsqueeze(0)

        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden, positions)

        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = hidden
            h = layer.input_layernorm(hidden)
            q = attn.q_proj(h).view(b, s, self.local_q_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(b, s, self.local_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(b, s, self.local_kv_heads, self.head_dim).transpose(1, 2)
            attn_out = self._attention(li, q, k, v, cos, sin, positions)
            partial = attn.o_proj(attn_out)
            hidden = residual + self._all_reduce(partial)

            residual = hidden
            h = layer.post_attention_layernorm(hidden)
            mlp = layer.mlp
            partial = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
            hidden = residual + self._all_reduce(partial)

        hidden = self.final_norm(hidden)
        if self.rank != 0:
            return None
        return self.lm_head(hidden).to(torch.float32)

    def close(self) -> None:
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
