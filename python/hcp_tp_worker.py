#!/usr/bin/env python3
"""HCP 抽象 worker N2：TP 逻辑 domain 接入外层精确 CP ring。

结构（tp_size=N，N1 引擎 + 本文件的 ring  glue）：
- rank 0：协议面 + 计算。作为 HcpWorkerBackend 挂进既有 QuicWorkerServer，
  ring wire 格式不变（发出去的是全头 KV f32 帧，与单进程 worker 完全一致）。
- rank 1..N-1：纯计算 follower（run_tp_follower），不连 coordinator/ring，
  只按确定顺序跟 collective。

逐层数据流（prefill）：
1. 两 rank 各算本地 chunk 的 Q/K/V head 切片（RoPE 全局位置）；
2. all_gather 两 rank 的 KV head 切片 → rank 0 得全头 KV 块送外层 ring；
3. rank 0 收到远端全头 KV 块 → broadcast 给 follower → 各 rank 切自己的 KV head；
4. 本地注意力（local_q_heads vs 全序列 1 个 KV head，repeat local_q_heads），
   掩码 key_pos <= query_pos（非方阵，同 prefill_ring_exact）；
5. o_proj / mlp.down 各 all_reduce(SUM)，hidden 保持复制语义。

cache：每 rank 只缓存自己 head 切片的全序列 KV（显存 1/tp_size）。

collective 与事件循环：collective 全部是同步阻塞调用，但与 ring I/O 严格
串行（exchange_layer await 完全结束后才进入下一个 collective），不存在
"事件循环被卡住导致 ring I/O 饿死"的窗口，故不需要 to_thread。

不支持的组合：ring_mode=rust（Q-ring packet decode 与 per-layer rust KV 流的
head 重复语义按单进程假设写的，tp_size>1 时无意义），在 main 入口处护栏。
"""

import asyncio
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from hcp_tp_engine import TensorParallelQwen2
from hcp_worker_sdk.backend import HcpWorkerBackend
from hcp_worker_sdk.types import KvBlock

_OP_PREFILL = 0
_OP_DECODE = 1
_OP_SHUTDOWN = 2


class TPTransformersBackend(HcpWorkerBackend):
    """HcpWorkerBackend 的 TP 实现：一个逻辑 domain = tp_size 个设备进程。"""

    def __init__(
        self,
        model_dir: str,
        device: str,
        tp_backend: str,
        rank: int,
        world_size: int,
        init_method: str,
        num_domains: int,
    ):
        self.engine = TensorParallelQwen2(
            model_dir,
            device=device,
            rank=rank,
            world_size=world_size,
            backend=tp_backend,
            init_method=init_method,
        )
        self.num_domains = num_domains
        self.device = self.engine.device
        self.rank = rank
        self.world_size = world_size
        self._cache_k: List[torch.Tensor] = []
        self._cache_v: List[torch.Tensor] = []
        self._cache_len = 0
        self._last_logits: Optional[torch.Tensor] = None
        print(
            f"[tp backend] rank={rank}/{world_size} device={self.device} "
            f"local_q_heads={self.engine.local_q_heads} "
            f"local_kv_heads={self.engine.local_kv_heads}",
            flush=True,
        )

    # ---- collective helpers（comm device = engine device：gloo→cpu, hccl→npu）----

    def _bcast(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
        dist.broadcast(t, src)
        return t

    def _all_gather_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[1, local_kv_heads, s, d] -> [1, num_kv_heads, s, d]（rank 序拼接即全局 head 序）。"""
        outs = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(outs, t.contiguous())
        return torch.cat(outs, dim=1)

    def broadcast_shutdown(self) -> None:
        """通知 follower 退出（best effort）。"""
        if self.rank != 0 or self.world_size <= 1:
            return
        try:
            hdr = torch.tensor([_OP_SHUTDOWN, 0, 0], dtype=torch.int32, device=self.device)
            dist.broadcast(hdr, 0)
        except Exception as e:
            print(f"[tp backend] shutdown broadcast warning: {e}", flush=True)

    # ---- prefill ----

    async def prefill_ring_exact(
        self,
        chunk: List[int],
        seq_offset: int,
        exchange_layer,
        domain_id: int = 0,
        num_domains: int = 1,
    ) -> Tuple[torch.Tensor, int]:
        """外层精确 CP prefill（rank 0：ring 交换 + 远端 KV 广播）。"""
        my_start = seq_offset
        ids32 = torch.tensor(chunk, dtype=torch.int32, device=self.device)
        hdr = torch.tensor([_OP_PREFILL, my_start, len(chunk)], dtype=torch.int32, device=self.device)
        self._bcast(hdr)
        self._bcast(ids32)
        return await self._tp_prefill(ids32, my_start, exchange_layer)

    def prefill(self, chunk: List[int], seq_offset: int, position_ids=None) -> Tuple[torch.Tensor, int]:
        """1-domain TP prefill（无远端块）。sync 上下文驱动 async 实现：
        exchange_layer=None 时协程从不挂起（collective 全同步），一次 send 跑完。"""
        my_start = seq_offset
        ids32 = torch.tensor(chunk, dtype=torch.int32, device=self.device)
        hdr = torch.tensor([_OP_PREFILL, my_start, len(chunk)], dtype=torch.int32, device=self.device)
        self._bcast(hdr)
        self._bcast(ids32)
        coro = self._tp_prefill(ids32, my_start, None)
        try:
            coro.send(None)
            raise RuntimeError("tp prefill coroutine suspended unexpectedly")
        except StopIteration as e:
            return e.value

    async def _tp_prefill(
        self, ids32: torch.Tensor, my_start: int, exchange_layer
    ) -> Tuple[torch.Tensor, int]:
        """两 rank 完全同构的 prefill 层循环；rank 0 额外做 ring 交换。

        collective 顺序（两 rank 必须一致）：每层 all_gather(k), all_gather(v)
        → 每个远端块 broadcast(hdr), broadcast(k), broadcast(v) → all_reduce(o)
        → all_reduce(mlp)。
        """
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        eng = self.engine
        my_len = ids32.numel()
        b = 1
        ids = ids32.to(torch.long).unsqueeze(0)
        positions = torch.arange(my_start, my_start + my_len, device=eng.device,
                                 dtype=torch.long).unsqueeze(0)

        hidden = eng.embed_tokens(ids)
        cos, sin = eng.rotary_emb(hidden, positions)

        cache_k: List[torch.Tensor] = []
        cache_v: List[torch.Tensor] = []
        prefill_total = my_start + my_len
        with torch.no_grad():
            for li, layer in enumerate(eng.layers):
                attn = layer.self_attn
                residual = hidden
                h = layer.input_layernorm(hidden)
                q = attn.q_proj(h).view(b, my_len, eng.local_q_heads, eng.head_dim).transpose(1, 2)
                k = attn.k_proj(h).view(b, my_len, eng.local_kv_heads, eng.head_dim).transpose(1, 2)
                v = attn.v_proj(h).view(b, my_len, eng.local_kv_heads, eng.head_dim).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

                # 出站：all_gather 全头 KV（rank0 送 ring，wire 格式不变）
                k_full = self._all_gather_heads(k)
                v_full = self._all_gather_heads(v)

                remote: List[KvBlock] = []
                if self.rank == 0 and exchange_layer is not None:
                    remote = await exchange_layer(li, k_full, v_full)
                remote = self._broadcast_remote(remote)

                # 各 rank 只保留自己 KV head 的全序列切片
                blocks = [(my_start, k, v)] + [
                    (
                        blk.global_seq_start,
                        blk.k[:, eng.rank : eng.rank + 1],
                        blk.v[:, eng.rank : eng.rank + 1],
                    )
                    for blk in remote
                ]
                blocks.sort(key=lambda t: t[0])
                prefill_total = max(prefill_total, max(t[0] + t[1].size(2) for t in blocks))
                k_all = torch.cat([t[1] for t in blocks], dim=2)
                v_all = torch.cat([t[2] for t in blocks], dim=2)
                key_pos = torch.cat([
                    torch.arange(t[0], t[0] + t[1].size(2), device=eng.device, dtype=torch.long)
                    for t in blocks
                ])

                mask = key_pos[None, :] <= positions[0, :, None]  # [s, total_kv]
                if eng.gqa_repeat > 1:
                    k_att = k_all.repeat_interleave(eng.gqa_repeat, dim=1)
                    v_att = v_all.repeat_interleave(eng.gqa_repeat, dim=1)
                else:
                    k_att, v_att = k_all, v_all
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q, k_att, v_att, attn_mask=mask[None, None, :, :]
                )
                attn_out = attn_out.transpose(1, 2).reshape(b, my_len, eng.local_q_dim)
                partial = attn.o_proj(attn_out)
                dist.all_reduce(partial, op=dist.ReduceOp.SUM)
                hidden = residual + partial

                residual = hidden
                h = layer.post_attention_layernorm(hidden)
                mlp = layer.mlp
                partial = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
                dist.all_reduce(partial, op=dist.ReduceOp.SUM)
                hidden = residual + partial

                cache_k.append(k_all)
                cache_v.append(v_all)

            hidden = eng.final_norm(hidden)
            logits = eng.lm_head(hidden[:, -1]) if self.rank == 0 else None

        self._cache_k = cache_k
        self._cache_v = cache_v
        self._cache_len = prefill_total
        if self.rank == 0:
            self._last_logits = logits[0].to(torch.float32).cpu()
            return self._last_logits, my_start + my_len
        return torch.zeros(0), my_start + my_len

    def _broadcast_remote(self, remote: List[KvBlock]) -> List[KvBlock]:
        """rank 0 把 ring 收到的远端全头 KV 块逐块广播给 follower。"""
        eng = self.engine
        n_remote = self.num_domains - 1
        if self.rank == 0:
            assert len(remote) == n_remote, f"expected {n_remote} remote blocks, got {len(remote)}"
        out: List[KvBlock] = []
        for r in range(n_remote):
            if self.rank == 0:
                blk = remote[r]
                start, ln = blk.global_seq_start, blk.global_seq_end - blk.global_seq_start
                hdr = torch.tensor([start, ln], dtype=torch.int32, device=self.device)
                k, v = blk.k.contiguous(), blk.v.contiguous()
            else:
                hdr = torch.empty(2, dtype=torch.int32, device=self.device)
                k = v = None
            self._bcast(hdr)
            start, ln = int(hdr[0].item()), int(hdr[1].item())
            if self.rank != 0:
                shape = (1, eng.num_kv_heads, ln, eng.head_dim)
                k = torch.empty(shape, dtype=torch.float32, device=self.device)
                v = torch.empty(shape, dtype=torch.float32, device=self.device)
            self._bcast(k)
            self._bcast(v)
            out.append(KvBlock(-1, start, start + ln, k, v))
        return out

    # ---- decode ----

    def decode(self, token: int) -> torch.Tensor:
        hdr = torch.tensor([_OP_DECODE, token, 0], dtype=torch.int32, device=self.device)
        self._bcast(hdr)
        return self._tp_decode(token)

    def _tp_decode(self, token: int) -> Optional[torch.Tensor]:
        """逐 token TP decode：各 rank 在自己 head 切片的全量 cache 上精确前向。

        单 query 的因果掩码退化为全可见，与 HF decode（cache 上全注意力）一致。
        """
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        eng = self.engine
        pos = self._cache_len
        ids = torch.tensor([[token]], dtype=torch.long, device=eng.device)
        pos_t = torch.tensor([[pos]], dtype=torch.long, device=eng.device)

        hidden = eng.embed_tokens(ids)
        cos, sin = eng.rotary_emb(hidden, pos_t)
        with torch.no_grad():
            for li, layer in enumerate(eng.layers):
                attn = layer.self_attn
                residual = hidden
                h = layer.input_layernorm(hidden)
                q = attn.q_proj(h).view(1, 1, eng.local_q_heads, eng.head_dim).transpose(1, 2)
                k = attn.k_proj(h).view(1, 1, eng.local_kv_heads, eng.head_dim).transpose(1, 2)
                v = attn.v_proj(h).view(1, 1, eng.local_kv_heads, eng.head_dim).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

                k_all = torch.cat([self._cache_k[li], k], dim=2)
                v_all = torch.cat([self._cache_v[li], v], dim=2)
                self._cache_k[li] = k_all
                self._cache_v[li] = v_all
                if eng.gqa_repeat > 1:
                    k_att = k_all.repeat_interleave(eng.gqa_repeat, dim=1)
                    v_att = v_all.repeat_interleave(eng.gqa_repeat, dim=1)
                else:
                    k_att, v_att = k_all, v_all
                attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_att, v_att)
                attn_out = attn_out.transpose(1, 2).reshape(1, 1, eng.local_q_dim)
                partial = attn.o_proj(attn_out)
                dist.all_reduce(partial, op=dist.ReduceOp.SUM)
                hidden = residual + partial

                residual = hidden
                h = layer.post_attention_layernorm(hidden)
                mlp = layer.mlp
                partial = mlp.down_proj(mlp.act_fn(mlp.gate_proj(h)) * mlp.up_proj(h))
                dist.all_reduce(partial, op=dist.ReduceOp.SUM)
                hidden = residual + partial

            hidden = eng.final_norm(hidden)
            self._cache_len = pos + 1
            if self.rank != 0:
                return None
            logits = eng.lm_head(hidden[:, -1])
        return logits[0].to(torch.float32).cpu()

    async def decode_ring_exact(self, token: int, packet_exchange) -> torch.Tensor:
        raise NotImplementedError(
            "TP abstract worker does not support Q-ring packet decode "
            "(decode 留给 PD 生态；tp_size>1 仅用 all-Python decode 路径)"
        )

    def recalculate_logits(self) -> torch.Tensor:
        """精确路径下 prefill 末位置 logits 已精确，直接返回缓存值。"""
        return self._last_logits

    # ---- KV block 接口（精确路径不走 _exchange_kv_ring，仅为接口完整）----

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        raise NotImplementedError(
            "TP backend caches only local head shards; ring exchange happens "
            "inside prefill_ring_exact, not via get_kv_block"
        )

    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        raise NotImplementedError("see get_kv_block")

    def load_model(self, model_dir: str, device: str) -> None:
        pass

    @property
    def capacity_mb(self) -> int:
        """逻辑 domain 容量 = 单芯空闲 × tp_size。

        权重与 KV 都按 head 切到 tp_size 个设备上，有效容量随芯数线性扩展；
        同构双芯下用 rank0 单芯值外推即可（避免额外 collective）。
        """
        try:
            import torch_npu  # noqa: F401
            if self.device.type == "npu":
                free, _ = torch_npu.npu.mem_get_info(self.device)
                return int(free // (1024 * 1024)) * self.world_size
        except ImportError:
            pass
        if self.device.type == "mps":
            return int(torch.mps.recommended_max_memory() // (1024 * 1024)) * self.world_size
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024)) * self.world_size
        return 4096 * self.world_size

    @property
    def num_layers(self) -> int:
        return self.engine.num_layers

    @property
    def num_heads(self) -> int:
        return self.engine.num_heads  # 全模型 Q head 数（协议/日志语义）

    @property
    def head_dim(self) -> int:
        return self.engine.head_dim

    def close(self) -> None:
        self.engine.close()


def run_tp_follower(
    model_dir: str,
    device: str,
    tp_backend: str,
    rank: int,
    world_size: int,
    init_method: str,
    num_domains: int,
) -> None:
    """TP follower 主循环：不连 coordinator/ring，只跟 collective。

    协议（rank 0 -> follower，int32 hdr[3] = [op, a, b]）：
      op=0 prefill: a=my_start, b=my_len，随后 broadcast ids int32[b]
      op=1 decode:  a=token
      op=2 shutdown
    """
    backend = TPTransformersBackend(
        model_dir, device, tp_backend, rank, world_size, init_method, num_domains
    )
    eng = backend.engine
    print(f"[tp follower rank {rank}] ready, num_domains={num_domains}", flush=True)
    while True:
        hdr = torch.empty(3, dtype=torch.int32, device=eng.device)
        dist.broadcast(hdr, 0)
        op, a, b = int(hdr[0].item()), int(hdr[1].item()), int(hdr[2].item())
        if op == _OP_SHUTDOWN:
            print(f"[tp follower rank {rank}] shutdown", flush=True)
            break
        if op == _OP_PREFILL:
            ids32 = torch.empty(b, dtype=torch.int32, device=eng.device)
            dist.broadcast(ids32, 0)
            # follower 无 ring 交换：协程从不挂起，同步驱动即可
            coro = backend._tp_prefill(ids32, a, None)
            try:
                coro.send(None)
                raise RuntimeError("tp prefill coroutine suspended unexpectedly")
            except StopIteration:
                pass
            print(f"[tp follower rank {rank}] prefill done (start={a}, len={b})", flush=True)
        elif op == _OP_DECODE:
            backend._tp_decode(a)
        else:
            raise RuntimeError(f"unknown tp op {op}")
    backend.close()
