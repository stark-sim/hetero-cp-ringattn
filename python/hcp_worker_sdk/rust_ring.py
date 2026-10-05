"""Rust 兼容的 per-layer-stream ring 传输（混合 Rust↔Python ring 用）。

与 Rust worker（rust/src/worker_sdk/runtime.rs + distributed/transport/quic.rs）
的连线模型完全一致：
- 每个 worker 向 successor 拨一条 QUIC 连接，按层序 0..L-1 开 L 条双向 stream，
  每条 stream 首字节写 0x00 dummy；同时从前驱接受一条连接上的 L 条 stream。
- 每条 stream 只用一个方向：outbound 写、inbound 读。
- 帧格式：[4B BE meta_len][JSON meta][payload...]
    - KV block（无 "type" 字段）：k_bytes/v_bytes + 可选 position_ids（i64 LE）
    - ring_packet（type="ring_packet"）：q/o/lse + scale（Q-ring decode）

入向帧由 per-stream reader task 按 meta.layer_idx 路由到 per-layer 队列，
因此对 stream 到达顺序不敏感（DERP 上 dummy 到达序可能与开流序不同）。
"""

import asyncio
import json
import os
import ssl
import struct
from typing import Dict, List, Optional, Tuple

import torch
from aioquic.asyncio.client import connect
from aioquic.asyncio.server import serve
from aioquic.quic.configuration import QuicConfiguration

from .quic_transport import get_cached_cert


def _t_to_f32le(t: torch.Tensor) -> bytes:
    return t.detach().cpu().to(torch.float32).numpy().tobytes()


def _f32le_to_t(data: bytes, shape: List[int], device: torch.device) -> torch.Tensor:
    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32)
    t = torch.from_numpy(arr.copy()).view(shape)
    return t.to(device)


def _i64le_to_t(data: bytes, shape: List[int]) -> torch.Tensor:
    import numpy as np
    arr = np.frombuffer(data, dtype=np.int64)
    return torch.from_numpy(arr.copy()).view(shape)


class RustRingPeer:
    """一个 worker 的 ring 邻接：out（到 successor）+ in（自 predecessor）。"""

    def __init__(self, num_layers: int, device: torch.device):
        self.num_layers = num_layers
        self.device = device
        self._out_writers: List[asyncio.StreamWriter] = []
        self._kv_queues: List[asyncio.Queue] = [asyncio.Queue() for _ in range(num_layers)]
        self._pkt_queues: List[asyncio.Queue] = [asyncio.Queue() for _ in range(num_layers)]
        self._accepted: List[Tuple] = []
        self._accepted_event = asyncio.Event()
        self._reader_tasks: List[asyncio.Task] = []
        self._server = None
        self._cert_tmp_files: List[str] = []
        self._conn_mgr = None
        self._peer_error: Optional[str] = None

    # ---------- 连接建立 ----------

    async def connect_out(self, host: str, port: int, retries: int = 30) -> None:
        """拨 successor，并按层序开 L 条 stream（每条立即写 dummy）。"""
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            conn_mgr = None
            try:
                # issue#6: 重试前清空，避免部分建流后 layer→stream 错位
                self._out_writers = []
                configuration = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)
                conn_mgr = connect(host, port, configuration=configuration)
                connection = await asyncio.wait_for(conn_mgr.__aenter__(), timeout=30.0)
                for _layer in range(self.num_layers):
                    _reader, writer = await connection.create_stream()
                    writer.write(b"\x00")
                    await writer.drain()
                    self._out_writers.append(writer)
                self._conn_mgr = conn_mgr
                return
            except (ConnectionError, asyncio.TimeoutError, OSError) as e:
                last_err = e
                if conn_mgr is not None:
                    try:
                        await conn_mgr.__aexit__(None, None, None)
                    except Exception:
                        pass
                print(f"[rust-ring] connect {host}:{port} attempt {attempt}/{retries} failed: {e}")
                await asyncio.sleep(2.0)
        raise ConnectionError(f"failed to connect to {host}:{port}: {last_err}")

    async def serve_in(self, host: str, port: int) -> None:
        """监听 predecessor 连接；每条接受的 stream 起一个 reader task。"""
        import tempfile
        cert_pem, key_pem = get_cached_cert()
        cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
        key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
        cert_file.write(cert_pem)
        key_file.write(key_pem)
        cert_file.close()
        key_file.close()
        self._cert_tmp_files = [cert_file.name, key_file.name]
        configuration = QuicConfiguration(is_client=False)
        configuration.load_cert_chain(cert_file.name, key_file.name)

        def stream_handler(reader, writer):
            self._accepted.append((reader, writer))
            self._reader_tasks.append(asyncio.create_task(self._reader_loop(reader)))
            if len(self._accepted) >= self.num_layers:
                self._accepted_event.set()

        # issue#5: serve() 是协程，须保留返回的 QuicServer 句柄才能关监听
        self._server = await serve(
            host, port, configuration=configuration, stream_handler=stream_handler,
        )
        await asyncio.wait_for(self._accepted_event.wait(), timeout=300.0)

    # ---------- 帧 IO ----------

    @staticmethod
    async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
        return await reader.readexactly(n)

    async def _reader_loop(self, reader: asyncio.StreamReader) -> None:
        """每个入向 stream 一个 reader：跳 dummy，逐帧解析并按 layer_idx 路由。"""
        try:
            await reader.readexactly(1)  # dummy
            while True:
                len_bytes = await reader.readexactly(4)
                meta_len = struct.unpack(">I", len_bytes)[0]
                meta = json.loads((await reader.readexactly(meta_len)).decode())
                layer = int(meta["layer_idx"])
                if meta.get("type") == "ring_packet":
                    q = _f32le_to_t(await reader.readexactly(meta["q_bytes"]), meta["q_shape"], self.device)
                    o = _f32le_to_t(await reader.readexactly(meta["o_bytes"]), meta["o_shape"], self.device)
                    lse = _f32le_to_t(await reader.readexactly(meta["lse_bytes"]), meta["lse_shape"], self.device)
                    await self._pkt_queues[layer].put((q, o, lse, float(meta["scale"])))
                else:
                    k = _f32le_to_t(await reader.readexactly(meta["k_bytes"]), meta["k_shape"], self.device)
                    v = _f32le_to_t(await reader.readexactly(meta["v_bytes"]), meta["v_shape"], self.device)
                    pos = meta.get("position_ids")
                    position_ids = None
                    if pos:
                        position_ids = _i64le_to_t(await reader.readexactly(pos["bytes"]), pos["shape"])
                    await self._kv_queues[layer].put({
                        "global_seq_start": int(meta["global_seq_start"]),
                        "global_seq_end": int(meta["global_seq_end"]),
                        "micro_block_idx": int(meta.get("micro_block_idx", 0)),
                        "total_micro_blocks": int(meta.get("total_micro_blocks", 1)),
                        "k": k, "v": v, "position_ids": position_ids,
                    })
        except (asyncio.IncompleteReadError, ConnectionResetError) as e:
            # reader 死亡要能被 recv 侧感知（issue#7），而非静默挂起
            self._peer_error = f"reader loop ended: {type(e).__name__}"
            return
        except Exception as e:
            self._peer_error = f"reader loop error: {e}"
            return

    @staticmethod
    def _write_frame(writer: asyncio.StreamWriter, meta: dict, payloads: List[bytes]) -> None:
        meta_bytes = json.dumps(meta).encode()
        frame = struct.pack(">I", len(meta_bytes)) + meta_bytes + b"".join(payloads)
        writer.write(frame)

    # ---------- KV block（prefill / legacy decode） ----------

    async def send_kv(self, layer: int, k: torch.Tensor, v: torch.Tensor,
                      seq_start: int, seq_end: int) -> None:
        """发送一个 KV block（k/v 须为 wire 形态：GQA 已扩展、f32）。单帧不微分块。"""
        kb = _t_to_f32le(k)
        vb = _t_to_f32le(v)
        # Rust 侧模型跑 bf16，接收端按 dtype 标签 cast；标 bfloat16 与
        # Rust<->Rust 行为一致（payload 始终是 f32 LE）。
        meta = {
            "layer_idx": layer,
            "global_seq_start": seq_start,
            "global_seq_end": seq_end,
            "micro_block_idx": 0,
            "total_micro_blocks": 1,
            "k_shape": list(k.shape), "v_shape": list(v.shape),
            "k_bytes": len(kb), "v_bytes": len(vb),
            "k_dtype": "bfloat16", "v_dtype": "bfloat16",
            "position_ids": None,
        }
        self._write_frame(self._out_writers[layer], meta, [kb, vb])
        await self._out_writers[layer].drain()

    async def recv_kv(self, layer: int) -> dict:
        """接收一个完整 KV block（必要时跨 micro-block 重组）。"""
        first = await self._queue_get(self._kv_queues[layer], layer, "kv")
        total = first["total_micro_blocks"]
        if total == 1:
            return first
        ks, vs = [first["k"]], [first["v"]]
        seq_start, seq_end = first["global_seq_start"], first["global_seq_end"]
        got = first["micro_block_idx"] + 1
        while got < total:
            nxt = await self._queue_get(self._kv_queues[layer], layer, "kv")
            assert nxt["micro_block_idx"] == got, f"micro block out of order on layer {layer}"
            ks.append(nxt["k"])
            vs.append(nxt["v"])
            seq_end = nxt["global_seq_end"]
            got += 1
        return {
            "global_seq_start": seq_start, "global_seq_end": seq_end,
            "micro_block_idx": 0, "total_micro_blocks": 1,
            "k": torch.cat(ks, dim=2), "v": torch.cat(vs, dim=2),
            "position_ids": None,
        }

    # ---------- ring_packet（Q-ring decode） ----------

    async def send_packet(self, layer: int, q: torch.Tensor, o: torch.Tensor,
                          lse: torch.Tensor, scale: float) -> None:
        # Rust 侧模型按 config.torch_dtype=bfloat16 运行，接收端按 *_dtype 标签
        # cast 回本地精度；q/o 必须标 bfloat16，否则 Rust 端 matmul bf16×f32 panic。
        # payload 始终 f32 LE（线格式约定），lse 在 Rust 侧本就是 float32。
        qb, ob, lb = _t_to_f32le(q), _t_to_f32le(o), _t_to_f32le(lse)
        meta = {
            "type": "ring_packet",
            "layer_idx": layer,
            "scale": scale,
            "q_shape": list(q.shape), "o_shape": list(o.shape), "lse_shape": list(lse.shape),
            "q_bytes": len(qb), "o_bytes": len(ob), "lse_bytes": len(lb),
            "q_dtype": "bfloat16", "o_dtype": "bfloat16", "lse_dtype": "float32",
        }
        self._write_frame(self._out_writers[layer], meta, [qb, ob, lb])
        await self._out_writers[layer].drain()

    async def recv_packet(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        return await self._queue_get(self._pkt_queues[layer], layer, "ring_packet")

    async def _queue_get(self, queue: asyncio.Queue, layer: int, what: str):
        """issue#7: 带超时的队列读取；reader task 全部死亡时快速失败而非永久挂起。
        超时与 Rust 侧 HCP_QUIC_TIMEOUT_SECS 默认 600s 对齐。"""
        timeout = float(os.environ.get("HCP_PY_RING_RECV_TIMEOUT", "600"))
        if self._peer_error is not None:
            raise ConnectionError(f"ring peer stream died earlier ({what} layer {layer}): {self._peer_error}")
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ConnectionError(f"recv {what} layer {layer} timeout after {timeout}s")

    async def close(self) -> None:
        for t in self._reader_tasks:
            t.cancel()
        if self._server is not None:
            self._server.close()
        if self._conn_mgr is not None:
            await self._conn_mgr.__aexit__(None, None, None)
        for path in self._cert_tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------- online softmax（与 Rust process_kv_block/decode_merge_packet 同公式） ----------

def online_partial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """单 query 对一段 KV 的 partial：(O 归一化输出 [B,H,1,D], LSE [B,H,1])。非因果。"""
    scores = torch.matmul(q, k.transpose(2, 3)) * scale      # [B,H,1,S]
    m = scores.amax(dim=3)                                   # [B,H,1]
    w = (scores - m.unsqueeze(3)).exp()
    rs = w.sum(dim=3)                                        # [B,H,1]
    pv = torch.matmul(w, v)                                  # [B,H,1,D]
    rs_safe = torch.where(rs != 0, rs, torch.ones_like(rs))
    o = pv / rs_safe.unsqueeze(3)
    lse = (m + rs.log()).to(torch.float32)
    return o, lse


def online_merge(q: torch.Tensor, o_acc: torch.Tensor, lse_acc: torch.Tensor,
                 k: torch.Tensor, v: torch.Tensor, scale: float
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """把环上 packet (o_acc, lse_acc) 与 q 对本地 segment 的 partial 合并。
    (rm=lse_acc, rs=1, obh=o_acc) 起点的 process_kv_block 等价形式。空 segment 原样通过。"""
    if k.size(2) == 0:
        return o_acc, lse_acc
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    local_max = scores.amax(dim=3)                           # [B,H,1]
    w = (scores - local_max.unsqueeze(3)).exp()
    local_sum = w.sum(dim=3)
    local_pv = torch.matmul(w, v)
    new_max = torch.maximum(lse_acc, local_max)
    exp_prev = (lse_acc - new_max).exp()
    exp_local = (local_max - new_max).exp()
    new_sum = exp_prev + exp_local * local_sum
    new_sum_safe = torch.where(new_sum != 0, new_sum, torch.ones_like(new_sum))
    o = (exp_prev.unsqueeze(3) * o_acc + exp_local.unsqueeze(3) * local_pv) / new_sum_safe.unsqueeze(3)
    lse = (new_max + new_sum.log()).to(torch.float32)
    return o, lse
