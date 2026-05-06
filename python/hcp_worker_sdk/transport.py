"""
HCP Worker SDK — KV Transport 实现

目前提供 TCP 版本。QUIC 版本可基于 aioquic 后续扩展。
"""

from abc import ABC, abstractmethod
import socket
import struct
import json
from typing import Optional
import torch
from .types import KvBlock


class KvTransport(ABC):
    """KV block 交换传输层抽象。"""

    @abstractmethod
    def exchange_kv_block(self, block: KvBlock) -> Optional[KvBlock]:
        """
        原子操作：发送本地 block 到 next peer，同时从 prev peer 接收 block。

        返回 None 表示 ring 遍历完成（没有更多 peer block）。
        """
        pass


class NoOpKvTransport(KvTransport):
    """不实际交换 KV 的 stub transport（Phase 1 控制面验证用）。"""

    def exchange_kv_block(self, block: KvBlock) -> Optional[KvBlock]:
        return None


class LinkedMockKvTransport(KvTransport):
    """内存队列连接的 Mock KV Transport（单进程 2-domain Phase 2 验证用）。"""

    def __init__(self, inbox, outbox, timeout: float = 5.0):
        """
        Args:
            inbox: 接收 peer KV block 的 queue.Queue
            outbox: 发送本地 KV block 的 queue.Queue
            timeout: queue get 超时秒数
        """
        self.inbox = inbox
        self.outbox = outbox
        self.timeout = timeout

    def exchange_kv_block(self, block: KvBlock) -> Optional[KvBlock]:
        """先发后收：将 block 放入 outbox，然后从 inbox 取 peer block。"""
        self.outbox.put(block)
        try:
            return self.inbox.get(timeout=self.timeout)
        except Exception:
            return None


class TcpKvTransport(KvTransport):
    """基于 TCP socket 的 KV block 传输。"""

    def __init__(self, stream: socket.socket, device: torch.device):
        self.stream = stream
        self.device = device
        stream.settimeout(120)  # 大 KV block 传输需要足够超时

    @staticmethod
    def _tensor_to_bytes(t: torch.Tensor) -> bytes:
        """将 tensor 序列化为 f32 LE bytes。"""
        return t.detach().cpu().to(torch.float32).numpy().tobytes()

    @staticmethod
    def _bytes_to_tensor(data: bytes, shape: list, device: torch.device) -> torch.Tensor:
        """从 f32 LE bytes 反序列化为 tensor。"""
        import numpy as np
        arr = np.frombuffer(data, dtype=np.float32)
        expected = 1
        for s in shape:
            expected *= s
        if arr.size != expected:
            raise ValueError(f"Shape mismatch: expected {expected} floats, got {arr.size}")
        t = torch.from_numpy(arr.copy()).view(shape).to(device)
        return t

    def send_kv_block(self, block: KvBlock) -> None:
        """发送单个 KV block。"""
        k_bytes = self._tensor_to_bytes(block.k)
        v_bytes = self._tensor_to_bytes(block.v)
        k_shape = list(block.k.shape)
        v_shape = list(block.v.shape)

        meta = json.dumps({
            "layer_idx": block.layer_idx,
            "global_seq_start": block.global_seq_start,
            "global_seq_end": block.global_seq_end,
            "k_shape": k_shape,
            "v_shape": v_shape,
            "k_bytes": len(k_bytes),
            "v_bytes": len(v_bytes),
        }).encode()

        # Frame: [meta_len: u32 BE] [meta] [k_bytes] [v_bytes]
        frame = struct.pack(">I", len(meta)) + meta + k_bytes + v_bytes
        self._sendall(frame)

    def recv_kv_block(self) -> Optional[KvBlock]:
        """接收单个 KV block。"""
        try:
            len_bytes = self._recvall(4)
        except (ConnectionResetError, ConnectionAbortedError):
            return None

        meta_len = struct.unpack(">I", len_bytes)[0]
        meta_bytes = self._recvall(meta_len)
        meta = json.loads(meta_bytes.decode())

        k_bytes_len = meta["k_bytes"]
        v_bytes_len = meta["v_bytes"]
        k_bytes = self._recvall(k_bytes_len)
        v_bytes = self._recvall(v_bytes_len)

        k = self._bytes_to_tensor(k_bytes, meta["k_shape"], self.device)
        v = self._bytes_to_tensor(v_bytes, meta["v_shape"], self.device)

        return KvBlock(
            layer_idx=meta["layer_idx"],
            global_seq_start=meta["global_seq_start"],
            global_seq_end=meta["global_seq_end"],
            k=k,
            v=v,
        )

    def exchange_kv_block(self, block: KvBlock) -> Optional[KvBlock]:
        """
        TCP 版本的 exchange：先发后收。

        注意：TCP 是双工流，先发后收不会死锁（只要缓冲区够大）。
        如果未来遇到死锁，可改用异步并发 send/recv。
        """
        self.send_kv_block(block)
        return self.recv_kv_block()

    def _sendall(self, data: bytes) -> None:
        total = len(data)
        sent = 0
        while sent < total:
            n = self.stream.send(data[sent:])
            if n == 0:
                raise ConnectionError("send: connection closed")
            sent += n

    def _recvall(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.stream.recv(n - len(data))
            if not chunk:
                raise ConnectionError("recv: connection closed")
            data += chunk
        return data
