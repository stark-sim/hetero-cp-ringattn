"""
HCP Worker SDK — 核心类型定义

与 Rust 端 distributed_protocol.rs + kv_transport.rs 对应。
"""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class KvBlock:
    """分布式 Ring Attention 中交换的 K/V Block。"""
    layer_idx: int
    global_seq_start: int
    global_seq_end: int
    k: torch.Tensor  # [batch, num_heads, seq_len, head_dim]
    v: torch.Tensor  # [batch, num_heads, seq_len, head_dim]

    def clone(self) -> "KvBlock":
        return KvBlock(
            layer_idx=self.layer_idx,
            global_seq_start=self.global_seq_start,
            global_seq_end=self.global_seq_end,
            k=self.k.clone(),
            v=self.v.clone(),
        )


@dataclass
class WorkerHandshake:
    """Worker 连接 Coordinator 后的握手包（固定 16 字节 LE）。"""
    domain_id: int
    capacity_mb: int

    def to_bytes(self) -> bytes:
        return (
            self.domain_id.to_bytes(8, "little") +
            self.capacity_mb.to_bytes(8, "little")
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "WorkerHandshake":
        return cls(
            domain_id=int.from_bytes(data[0:8], "little"),
            capacity_mb=int.from_bytes(data[8:16], "little"),
        )


@dataclass
class WorkerCommand:
    """Coordinator → Worker 控制指令。"""
    kind: str  # "Prefill" | "Decode" | "SyncGlobalSeqLen" | "Shutdown"
    chunk: Optional[List[int]] = None       # Prefill 用
    seq_offset: Optional[int] = None        # Prefill 用
    token: Optional[int] = None             # Decode 用
    global_seq_len: Optional[int] = None    # SyncGlobalSeqLen 用

    def to_dict(self) -> dict:
        d = {"kind": self.kind}
        if self.chunk is not None:
            d["chunk"] = self.chunk
        if self.seq_offset is not None:
            d["seq_offset"] = self.seq_offset
        if self.token is not None:
            d["token"] = self.token
        if self.global_seq_len is not None:
            d["global_seq_len"] = self.global_seq_len
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerCommand":
        return cls(
            kind=d["kind"],
            chunk=d.get("chunk"),
            seq_offset=d.get("seq_offset"),
            token=d.get("token"),
            global_seq_len=d.get("global_seq_len"),
        )

    @classmethod
    def prefill(cls, chunk: List[int], seq_offset: int) -> "WorkerCommand":
        return cls(kind="Prefill", chunk=chunk, seq_offset=seq_offset)

    @classmethod
    def decode(cls, token: int) -> "WorkerCommand":
        return cls(kind="Decode", token=token)

    @classmethod
    def sync_global_seq_len(cls, length: int) -> "WorkerCommand":
        return cls(kind="SyncGlobalSeqLen", global_seq_len=length)

    @classmethod
    def shutdown(cls) -> "WorkerCommand":
        return cls(kind="Shutdown")


@dataclass
class WorkerResponse:
    """Worker → Coordinator 响应。"""
    kind: str  # "PrefillDone" | "DecodeDone" | "Error"
    last_logits_bytes: Optional[bytes] = None   # PrefillDone: f32 LE bytes
    logits_bytes: Optional[bytes] = None        # DecodeDone: f32 LE bytes
    global_seq_len: Optional[int] = None        # PrefillDone
    error: Optional[str] = None                 # Error

    def to_dict(self) -> dict:
        d = {"kind": self.kind}
        if self.last_logits_bytes is not None:
            d["last_logits_bytes"] = list(self.last_logits_bytes)
        if self.logits_bytes is not None:
            d["logits_bytes"] = list(self.logits_bytes)
        if self.global_seq_len is not None:
            d["global_seq_len"] = self.global_seq_len
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerResponse":
        return cls(
            kind=d["kind"],
            last_logits_bytes=bytes(d["last_logits_bytes"]) if "last_logits_bytes" in d else None,
            logits_bytes=bytes(d["logits_bytes"]) if "logits_bytes" in d else None,
            global_seq_len=d.get("global_seq_len"),
            error=d.get("error"),
        )

    @classmethod
    def prefill_done(cls, last_logits: torch.Tensor, global_seq_len: int) -> "WorkerResponse":
        """从 torch.Tensor 自动转换为 f32 LE bytes。"""
        logits_np = last_logits.detach().cpu().to(torch.float32).numpy()
        return cls(
            kind="PrefillDone",
            last_logits_bytes=logits_np.tobytes(),
            global_seq_len=global_seq_len,
        )

    @classmethod
    def decode_done(cls, logits: torch.Tensor) -> "WorkerResponse":
        logits_np = logits.detach().cpu().to(torch.float32).numpy()
        return cls(
            kind="DecodeDone",
            logits_bytes=logits_np.tobytes(),
        )

    @classmethod
    def error(cls, message: str) -> "WorkerResponse":
        return cls(kind="Error", error=message)
