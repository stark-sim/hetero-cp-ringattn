"""
QUIC control-plane client for Python workers connecting to Rust coordinator.

Protocol (matches Rust distributed_protocol):
- Handshake: 16 bytes LE (domain_id u64 + capacity_mb u64), no length prefix
- Commands/Responses: [4-byte BE length][bincode payload]
- Bincode format: little-endian, fixed-width integers
  - enum tag: u32 (4 bytes)
  - usize/i64/u64: 8 bytes
  - Vec<T>: 8-byte len prefix + elements
"""

import asyncio
import ssl
import struct
from typing import Optional, Tuple

from aioquic.asyncio.client import connect
from aioquic.quic.configuration import QuicConfiguration

from .bincode import (
    encode_command, decode_command,
    encode_response, decode_response,
    encode_handshake, decode_handshake,
)


class QuicControlClient:
    """
    QUIC control-plane client for HCP worker.

    Usage:
        client = QuicControlClient()
        await client.connect("127.0.0.1", 26001)
        await client.send_handshake(domain_id=0, capacity_mb=4096)
        while True:
            cmd = await client.recv_command()
            if cmd["kind"] == "Shutdown":
                break
            # ... process cmd ...
            await client.send_response({"kind": "PrefillDone", ...})
        await client.close()
    """

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout
        self.connection = None
        self.reader = None
        self.writer = None
        self._dummy_sent = False
        self._dummy_recv = False

    async def connect(self, host: str, port: int) -> None:
        """Connect to coordinator via QUIC."""
        configuration = QuicConfiguration(
            is_client=True,
            verify_mode=ssl.CERT_NONE,
        )
        self._conn_ctx = connect(host, port, configuration=configuration)
        self.connection = await self._conn_ctx.__aenter__()
        self.reader, self.writer = await self.connection.create_stream()
        print(f"[quic client] connected to {host}:{port}")

    async def close(self) -> None:
        """Close QUIC connection."""
        if self.writer:
            self.writer.write_eof()
            await self.writer.drain()
        if self.connection:
            self.connection.close()
        if hasattr(self, '_conn_ctx'):
            await self._conn_ctx.__aexit__(None, None, None)
        print("[quic client] connection closed")

    async def send_handshake(self, domain_id: int, capacity_mb: int) -> None:
        """Send 16-byte handshake."""
        data = encode_handshake(domain_id, capacity_mb)
        self.writer.write(data)
        await self.writer.drain()
        print(f"[quic client] handshake sent: domain_id={domain_id}, capacity_mb={capacity_mb}")

    async def recv_handshake(self) -> Tuple[int, int]:
        """Receive 16-byte handshake (for coordinator mode)."""
        data = await self._read_exact(16)
        return decode_handshake(data)

    async def send_command(self, kind: str, **kwargs) -> None:
        """Send a length-prefixed bincode command."""
        payload = encode_command(kind, **kwargs)
        frame = struct.pack(">I", len(payload)) + payload
        self.writer.write(frame)
        await self.writer.drain()
        print(f"[quic client] command sent: {kind}")

    async def recv_command(self) -> dict:
        """Receive a length-prefixed bincode command."""
        len_bytes = await self._read_exact(4)
        length = struct.unpack(">I", len_bytes)[0]
        if length > 64 * 1024 * 1024:
            raise ValueError(f"frame too large: {length} bytes")
        payload = await self._read_exact(length)
        cmd = decode_command(payload)
        print(f"[quic client] command received: {cmd['kind']}")
        return cmd

    async def send_response(self, kind: str, **kwargs) -> None:
        """Send a length-prefixed bincode response."""
        payload = encode_response(kind, **kwargs)
        frame = struct.pack(">I", len(payload)) + payload
        self.writer.write(frame)
        await self.writer.drain()
        print(f"[quic client] response sent: {kind}")

    async def recv_response(self) -> dict:
        """Receive a length-prefixed bincode response."""
        len_bytes = await self._read_exact(4)
        length = struct.unpack(">I", len_bytes)[0]
        if length > 64 * 1024 * 1024:
            raise ValueError(f"frame too large: {length} bytes")
        payload = await self._read_exact(length)
        resp = decode_response(payload)
        print(f"[quic client] response received: {resp['kind']}")
        return resp

    async def _read_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the stream."""
        data = b""
        while len(data) < n:
            chunk = await self.reader.read(n - len(data))
            if not chunk:
                raise ConnectionError(f"read_exact: connection closed (needed {n}, got {len(data)})")
            data += chunk
        return data
