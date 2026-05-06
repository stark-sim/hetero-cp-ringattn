"""
QUIC KV Transport — 基于 aioquic，与 Rust quinn 互通。

帧格式（与 Rust 侧完全兼容）：
    [4-byte BE meta_len] [JSON metadata] [k_bytes] [v_bytes]

metadata 字段：
    layer_idx, global_seq_start, global_seq_end,
    k_shape, v_shape, k_bytes, v_bytes
"""

import asyncio
import ipaddress
import json
import ssl
import struct
from typing import Optional

import torch
from aioquic.asyncio.client import connect
from aioquic.asyncio.server import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.tls import SessionTicket

from .transport import KvTransport
from .types import KvBlock


def generate_self_signed_cert():
    """生成自签名证书和私钥（PEM bytes）。"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HCP"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


# 缓存证书，避免重复生成
_cert_cache = None


def get_cached_cert():
    global _cert_cache
    if _cert_cache is None:
        _cert_cache = generate_self_signed_cert()
    return _cert_cache


class QuicKvTransport(KvTransport):
    """基于 QUIC stream 的 KV block 传输。

    使用 asyncio StreamReader/Writer 进行读写。
    exchange_kv_block() 是同步接口，内部桥接 async。
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, device: torch.device):
        self.reader = reader
        self.writer = writer
        self.device = device

    @staticmethod
    def _tensor_to_bytes(t: torch.Tensor) -> bytes:
        return t.detach().cpu().to(torch.float32).numpy().tobytes()

    @staticmethod
    def _bytes_to_tensor(data: bytes, shape: list, device: torch.device) -> torch.Tensor:
        import numpy as np
        arr = np.frombuffer(data, dtype=np.float32)
        expected = 1
        for s in shape:
            expected *= s
        if arr.size != expected:
            raise ValueError(f"Shape mismatch: expected {expected} floats, got {arr.size}")
        t = torch.from_numpy(arr.copy()).view(shape).to(device)
        return t

    async def _send_kv_block(self, block: KvBlock) -> None:
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

        frame = struct.pack(">I", len(meta)) + meta + k_bytes + v_bytes
        self.writer.write(frame)
        await self.writer.drain()

    async def _recv_kv_block(self) -> Optional[KvBlock]:
        try:
            len_bytes = await self.reader.readexactly(4)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

        meta_len = struct.unpack(">I", len_bytes)[0]
        meta_bytes = await self.reader.readexactly(meta_len)
        meta = json.loads(meta_bytes.decode())

        k_bytes_len = meta["k_bytes"]
        v_bytes_len = meta["v_bytes"]
        k_bytes = await self.reader.readexactly(k_bytes_len)
        v_bytes = await self.reader.readexactly(v_bytes_len)

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
        """同步接口：先发后收。"""
        async def _exchange():
            await self._send_kv_block(block)
            return await self._recv_kv_block()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_exchange())

        # 如果已有运行中的事件循环，用线程池桥接
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _exchange())
            return future.result()


async def create_quic_server(host: str, port: int) -> tuple:
    """创建 QUIC server，返回 (stop_event, task)。

    stream_handler 负责处理每个连接上的 stream。
    """
    cert_pem, key_pem = get_cached_cert()

    import tempfile
    cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    cert_file.write(cert_pem)
    key_file.write(key_pem)
    cert_file.close()
    key_file.close()

    configuration = QuicConfiguration(is_client=False)
    configuration.load_cert_chain(cert_file.name, key_file.name)

    # 用于存储接受的 stream
    accepted_streams = []
    connected_event = asyncio.Event()

    def stream_handler(reader, writer):
        accepted_streams.append((reader, writer))
        connected_event.set()

    server_task = asyncio.create_task(serve(
        host, port,
        configuration=configuration,
        stream_handler=stream_handler,
    ))

    return connected_event, accepted_streams, server_task


async def create_quic_client(host: str, port: int) -> tuple:
    """创建 QUIC client，返回 (reader, writer, connection_manager)。

    使用 async with 管理 connection 生命周期：
        async with create_quic_client(...) as (reader, writer, conn):
            ...
    """
    configuration = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)
    connection_manager = connect(host, port, configuration=configuration)
    connection = await connection_manager.__aenter__()
    reader, writer = await connection.create_stream()
    return reader, writer, connection_manager
