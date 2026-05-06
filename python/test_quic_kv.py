#!/usr/bin/env python3
"""
QUIC KV Transport 闭环测试 — 验证 Python QuicKvTransport 正确性。
"""

import asyncio
import torch

from hcp_worker_sdk.quic_transport import (
    get_cached_cert,
    QuicKvTransport,
)
from hcp_worker_sdk.types import KvBlock
from aioquic.asyncio.server import serve
from aioquic.asyncio.client import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.tls import SessionTicket
import ssl
import tempfile


ticket_store = {}


def store_ticket(ticket: SessionTicket) -> None:
    ticket_store[ticket.ticket] = ticket


async def server_task(host: str, port: int):
    cert_pem, key_pem = get_cached_cert()
    cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    cert_file.write(cert_pem)
    key_file.write(key_pem)
    cert_file.close()
    key_file.close()

    configuration = QuicConfiguration(is_client=False, max_datagram_frame_size=65536)
    configuration.load_cert_chain(cert_file.name, key_file.name)

    result = {"block": None, "done": asyncio.Event()}

    def handler(reader, writer):
        async def handle():
            device = torch.device("cpu")
            transport = QuicKvTransport(reader, writer, device)

            block_in = await transport._recv_kv_block()
            print(f"[server] received: layer={block_in.layer_idx}, k_shape={list(block_in.k.shape)}")

            block_out = KvBlock(
                layer_idx=block_in.layer_idx,
                global_seq_start=block_in.global_seq_start,
                global_seq_end=block_in.global_seq_end,
                k=block_in.k + 1.0,
                v=block_in.v + 1.0,
            )
            await transport._send_kv_block(block_out)
            print("[server] sent back")
            await writer.drain()
            await asyncio.sleep(0.2)
            writer.close()
            await writer.wait_closed()
            result["done"].set()
        asyncio.create_task(handle())

    await serve(
        host, port,
        configuration=configuration,
        stream_handler=handler,
        session_ticket_handler=store_ticket,
    )
    print(f"[server] listening on {host}:{port}")
    await asyncio.sleep(20)


async def client_task(host: str, port: int):
    await asyncio.sleep(1)

    configuration = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)

    async with connect(host, port, configuration=configuration) as connection:
        reader, writer = await connection.create_stream()
        device = torch.device("cpu")
        transport = QuicKvTransport(reader, writer, device)

        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        block = KvBlock(layer_idx=3, global_seq_start=10, global_seq_end=14, k=k, v=v)

        print(f"[client] send k_sum={k.sum().item():.4f}, v_sum={v.sum().item():.4f}")
        await transport._send_kv_block(block)
        block_back = await transport._recv_kv_block()
        print(f"[client] recv k_sum={block_back.k.sum().item():.4f}, v_sum={block_back.v.sum().item():.4f}")

        k_diff = (block_back.k - (k + 1.0)).abs().max().item()
        v_diff = (block_back.v - (v + 1.0)).abs().max().item()
        print(f"[client] k_diff={k_diff:.6f}, v_diff={v_diff:.6f}")

        if k_diff < 1e-5 and v_diff < 1e-5:
            print("[client] ✅ QUIC KV roundtrip PASS")
        else:
            print("[client] ❌ FAIL")
            raise AssertionError(f"diff too large")

        writer.close()


async def main():
    host, port = "127.0.0.1", 29596
    await asyncio.gather(
        server_task(host, port),
        client_task(host, port),
    )
    print("[main] done")


if __name__ == "__main__":
    asyncio.run(main())
