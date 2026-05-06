#!/usr/bin/env python3
"""
最小 QUIC echo 测试 — 验证 aioquic API + 自签名证书。
"""

import asyncio
import ssl
from aioquic.asyncio.client import connect
from aioquic.asyncio.server import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.tls import SessionTicket


def generate_cert():
    """生成自签名证书（内存中）。"""
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
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=1)
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName("localhost")]),
        critical=False,
    ).sign(key, hashes.SHA256())

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


ticket_store = {}


def store_ticket(ticket: SessionTicket) -> None:
    ticket_store[ticket.ticket] = ticket


async def server_task(host: str, port: int, cert_pem: bytes, key_pem: bytes):
    """QUIC server：接收消息并 echo 回去。"""
    configuration = QuicConfiguration(
        is_client=False,
        max_datagram_frame_size=65536,
    )
    # Write certs to temp files (aioquic requires file paths)
    import tempfile
    cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    cert_file.write(cert_pem)
    key_file.write(key_pem)
    cert_file.close()
    key_file.close()
    configuration.load_cert_chain(cert_file.name, key_file.name)

    def handler(reader, writer):
        async def handle():
            data = await reader.read(1024)
            print(f"[server] received: {data.decode()}")
            writer.write(data)
            await writer.drain()
            writer.close()
        asyncio.create_task(handle())

    await serve(
        host, port,
        configuration=configuration,
        stream_handler=handler,
        session_ticket_handler=store_ticket,
    )
    print(f"[server] listening on {host}:{port}")
    await asyncio.sleep(10)


async def client_task(host: str, port: int, cert_pem: bytes):
    """QUIC client：发送消息并接收 echo。"""
    await asyncio.sleep(0.5)  # 等 server 启动

    configuration = QuicConfiguration(
        is_client=True,
        verify_mode=ssl.CERT_NONE,  # 开发环境跳过验证
    )

    # For client, also write cert to temp file if needed (but we use verify_mode=NONE)
    async with connect(host, port, configuration=configuration) as connection:
        reader, writer = await connection.create_stream()
        msg = b"hello from python quic"
        writer.write(msg)
        await writer.drain()

        data = await reader.read(1024)
        print(f"[client] received echo: {data.decode()}")

        if data == msg:
            print("[client] ✅ echo match")
        else:
            print(f"[client] ❌ mismatch: expected {msg}, got {data}")

        writer.close()


async def main():
    cert_pem, key_pem = generate_cert()
    host, port = "127.0.0.1", 29499

    await asyncio.gather(
        server_task(host, port, cert_pem, key_pem),
        client_task(host, port, cert_pem),
    )


if __name__ == "__main__":
    asyncio.run(main())
