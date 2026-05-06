#!/usr/bin/env python3
"""
Python QUIC client ↔ Rust QUIC server 互通验证。

Rust server: cargo run --bin quic-echo-server 127.0.0.1:29590
Python client: python test_quic_python_rust.py
"""

import asyncio
import ssl
import struct
import json
import numpy as np

from aioquic.asyncio.client import connect
from aioquic.quic.configuration import QuicConfiguration


def tensor_to_bytes(t: np.ndarray) -> bytes:
    return t.astype(np.float32).tobytes()


def bytes_to_tensor(data: bytes, shape: list) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.float32)
    expected = 1
    for s in shape:
        expected *= s
    assert arr.size == expected, f"Shape mismatch: expected {expected} floats, got {arr.size}"
    return arr.reshape(shape)


async def client_task(host: str, port: int):
    configuration = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)

    print(f"[python client] connecting to {host}:{port}")
    async with connect(host, port, configuration=configuration) as connection:
        reader, writer = await connection.create_stream()
        print("[python client] connected, stream created")

        # Rust quinn workaround: send dummy on first write
        writer.write(b"\x00")
        await writer.drain()

        # Build test KV block
        k = np.random.randn(1, 2, 4, 8).astype(np.float32)
        v = np.random.randn(1, 2, 4, 8).astype(np.float32)

        meta = {
            "layer_idx": 3,
            "global_seq_start": 10,
            "global_seq_end": 14,
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "k_bytes": k.size * 4,
            "v_bytes": v.size * 4,
        }
        meta_bytes = json.dumps(meta).encode()
        k_bytes = tensor_to_bytes(k)
        v_bytes = tensor_to_bytes(v)

        frame = struct.pack(">I", len(meta_bytes)) + meta_bytes + k_bytes + v_bytes
        writer.write(frame)
        await writer.drain()
        print(f"[python client] sent block: layer={meta['layer_idx']}, k_sum={k.sum():.4f}")

        # Receive response (read all available data)
        all_data = b""
        while True:
            try:
                chunk = await asyncio.wait_for(reader.read(1024), timeout=3.0)
                if not chunk:
                    break
                all_data += chunk
            except asyncio.TimeoutError:
                break
            except Exception as e:
                print(f"[python client] read error: {e}")
                break
        print(f"[python client] total received: {len(all_data)} bytes, hex={all_data[:30].hex()}")

        # Parse frame: skip dummy, then read frame
        if len(all_data) < 1:
            raise ValueError("no data received")
        dummy = all_data[0:1]
        print(f"[python client] dummy={dummy.hex()}")
        if len(all_data) < 5:
            raise ValueError("data too short")
        len_bytes = all_data[1:5]
        print(f"[python client] len_bytes={len_bytes.hex()}")
        meta_len = struct.unpack(">I", len_bytes)[0]
        print(f"[python client] meta_len={meta_len}")
        if len(all_data) < 5 + meta_len:
            raise ValueError(f"data too short for meta: {len(all_data)} < {5 + meta_len}")
        meta_bytes_back = all_data[5:5+meta_len]
        meta_back = json.loads(meta_bytes_back.decode())

        k_bytes_len = meta_back["k_bytes"]
        v_bytes_len = meta_back["v_bytes"]
        k_offset = 5 + meta_len
        v_offset = k_offset + k_bytes_len
        if len(all_data) < v_offset + v_bytes_len:
            raise ValueError(f"data too short for k/v: {len(all_data)} < {v_offset + v_bytes_len}")
        k_bytes_back = all_data[k_offset:k_offset+k_bytes_len]
        v_bytes_back = all_data[v_offset:v_offset+v_bytes_len]

        k_back = bytes_to_tensor(k_bytes_back, meta_back["k_shape"])
        v_back = bytes_to_tensor(v_bytes_back, meta_back["v_shape"])

        print(f"[python client] received back: k_sum={k_back.sum():.4f}, v_sum={v_back.sum():.4f}")

        # Verify: server adds 1.0 to k/v
        k_diff = np.abs(k_back - (k + 1.0)).max()
        v_diff = np.abs(v_back - (v + 1.0)).max()
        print(f"[python client] k_diff={k_diff:.6f}, v_diff={v_diff:.6f}")

        if k_diff < 1e-5 and v_diff < 1e-5:
            print("[python client] ✅ Python↔Rust QUIC互通验证通过")
        else:
            print("[python client] ❌ FAIL")
            raise AssertionError(f"diff too large: k={k_diff}, v={v_diff}")

        writer.close()


if __name__ == "__main__":
    asyncio.run(client_task("127.0.0.1", 29590))
