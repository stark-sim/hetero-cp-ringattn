#!/usr/bin/env python3
"""
HCP 快速原型 Worker。
继承 phase1_poc_tcp/worker.py 的思路：
- 监听 TCP 端口
- 接收 controller 发来的 Q chunk + 全局 K/V
- 在本地执行 Attention 计算（当前为 placeholder）
- 返回状态
"""

import argparse
import json
import socket
import struct

import numpy as np


def send_msg(sock, data_bytes: bytes):
    sock.sendall(struct.pack(">I", len(data_bytes)))
    sock.sendall(data_bytes)


def recv_msg(sock) -> bytes:
    raw_len = sock.recv(4)
    if not raw_len:
        return b""
    msg_len = struct.unpack(">I", raw_len)[0]
    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if not chunk:
            break
        data += chunk
    return data


def handle_client(conn, addr):
    print(f"[worker] connection from {addr}")
    data = recv_msg(conn)
    if not data:
        return
    payload = json.loads(data.decode())
    domain_id = payload["domain_id"]
    seq_chunk_len = payload["seq_chunk_len"]
    print(f"[worker] {domain_id} received seq_chunk_len={seq_chunk_len}")

    # TODO: 真实执行本地 Attention 计算或 ring 通信占位
    result = {"status": "ok", "domain_id": domain_id, "seq_chunk_len": seq_chunk_len}
    send_msg(conn, json.dumps(result).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ringattn-worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26001)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.host, args.port))
        s.listen(1)
        print(f"[worker] {args.name} listening on {args.host}:{args.port}")
        while True:
            conn, addr = s.accept()
            with conn:
                handle_client(conn, addr)


if __name__ == "__main__":
    main()
