#!/usr/bin/env python3
"""
HCP 快速原型控制器。
继承 phase1_poc_tcp/controller.py 的思路：
- 生成输入 Q/K/V
- 按 domain 的 seq_chunk_len 不均分切分
- 通过 TCP 分发给各 worker
- 收集结果并验证 online softmax 聚合后的数值
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report-path", default="reports/ringattn_smoke.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    global_seq_len = config["global_seq_len"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    domains = config["domains"]

    print(f"[controller] global_seq_len={global_seq_len}, heads={num_heads}, dim={head_dim}")
    print(f"[controller] domains={len(domains)}")

    # 生成全局 Q/K/V（仅用于原型验证）
    np.random.seed(42)
    Q = np.random.randn(global_seq_len, num_heads, head_dim).astype(np.float32)
    K = np.random.randn(global_seq_len, num_heads, head_dim).astype(np.float32)
    V = np.random.randn(global_seq_len, num_heads, head_dim).astype(np.float32)

    # 按各 domain 的 seq_chunk_len 切分 Q
    offset = 0
    for d in domains:
        chunk_len = d["seq_chunk_len"]
        q_chunk = Q[offset : offset + chunk_len]
        payload = {
            "domain_id": d["domain_id"],
            "global_seq_len": global_seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "seq_chunk_len": chunk_len,
            "block_size": d["block_size"],
            "q_chunk": q_chunk.tobytes(),
            "k_global": K.tobytes(),
            "v_global": V.tobytes(),
        }
        data = json.dumps(payload).encode()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((d["host"], d["port"]))
            send_msg(s, data)
            resp = recv_msg(s)
            if resp:
                result = json.loads(resp.decode())
                print(f"[controller] {d['domain_id']} -> {result.get('status', 'unknown')}")
            else:
                print(f"[controller] {d['domain_id']} -> no response")
        offset += chunk_len

    report = {"status": "ok", "config": config}
    import os
    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[controller] report saved to {args.report_path}")


if __name__ == "__main__":
    main()
