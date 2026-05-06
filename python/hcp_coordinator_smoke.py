#!/usr/bin/env python3
"""
简化 Python Coordinator — Phase 1 控制面验证用。

与 Rust Coordinator 协议不同（TCP+JSON vs QUIC+bincode），
仅用于验证 Python Worker 的 prefill/decode/shutdown 命令处理逻辑。

用法：
    python hcp_coordinator_smoke.py \
        --model-dir ~/models/Qwen2-0.5B \
        --prompt "Hello world" \
        --max-tokens 5 \
        --worker-addr 127.0.0.1:29451
"""

import argparse
import socket
import struct
import json
import time
import numpy as np

from hcp_worker_sdk.types import WorkerHandshake, WorkerCommand, WorkerResponse


def recvall(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("recvall: connection closed")
        data += chunk
    return data


def send_command(sock: socket.socket, cmd: WorkerCommand) -> None:
    data = json.dumps(cmd.to_dict()).encode()
    frame = struct.pack(">I", len(data)) + data
    sock.sendall(frame)


def recv_response(sock: socket.socket) -> WorkerResponse:
    len_bytes = recvall(sock, 4)
    length = struct.unpack(">I", len_bytes)[0]
    data = recvall(sock, length)
    return WorkerResponse.from_dict(json.loads(data.decode()))


def tokenize(prompt: str, tokenizer_path: str) -> list:
    """用 tokenizer.json 做简单分词（空格分词作为 fallback）。"""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return tok.encode(prompt, add_special_tokens=False)
    except Exception:
        print("WARNING: tokenizer load failed, using simple whitespace split")
        return prompt.split()


def sample_token(logits_bytes: bytes, temperature: float = 0.0) -> int:
    """从 f32 bytes 采样 token。temperature=0 时 greedy。"""
    logits = np.frombuffer(logits_bytes, dtype=np.float32)
    if temperature == 0.0:
        return int(np.argmax(logits))
    # temperature sampling
    probs = np.exp(logits / temperature)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Hello, world!")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--worker-addr", default="127.0.0.1:29451")
    args = parser.parse_args()

    # Tokenize prompt
    token_ids = tokenize(args.prompt, args.model_dir)
    print(f"[coordinator] prompt: '{args.prompt}'")
    print(f"[coordinator] tokens: {len(token_ids)}")

    # Connect to worker
    host, port = args.worker_addr.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, int(port)))
    print(f"[coordinator] connected to worker {args.worker_addr}")

    # Read handshake (16 bytes LE)
    handshake_bytes = recvall(sock, 16)
    handshake = WorkerHandshake.from_bytes(handshake_bytes)
    print(f"[coordinator] worker handshake: domain_id={handshake.domain_id}, "
          f"capacity={handshake.capacity_mb} MB")

    # Send Prefill
    print(f"[coordinator] sending Prefill ({len(token_ids)} tokens)")
    send_command(sock, WorkerCommand.prefill(token_ids, seq_offset=0))

    resp = recv_response(sock)
    if resp.kind == "PrefillDone":
        logits = np.frombuffer(resp.last_logits_bytes, dtype=np.float32)
        global_seq_len = resp.global_seq_len or len(token_ids)
        print(f"[coordinator] PrefillDone, global_seq_len={global_seq_len}, "
              f"logits shape={logits.shape}")
    elif resp.kind == "Error":
        print(f"[coordinator] worker error: {resp.error}")
        sock.close()
        return
    else:
        print(f"[coordinator] unexpected response: {resp.kind}")
        sock.close()
        return

    # SyncGlobalSeqLen
    send_command(sock, WorkerCommand.sync_global_seq_len(global_seq_len))

    # Decode loop
    generated = []
    for step in range(args.max_tokens):
        next_token = sample_token(resp.last_logits_bytes if step == 0 else resp.logits_bytes,
                                   args.temperature)
        generated.append(next_token)
        print(f"[coordinator] step {step}: token_id={next_token}")

        send_command(sock, WorkerCommand.decode(next_token))
        resp = recv_response(sock)

        if resp.kind == "DecodeDone":
            logits = np.frombuffer(resp.logits_bytes, dtype=np.float32)
        elif resp.kind == "Error":
            print(f"[coordinator] worker error: {resp.error}")
            break
        else:
            print(f"[coordinator] unexpected response: {resp.kind}")
            break

    print(f"[coordinator] generated tokens: {generated}")

    # Shutdown
    send_command(sock, WorkerCommand.shutdown())
    print("[coordinator] shutdown sent")
    sock.close()

    # Try decode tokens
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        text = tok.decode(generated, skip_special_tokens=True)
        print(f"[coordinator] generated text: '{text}'")
    except Exception as e:
        print(f"[coordinator] decode text failed: {e}")


if __name__ == "__main__":
    main()
