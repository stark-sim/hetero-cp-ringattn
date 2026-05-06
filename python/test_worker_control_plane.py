#!/usr/bin/env python3
"""
本地控制面通信验证 — 使用 transformers 作为 backend。

验证目标：
1. HcpWorkerServer 能正确接收 Prefill/Decode/Shutdown 命令
2. Worker 返回的 logits 格式正确（f32 bytes）
3. Coordinator 能从 logits 采样 token
4. 端到端生成流程正常

用法（两个 terminal）：
    # Terminal 1: 启动 Coordinator（先启动，监听等待 worker 连接）
    python test_worker_control_plane.py --coordinator \
        --model-dir ../models/Qwen2-0.5B \
        --prompt "The answer to life, the universe, and everything is" \
        --max-tokens 5 \
        --listen-addr 127.0.0.1:29500

    # Terminal 2: 启动 Worker（连接 coordinator）
    python test_worker_control_plane.py --worker \
        --model-dir ../models/Qwen2-0.5B \
        --coordinator-addr 127.0.0.1:29500 \
        --listen-addr 127.0.0.1:29501
"""

import argparse
import json
import struct
import sys
import torch
import numpy as np
from typing import List, Tuple

from hcp_worker_sdk import (
    HcpWorkerBackend,
    KvBlock,
    HcpWorkerServer,
    NoOpKvTransport,
)
from hcp_worker_sdk.types import WorkerCommand, WorkerResponse, WorkerHandshake


class TransformersBackend(HcpWorkerBackend):
    """用 transformers 实现的简化 backend，用于本地控制面验证。"""

    def __init__(self, model_dir: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        print(f"[transformers backend] loading model from {model_dir} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.model.eval()

        config = self.model.config
        self._num_layers = getattr(config, "num_hidden_layers", 24)
        self._num_heads = getattr(config, "num_attention_heads", 14)
        self._head_dim = getattr(config, "hidden_size", 896) // self._num_heads
        self._history: List[int] = []
        print(f"[transformers backend] loaded: {self._num_layers} layers")

    def load_model(self, model_dir: str, device: str) -> None:
        pass

    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        self._history = list(chunk)
        logits = self._forward(self._history)
        return logits, len(self._history) + seq_offset

    def decode(self, token: int) -> torch.Tensor:
        self._history.append(token)
        return self._forward(self._history)

    def _forward(self, token_ids: List[int]) -> torch.Tensor:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1]
        return logits.to(torch.float32).cpu()

    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        k = torch.empty(0)
        v = torch.empty(0)
        return KvBlock(layer_idx, seq_start, seq_end, k, v)

    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        pass

    @property
    def capacity_mb(self) -> int:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
        return 4096

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim


def run_worker(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backend = TransformersBackend(args.model_dir, device)
    transport = NoOpKvTransport()
    server = HcpWorkerServer(
        backend=backend,
        transport=transport,
        domain_id=0,
        num_domains=1,
    )
    print(f"[worker] connecting to coordinator {args.coordinator_addr}")
    server.run(
        coordinator_addr=args.coordinator_addr,
        listen_addr=args.listen_addr,
        next_peer_addr="127.0.0.1:0",
    )


def run_coordinator(args):
    import socket
    import struct
    import json

    # Tokenize
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    token_ids = tok.encode(args.prompt, add_special_tokens=False)
    print(f"[coordinator] prompt tokens: {token_ids}")

    # Listen for worker connection
    host, port = args.listen_addr.rsplit(":", 1)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(1)
    print(f"[coordinator] listening on {args.listen_addr}")

    sock, addr = listener.accept()
    print(f"[coordinator] worker connected from {addr}")
    listener.close()

    # Read handshake (16 bytes LE)
    data = b""
    while len(data) < 16:
        chunk = sock.recv(16 - len(data))
        if not chunk:
            raise ConnectionError("handshake recv failed")
        data += chunk
    handshake = WorkerHandshake.from_bytes(data)
    print(f"[coordinator] handshake: domain={handshake.domain_id}, cap={handshake.capacity_mb} MB")

    # Prefill
    send_command(sock, WorkerCommand.prefill(token_ids, seq_offset=0))
    resp = recv_response(sock)
    assert resp.kind == "PrefillDone", f"unexpected: {resp.kind}"
    logits = np.frombuffer(resp.last_logits_bytes, dtype=np.float32)
    global_seq_len = resp.global_seq_len or len(token_ids)
    print(f"[coordinator] PrefillDone, logits shape={logits.shape}, global_seq_len={global_seq_len}")

    # Sync
    send_command(sock, WorkerCommand.sync_global_seq_len(global_seq_len))

    # Decode loop
    generated = []
    for step in range(args.max_tokens):
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        print(f"[coordinator] step {step}: token={next_token}")

        send_command(sock, WorkerCommand.decode(next_token))
        resp = recv_response(sock)
        assert resp.kind == "DecodeDone", f"unexpected: {resp.kind}"
        logits = np.frombuffer(resp.logits_bytes, dtype=np.float32)

    # Shutdown
    send_command(sock, WorkerCommand.shutdown())
    print("[coordinator] shutdown sent")
    sock.close()

    text = tok.decode(generated, skip_special_tokens=True)
    print(f"[coordinator] generated: '{text}'")


def send_command(sock, cmd):
    data = json.dumps(cmd.to_dict()).encode()
    frame = struct.pack(">I", len(data)) + data
    sock.sendall(frame)


def recv_response(sock):
    len_bytes = b""
    while len(len_bytes) < 4:
        chunk = sock.recv(4 - len(len_bytes))
        if not chunk:
            raise ConnectionError("recv len failed")
        len_bytes += chunk
    length = struct.unpack(">I", len_bytes)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("recv data failed")
        data += chunk
    return WorkerResponse.from_dict(json.loads(data.decode()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--worker", action="store_true", help="run as worker")
    parser.add_argument("--coordinator", action="store_true", help="run as coordinator")
    parser.add_argument("--coordinator-addr", default="127.0.0.1:29500")
    parser.add_argument("--listen-addr", default="127.0.0.1:29501")
    parser.add_argument("--prompt", default="The answer to life, the universe, and everything is")
    parser.add_argument("--max-tokens", type=int, default=5)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    elif args.coordinator:
        run_coordinator(args)
    else:
        print("ERROR: specify --worker or --coordinator")
        sys.exit(1)


if __name__ == "__main__":
    main()
