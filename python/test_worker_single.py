#!/usr/bin/env python3
"""
单 Worker 控制面验证 — 自动启动 coordinator + 1 worker。

验证目标：
1. 单 vLLM / transformers worker 在 HCP Worker SDK 协议下端到端正确
2. Prefill → Decode → Shutdown 全链路通过
3. 输出与单节点参考一致

用法：
    # transformers (本地 CPU/MPS)
    python test_worker_single.py --model-dir models/Qwen2-0.5B --backend transformers

    # vLLM (远程 GPU，需先安装 vllm)
    python test_worker_single.py --model-dir /home/stark/models/Qwen2-0.5B --backend vllm
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer


def compute_reference(args) -> List[int]:
    """用 transformers 计算贪婪解码参考结果。"""
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    token_ids = tok.encode(args.prompt, add_special_tokens=False)
    generated = []
    for _ in range(args.max_tokens):
        input_ids = torch.tensor([token_ids + generated], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]
        next_token = int(torch.argmax(logits))
        generated.append(next_token)
    return generated


def run_coordinator(args, ref_tokens: List[int]):
    """Coordinator：接受 1 个 worker，完整 prompt prefill，广播 decode。"""
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    token_ids = tok.encode(args.prompt, add_special_tokens=False)
    print(f"[coordinator] prompt tokens: {token_ids}")

    import socket
    host, port = args.listen_addr.rsplit(":", 1)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(1)
    print(f"[coordinator] listening on {args.listen_addr}")

    sock, addr = listener.accept()
    print(f"[coordinator] worker connected from {addr}")
    listener.close()

    # Handshake
    from hcp_worker_sdk.types import WorkerHandshake
    data = b""
    while len(data) < 16:
        chunk = sock.recv(16 - len(data))
        if not chunk:
            raise ConnectionError("handshake failed")
        data += chunk
    hs = WorkerHandshake.from_bytes(data)
    print(f"[coordinator] handshake domain={hs.domain_id} cap={hs.capacity_mb} MB")

    def send_cmd(sock, cmd):
        data = json.dumps(cmd.to_dict()).encode()
        frame = struct.pack(">I", len(data)) + data
        sock.sendall(frame)

    def recv_resp(sock):
        from hcp_worker_sdk.types import WorkerResponse
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

    # Prefill
    from hcp_worker_sdk.types import WorkerCommand
    send_cmd(sock, WorkerCommand.prefill(token_ids, seq_offset=0))
    resp = recv_resp(sock)
    assert resp.kind == "PrefillDone", f"unexpected: {resp.kind}"
    logits = np.frombuffer(resp.last_logits_bytes, dtype=np.float32)
    global_seq_len = resp.global_seq_len or len(token_ids)
    print(f"[coordinator] PrefillDone, global_seq_len={global_seq_len}")

    # Sync
    send_cmd(sock, WorkerCommand.sync_global_seq_len(global_seq_len))

    # Decode loop
    generated = []
    for step in range(args.max_tokens):
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        print(f"[coordinator] step {step}: token={next_token}")

        send_cmd(sock, WorkerCommand.decode(next_token))
        resp = recv_resp(sock)
        assert resp.kind == "DecodeDone", f"unexpected: {resp.kind}"
        logits = np.frombuffer(resp.logits_bytes, dtype=np.float32)

    # Shutdown
    send_cmd(sock, WorkerCommand.shutdown())
    sock.close()

    text = tok.decode(generated, skip_special_tokens=True)
    print(f"[coordinator] generated: '{text}'")

    if generated == ref_tokens:
        print(f"[coordinator] ✅ MATCH: output identical to single-node reference")
    else:
        print(f"[coordinator] ❌ MISMATCH: ref={ref_tokens}, got={generated}")
        sys.exit(1)


def run_worker_subprocess(args) -> subprocess.Popen:
    """用 subprocess 启动 worker 进程（避免 vLLM 与 multiprocessing 冲突）。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    cmd = [
        sys.executable,
        "-c",
        f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')

from hcp_worker_sdk import NoOpKvTransport, HcpWorkerServer

backend_name = '{args.backend}'
model_dir = '{args.model_dir}'

if backend_name == 'vllm':
    from hcp_vllm_worker import VllmBackend
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = VllmBackend(model_dir, device)
else:
    from test_worker_control_plane import TransformersBackend
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    backend = TransformersBackend(model_dir, device)

transport = NoOpKvTransport()
server = HcpWorkerServer(
    backend=backend,
    transport=transport,
    domain_id=0,
    num_domains=1,
)
server.run(
    coordinator_addr='{args.listen_addr}',
    listen_addr='127.0.0.1:29501',
    next_peer_addr='127.0.0.1:0',
)
""",
    ]
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="The answer to life, the universe, and everything is")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--listen-addr", default="127.0.0.1:29500")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    args = parser.parse_args()

    print(f"[main] computing single-node reference...")
    ref_tokens = compute_reference(args)
    print(f"[main] reference tokens: {ref_tokens}")

    # 启动 worker 子进程
    proc = run_worker_subprocess(args)
    time.sleep(2)

    try:
        run_coordinator(args, ref_tokens)
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    print("[main] done")


if __name__ == "__main__":
    main()
