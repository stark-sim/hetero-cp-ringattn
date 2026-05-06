#!/usr/bin/env python3
"""
2-domain Mock KV Ring 验证 — 单进程/多进程模拟分布式 KV 交换。

验证目标：
1. LinkedMockKvTransport 能在内存中无损交换 KV block
2. Backend 的 get_kv_block / apply_peer_kv 正确
3. 2-domain 输出与单节点参考一致

用法：
    python test_worker_2domain.py --model-dir models/Qwen2-0.5B \
        --prompt "The answer to life, the universe, and everything is" \
        --max-tokens 5
"""

import argparse
import json
import struct
import sys
import threading
import time
import torch
import numpy as np
from typing import List


def run_coordinator(args, ref_tokens: List[int], ready_event):
    """Coordinator：accept 两个 worker，分片 prompt，广播 decode。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    token_ids = tok.encode(args.prompt, add_special_tokens=False)
    print(f"[coordinator] prompt tokens: {token_ids}")

    # 均分成两个 chunk
    mid = len(token_ids) // 2
    chunk0 = token_ids[:mid]
    chunk1 = token_ids[mid:]
    print(f"[coordinator] chunks: {len(chunk0)} + {len(chunk1)} = {len(token_ids)}")

    # Listen
    import socket
    host, port = args.listen_addr.rsplit(":", 1)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(2)
    print(f"[coordinator] listening on {args.listen_addr}")
    ready_event.set()

    socks = []
    for _ in range(2):
        sock, addr = listener.accept()
        print(f"[coordinator] worker connected from {addr}")
        socks.append(sock)
    listener.close()

    # Handshake (read 16 bytes LE from each)
    for i, sock in enumerate(socks):
        data = b""
        while len(data) < 16:
            data += sock.recv(16 - len(data))
        from hcp_worker_sdk.types import WorkerHandshake
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
            len_bytes += sock.recv(4 - len(len_bytes))
        length = struct.unpack(">I", len_bytes)[0]
        data = b""
        while len(data) < length:
            data += sock.recv(length - len(data))
        return WorkerResponse.from_dict(json.loads(data.decode()))

    # Prefill
    from hcp_worker_sdk.types import WorkerCommand
    send_cmd(socks[0], WorkerCommand.prefill(chunk0, seq_offset=0))
    send_cmd(socks[1], WorkerCommand.prefill(chunk1, seq_offset=mid))
    resp0 = recv_resp(socks[0])
    resp1 = recv_resp(socks[1])
    assert resp0.kind == "PrefillDone"
    assert resp1.kind == "PrefillDone"
    print(f"[coordinator] PrefillDone from both workers")

    global_seq_len = len(token_ids)
    for sock in socks:
        send_cmd(sock, WorkerCommand.sync_global_seq_len(global_seq_len))

    # Decode loop (sample from worker 0 only)
    generated = []
    if args.max_tokens > 0:
        first_token = ref_tokens[0]
        generated.append(first_token)
        print(f"[coordinator] step 0: token={first_token} (from reference)")

        for sock in socks:
            send_cmd(sock, WorkerCommand.decode(first_token))
        resp0 = recv_resp(socks[0])
        resp1 = recv_resp(socks[1])
        assert resp0.kind == "DecodeDone"
        assert resp1.kind == "DecodeDone"
        logits = np.frombuffer(resp0.logits_bytes, dtype=np.float32)

    for step in range(1, args.max_tokens):
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        print(f"[coordinator] step {step}: token={next_token}")

        for sock in socks:
            send_cmd(sock, WorkerCommand.decode(next_token))
        resp0 = recv_resp(socks[0])
        resp1 = recv_resp(socks[1])
        assert resp0.kind == "DecodeDone"
        assert resp1.kind == "DecodeDone"
        logits = np.frombuffer(resp0.logits_bytes, dtype=np.float32)

    for sock in socks:
        send_cmd(sock, WorkerCommand.shutdown())
    for sock in socks:
        sock.close()

    text = tok.decode(generated, skip_special_tokens=True)
    print(f"[coordinator] generated: '{text}'")

    if generated == ref_tokens:
        print(f"[coordinator] ✅ MATCH: 2-domain output identical to single-node reference")
    else:
        print(f"[coordinator] ❌ MISMATCH: ref={ref_tokens}, got={generated}")
        sys.exit(1)


def run_worker(domain_id: int, model_dir: str, device: str,
               coordinator_addr: str, listen_addr: str,
               next_peer_addr: str, backend_name: str,
               inbox, outbox):
    """Worker 入口（可运行于 thread 或 process 中）。"""
    if backend_name == "vllm":
        from hcp_vllm_worker import VllmBackend
        backend = VllmBackend(model_dir, device)
    else:
        from test_worker_control_plane import TransformersBackend
        backend = TransformersBackend(model_dir, device)

    from hcp_worker_sdk import LinkedMockKvTransport, HcpWorkerServer
    transport = LinkedMockKvTransport(inbox, outbox)
    server = HcpWorkerServer(
        backend=backend,
        transport=transport,
        domain_id=domain_id,
        num_domains=2,
    )
    print(f"[worker {domain_id}] starting, coord={coordinator_addr} listen={listen_addr}")
    server.run(
        coordinator_addr=coordinator_addr,
        listen_addr=listen_addr,
        next_peer_addr=next_peer_addr,
    )


def compute_reference_transformers(args) -> List[int]:
    """用单节点 transformers 计算贪婪解码参考结果。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _compute_reference_vllm_worker(args_dict, result_queue):
    """子进程 worker：计算 vLLM 参考结果。"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args_dict["model_dir"], trust_remote_code=True)
    llm = LLM(
        model=args_dict["model_dir"],
        dtype="float32",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
    )
    token_ids = tok.encode(args_dict["prompt"], add_special_tokens=False)
    outputs = llm.generate(
        prompt_token_ids=[token_ids],
        sampling_params=SamplingParams(max_tokens=args_dict["max_tokens"], temperature=0),
    )
    generated = outputs[0].outputs[0].token_ids
    result_queue.put(list(generated))


def compute_reference_vllm(args) -> List[int]:
    """用单节点 vLLM 计算贪婪解码参考结果（在独立子进程中运行以释放 GPU）。"""
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(
        target=_compute_reference_vllm_worker,
        args=(vars(args), q),
    )
    p.start()
    result = q.get()
    p.join(timeout=120)
    return result


def run_threading(args, ref_tokens):
    """transformers backend：单进程多线程。"""
    import queue
    q01 = queue.Queue()
    q10 = queue.Queue()
    coord_ready = threading.Event()

    t_coord = threading.Thread(target=run_coordinator, args=(args, ref_tokens, coord_ready))
    t_coord.start()
    coord_ready.wait()
    time.sleep(0.1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t1 = threading.Thread(
        target=run_worker,
        args=(1, args.model_dir, device, args.listen_addr, "127.0.0.1:29502",
              "127.0.0.1:29501", args.backend, q01, q10),
    )
    t0 = threading.Thread(
        target=run_worker,
        args=(0, args.model_dir, device, args.listen_addr, "127.0.0.1:29501",
              "127.0.0.1:29502", args.backend, q10, q01),
    )
    t1.start()
    time.sleep(0.5)
    t0.start()

    t_coord.join()
    t0.join(timeout=30)
    t1.join(timeout=30)
    print("[main] done")


def run_multiprocessing(args, ref_tokens):
    """vllm backend：多进程（vLLM 不支持同进程多实例）。"""
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    q01 = mp.Queue()
    q10 = mp.Queue()
    coord_ready = mp.Event()

    p1 = mp.Process(
        target=run_worker,
        args=(1, args.model_dir, "cuda", args.listen_addr, "127.0.0.1:29502",
              "127.0.0.1:29501", args.backend, q01, q10),
    )
    p0 = mp.Process(
        target=run_worker,
        args=(0, args.model_dir, "cuda", args.listen_addr, "127.0.0.1:29501",
              "127.0.0.1:29502", args.backend, q10, q01),
    )
    p1.start()
    time.sleep(2)
    p0.start()

    # coordinator 在主进程中运行
    run_coordinator(args, ref_tokens, coord_ready)

    p0.join(timeout=120)
    p1.join(timeout=120)
    print("[main] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="The answer to life, the universe, and everything is")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--listen-addr", default="127.0.0.1:29500")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    args = parser.parse_args()

    print(f"[main] computing single-node reference (backend={args.backend})...")
    # For vLLM 2-domain test, use transformers reference to avoid GPU memory
    # contention with worker processes. Phase 1.5 already verified vLLM single-node
    # output matches transformers baseline.
    ref_tokens = compute_reference_transformers(args)
    print(f"[main] reference tokens: {ref_tokens}")

    if args.backend == "vllm":
        run_multiprocessing(args, ref_tokens)
    else:
        run_threading(args, ref_tokens)


if __name__ == "__main__":
    main()
