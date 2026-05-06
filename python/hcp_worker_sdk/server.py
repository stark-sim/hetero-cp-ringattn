"""
HCP Worker SDK — 通用 Worker 服务器

框架开发者实现 HcpWorkerBackend 后，用此类启动事件循环即可。
"""

import socket
import struct
import json
import threading
from typing import Optional
from .types import WorkerCommand, WorkerResponse, WorkerHandshake
from .backend import HcpWorkerBackend
from .transport import KvTransport


class HcpWorkerServer:
    """
    HCP Worker 协议服务器。

    职责：
    1. 连接 Coordinator（TCP/QUIC）
    2. 发送 Handshake
    3. 循环接收 WorkerCommand，调用 backend 处理，返回 WorkerResponse
    4. 在 forward 过程中通过 KvTransport 交换 KV block
    """

    def __init__(
        self,
        backend: HcpWorkerBackend,
        transport: KvTransport,
        domain_id: int,
        num_domains: int,
    ):
        self.backend = backend
        self.transport = transport
        self.domain_id = domain_id
        self.num_domains = num_domains
        self.coord_sock: Optional[socket.socket] = None
        self.peer_sock: Optional[socket.socket] = None
        self.global_seq_len = 0
        self.seq_offset = 0

    def run(
        self,
        coordinator_addr: str,
        listen_addr: str,
        next_peer_addr: str,
    ) -> None:
        """
        启动 Worker 主循环。

        Args:
            coordinator_addr: "host:port"，coordinator 监听地址
            listen_addr: "host:port"，本 worker 监听 peer 连接的地址
            next_peer_addr: "host:port"，下一个 peer 的地址
        """
        # 1. 建立 peer 连接（先 listen 或先 connect，取决于 domain_id）
        self._setup_peer_connections(listen_addr, next_peer_addr)

        # 2. 连接 coordinator
        self._connect_coordinator(coordinator_addr)

        # 3. 发送 handshake
        handshake = WorkerHandshake(
            domain_id=self.domain_id,
            capacity_mb=self.backend.capacity_mb,
        )
        self._send_handshake(handshake)
        print(f"[worker {self.domain_id}] handshake sent, capacity={handshake.capacity_mb} MB")

        # 4. 命令循环
        while True:
            cmd = self._recv_command()
            print(f"[worker {self.domain_id}] received command: {cmd.kind}")

            if cmd.kind == "Prefill":
                resp = self._handle_prefill(cmd)
            elif cmd.kind == "Decode":
                resp = self._handle_decode(cmd)
            elif cmd.kind == "SyncGlobalSeqLen":
                self.global_seq_len = cmd.global_seq_len or 0
                print(f"[worker {self.domain_id}] synced global_seq_len = {self.global_seq_len}")
                continue  # no response needed for sync
            elif cmd.kind == "Shutdown":
                print(f"[worker {self.domain_id}] shutting down")
                break
            else:
                resp = WorkerResponse.error(f"unknown command: {cmd.kind}")

            if resp is not None:
                self._send_response(resp)

    def _handle_prefill(self, cmd: WorkerCommand) -> WorkerResponse:
        """处理 Prefill 命令。"""
        chunk = cmd.chunk or []
        self.seq_offset = cmd.seq_offset or 0

        # 执行框架 prefill
        logits, seq_len = self.backend.prefill(chunk, self.seq_offset)
        self.global_seq_len = seq_len

        # KV Ring 交换（prefill 阶段需要交换所有层的 KV）
        self._exchange_kv_ring(prefill=True)

        return WorkerResponse.prefill_done(logits, self.global_seq_len)

    def _handle_decode(self, cmd: WorkerCommand) -> WorkerResponse:
        """处理 Decode 命令。"""
        token = cmd.token or 0

        # 执行框架 decode
        logits = self.backend.decode(token)

        # KV Ring 交换（decode 阶段只交换 prefill 历史 KV）
        self._exchange_kv_ring(prefill=False)

        return WorkerResponse.decode_done(logits)

    def _exchange_kv_ring(self, prefill: bool) -> None:
        """
        执行 KV Ring 交换。

        对每一层：
        1. 提取本 domain 的 KV block
        2. 通过 transport 与 peer 交换
        3. 将收到的 peer KV 合并到当前层
        """
        for layer_idx in range(self.backend.num_layers):
            seq_start = self.seq_offset
            seq_end = self.global_seq_len if prefill else self.seq_offset + self.backend.num_layers
            # 注意：decode 阶段 seq_end 需要特殊处理，这里简化

            local_block = self.backend.get_kv_block(layer_idx, seq_start, seq_end)

            for _round in range(self.num_domains - 1):
                peer_block = self.transport.exchange_kv_block(local_block)
                if peer_block is None:
                    break
                self.backend.apply_peer_kv(layer_idx, peer_block)
                local_block = peer_block  # 转发给下一个 peer

    def _setup_peer_connections(self, listen_addr: str, next_peer_addr: str) -> None:
        """建立 peer 连接（简化版 TCP）。"""
        host, port = listen_addr.rsplit(":", 1)
        listen_port = int(port)

        next_host, next_port = next_peer_addr.rsplit(":", 1)
        next_port = int(next_port)

        if self.domain_id == 0:
            # domain 0 先 connect 到 next_peer
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((next_host, next_port))
            self.peer_sock = sock
            print(f"[worker {self.domain_id}] connected to peer {next_peer_addr}")
        else:
            # domain 1 先 listen
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, listen_port))
            listener.listen(1)
            print(f"[worker {self.domain_id}] listening for peer on {listen_addr}")
            sock, addr = listener.accept()
            self.peer_sock = sock
            print(f"[worker {self.domain_id}] accepted peer connection from {addr}")
            listener.close()

    def _connect_coordinator(self, coordinator_addr: str) -> None:
        """连接 coordinator。"""
        host, port = coordinator_addr.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
        self.coord_sock = sock
        print(f"[worker {self.domain_id}] connected to coordinator {coordinator_addr}")

    def _send_handshake(self, handshake: WorkerHandshake) -> None:
        assert self.coord_sock is not None
        self.coord_sock.sendall(handshake.to_bytes())

    def _recv_command(self) -> WorkerCommand:
        assert self.coord_sock is not None
        # Length-prefixed JSON
        len_bytes = self._recvall(self.coord_sock, 4)
        length = struct.unpack(">I", len_bytes)[0]
        data = self._recvall(self.coord_sock, length)
        return WorkerCommand.from_dict(json.loads(data.decode()))

    def _send_response(self, resp: WorkerResponse) -> None:
        assert self.coord_sock is not None
        data = json.dumps(resp.to_dict()).encode()
        frame = struct.pack(">I", len(data)) + data
        self.coord_sock.sendall(frame)

    @staticmethod
    def _recvall(sock: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("recvall: connection closed")
            data += chunk
        return data
