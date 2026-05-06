"""
HCP Worker SDK — Python 版

让框架开发者（vLLM、TensorRT-LLM、MLX 等）只需实现 HcpWorkerBackend，
即可接入 HCP 分布式推理网络。

用法:
    from hcp_worker_sdk import HcpWorkerBackend, KvBlock, HcpWorkerServer

    class MyBackend(HcpWorkerBackend):
        def prefill(self, chunk, seq_offset):
            ...

    server = HcpWorkerServer(MyBackend(), transport)
    server.run(coordinator_addr="10.0.0.1:29450", listen_addr="0.0.0.0:29451")
"""

from .types import KvBlock, WorkerCommand, WorkerResponse, WorkerHandshake
from .backend import HcpWorkerBackend
from .transport import KvTransport, TcpKvTransport, NoOpKvTransport, LinkedMockKvTransport
from .server import HcpWorkerServer

__all__ = [
    "KvBlock",
    "WorkerCommand",
    "WorkerResponse",
    "WorkerHandshake",
    "HcpWorkerBackend",
    "KvTransport",
    "TcpKvTransport",
    "NoOpKvTransport",
    "LinkedMockKvTransport",
    "HcpWorkerServer",
]
