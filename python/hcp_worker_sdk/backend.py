"""
HCP Worker SDK — 后端抽象接口

框架适配器（vLLM、TensorRT-LLM、MLX 等）必须实现此接口。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from .types import KvBlock


class HcpWorkerBackend(ABC):
    """
    HCP Worker 后端抽象。

    实现者只需关注：
    1. 如何加载模型权重
    2. 如何执行 prefill / decode forward
    3. 如何从框架 KV cache 提取 KvBlock
    4. 如何将 peer KvBlock 合并到当前状态

    协议层（序列化、网络、事件循环）由 HcpWorkerServer 处理。
    """

    @abstractmethod
    def load_model(self, model_dir: str, device: str) -> None:
        """加载模型权重到指定设备。"""
        pass

    @abstractmethod
    def prefill(self, chunk: List[int], seq_offset: int) -> Tuple[torch.Tensor, int]:
        """
        执行 prefill forward。

        Args:
            chunk: token ID 列表，本 domain 负责的 prompt 分片
            seq_offset: 本 chunk 在全局序列中的起始位置

        Returns:
            last_token_logits: [vocab_size] 的 float32 tensor
            global_seq_len: 当前全局序列总长度（通常等于 len(chunk) + seq_offset）
        """
        pass

    @abstractmethod
    def decode(self, token: int) -> torch.Tensor:
        """
        执行单 token decode forward。

        Args:
            token: 当前要解码的 token ID

        Returns:
            logits: [vocab_size] 的 float32 tensor
        """
        pass

    @abstractmethod
    def get_kv_block(self, layer_idx: int, seq_start: int, seq_end: int) -> KvBlock:
        """
        从框架 KV cache 提取指定层的 K/V block。

        Args:
            layer_idx: 层索引
            seq_start: 全局序列起始位置
            seq_end: 全局序列结束位置

        Returns:
            KvBlock: 包含 K [batch, heads, seq, dim] 和 V
        """
        pass

    @abstractmethod
    def apply_peer_kv(self, layer_idx: int, peer_block: KvBlock) -> None:
        """
        将 peer KV block 的注意力贡献合并到当前层。

        这是 Ring Attention 的核心：用 online softmax 将 peer KV 的
        attention 输出增量合并到当前状态。

        Args:
            layer_idx: 当前层索引
            peer_block: 从 peer worker 收到的 KvBlock
        """
        pass

    @property
    @abstractmethod
    def capacity_mb(self) -> int:
        """
        上报本节点的可用计算资源（显存或内存），单位 MB。

        Coordinator 用此信息做 capacity-aware 分片。
        """
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """模型层数。"""
        pass

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Attention head 数。"""
        pass

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """每个 head 的维度。"""
        pass
