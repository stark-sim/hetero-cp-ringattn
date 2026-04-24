#!/usr/bin/env python3
"""
Ring Attention kernel 占位实现。
用于在不依赖真实 CUDA/MLX/NPU kernel 的情况下，验证 online softmax 的数值正确性。
"""

import numpy as np


def online_softmax_update(
    o_prev: np.ndarray,
    l_prev: np.ndarray,
    m_prev: np.ndarray,
    q_chunk: np.ndarray,
    k_block: np.ndarray,
    v_block: np.ndarray,
) -> tuple:
    """
    执行一个 block 的 online softmax 更新。
    Args:
        o_prev: [chunk_len, num_heads, head_dim]
        l_prev: [num_heads, chunk_len]
        m_prev: [num_heads, chunk_len]
        q_chunk: [chunk_len, num_heads, head_dim]
        k_block: [block_len, num_heads, head_dim]
        v_block: [block_len, num_heads, head_dim]
    Returns:
        (o_new, l_new, m_new)
    """
    chunk_len, num_heads, head_dim = q_chunk.shape
    # scores: [chunk_len, num_heads, block_len]
    scores = np.einsum("qhd,khd->qhk", q_chunk, k_block)
    m_local = np.max(scores, axis=-1)  # [chunk_len, num_heads]
    m_local = m_local.transpose(1, 0)   # [num_heads, chunk_len]

    m_new = np.maximum(m_prev, m_local)
    exp_prev = np.exp(m_prev - m_new)
    exp_local = np.exp(m_local - m_new)

    p_local = np.exp(scores - m_local[:, None, :])  # [chunk_len, num_heads, block_len]
    l_local = np.sum(p_local, axis=-1)               # [chunk_len, num_heads]
    l_local = l_local.transpose(1, 0)                # [num_heads, chunk_len]

    l_new = exp_prev * l_prev + exp_local * l_local

    # pv: [chunk_len, num_heads, head_dim]
    pv = np.einsum("qhk,khd->qhd", p_local, v_block)
    pv = pv.transpose(1, 0, 2)  # [num_heads, chunk_len, head_dim]
    o_prev_t = o_prev.transpose(1, 0, 2)  # [num_heads, chunk_len, head_dim]

    o_new_t = (exp_prev[..., None] * o_prev_t + exp_local[..., None] * pv) / l_new[..., None]
    o_new = o_new_t.transpose(1, 0, 2)  # [chunk_len, num_heads, head_dim]

    return o_new, l_new, m_new


def ring_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    标准 Attention 的参考实现，用于对比 ring attention 结果。
    """
    scores = np.einsum("qhd,khd->qhk", q, k)
    p = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    p = p / np.sum(p, axis=-1, keepdims=True)
    out = np.einsum("qhk,khd->qhd", p, v)
    return out
