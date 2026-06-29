import sqlite3
import re
from datetime import datetime
from pathlib import Path

DB = Path('graph-memory/graph.db')
conn = sqlite3.connect(DB)
c = conn.cursor()

def node_id(prefix, slug):
    return f'{prefix}-{slug}'

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.8, status='held', source=None):
    c.execute('''
        INSERT OR REPLACE INTO nodes
        (id, type, layer, project, title, content, importance, confidence, status, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ''', (id_, type_, layer, 'hetero-cp-ringattn', title, content, importance, confidence, status, source))

def insert_edge(src, tgt, type_, weight=1.0, note=None):
    c.execute('''
        INSERT OR REPLACE INTO edges (source, target, type, weight, note)
        VALUES (?, ?, ?, ?, ?)
    ''', (src, tgt, type_, weight, note))

PROJECT = node_id('proj', 'hetero-cp-ringattn')

# ---------- blueprint layer ----------
insert_node(
    PROJECT, 'project', 'blueprint',
    'HCP Ring Attention Repo',
    '独立研究仓，聚焦 HCP（Heterogeneous Context Parallelism）在超长 context 场景下的 intra-layer / low-boundary Ring Attention 路线。'
    '愿景：当 context 从 200k 走向 1M 乃至 10M 时，允许多个异构 domain 以不均分方式共同承担同一个 attention layer。',
    importance=1.0, confidence=1.0, source='memory-bank/projectbrief.md'
)

insert_node(
    node_id('bp', 'problem'), 'blueprint', 'blueprint',
    '产品问题：异构设备协作支撑超长 context',
    '长上下文需求持续增长，但单卡显存和同构高端集群供给无法无限增长。'
    '现实资源通常是混合的（CUDA、Apple Silicon/MLX、其他加速器）。'
    'HCP 的问题是：能否通过增加异构 domain / 设备继续支撑任务，而不是受制于最强单卡。',
    importance=0.9, confidence=0.85, source='memory-bank/productContext.md'
)
insert_edge(node_id('bp', 'problem'), PROJECT, 'PART_OF')

insert_node(
    node_id('bp', 'product-decisions'), 'decision', 'blueprint',
    '产品决策：P2P、correctness 优先、结构化实验产物',
    'HCP 不是 HLPP 的细粒度版本，而是 intra-layer / low-boundary 路线。'
    '跨异构域坚持 P2P，不把 all-gather / reduce-scatter / all-to-all / all-reduce 作为主假设。'
    'correctness 和协议闭环优先于性能图。每个阶段输出结构化实验产物。',
    importance=0.85, confidence=0.85, source='memory-bank/productContext.md'
)
insert_edge(node_id('bp', 'product-decisions'), PROJECT, 'PART_OF')

insert_node(
    node_id('bp', 'success-criteria'), 'fact', 'blueprint',
    '成功标准：online softmax 对齐 + RingAttnMessage 可传输 + remote heterogeneous smoke',
    'online softmax 在不均分 seq_chunk_len / block_size 下与 reference attention 对齐。'
    'RingAttnMessage 可以稳定编码、传输、解码。'
    '2-domain remote heterogeneous smoke 可复现，并产出 correctness、transport、failure summary。',
    importance=0.8, confidence=0.8, source='memory-bank/productContext.md'
)
insert_edge(node_id('bp', 'success-criteria'), PROJECT, 'PART_OF')

insert_node(
    node_id('bp', 'tech-stack'), 'component', 'blueprint',
    '技术栈：Rust + C++ + Python 原型',
    'Core: C++17, CMake 3.16+, Rust 2021, Python 3。\n'
    'Libtorch/PyTorch 2.11.0, tch-rs 0.24.0（可选 tch-backend）。\n'
    'QUIC: quinn 0.11 + rustls 0.23 + rcgen 0.13。\n'
    '模型权重：safetensors, tokenizers, half。',
    importance=0.85, confidence=0.9, source='memory-bank/techContext.md'
)
insert_edge(node_id('bp', 'tech-stack'), PROJECT, 'PART_OF')

insert_node(
    node_id('bp', 'quic-config'), 'decision', 'blueprint',
    'QUIC Transport 配置：512MB stream window / 1GB connection window / 300s idle timeout',
    '显式覆盖 quinn 默认值：max_concurrent_bidi/uni_streams=256, keep_alive_interval=1s, max_idle_timeout=300s, '
    'stream_receive_window=512MB, receive_window=1GB, send_window=1GB。'
    '历史上因 send_window 和 stream_receive_window 不足导致 16K/64K 死锁。',
    importance=0.8, confidence=0.9, source='memory-bank/techContext.md'
)
insert_edge(node_id('bp', 'quic-config'), node_id('bp', 'tech-stack'), 'PART_OF')

insert_node(
    node_id('bp', 'deployment-rule'), 'decision', 'blueprint',
    '部署铁律：1 GPU = 1 worker',
    '每个 worker 加载完整模型权重。3B bf16 × 2 workers 在 RTX 4090 loopback 上实测 OOM。'
    '--local-domain-ids 仅限 <1GB 小模型的本地协议验证；生产/大规模验证必须每平台一 worker。',
    importance=0.85, confidence=0.9, source='memory-bank/techContext.md'
)
insert_edge(node_id('bp', 'deployment-rule'), PROJECT, 'PART_OF')

insert_node(
    node_id('bp', 'arch-overview'), 'blueprint', 'blueprint',
    '架构概览：Rust + C++ 为主、Python 原型为历史对照',
    'C++ 部分定义 HCP Ring Attention 低边界 runtime 抽象和 libtorch bridge。'
    'Rust 部分负责 correctness model、report、可序列化协议 schema 和 P2P transport smoke。'
    '每个 domain 持有本地 Q chunk，ring 中持续传递 K/V block，每个 domain 更新 online softmax state。',
    importance=0.9, confidence=0.9, source='memory-bank/systemPatterns.md'
)
insert_edge(node_id('bp', 'arch-overview'), PROJECT, 'PART_OF')

insert_node(
    node_id('bp', 'p2p-decision'), 'decision', 'blueprint',
    '架构决策：采用原始论文 P2P 而非 PyTorch CP Collective',
    'Ring Attention 原始论文（Liu et al. 2023）的通信本就是 P2P send/recv。'
    'PyTorch 2.7+ Context Parallel 改用 all-gather/all-to-all 是对同构 NVLink 集群的工程优化，不是数学必须。'
    'P2P 支持异构、非均分、任意拓扑，更符合 HCP 定位。',
    importance=0.9, confidence=0.9, source='memory-bank/systemPatterns.md'
)
insert_edge(node_id('bp', 'p2p-decision'), node_id('bp', 'arch-overview'), 'PART_OF')

insert_node(
    node_id('bp', 'correctness-first'), 'decision', 'blueprint',
    'Correctness-First 开发纪律',
    '当前处于 correctness 验证阶段，尚未进入性能调优。'
    '在全部 target 设备上稳定通过前，禁止实施量化、近似 attention、非 deterministic kernel、投机/跳过层优化。'
    '每次提出优化前必须写 trade-off 分析。',
    importance=0.85, confidence=0.85, source='memory-bank/systemPatterns.md'
)
insert_edge(node_id('bp', 'correctness-first'), PROJECT, 'PART_OF')

insert_node(
    node_id('bp', 'heterogeneous-validation'), 'belief', 'blueprint',
    '异构分布式推理的数值验证策略',
    'BF16 场景下，跨平台 BLAS 差异导致 logits 数值对比不是有意义的 correctness 指标。'
    'Correctness 应分层：L1 float32 数学正确性（cargo test synthetic weights）、L2 工程正确性（argmax 一致性/文本任务指标）、L3 端到端冒烟。'
    '强证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异，证明差异主要来自 BF16 online softmax block-wise 处理顺序，而非跨平台 bug。',
    importance=0.85, confidence=0.8, source='memory-bank/systemPatterns.md'
)
insert_edge(node_id('bp', 'heterogeneous-validation'), node_id('bp', 'correctness-first'), 'PART_OF')

insert_node(
    node_id('bp', 'plugin-architecture'), 'decision', 'blueprint',
    '可插拔域内后端架构',
    'HCP 的边界是跨域低层协议（P2P KV ring + online softmax），域内实现是黑盒。'
    '同构域内可通过接口实现替换为 vLLM、TensorRT-LLM、MLX 等社区框架。'
    'Python Worker SDK 和 Rust Worker SDK 提供标准接口。',
    importance=0.8, confidence=0.8, source='memory-bank/systemPatterns.md'
)
insert_edge(node_id('bp', 'plugin-architecture'), node_id('bp', 'arch-overview'), 'PART_OF')

insert_node(
    node_id('bp', 'uneven-cp'), 'decision', 'blueprint',
    '容量感知非均等 CP 分片是异构长 context 的必需',
    '2026-06-19 1M context 验证：24GB CUDA + 16GB HIP 无法通过 1:1 分片完成 1M。'
    '必须使用 capacity-aware 不均等分片（white 750K / pearl 250K，即 3:1）。'
    '均匀分片在异构显存下会因小显存设备 OOM 而失败；按可用显存比例分配 chunk 才能使 heterogeneous ring 达到可行性边界。',
    importance=0.9, confidence=0.9, source='memory-bank/systemPatterns.md'
)
insert_edge(node_id('bp', 'uneven-cp'), node_id('bp', 'arch-overview'), 'PART_OF')

# ---------- progress layer ----------
insert_node(
    node_id('prog', '1m-white-pearl'), 'session', 'progress',
    '[2026-06-19] 1M context 本地异构分布式推理成功',
    'white RTX 4090 CUDA + pearl RX 9060 XT HIP，2.5G 有线直连。'
    'Qwen2-0.5B-1M，capacity-aware 3:1 分片（white 750K / pearl 250K）。'
    'Prefill 24/24 全通，decode 5 tokens 全通，exit=0。'
    '总耗时 ~2h 11m，white 显存峰值 23,999 MB。'
    '攻克：KV channel buffer 512、QUIC timeout 14400s、max_position_embeddings=1048576 patch、pearl 碎片化 OOM 通过 3:1 分片缓解。',
    importance=1.0, confidence=1.0, status='closed', source='memory-bank/progress.md'
)
insert_edge(node_id('prog', '1m-white-pearl'), PROJECT, 'PART_OF')
insert_edge(node_id('prog', '1m-white-pearl'), node_id('bp', 'uneven-cp'), 'SUPPORTS', weight=0.9,
            note='1M 端到端成功验证 capacity-aware uneven CP 的必要性')

insert_node(
    node_id('prog', 'npu-poc'), 'session', 'progress',
    '[2026-06-17] 昇腾 910B NPU 控制面 E2E 打通',
    '单机 1× Ascend 910B4 (32 GB HBM) 上完成 Python vLLM worker ↔ Rust coordinator 控制面 E2E。'
    'Rust coordinator 脱离 libtorch feature 可编译运行，纯 Rust 采样替代 tch::Tensor。'
    'Coordinator 输出 generated: ! I\'m。',
    importance=0.75, confidence=1.0, status='closed', source='memory-bank/progress.md'
)
insert_edge(node_id('prog', 'npu-poc'), PROJECT, 'PART_OF')

# ---------- active layer ----------
insert_node(
    node_id('active', 'focus'), 'task', 'active',
    '当前焦点：1M 异构分布式推理已闭环',
    '1M v9（3:1 split）成功，prefill 24/24 + decode 5/5，exit=0。'
    '文档已同步：1M_CONTEXT_THUNDERBOLT_PLAN.md、SCALING_ARGUMENT.md、systemPatterns.md。'
    '当前无未完成的 1M 攻坚任务；下一步决定是否需要更大模型 / 更多 domain 验证。',
    importance=0.95, confidence=0.95, status='ongoing', source='memory-bank/activeContext.md'
)
insert_edge(node_id('active', 'focus'), PROJECT, 'PART_OF')
insert_edge(node_id('active', 'focus'), node_id('prog', '1m-white-pearl'), 'BASED_ON')

insert_node(
    node_id('active', 'next-decision'), 'uncertainty', 'active',
    '下一步决策：更大模型 / 更多 domain？',
    '1M 里程碑已达成，需决定后续方向：'
    '1) 引入第三台设备做 3-domain 1M 降低单设备压力；'
    '2) 7B 1M context 可行性评估；'
    '3) KV cache 量化/压缩以缩短 decode 时间和降低显存占用。',
    importance=0.8, confidence=0.5, status='open', source='memory-bank/activeContext.md'
)
insert_edge(node_id('active', 'next-decision'), node_id('active', 'focus'), 'QUESTIONS', weight=0.5)

# ---------- evidence links ----------
insert_node(
    node_id('ev', '1m-oom-even'), 'evidence', 'progress',
    '证据：1:1/2:1/3:2 split 均导致 pearl OOM',
    '在 1M context 尝试中，均分 500K 及 2:1、3:2 split 均在 layer 23/24 因 pearl 16GB 显存分配失败而 OOM。'
    '只有 3:1 split 成功。',
    importance=0.85, confidence=0.9, source='memory-bank/progress.md'
)
insert_edge(node_id('ev', '1m-oom-even'), node_id('bp', 'uneven-cp'), 'CONFIRMS', weight=0.9)

insert_node(
    node_id('ev', 'blas-diff'), 'evidence', 'progress',
    '证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异',
    'White CUDA loopback 双 domain 3B max_diff=0.406，0.5B max_diff=0.344，argmax=10/10。'
    '跨平台单节点 0.438，异构分布式 0.484。证明跨平台 BLAS 仅贡献 ~0.1 额外差异，不是 logits 差异主导因素。',
    importance=0.8, confidence=0.85, source='memory-bank/systemPatterns.md'
)
insert_edge(node_id('ev', 'blas-diff'), node_id('bp', 'heterogeneous-validation'), 'CONFIRMS', weight=0.8)

conn.commit()
conn.close()
print('Migration complete')
