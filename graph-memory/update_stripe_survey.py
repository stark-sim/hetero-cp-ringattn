import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source=None):
    c.execute('''
        INSERT OR REPLACE INTO nodes
        (id, type, layer, project, title, content, importance, confidence, status, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ''', (id_, type_, layer, PROJECT, title, content, importance, confidence, status, source))

def insert_edge(src, tgt, type_, weight=1.0, note=None):
    c.execute('''
        INSERT OR REPLACE INTO edges (source, target, type, weight, note)
        VALUES (?, ?, ?, ?, ?)
    ''', (src, tgt, type_, weight, note))

# Evidence: Stripe Attention paper
insert_node(
    'ev-stripe-attention-paper', 'evidence', 'active',
    '[论文] Striped Attention: Faster Ring Attention for Causal Transformers',
    '作者：William Brandon 等 (MIT)，arXiv:2311.09431，2023。\n'
    '核心发现：因果 attention 的三角结构导致 Ring Attention 工作负载不均。\n'
    '方案：每个 device 持有均匀分布在整个序列上的 token 子集（striped permutation），而非连续 chunk。\n'
    '效果：A100 256K 序列上端到端吞吐提升最高 1.45×；16×TPUv4 786K 序列上 1.65×。\n'
    '实现复杂度：只需在 forward 开始前对输入序列做一次 permutation，并调整 attention mask 结构。\n'
    '与 HCP 相关性：直接相关，可能缓解 pearl 等小/慢 domain 在 Phase 2 成为瓶颈的问题。',
    importance=0.9, confidence=0.85, status='held', source='https://arxiv.org/abs/2311.09431'
)
insert_edge('ev-stripe-attention-paper', 'hyp-stripe-ring', 'CONFIRMS', weight=0.9,
            note='MIT 论文验证 striped permutation 可平衡因果 ring attention 负载')
insert_edge('ev-stripe-attention-paper', 'bp-arch-overview', 'RELATES_TO', weight=0.7,
            note='实现需兼容 HCP P2P + online softmax')

# Evidence: Ring Attention original paper
insert_node(
    'ev-ring-attention-paper', 'evidence', 'active',
    '[论文] Ring Attention with Blockwise Transformers for Near-Infinite Context',
    '作者：Hao Liu, Matei Zaharia, Pieter Abbeel (UC Berkeley)，arXiv:2310.01889，ICLR 2024。\n'
    '提出 blockwise attention + online softmax，使 self-attention 计算可分布到多个设备；KV block 沿 ring 传递。\n'
    'HCP 的数学基础即来源于此。',
    importance=0.85, confidence=0.9, status='held', source='https://arxiv.org/abs/2310.01889'
)
insert_edge('ev-ring-attention-paper', 'bp-p2p-decision', 'BASED_ON', weight=0.9)

# Evidence / claim: other mainstream derivatives
insert_node(
    'claim-ring-derivatives', 'claim', 'active',
    'Ring Attention 主流衍生方案综述',
    '除原始 Ring Attention 与 Striped Attention 外，当前主流/相关方案包括：\n'
    '- Ring Flash Attention（zhuzilin 等开源）：将 FlashAttention kernel 与 Ring 通信重叠。\n'
    '- ZigZag Ring Attention（ring-flash-attention issue #2 / Megatron-Core）：通过折叠 query 维度并在 worker 间镜像 block 平衡负载。\n'
    '- DeepSpeed Ulysses（Microsoft, arXiv:2309.14509）：用 all-to-all 替代 all-gather/reduce-scatter 的序列并行，聚焦同构集群通信效率。\n'
    '- USP（Tencent, arXiv:2405.07719）：统一 Ulysses + Ring Attention 的序列并行框架。\n'
    '- Context Parallelism for Scalable Million-Token Inference（arXiv:2411.01783）：面向推理的 context parallelism。\n'
    '- MoBA (Mixture of Block Attention, arXiv:2502.13189)：块级稀疏 attention，可与 ring 结合。\n'
    '- XAttention (arXiv:2502.xxxxx)：block sparse attention with antidiagonal scoring。\n'
    '- MTraining (arXiv:2510.18830)：基于 Striped Ring Attention 的动态稀疏 attention 训练系统。\n'
    '- Mnemosyne (arXiv:2409.17264)：多百万 token 推理服务系统，讨论 Ring/Striped 在推理中的 head-of-line blocking 与 batching 局限。',
    importance=0.85, confidence=0.75, status='held', source='web-search-survey'
)
insert_edge('claim-ring-derivatives', 'hyp-stripe-ring', 'RELATES_TO', weight=0.8,
            note='striped 是其中最贴近 HCP 的衍生之一')
insert_edge('claim-ring-derivatives', 'active-next-lines', 'PART_OF')

# Task: start with stripe survey
insert_node(
    'task-stripe-survey', 'task', 'active',
    '先做：Stripe Attention 调研与 HCP 适配性分析',
    '具体步骤：\n'
    '1. 精读 Striped Attention 论文（arXiv:2311.09431），提取 permutation + mask 调整细节。\n'
    '2. 在 HCP Rust correctness model 中实现 striped permutation 原型，验证与同构/异构 uneven chunk 的兼容性。\n'
    '3. 对比 vanilla ring vs striped 在因果 attention 下的 per-domain 计算量（理论 + 模拟）。\n'
    '4. 输出设计文档：如何在 capacity-aware 不均等分片下应用 stripe（或不适用）。',
    importance=0.9, confidence=0.75, status='ongoing', source='user-direction'
)
insert_edge('task-stripe-survey', 'active-next-lines', 'PART_OF')
insert_edge('task-stripe-survey', 'hyp-stripe-ring', 'LEADS_TO', weight=0.9)
insert_edge('task-stripe-survey', 'ev-stripe-attention-paper', 'BASED_ON', weight=0.9)

# Mark network speed and vLLM hypotheses as still open but secondary for now
c.execute("UPDATE nodes SET status='open', updated_at=datetime('now') WHERE id IN ('hyp-net-speed', 'hyp-block-kv-vllm')")

conn.commit()
conn.close()
print('Stripe survey nodes added')
