import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source='user-input'):
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

# Reflection node: scope of 1M milestone
insert_node(
    'belief-1m-scope', 'belief', 'active',
    '1M white+pearl 是可行性里程碑，而非生产实用配置',
    '在 16GB + 24GB 两台消费级机器上跑通 1M context 证明了 HCP 异构不均等 CP 的可行性边界，'
    '但 decode 每 token ~3 分钟、white 显存几乎满载，距离实际生产部署仍有显著差距。'
    '其价值在于验证架构路径，而非直接作为产品配置。',
    importance=0.85, confidence=0.9, status='held', source='user-reflection'
)
insert_edge('belief-1m-scope', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('belief-1m-scope', 'prog-1m-white-pearl', 'BASED_ON', weight=0.9,
            note='1M 成功证明可行性，但暴露了性能与成本问题')

# Next phase umbrella task
insert_node(
    'active-next-lines', 'task', 'active',
    '下一阶段：从 1M 可行性验证走向多条扩展线探索',
    '1M 只是众多验证线中的一条。接下来需要并行探索：\n'
    '1. 网络速度对异构 CP 收益的影响（CXL / 类 RDMA 方向）。\n'
    '2. Stripe Ring Attention 等算法升级在 HCP 框架中的适用性。\n'
    '3. Block KV cache + vLLM 集成：插件解耦 vs HCP 内联 PageAttention 两条路线。',
    importance=0.95, confidence=0.8, status='ongoing', source='user-direction'
)
insert_edge('active-next-lines', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('active-next-lines', 'belief-1m-scope', 'BASED_ON', weight=0.9)

# Hypothesis 1: network speed / CXL
insert_node(
    'hyp-net-speed', 'hypothesis', 'active',
    '异构 CP 对网络速度敏感，CXL / 类 RDMA 互联可显著突破网线局限',
    '当前 2.5G 有线以太网在 1M context 下不是带宽瓶颈（prefill 受显存与 memory-bound compute 主导），'
    '但随着 chunk 缩小、domain 增多或模型变大，KV ring 的通信量会快速上升。'
    '需要系统测试不同 RTT/带宽（WiFi、2.5G、10G、RDMA、CXL）对 prefill/decode 的边际收益，'
    '论证在异构节点上投资高速互联（CXL / GPU Direct / RDMA）能否取得与增加显存同量级的回报。',
    importance=0.85, confidence=0.6, status='open', source='user-direction'
)
insert_edge('hyp-net-speed', 'active-next-lines', 'PART_OF')
insert_edge('hyp-net-speed', 'bp-p2p-decision', 'DEPENDS_ON', weight=0.8,
            note='P2P ring 对点对点延迟/带宽敏感')
insert_edge('hyp-net-speed', 'bp-quic-config', 'RELATES_TO', weight=0.5,
            note='QUIC window 配置与网络能力直接相关')

# Hypothesis 2: stripe ring attention
insert_node(
    'hyp-stripe-ring', 'hypothesis', 'active',
    'Stripe Ring Attention 可适配 HCP 并改善异构负载均衡',
    '传统 Ring Attention 按 chunk 顺序遍历，小显存 domain 在 Phase 2 接收更多 remote block，容易成为瓶颈。'
    'Stripe Ring Attention 通过更细粒度或交错式的 KV block 调度，把负载分配得更均匀。'
    '需要评估其是否兼容 HCP 的 P2P / online-softmax / 非均等 chunk 设计，以及能否缓解 pearl 类慢节点的瓶颈。',
    importance=0.8, confidence=0.55, status='open', source='user-direction'
)
insert_edge('hyp-stripe-ring', 'active-next-lines', 'PART_OF')
insert_edge('hyp-stripe-ring', 'bp-uneven-cp', 'RELATES_TO', weight=0.7,
            note='两者都试图解决异构负载不均')
insert_edge('hyp-stripe-ring', 'bp-arch-overview', 'DEPENDS_ON', weight=0.8,
            note='必须兼容 HCP ring + online softmax 语义')

# Hypothesis 3: block KV cache + vLLM
insert_node(
    'hyp-block-kv-vllm', 'hypothesis', 'active',
    'Block KV cache + vLLM 集成：插件解耦 vs HCP 内联 PageAttention 双路线',
    '当前 HCP 主要关注整段 KV cache 的 P2P 传输。下一步探索与 vLLM 生态结合：\n'
    '路线 A（插件解耦）：HCP 作为 vLLM 外部的 context-parallel 插件，通过标准接口交换 block-level KV，保持 vLLM 内部完整。\n'
    '路线 B（HCP 为主 + 内联 PageAttention）：HCP 自身管理 page/block 粒度的 KV，内联 PageAttention 的 scheduling/block 机制，深度整合以获得最佳性能。\n'
    '需要并行验证两条路线的工程可行性、correctness 风险和对 vLLM 版本升级的耦合度。',
    importance=0.9, confidence=0.5, status='open', source='user-direction'
)
insert_edge('hyp-block-kv-vllm', 'active-next-lines', 'PART_OF')
insert_edge('hyp-block-kv-vllm', 'bp-plugin-architecture', 'DEPENDS_ON', weight=0.9,
            note='插件解耦路线直接依赖可插拔后端设计')
insert_edge('hyp-block-kv-vllm', 'bp-uneven-cp', 'RELATES_TO', weight=0.6,
            note='block 粒度分片可能与 capacity-aware 分片互补')

# Close / supersede the previous active focus
insert_edge('belief-1m-scope', 'active-focus', 'UPDATES', weight=0.8,
            note='重新定义 1M 成功的意义，并引出下阶段多线探索')

c.execute("UPDATE nodes SET status = 'closed', updated_at = datetime('now') WHERE id = 'active-focus'")

c.execute("""
    UPDATE nodes SET status = 'superseded', updated_at = datetime('now')
    WHERE id = 'active-focus' AND status = 'closed'
""")
# Actually keep it closed; superseded status is also fine. Let's set superseded and add replaced_by.
c.execute("""
    UPDATE nodes SET status = 'superseded', replaced_by = 'active-next-lines', updated_at = datetime('now')
    WHERE id = 'active-focus'
""")
insert_edge('active-focus', 'active-next-lines', 'REPLACED_BY', weight=1.0,
            note='下阶段多线探索替代单一 1M 焦点')

conn.commit()
conn.close()
print('Next lines captured in graph.db')
