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

insert_node(
    'claim-two-scheduling-strategies', 'claim', 'active',
    'HCP 调度策略对比：capacity-aware 连续分片 vs 加权 Striped',
    '之前的 3:1 capacity-aware 连续分片是在尚未研究 Striped Attention 时提出的方案；'
    '它不一定比 Striped 更简洁或更优。两者应作为 HCP 的两种候选调度策略并行推进、对比评估。\n\n'
    '方案 A：capacity-aware 连续分片（current）\n'
    '- 每个 domain 持有原始序列中一段连续的 token。\n'
    '- 优点：实现简单，与 RoPE/位置编码天然对齐，decode 时新 token 追加逻辑直观。\n'
    '- 缺点：因果 attention 下 early-return 导致负载不均，小 domain 可能成为瓶颈。\n\n'
    '方案 B：加权 Striped permutation\n'
    '- 每个 domain 的 token 按容量比例均匀散布在原始序列中。\n'
    '- 优点：消除 early-return 不对称，负载按 capacity 比例平滑分配。\n'
    '- 缺点：需要位置 id permutation、inverse permutation、按原始位置构造 mask，实现更复杂。\n\n'
    '评估维度：\n'
    '1. 同构/异构设备下的 wall-time 均衡性\n'
    '2. 不同网络带宽下的通信开销（striped 不增加总通信量，但可能改变 micro block 粒度）\n'
    '3. decode 阶段新 token 归属与 inverse permutation 的复杂度\n'
    '4. 与 FlashAttention / PageAttention 等 kernel 的兼容性\n'
    '5. 实现复杂度和可维护性',
    importance=0.85, confidence=0.8, status='held', source='user-direction + design-reasoning'
)
insert_edge('claim-two-scheduling-strategies', 'bp-uneven-cp', 'RELATES_TO', weight=0.9)
insert_edge('claim-two-scheduling-strategies', 'hyp-stripe-ring', 'RELATES_TO', weight=0.9)
insert_edge('claim-two-scheduling-strategies', 'ev-baseline-ring-perf', 'BASED_ON', weight=0.7)

# Update the stripe task to be one of two comparison tasks
c.execute("""
    UPDATE nodes SET status='ongoing', updated_at=datetime('now')
    WHERE id='task-implement-stripe-correctness'
""")

insert_node(
    'task-compare-scheduling-strategies', 'task', 'active',
    '任务：实现并对比两种 HCP 调度策略',
    '把 capacity-aware 连续分片和加权 Striped 作为两条线同时推进：\n'
    '1. 保持当前 3:1 连续分片作为 baseline，优化空间较小但可作为参照。\n'
    '2. 实现加权 Striped correctness 原型，复跑同一 perf 测试。\n'
    '3. 在相同 seq_len、chunk 比例、设备配置下对比 HCP_PERF_LOG。\n'
    '4. 输出对比报告：wall-time 差距、per-token compute 成本、通信 bytes、decode 复杂度。\n'
    '5. 根据结果决定 HCP 默认调度策略，或保留两者作为配置选项。',
    importance=0.9, confidence=0.75, status='ongoing', source='user-direction'
)
insert_edge('task-compare-scheduling-strategies', 'claim-two-scheduling-strategies', 'LEADS_TO', weight=0.9)
insert_edge('task-compare-scheduling-strategies', 'task-implement-stripe-correctness', 'PART_OF')

conn.commit()
conn.close()
print('Scheduling comparison recorded')
