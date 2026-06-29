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
    'claim-stripe-uneven-feasibility', 'claim', 'active',
    'Striped Attention 可以推广到 capacity-aware 不均等分片',
    '原始 Striped Attention 论文假设每个 device 持有 L/N 个 token（均分），'
    '但其核心思想——让各 device 的 token 均匀散布在原始序列中——可以推广到任意比例。\n\n'
    '推广方式：加权循环调度（weighted round-robin scheduling unit）\n'
    '- 对 3:1 的 2-domain 场景，调度周期为 4，模式为 [0,0,0,1]。\n'
    '- device 0 持有所有满足 p mod 4 ∈ {0,1,2} 的位置，占 75%。\n'
    '- device 1 持有所有满足 p mod 4 = 3 的位置，占 25%。\n'
    '- 当 scheduling unit 足够小（如 1 token 或几十 tokens）时，每个 device 的位置在原始序列中近似均匀散布。\n\n'
    '为什么这能保留 Striped 的好处：\n'
    '1. early-return 不对称性被消除：domain 0 的 Q 会“看到”domain 1 的部分历史 KV，'
    '   不再像连续 chunk 那样整 block 被跳过。\n'
    '2. 负载按容量比例分配：domain 0 处理约 75% 的有效 attention pair，domain 1 处理约 25%，'
    '   与它们的 chunk 比例一致，符合 capacity-aware 的初衷。\n'
    '3. 不需要改变通信原语：仍然是 Q 固定、KV 沿 ring P2P 传递。\n\n'
    '需要放弃原始论文的简单 block-triangular mask：\n'
    '- 加权 stripe 的 residue 关系不再是简单的 j<k 或 j>k。\n'
    '- 必须改用原始位置 id 比较来构造 causal mask（已在适配计划中提出）。\n\n'
    '限制：\n'
    '- 论文中的理论 2× speedup 上界仅在均分且 N 较大时严格成立；'
    '  不均等场景下收益是启发式的，取决于 scheduling unit 大小和具体比例。\n'
    '- scheduling unit 过大时，device 的 token 会局部聚集，early-return 会重新出现。',
    importance=0.85, confidence=0.75, status='held', source='paper-analysis + design-reasoning'
)
insert_edge('claim-stripe-uneven-feasibility', 'hyp-stripe-ring', 'SUPPORTS', weight=0.85,
            note='striped 可推广到 capacity-aware 不均等分片')
insert_edge('claim-stripe-uneven-feasibility', 'bp-stripe-adaptation-plan', 'PART_OF')
insert_edge('claim-stripe-uneven-feasibility', 'bp-uneven-cp', 'RELATES_TO', weight=0.8,
            note='加权 stripe 是实现 capacity-aware 的一种可行手段')

# Update the expected-impact hypothesis with nuance
c.execute('''
    UPDATE nodes SET content = ?, confidence = ?, updated_at = datetime('now')
    WHERE id = 'hyp-stripe-expected-impact'
''', (
    '在 3:1 不均等分片下，加权 Striped 预计能消除 vanilla ring 的 early-return 不对称性，'
    '使 domain 0/1 的 wall-time 比例从实测 3.6:1 向容量比例 3:1（同构设备）或更接近设备能力比例收敛。\n'
    '关键判断：striped 不会让两个 domain 耗时完全相等（因为它们本来就持有不同 token 数），'
    '而是让“每 token 的 compute 成本”在两个 domain 上更均衡，避免小 domain 因处理全部历史 KV 而额外过载，'
    '也避免大 domain 因 peer block 被整段跳过而长时间空闲等待。\n'
    '最终效果取决于：scheduling unit 粒度、device 相对算力、KV cache memory bandwidth。',
    0.65
))

# Update task to reflect generalized stripe implementation
c.execute('''
    UPDATE nodes SET content = ?, updated_at = datetime('now')
    WHERE id = 'task-implement-stripe-correctness'
''', (
    '实现 capacity-aware Striped correctness 原型：\n'
    '1. 设计加权 permutation：给定 chunk_sizes，生成周期为 sum(chunk_sizes) 的循环调度，'
    '   如 3:1 对应模式 [0,0,0,1]。\n'
    '2. 在 test_ring_attention_uneven_perf 中新增 striped 模式，比较 vanilla vs striped 的 HCP_PERF_LOG。\n'
    '3. 用原始位置 id 构造 causal mask，确保 correctness diff < 1e-4。\n'
    '4. 评估 scheduling unit 粒度（1 token vs 64 tokens vs 256 tokens）对负载均衡和 mask 开销的影响。',
))

conn.commit()
conn.close()
print('Stripe uneven feasibility recorded')
