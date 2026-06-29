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
    'bp-simplicity-principle', 'decision', 'blueprint',
    'HCP 设计原则：简洁性 / Occam\'s Razor',
    '全局 AGENTS.md 已加入简洁性原则：如果更复杂的设计没有可验证的明显收益，就选择更简单的方案。\n'
    '对 HCP 当前工作的影响：\n'
    '- Striped Attention 的引入必须通过与 capacity-aware 连续分片的实际对比来证明其价值。\n'
    '- 如果 Striped 在 wall-time、代码复杂度、decode 复杂度、kernel 兼容性上没有明显优势，'
    '  则保留更简单的 capacity-aware 连续分片。\n'
    '- 决策必须基于同一测试配置下的 HCP_PERF_LOG 数据，而不是论文理论 speedup。\n'
    '- 最终结论需记录到 graph-memory 和 commit message。',
    importance=0.85, confidence=0.9, status='held', source='~/.agents/AGENTS.md'
)
insert_edge('bp-simplicity-principle', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('bp-simplicity-principle', 'task-compare-scheduling-strategies', 'GOVERNS', weight=0.9)

# Update comparison claim to include simplicity criterion
c.execute('''
    UPDATE nodes SET content = ?, updated_at = datetime('now')
    WHERE id = 'claim-two-scheduling-strategies'
''', (
    '之前的 3:1 capacity-aware 连续分片是在尚未研究 Striped Attention 时提出的方案；'
    '它不一定比 Striped 更简洁或更优。两者应作为 HCP 的两种候选调度策略并行推进、对比评估。\n'
    '根据全局 AGENTS.md 的简洁性原则，如果 Striped 没有可验证的明显收益，应选择更简单的连续分片。\n\n'
    '方案 A：capacity-aware 连续分片（current）\n'
    '- 优点：实现简单，与 RoPE/位置编码天然对齐，decode 时新 token 追加逻辑直观。\n'
    '- 缺点：因果 attention 下 early-return 导致负载不均，小 domain 可能成为瓶颈。\n\n'
    '方案 B：加权 Striped permutation\n'
    '- 优点：消除 early-return 不对称，负载按 capacity 比例平滑分配。\n'
    '- 缺点：需要位置 id permutation、inverse permutation、按原始位置构造 mask，实现更复杂。\n\n'
    '评估维度（按简洁性原则加权）：\n'
    '1. 同构/异构设备下的 wall-time 均衡性（必须有数据）\n'
    '2. 不同网络带宽下的通信开销（striped 不增加总通信量，但可能改变 micro block 粒度）\n'
    '3. decode 阶段新 token 归属与 inverse permutation 的复杂度\n'
    '4. 与 FlashAttention / PageAttention 等 kernel 的兼容性\n'
    '5. 实现复杂度和可维护性\n'
    '6. 如果以上维度没有明显 winner，默认选择方案 A。',
))

# Update expected-impact hypothesis to include simplicity fallback
c.execute('''
    UPDATE nodes SET content = ?, updated_at = datetime('now')
    WHERE id = 'hyp-stripe-expected-impact'
''', (
    '在 3:1 不均等分片下，加权 Striped 预计能消除 vanilla ring 的 early-return 不对称性，'
    '使 domain 0/1 的 wall-time 比例从实测 3.6:1 向容量比例 3:1（同构设备）或更接近设备能力比例收敛。\n'
    '关键判断：striped 不会让两个 domain 耗时完全相等（因为它们本来就持有不同 token 数），'
    '而是让“每 token 的 compute 成本”在两个 domain 上更均衡。\n'
    '但如果实测显示 wall-time 收益不足以抵消实现复杂度、decode 复杂度或 kernel 兼容性问题，'
    '根据简洁性原则，应回退到 capacity-aware 连续分片并寻找其他优化点（如 network speed、kernel fusion）。',
))

conn.commit()
conn.close()
print('Simplicity principle recorded')
