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
    'bp-stripe-adaptation-plan', 'decision', 'blueprint',
    'Striped Attention HCP 适配计划',
    '已将详细实现计划写入 docs/STRIPE_ATTENTION_ADAPTATION_PLAN.md。\n'
    '核心思路：通过细粒度 scheduling unit 实现 capacity-aware 不均等 stripe；'
    '用原始位置 id 计算 causal mask；worker 输入/输出做 permutation / inverse-permutation；'
    'online softmax 与 KV transport 不变。\n'
    '实施顺序：先在 correctness model 验证，再改 coordinator/worker，最后跑 uneven 分布式 smoke。',
    importance=0.85, confidence=0.8, status='held', source='docs/STRIPE_ATTENTION_ADAPTATION_PLAN.md'
)
insert_edge('bp-stripe-adaptation-plan', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('bp-stripe-adaptation-plan', 'hyp-stripe-ring', 'IMPLIES', weight=0.9)
insert_edge('bp-stripe-adaptation-plan', 'ev-stripe-attention-deep-read', 'BASED_ON', weight=0.9)
insert_edge('bp-stripe-adaptation-plan', 'decision-p2p-feasibility', 'PART_OF')

# Mark deep-read and survey tasks complete, plan drafted
c.execute("UPDATE nodes SET status='closed', updated_at=datetime('now') WHERE id='task-stripe-survey'")

conn.commit()
conn.close()
print('Stripe adaptation plan node added')
