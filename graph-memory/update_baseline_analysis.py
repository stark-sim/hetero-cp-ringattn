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
    'ev-baseline-ring-perf', 'evidence', 'active',
    '基线测量：vanilla Ring Attention 在 3:1 不均等分片下的 compute 失衡',
    '测试配置：seq_len=4096，2 domain，chunk0=3072 (75%)，chunk1=1024 (25%)，num_heads=8，head_dim=128，float32 CPU。\n'
    '命令：DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib HCP_PERF_LOG=/tmp/ring_perf_8192.jsonl '
    'cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture\n\n'
    '测量结果（单次 layer，mock transport）：\n'
    '- domain 0 (大 domain): total 150.3 ms，local_compute 148.0 ms，peer_compute 0.001 ms\n'
    '- domain 1 (小 domain): total 41.4 ms，local_compute 14.7 ms，peer_compute 26.2 ms\n'
    '- domain 0 总耗时约为 domain 1 的 3.6 倍\n\n'
    '解读：\n'
    '- 由于 chunk 连续且因果 mask，domain 0 的 peer KV（来自 domain 1，全局位置 3072-4096）'
    '全部位于 Q0 的“未来”，触发 early-return，几乎不耗计算。\n'
    '- domain 1 的 Q 需要 attend 到 domain 0 的全部 3072 个位置，因此 peer_compute 占其总时间 63%。\n'
    '- 在相同算力设备上，大 domain 成为瓶颈；在异构设备上，若小 domain 算力更慢，瓶颈会进一步恶化。',
    importance=0.85, confidence=0.8, status='held', source='HCP_PERF_LOG /tmp/ring_perf_8192.jsonl'
)
insert_edge('ev-baseline-ring-perf', 'bp-uneven-cp', 'SUPPORTS', weight=0.8,
            note='不均等分片下 vanilla ring 计算严重失衡')
insert_edge('ev-baseline-ring-perf', 'hyp-stripe-ring', 'CONFIRMS', weight=0.8,
            note='证实了 striped 可以缓解的瓶颈')

insert_node(
    'note-early-return-asymmetry', 'claim', 'active',
    'Vanilla Ring Attention 的 early-return 在不均等分片下加剧负载不均',
    'process_kv_block 在因果路径下会跳过 kv_global_start >= q_global_end 的 block。'
    '连续 chunk 场景下，持有靠前 token 的大 domain 会跳过来自后续小 domain 的 peer block，'
    '导致其 peer_compute 接近零；而小 domain 必须处理来自大 domain 的全部历史 KV。'
    '这是 vanilla ring 在 capacity-aware 不均等分片下出现 3.6× 耗时差距的根本原因。',
    importance=0.8, confidence=0.85, status='held', source='code-inspection + baseline measurement'
)
insert_edge('note-early-return-asymmetry', 'ev-baseline-ring-perf', 'BASED_ON', weight=0.9)

insert_node(
    'hyp-stripe-expected-impact', 'hypothesis', 'active',
    'Striped 预计能将 3:1 分片下的 domain 总耗时差距从 ~3.6× 降到 ~1.2× 以内',
    'Striped permutation 让每个 domain 的 Q 和 KV 均匀散布在原始序列中，'
    'early-return 条件对两个 domain 大致对称，每个 domain 都会处理大量 peer block。\n'
    '在 seq_len=4096、3:1 分片的理想模型下：\n'
    '- vanilla 有效 attention pair 数：domain0 ≈ 4.72M，domain1 ≈ 3.69M，比例 1.28:1\n'
    '- 实测 wall-time 比例 3.6:1，主要来自 local chunk 大小差异与 early-return 的非对称性\n'
    '- striped 下每个 domain 处理的 pair 数应接近 50%/50%，wall-time 比例预计接近 1:1（同构设备）\n'
    '需要实现后在相同测试上验证。',
    importance=0.8, confidence=0.6, status='open', source='theoretical projection'
)
insert_edge('hyp-stripe-expected-impact', 'ev-baseline-ring-perf', 'BASED_ON', weight=0.8)
insert_edge('hyp-stripe-expected-impact', 'hyp-stripe-ring', 'PART_OF')

insert_node(
    'task-implement-stripe-correctness', 'task', 'active',
    '下一步：实现 Striped correctness model 原型并复跑基线测试',
    '基于 docs/STRIPE_ATTENTION_ADAPTATION_PLAN.md，先在 test_ring_attention_uneven_perf 中增加 striped 模式：\n'
    '1. 添加 permutation/inverse_permutation 生成（细粒度 scheduling unit）。\n'
    '2. 修改 causal mask 使用原始位置 id。\n'
    '3. 保持 online softmax 不变，验证 correctness diff < 1e-4。\n'
    '4. 收集 striped 模式下的 HCP_PERF_LOG，与 vanilla 对比 domain 0/1 的 total_ms。',
    importance=0.9, confidence=0.75, status='ongoing', source='baseline-analysis'
)
insert_edge('task-implement-stripe-correctness', 'hyp-stripe-ring', 'LEADS_TO', weight=0.9)
insert_edge('task-implement-stripe-correctness', 'bp-stripe-adaptation-plan', 'BASED_ON', weight=0.9)

conn.commit()
conn.close()
print('Baseline analysis recorded')
