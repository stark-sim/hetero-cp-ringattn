import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source='analysis'):
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
    'task-kernel-hardening-backlog-20260721', 'task', 'active',
    'kernel-hardening backlog(性能/规模,非正确性;128K+ 启动时按序做)',
    '第 2 步 kernel 化完成后,正确性层面无遗留;以下均为性能/规模项,128K+/1M 规模测试启动时按此顺序做:\n'
    '1. local 段 paged 直读(中-高):现 forward 每请求每层先把本地 KV 从分页池 gather 成连续张量再进 kernel,'
    '128K/1M 规模下每层每步多出上百 MB 线性拷贝流量;应让 ring_triton_attn 直接吃 block_table 对 local 段直读'
    '(peer 段连续 staging 不变)。这是 PagedAttention 整合未完成的一半。\n'
    '2. 长上下文 decode split-KV(中-高):decode(Tq=1) 现走 prefill 风格 kernel 沿 KV 串行扫;'
    '参照 vllm unified_attention 的 softmax_segm split-K 分段并行。1M 教训中 decode 本是瓶颈。\n'
    '3. 批量 kernel 启动(中):per-request Python 循环,每请求每层 2 次 launch;local 段按 batch 合并 '
    '(cu_seqlens + block table) 一次 launch。纯性能。\n'
    '4. N>2 真 ring 多路合并(将来):merge_attn_states 两路迭代可行;connector 每请求限一个 peer chunk '
    '(PoC 限制),配置面可向后兼容地追加 chunk_ids。v0.1 插件明确声明不支持。\n'
    '不做这些的理由(当下):均为 kernel 内部实现或配置追加项,不影响 v0.1 插件对外配置面冻结;'
    '正确性已由三件套 + 16k + 跨节点验证覆盖。',
    importance=0.85, confidence=0.85, status='ongoing', source='analysis'
)
insert_edge('task-kernel-hardening-backlog-20260721', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('task-kernel-hardening-backlog-20260721', 'ev-ring-triton-kernel-20260721', 'BASED_ON', weight=0.9,
            note='kernel 化完成后盘点的性能/规模遗留')
insert_edge('task-kernel-hardening-backlog-20260721', 'active-vllm-ecosystem-plugin-20260721', 'RELATES_TO', weight=0.7,
            note='不阻塞插件化:均为内部实现或配置追加项,不动 v0.1 配置面')
insert_edge('task-kernel-hardening-backlog-20260721', 'hyp-stripe-ring', 'RELATES_TO', weight=0.5,
            note='N>2 真 ring 与 stripe/负载均衡议题会合时再统一考虑')

conn.commit()
conn.close()
print('kernel-hardening backlog recorded')
