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
    'active-n3-ring-route-20260722', 'task', 'active',
    '后续路线:N>2 真 ring(三机:white CUDA + pearl ROCm + laptop CPU)',
    '基于三步收官后的现状,N>2 ring 的技术改动已定位为三处小改:\n'
    '1. connector 加 ring_role=relay:中间节点同时标前序 external(get_num_new_matched_tokens)'
    '并存自己 chunk(build_connector_meta 的 store/load 两条路径本来就分开),就绪状态级联;\n'
    '2. backend 多 peer 合并:N-1 个 peer chunk 是全局连续前缀,cat 成一段连续 KV 做一次 peer pass '
    '(线性拷贝,数学等价),merge_attn_states 调用不变;\n'
    '3. hcp_ring 每请求参数加 chunk_ids(复数,向后兼容追加);staging 已按 chunk 键,'
    '请求→chunk 映射从单值变列表。\n'
    '验证拓扑两步走:\n'
    '(a) white 当 relay(吃 c0 产 c1) + pearl 当 consumer(2 前缀 chunk)——证明 N>2 机制,不依赖 Mac;\n'
    '(b) 三机真异构:laptop(Mac)需自建 vLLM CPU(无 macOS wheel,VLLM_TARGET_DEVICE=cpu),'
    '担任 chunk0 producer(不吃前缀、计算量最小)——满足"每平台必须跑 worker"纪律的最小可行角色。\n'
    '前置:pearl 恢复可达,完成 task-gfx1200-repo-extraction。',
    importance=0.9, confidence=0.8, status='ongoing', source='user-direction'
)
insert_edge('active-n3-ring-route-20260722', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('active-n3-ring-route-20260722', 'task-kernel-hardening-backlog-20260721', 'RELATES_TO', weight=0.7,
            note='backlog 第 4 项(N>2 多路合并)由本路线落地;kernel hardening 1-3 可并行')
insert_edge('active-n3-ring-route-20260722', 'task-gfx1200-repo-extraction', 'DEPENDS_ON', weight=0.8,
            note='pearl 是 N>2 拓扑的必要节点,先完成其迁移')
insert_edge('active-n3-ring-route-20260722', 'active-next-lines', 'PART_OF', weight=0.7)

conn.commit()
conn.close()
print('N>2 ring route recorded in graph.db')
