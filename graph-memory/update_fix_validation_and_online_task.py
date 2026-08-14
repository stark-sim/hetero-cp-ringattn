import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'phase3-10-fix-validation-20260815'
METHOD = 'preference-motivation-analysis-20260721'
TASK_FIX = 'task-phase3-admission-activation-headroom-20260814'
TASK_ONLINE = 'task-phase3-online-attention-workspace-20260815'
EV = 'evidence-phase3-10-fix-and-8k-20260815'
DEC = 'decision-phase3-core-vs-reusable-wheels-20260814'

EV_CONTENT = (
'admission 激活余量修复落地并验证（commit ec59df8，源头扣减 worker_capacities；首版只扣 HTTP ledger 被重跑证伪后修正）。'
'30k 重跑（routeb-p3-kvwall-20260814-232932）：预算正确缩至 [15586,8406] MiB 但仍 OOM——1.5GB 余量盖不住 30k prefill 激活峰值；'
'根因深一层：ring 每跳物化全量 [heads, shard, kv_len] 分数矩阵（无 online softmax），30k 时约 7.5GB/跳 bf16——激活工作区而非 KV 字节是 30k 的真实约束。'
'8k 变体（routeb-p3-kvwall-20260815-000103，mc 8/16/32/64）：HCP 120/120 全完成零 OOM，账本健康（reserved==released 全等）；'
'vLLM PD 同样 120/120 但 64x8k=524k tokens 超其 229.5k 硬池，排队加深（p99 TTFT 48->95s）零 preemption。'
'HCP mc=64 TPOT 14.5s vs vLLM 0.36s——A 类引擎差距的高并发体感。'
'结论：能力账（2.7x 聚合）机制真实且入账正确；30k 兑现被激活工作区卡住，路径=online attention 轮子。VERDICT: VERIFIED。')

TASK_ONLINE_CONTENT = (
'实现 online/chunked attention（flash 式分块+在线 softmax），消除 O(shard^2) 激活工作区。'
'1. 问题：ring 每跳物化全量分数矩阵，30k 长上下文下激活峰值 ~7.5GB/跳，使 KV 账面容量（2.7x 优势）无法兑现。'
'2. 现状：attention.rs 第六步 scores=q.matmul(k^T) 全量物化；PROJ_CHUNK_SIZE 只分块投影，不分块注意力。'
'3. 终态：注意力按 (q_chunk x kv_chunk) 分块 + 在线 softmax 累加，激活峰值与 shard 长度解耦（有界工作区）；'
'    验证：30k mc=16 重跑 16/16 无 OOM，mc=32 = 16 完成 + 16 fail-closed 拒绝；数值对照不回归（argmax 一致、漂移不超管线固有值）。'
'4. 他者：flash-attention/online softmax 是所有主流引擎的标准轮子；ring-attention 论文本身即基于分块在线累加。'
'5. 本方案：在 tch_backend/attention 路径实现分块在线 softmax（Rust/tch-rs 手写，或评估复用 flash-attn kernel 绑定——属 wheel-reuse 评估的交集）。'
'6. 为什么：这是把 KV 容量账面优势变成可兑现服务能力的闸门；属 (A) 类轮子但直接解锁 (B) 类核心主张的实测。'
'VERDICT: IMPLEMENT（优先级高于其他生态轮子，因为它解锁核心验证）。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
def upsert(nid, ntype, title, content, status):
    conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,1.0,?,?,datetime(\'now\'),datetime(\'now\'))
        ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,status=excluded.status,
        importance=excluded.importance,source=excluded.source,updated_at=datetime(\'now\')''',
        (nid, ntype, 'active', PROJECT, title, content, 1.0, status, SOURCE))
upsert(EV, 'evidence', 'admission 修复验证 + 8k 墙扫描：120/120 全绿；30k 真约束=激活工作区', EV_CONTENT, 'verified')
upsert(TASK_ONLINE, 'task', 'online/chunked attention：消除 O(shard^2) 激活工作区（解锁 30k 容量兑现）', TASK_ONLINE_CONTENT, 'pending')
conn.execute("UPDATE nodes SET status='completed', updated_at=datetime('now'), content=content||' [2026-08-15 收尾：机制已落地并于 8k 验证；30k 场景的激活峰值超出任何合理固定余量，转入 task-phase3-online-attention-workspace-20260815]' WHERE id=?", (TASK_FIX,))
def edge(s, t, ty, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (s, t, ty, note))
edge(TASK_FIX, EV, 'PRODUCES', 'reserve mechanism landed (ec59df8), verified at 8k')
edge(EV, TASK_ONLINE, 'SUPPORTS', '30k OOM root-caused to full-score materialization')
edge(TASK_ONLINE, DEC, 'DEPENDS_ON', 'A-class wheel that unlocks the B-class capacity claim')
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('graph updated')