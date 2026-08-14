import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'user-direction-2026-08-14-kv-wall-boundary'
METHOD = 'preference-motivation-analysis-20260721'
DECISION = 'decision-phase3-baseline-overhead-vs-capability-20260814'
TASK_PD = 'task-phase3-vllm-pd-baseline-20260814'
TASK_WALL = 'task-phase3-kv-wall-capacity-20260814'

DECISION_CONTENT = (
'owner 裁决（2026-08-14）：三期 A 对照实验必须区分两个维度，互不替代：'
'(a) 引擎开销对照：同负载（小 context）下 HCP vs vLLM PD 的延迟/吞吐——vLLM PD 更快是预期结果，量化的是 HCP 执行器成熟度差距；'
'(b) 能力对照：KV 容量墙——vLLM PD 把单请求全量 KV 压在 decode 单节点（天花板=单节点 VRAM 池，触墙表现为 preemption/重排队），HCP ring 把每请求 KV 分片到全环（天花板=聚合 VRAM，触墙表现为 fail-closed admission 拒绝）；'
'ring attention 的核心价值是 (b) 的长上下文支撑能力，(a) 的数字不得脱离该边界被引用。'
'docs/PHASE3_N2_PERF_BASELINE.md 与 phase3-9 comparison.json 必须携带此边界声明。')

TASK_CONTENT = (
'三期 A 补：KV 容量墙对照实验（context x concurrency 扫描）。'
'动机剖析六问：'
'1. 问题：PD bench（32-token 输入）只测引擎开销，完全没碰 HCP 存在的理由——异构合作缓解 KV 显存压力；需要量化两栈的 KV 容量天花板差异。'
'2. 现状：N=2 有线基线 + PD 对照 harness 就绪；HCP N=3 16k-token ring E2E 已证；vLLM PD 的 KV 全在 decode 节点（pearl 16GB，池约 13GB），HCP N=2 分片后聚合约 2x。'
'3. 终态：input_len 长 context（如 4k/16k/64k）x 并发档位递增的扫描曲线：vLLM PD 侧观测到 KV 池耗尽后的 preemption/延迟崩塌点，HCP 侧观测到 fail-closed 拒绝点；两曲线并排，容量天花板比值可读出；过墙行为差异（preempt vs reject）明确记录。'
'4. 他者：vLLM 触墙行为是 preemption（recompute 模式）——块池耗尽时换出重算，TTFT 尾部爆炸；HCP 是 byte-level admission fail-closed——直接拒绝。两者都是合法策略，比较的是墙的位置不是策略优劣。'
'5. 本方案：复用 phase3_9 交错 harness 形态；每档先单发 sanity 再 ramp；vLLM 侧从 /metrics 或日志抓 preemption 计数，HCP 侧从 metrics failed/admission 拒绝计数；记录每档 VRAM 占用与 KV 池水位。'
'6. 为什么：这是唯一能在当前硬件（16+24GB）上把蓝图主张（显存墙->可调度问题）变成可测量曲线的方法；单机绳长上限内两栈都能跑满，唯有并发长 context 能把墙逼出来。'
'VERDICT: IMPLEMENT，排在 phase3-9 交错基线完成之后（避免 GPU 争用）。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
def upsert(conn, nid, ntype, layer, title, content, status, imp):
    conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,1.0,?,?,datetime(\'now\'),datetime(\'now\'))
        ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,status=excluded.status,
        importance=excluded.importance,source=excluded.source,updated_at=datetime(\'now\')''',
        (nid, ntype, layer, PROJECT, title, content, imp, status, SOURCE))
upsert(conn, DECISION, 'decision', 'active', '三期对照分两层：引擎开销对照与 KV 容量能力对照互不替代', DECISION_CONTENT, 'held', 1.0)
upsert(conn, TASK_WALL, 'task', 'active', '三期 A 补：KV 容量墙对照实验（长 context x 并发扫描）', TASK_CONTENT, 'pending', 1.0)
def edge(conn, s, t, ty, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (s, t, ty, note))
edge(conn, METHOD, DECISION, 'GOVERNS', 'owner boundary ruling recorded via six-question discipline')
edge(conn, DECISION, TASK_PD, 'REFINES', 'PD bench results must be quoted with the capability boundary')
edge(conn, TASK_WALL, DECISION, 'DEPENDS_ON', 'capability-axis comparison defined by this decision')
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('boundary decision + kv-wall task recorded')