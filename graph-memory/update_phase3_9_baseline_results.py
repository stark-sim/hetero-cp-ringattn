import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'phase3-9-interleaved-baseline-20260814'
METHOD = 'preference-motivation-analysis-20260721'
EV = 'evidence-phase3-9-interleaved-baseline-20260814'
TASK = 'task-phase3-vllm-pd-baseline-20260814'
DECISION = 'decision-phase3-baseline-overhead-vs-capability-20260814'

EV_CONTENT = (
'三期 A 受控对照完成：routeb-p3-pd-baseline-20260814-194248，10+10 交错 rep 全 PASS（无重试触发）。'
'median 对照（HCP/PD）：L1 TTFT 333.6/49.3ms TPOT 72.6/8.3ms 吞吐 15.1/17.2 tok/s；'
'L2 TTFT 150.8/30.5ms TPOT 57.4/8.6ms 吞吐 31.0/191.4 tok/s；'
'L3 TTFT 282.6/33.6ms TPOT 114.9/8.7ms 吞吐 30.3/355.5 tok/s。'
'HCP 离散 1-10%、PD 离散 0.2-9.5%，HCP 落在自身 20-rep 基线 min-max 带内。'
'网络门等价：RTT 0.193ms、iperf3 2350Mbps、0 重传。'
'解读：同负载开销维 vLLM PD 快 5-13x 属引擎成熟度差距（连续批处理是最大单项，并发吞吐差 6-11x）；'
'L1 单流吞吐持平（0.88x）给出干净的每 token 固定开销读数。'
'边界：此结果仅覆盖引擎开销维；KV 容量能力维由 task-phase3-kv-wall-capacity-20260814 承接（3B 模型）。'
'完整报告：docs/PHASE3_VLLM_PD_COMPARISON.md；真源 comparison.json 在 reports/routeb-p3-pd-baseline-20260814-194248/。'
'VERDICT: VERIFIED — 交错、双门、网络等价门、20/20 PASS。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
    VALUES (?,?,?,?,?,?,?,1.0,?,?,datetime(\'now\'),datetime(\'now\'))
    ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,status=excluded.status,
    importance=excluded.importance,source=excluded.source,updated_at=datetime(\'now\')''',
    (EV, 'evidence', 'active', PROJECT, '三期A交错对照：HCP N=2 vs vLLM PD 10+10 rep 全绿，开销维差距量化', EV_CONTENT, 1.0, 'verified', SOURCE))
conn.execute("UPDATE nodes SET status='completed', updated_at=datetime('now') WHERE id=?", (TASK,))
def edge(s, t, ty, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (s, t, ty, note))
edge(TASK, EV, 'PRODUCES', '10+10 interleaved baseline all PASS')
edge(DECISION, EV, 'GOVERNS', 'results must be quoted with overhead-vs-capability boundary')
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('evidence recorded, task completed')