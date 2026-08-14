import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'owner-strategy-2026-08-14-core-vs-wheels'
METHOD = 'preference-motivation-analysis-20260721'
DEC = 'decision-phase3-core-vs-reusable-wheels-20260814'
DEC_BOUNDARY = 'decision-phase3-baseline-overhead-vs-capability-20260814'
EV_WALL = 'evidence-phase3-10-kv-wall-scan-20260814'
EV_PD = 'evidence-phase3-9-interleaved-baseline-20260814'
TASK_FIX = 'task-phase3-admission-activation-headroom-20260814'
TASK_WHEEL = 'task-phase3-wheel-reuse-route-eval-20260814'

DEC_CONTENT = (
'owner 战略框架（2026-08-14）：三期对比与建设必须区分两层——'
'(A) 可复用工程轮子：排队、连续批处理、激活余量入账、cudagraph 等推理框架工程特性。'
'    这些不是任何一方的架构属性：HCP 可以自己实现（生态完善工作），也可以走复用 vLLM 轮子的路线'
'    （把 vLLM 当执行引擎/组件嵌入 HCP 拓扑）。因此开销维差距（phase3-9 的 5-13x）是可关闭的工程债，'
'    不构成对 HCP 架构的否定。'
'(B) HCP 架构核心（真正的差异化，必须聚焦验证与放大）：'
'    1) 异构线性网络拓扑（CUDA+ROCm 混编入环）；'
'    2) 长上下文无限合作（KV 分片随 N 聚合扩展，phase3-10 实测账面 28.4GB vs PD 单点池 7.88GiB = 2.7x）；'
'    3) KV 搬运量大幅减少（PD 把全量 KV ~1.09GB/请求从 prefill 整包搬运到 decode；'
'       HCP 每域 KV 留在本域，admission 实测每域 543MB/请求且无需跨机整包转移）。'
'比较纪律：任何方案对比必须显式标注每个差异点属于 (A) 还是 (B)；'
'(A) 类差距只记录量级与关闭成本，不作为架构裁决依据；(B) 类指标才是 HCP 的胜负手。'
'三期生态完善的内涵由此明确：补齐/复用 (A) 类轮子（含评估 vLLM 轮子复用路线），让 (B) 类优势可测量地呈现。')

TASK_WHEEL_CONTENT = (
'三期生态线：评估复用 vLLM 轮子的路线。'
'1. 问题：HCP 缺主流引擎的工程特性（排队/连续批处理/cudagraph），自研每一项成本高。'
'2. 现状：tch-rs eager 执行器研究级；vLLM 同 commit 双端已能跑（含 ROCm NIXL 链）。'
'3. 终态：一份路线评估文档——哪些轮子可复用（vLLM 作为 HCP 域内执行引擎？scheduler？kernel 库？），'
'    复用界面在哪，与 HCP 的 QUIC ring/KVring 协议如何拼接，工作量分级。'
'4. 他者：Dynamo 走的是 vLLM 之上的编排层路线（已裁决延后）；llm-d 等同理。'
'5. 本方案：先做书面评估（不动代码），按 decision-phase3-core-vs-reusable-wheels-20260814 的 A/B 分类逐项映射。'
'6. 为什么：若轮子可复用，(A) 类差距的关闭成本从自研数月降为集成数周，让团队火力集中在 (B) 类核心。'
'VERDICT: EVALUATE-FIRST（书面评估，owner 裁决后再动代码）。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
def upsert(nid, ntype, title, content, status, imp=1.0):
    conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,1.0,?,?,datetime(\'now\'),datetime(\'now\'))
        ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,status=excluded.status,
        importance=excluded.importance,source=excluded.source,updated_at=datetime(\'now\')''',
        (nid, ntype, 'active', PROJECT, title, content, imp, status, SOURCE))
upsert(DEC, 'decision', '三期战略框架：可复用轮子(A) vs HCP 架构核心(B)，对比必须分类标注', DEC_CONTENT, 'held')
upsert(TASK_WHEEL, 'task', '评估复用 vLLM 轮子路线（书面评估先行）', TASK_WHEEL_CONTENT, 'pending', 0.9)
def edge(s, t, ty, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (s, t, ty, note))
edge(METHOD, DEC, 'GOVERNS', 'owner strategy recorded via six-question discipline')
edge(DEC, DEC_BOUNDARY, 'REFINES', 'overhead-vs-capability boundary extended with A/B wheel classification')
edge(DEC, EV_PD, 'EXPLAINS', 'overhead-axis gap classified as closable engineering debt (A), not architecture verdict')
edge(DEC, EV_WALL, 'EXPLAINS', 'capacity axis (B) confirmed HCP core; OOM gap is (A)-class admission accounting')
edge(TASK_FIX, DEC, 'DEPENDS_ON', 'admission headroom fix is (A)-class wheel completion')
edge(TASK_WHEEL, DEC, 'DEPENDS_ON', 'wheel-reuse route evaluation defined by this strategy')
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('strategy decision recorded')