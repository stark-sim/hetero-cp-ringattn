import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'user-direction-2026-08-14-pd-baseline'
METHOD = 'preference-motivation-analysis-20260721'
DECISION = 'decision-phase3-vllm-pd-baseline-20260814'
TASK = 'task-phase3-vllm-pd-baseline-20260814'
UMBRELLA = 'task-phase3-vllm-bench-and-ecosystem-20260812'
ECO = 'task-route-b-phase3-ecosystem-20260809'
BASELINE = 'evidence-phase3-wired-n2-baseline-20rep-20260814'
PD_BELIEF = 'belief-vllm-pd-full-kv-20260721'

DECISION_CONTENT = (
'动机剖析六问：'
'1. 问题：HCP 已有 20-rep 有线稳定基线但无主流引擎对照，绝对值无法回答 P2P ring 开销高低；核心论点（高速互联必要性）缺比较证据。'
'2. 现状：vLLM 两端同 commit(3f99883d9) 源码构建就位（white vllm-v1/cu131，pearl vllm-rocm/rocm713）；但该 checkout 跨节点 executor 只有 Ray（uniproc/multiproc 均单机）。'
'3. 终态：同链路、同模型(Qwen2-0.5B bf16)、同负载阶梯(L1/L2/L3)、同客户端(vllm bench serve)、同统计纪律(10 reps + network.json 门禁)下 vLLM PD 与 HCP N=2 并排性能表；为消除时间窗混杂，HCP/vLLM 逐 rep 交错采集；结论落 docs + graph。'
'4. 他者：vLLM 官方跨节点形态：PP/TP 需 Ray（进程编排层，与张量通信无关）；PD 分离不需 Ray——两个 vllm serve + ~200 行 round-robin 转发 proxy（无调度逻辑），KV 走官方 connector（nixl/lmcache/mooncake/moriio）。graph 既有记录：vLLM 官方长上下文分布路线即 disaggregated prefill。'
'5. 本方案：采用 PD 分离作为对照形态——white=prefill(kv_both/producer)+pearl=decode(consumer)，proxy+bench client 在 white，全部端口钉在 192.168.100.x；step0 尖峰验证 connector 在 CUDA-ROCm 有线上可行（nixl 优先，UCX host-staging 可绕开 NCCL-RCCL 线协议风险；备选 moriio/lmcache）。'
'6. 为什么：owner 第一性原理裁决——HCP 的调度面源于 KV ring 合作复杂，比较对象不应被强加额外调度面；无必要时勿增实体（不引 Ray）。且 HCP 是 context parallel（全量权重+KV 过网），PP 是层切分，PD 才是同构对照：比较变为 KV 过网两实现——整段一次搬移 vs 分层 ring 流水。'
'VERDICT: IMPLEMENT（owner 在 ask_user_question 中确认 PD 形态，取代其早前 PP 口径）')

TASK_CONTENT = (
'三期 A：受控 vLLM PD 对照基线（无 Ray）。步骤：'
'0. 尖峰：两端装 nixl，PD pair 起在 192.168.100.x，单请求 curl 验证 token 合理 + KV 确实跨有线（日志+接口字节计数）；不可行则带证据回来定备选（moriio/lmcache/PP+Ray）。'
'1. vLLM 侧 ladder 单跑 sanity（L1/L2/L3 32 prompts 全完成）。'
'2. 交错战役：单一 harness 逐 rep 交替 HCP 栈(phase3_7a_n2_driver) 与 vLLM PD 栈，各 10 reps；每 rep 全新栈；HCP rep 过 7a 门禁，vLLM rep 过 32/32+指标 sane 门禁；战役级 network.json。'
'3. 聚合：side-by-side 表（中位/min/max/spread），按 docs 比较规则出结论；graph evidence；docs 更新。'
'commit 边界：step0 证据独立 commit；harness 脚本独立 commit；结论+docs+graph 收尾 commit。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
def upsert(conn, nid, ntype, layer, title, content, status):
    conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,1.0,1.0,?,?,datetime(\'now\'),datetime(\'now\'))
        ON CONFLICT(id) DO UPDATE SET type=excluded.type,layer=excluded.layer,title=excluded.title,
        content=excluded.content,status=excluded.status,source=excluded.source,updated_at=datetime(\'now\')''',
        (nid, ntype, layer, PROJECT, title, content, status, SOURCE))
upsert(conn, DECISION, 'decision', 'active', '三期对照基线采用 vLLM PD 分离形态（无 Ray），取代 PP 口径', DECISION_CONTENT, 'held')
upsert(conn, TASK, 'task', 'active', '三期 A：受控 vLLM PD 对照基线（无 Ray，交错 10+10 reps）', TASK_CONTENT, 'active')
def edge(conn, s, t, ty, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (s, t, ty, note))
edge(conn, METHOD, DECISION, 'GOVERNS', 'six-question analysis before implementation')
edge(conn, TASK, DECISION, 'DEPENDS_ON', 'implements the PD-form baseline decision')
edge(conn, TASK, UMBRELLA, 'PART_OF', 'umbrella step 2: controlled vLLM baseline comparison')
edge(conn, TASK, ECO, 'PART_OF', 'phase-3 ecosystem exit criterion 1: fair benchmark')
edge(conn, TASK, BASELINE, 'DEPENDS_ON', 'HCP side of the comparison is the verified 20-rep wired baseline')
edge(conn, DECISION, PD_BELIEF, 'BASED_ON', 'PD is vLLM official long-context distribution route')
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('decision+task recorded')