import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'phase3-10-kv-wall-scan-20260814'
METHOD = 'preference-motivation-analysis-20260721'
EV = 'evidence-phase3-10-kv-wall-scan-20260814'
TASK_WALL = 'task-phase3-kv-wall-capacity-20260814'
TASK_FIX = 'task-phase3-admission-activation-headroom-20260814'
DECISION = 'decision-phase3-baseline-overhead-vs-capability-20260814'

EV_CONTENT = (
'KV 容量墙扫描完成（routeb-p3-kvwall-20260814-211513，Qwen2.5-3B，30k prompts，mc 4/8/16/32）。'
'地面真值：vLLM PD decode 池 7.88GiB=229504 tokens（自报 max concurrency 7.00x @32k）；'
'HCP ring 聚合预算 17.99+10.42GB≈28.4GB（账面约 19 并发 30k 会话，理论容量比 2.7x）。'
'结果：mc4 双方 4/4；mc8 双方 8/8；mc16 HCP 14/16 vs PD 16/16；mc32 HCP 0/32 vs PD 32/32。'
'意外一：vLLM 触墙是准入排队而非 preemption（num_preemptions 全程 0），p99 TTFT 53->408s 线性排队，60/60 全完成——低而稳的软墙。'
'意外二：HCP admission 只记 KV 字节不预留激活工作区，mc16 时 pearl KV 分配压满 16GB 后 172MB 激活分配 OOM 崩溃（worker panic->级联），mc32 全灭——账面容量优势当前无法兑现，墙以崩溃形式出现。'
'报告：docs/PHASE3_KV_WALL_SCAN.md。VERDICT: VERIFIED——能力维对照完成并产出一个真实工程缺口。')

TASK_FIX_CONTENT = (
'修 admission 激活余量：让 HCP 的 fail-closed 拒绝发生在正确位置。'
'1. 问题：KV byte admission 账面放行超出可执行容量的负载（phase3-10 mc16/32 OOM 崩溃实证）。'
'2. 现状：admission 预算=握手时各域空闲 VRAM，仅扣 KV 字节；prefill 激活工作区（logits/attention 中间量）未入账，pearl 16GB 卡被 KV 压到 36MB 空闲后 OOM。'
'3. 终态：预算=空闲VRAM-KV-激活工作区估计（按模型 config+当前 in-flight prefill 数）或至少固定安全余量（约2GB/域）；mc=32 重跑=16 完成+16 status=rejected，无 worker panic。'
'4. 他者：vLLM 的做法是 KV 池在启动时按 gpu_memory_utilization 预分配固定大小，激活余量在池外天然保留；块不够时准入排队。'
'5. 本方案：worker 握手 capacity 上报或 coordinator 入账时扣除激活余量；拒绝路径已有（status=rejected），只需预算正确。'
'6. 为什么：这是 ring 聚合容量优势（2.7x 账面）能否兑现的闸门；崩溃式触墙违背 fail-closed 设计承诺。VERDICT: IMPLEMENT。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
def upsert(nid, ntype, title, content, status):
    conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,1.0,?,?,datetime(\'now\'),datetime(\'now\'))
        ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,status=excluded.status,
        importance=excluded.importance,source=excluded.source,updated_at=datetime(\'now\')''',
        (nid, ntype, 'active', PROJECT, title, content, 1.0, status, SOURCE))
upsert(EV, 'evidence', 'KV 墙扫描：vLLM 软墙排队 60/60，HCP 账面 2.7x 优势但 admission 缺激活余量致 OOM', EV_CONTENT, 'verified')
upsert(TASK_FIX, 'task', 'admission 预留激活工作区余量（修复 KV 墙 OOM）', TASK_FIX_CONTENT, 'pending')
conn.execute("UPDATE nodes SET status='completed', updated_at=datetime('now') WHERE id=?", (TASK_WALL,))
def edge(s, t, ty, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (s, t, ty, note))
edge(TASK_WALL, EV, 'PRODUCES', 'wall scan executed, both axes now measured')
edge(DECISION, EV, 'GOVERNS', 'capability-axis measurement under the two-axis boundary')
edge(TASK_FIX, EV, 'DEPENDS_ON', 'fix derived directly from the mc16/32 OOM evidence')
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('graph updated')