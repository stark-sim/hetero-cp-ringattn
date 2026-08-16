import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'hetero-cp-ringattn@e3addf5'

EV = 'evidence-nixl-s4-2a-ring-transfer-bug-20260816'
DEC = 'decision-nixl-s4-2a-defer-20260816'
S4_DEC = 'decision-nixl-s4-ring-pagedkv-20260816'

EV_CONTENT = """NIXL S4-2a 三机环逐 hop transfer 探针（nixl-ring-probe, 2 rounds）数值错误：round 0 后三个节点的 recv 缓冲区全部等于 laptop 的 current（seed 200 数据），仅 white.recv（其前驱=laptop）碰巧正确；pearl.recv 应为 white.current(seed 0)、laptop.recv 应为 pearl.current(seed 100)，实际均为 [200,201,...]。三个 recv.bin 文件 byte-identical（MD5 相同），且与 bf16(arange+200) 逐值精确匹配（8192/8192）。【已排除项】desc 交换方向与地址（succ_desc.addr == succ 自己导出的 recv data_ptr，md 字节级验证一致）、seed 传参、initial current / register 后 current 值、双向 md load（md-in2 无效）、pkill/ssh -f/scp 原子性、UCX_TLS=tcp 与 UCX_NET_DEVICES 网络配置（三机 192.168.8.x 互通、连接建立日志全部 CONNECTED）。【地址级调试确认】在 transport.submit_transfer 打印 local/remote desc 地址：white local=white.current(0x62f420994440)->remote=pearl.recv(0x58effab34f40)，pearl local=pearl.current->laptop.recv，laptop local=laptop.current->white.recv，全部与 probe 注册地址一致；即 NIXL 收到的 local/remote 地址、mem_type、agent 名都正确。【对照】同一 NixlBlockTransport 的 2 节点双向探针（nixl_transfer_pair.sh white<->pearl）PASS（max|diff|=0.0），证明 register/submit_transfer/poll 本身正确。【剩余假设】问题出在 NIXL 内部（UCX put 的 rkey/addr 解析或三 agent 场景的 endpoint 复用），不是 HCP transport 代码——transport 层已地址级核对无误。该 bug 未修复即延后。"""

DEC_CONTENT = """【NIXL 三机环 transfer bug 延后决策（2026-08-16，用户裁定）】

1. 问题：S4-2a（三机环逐 hop transfer 机制验证）卡在数值 bug 上——round 0 后 pearl.recv/laptop.recv 收到 laptop 的数据（见 evidence-nixl-s4-2a-ring-transfer-bug-20260816）。已多轮调试（desc 交换方向、地址级核对、双向 md load、日志/telemetry），transport 代码与地址全部正确，2 节点对照 PASS，但 NIXL 内部三 agent 场景行为异常，根因未定位。

2. 现状：NIXL block 数据面已闭环到「两两 transfer 字节级一致」（S3a/S4-1），三机环是 S4-2 的下一验证点；卡在 NIXL 内部（疑似 UCX put rkey/addr 或多 agent endpoint 解析），非 HCP transport 代码。

3. 终态（延后后的目标）：NIXL 不是关键路径；block 数据面抽象（KvBlockTransport/BlockDesc/mem_type/register-transfer 生命周期）与 S4 paged-KV 化不依赖 NIXL 三机环 bug 修复。优先继续 S4 主线（paged-KV block 抽象 + vLLM 形状对齐），NIXL 三机环机制验证挂起，待有明确收益场景（如真有 GPU-direct 网络）或 NIXL/UCX 版本更新后再回访。

4. 他者：vLLM 的 NixlConnector 是 prefill->decode 整段 KV 一次搬移（block 级 + 独立 TCP side channel），非 ring 逐 hop；其单跳语义与我们两两验证一致（已验证可行），三机环的逐 hop 转发是 HCP 特有编排。

5. 本方案：把 S4-2a 的 NIXL 环 bug 标为 deferred（不在本节点继续投入）；S4 继续走 block 数据面抽象 + paged-KV 化，三机环验证等 NIXL 侧有明确修复/升级（或换真 GPU-direct 网络）再回访。

6. 为什么延后：NIXL 收益定位早已收敛为「block 数据面抽象经验」（跨异构 host staging + 消费级 WiFi 无 RDMA，NIXL 零拷贝在现有网络无法兑现，S4-1 已证）；三机环 bug 修复的边际价值低于继续推进 S4 paged-KV 主线。用户 2026-08-16 明确裁定「NIXL 不是关键路径，先延后」。

VERDICT: DEFER（NIXL 三机环 transfer 机制验证挂起；S4 主线继续，block 数据面抽象不受影响）。"""

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')

def upsert(nid, ntype, title, content, status, importance=0.9, confidence=1.0):
    conn.execute('''INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,status=excluded.status,
        importance=excluded.importance,confidence=excluded.confidence,source=excluded.source,updated_at=datetime('now')''',
        (nid, ntype, 'active', PROJECT, title, content, importance, confidence, status, SOURCE))

upsert(EV, 'evidence', 'NIXL S4-2a 三机环 transfer bug：round 0 后三个 recv 全为 laptop 数据，transport 地址级已排除', EV_CONTENT, 'held', 0.9, 1.0)
upsert(DEC, 'decision', 'NIXL 三机环 transfer bug 延后：不是关键路径，S4 主线继续（paged-KV 抽象不受影响）', DEC_CONTENT, 'held', 0.95, 1.0)

def edge(s, t, ty, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (s, t, ty, note))

edge(DEC, EV, 'MOTIVATES', 'defer decision records the unrooted S4-2a ring transfer bug')
edge(EV, 'decision-nixl-s4-ring-pagedkv-20260816', 'SUPPORTS', 'block dataplane abstraction value is independent of the 3-node ring bug')
edge(DEC, S4_DEC, 'SUPERSEDES', 'S4-2 NIXL ring hop-by-hop mechanism verification deferred; S4-3 paged-KV mainline continues')

conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('recorded:', EV, DEC)
