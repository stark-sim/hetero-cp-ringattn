import sqlite3, subprocess, sys
from pathlib import Path
DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
SOURCE = 'hetero-cp-ringattn@phase3-wired-baseline-20rep'
BASELINE_TASK = 'task-phase3-wired-n2-baseline-20260814'
EVIDENCE = 'evidence-phase3-wired-n2-baseline-20rep-20260814'
REBOOT = 'evidence-white-reboot-persistence-20260814'
WIRED = 'evidence-white-pearl-wired-2p5g-20260814'
NETWORK_TASK = 'task-white-single-network-mode-20260814'

def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence):
    conn.execute('''INSERT INTO nodes
        (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET type=excluded.type,layer=excluded.layer,project=excluded.project,
          title=excluded.title,content=excluded.content,importance=excluded.importance,
          confidence=excluded.confidence,status=excluded.status,source=excluded.source,updated_at=datetime('now')''',
        (node_id, node_type, layer, PROJECT, title, content, importance, confidence, status, SOURCE))

def upsert_edge(conn, source, target, edge_type, note):
    conn.execute('''INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note''', (source, target, edge_type, note))

EV_CONTENT = (
'[2026-08-14] 三档负载(L1 rate=1/L2 mc=2/L3 mc=4)x20 次独立 repetition 全部首过 7a 正确性门禁'
'(bench 32/32 完成、trace reserved==released、prefill_hops=24、decode_hops=steps*24、metrics failed=0)。'
'run1 reports/routeb-p3-baseline-20260814-134121 (05:41Z, commit e07be07)；'
'run2 reports/routeb-p3-baseline-20260814-155401 (07:54Z, commit 79cb7a7，white 无人值守重启之后)。'
'两轮 network.json 等价：enp10s0<->enp8s0 2500Mb/s Full，RTT avg 0.172/0.185ms，iperf3 receiver 2.35 Gbit/s、0 retransmit。'
'跨轮中位差全部在 1.4% 以内。20-rep 合并中位数：L1 TTFT 334ms/TPOT 70.4ms/out 15.2 tok/s；'
'L2 TTFT 151ms/TPOT 55.8ms/out 31.6 tok/s；L3 TTFT 279ms/TPOT 112.4ms/out 30.8 tok/s。'
'合并 min-max spread 最大 10.0%(L2 TTFT)，吞吐档最小 0.8%。'
'对比被取代的 WiFi 基线(44 Mbit/s)：TTFT 低 26-66x、TPOT 低 20-31x、吞吐高 10-23x——网络敏感性证据，非算法加速宣称。'
'文档 docs/PHASE3_N2_PERF_BASELINE.md；原始报告不入 git。')

REBOOT_CONTENT = (
'[2026-08-14] white 于 ~12:38Z 重启(非本会话触发)，重启后全部自动恢复、零人工干预：'
'enp10s0 静态 192.168.100.1/24(networkctl configured)、wlp11s0 DHCP 192.168.8.173'
'(netplan-wpa-wlp11s0 active, routable/online)、tailscaled active(100.118.253.68 可达)、kubelet active。'
'单 networkd 模式(NM 已 purge，/etc/netplan/00-renderer-networkd.yaml 覆盖 ubuntu-settings 全局 NM 默认)'
'重启持久性验收通过，关闭 task-white-single-network-mode-20260814 的最后待办。')

conn = sqlite3.connect(DB)
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('BEGIN IMMEDIATE')
upsert_node(conn, EVIDENCE, 'evidence', 'progress',
    'white-pearl 2.5GbE N=2 vllm bench 基线建成：两轮独立 10-rep 全 PASS，跨轮中位差 <=1.4%',
    EV_CONTENT, 'verified', 1.0, 1.0)
upsert_node(conn, REBOOT, 'evidence', 'progress',
    'white 单 networkd 配置通过无人值守重启持久性验证',
    REBOOT_CONTENT, 'verified', 0.95, 1.0)
upsert_edge(conn, EVIDENCE, WIRED, 'CONFIRMS', '20-rep run network gates match the 2.5GbE link evidence')
upsert_edge(conn, EVIDENCE, BASELINE_TASK, 'CONFIRMS', 'baseline task acceptance met: 2x10 reps, gates green, network.json recorded')
upsert_edge(conn, REBOOT, NETWORK_TASK, 'CONFIRMS', 'final acceptance of single-networkd mode: unattended reboot self-recovery')
conn.execute("UPDATE nodes SET status='completed', updated_at=datetime('now') WHERE id=?", (BASELINE_TASK,))
conn.execute("UPDATE nodes SET content=replace(content, '当前运行中。', '已完成：两轮独立 10-rep 全 PASS，见 evidence-phase3-wired-n2-baseline-20rep-20260814。'), updated_at=datetime('now') WHERE id=?", (BASELINE_TASK,))
conn.commit(); conn.close()
subprocess.run([sys.executable, 'graph-memory/export.py'], check=True)
print('graph-memory updated')