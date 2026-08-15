import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@k10-kv-byte-ledger-20260816"

TASK = "inquiry-kv-transport-quantitative-ledger-20260815"
DECISION = "decision-k10-kv-byte-ledger-20260816"

def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence, source):
    conn.execute(
        "INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now')) ON CONFLICT(id) DO UPDATE SET type=excluded.type,layer=excluded.layer,project=excluded.project,title=excluded.title,content=excluded.content,importance=excluded.importance,confidence=excluded.confidence,status=excluded.status,source=excluded.source,updated_at=datetime('now')",
        (node_id, node_type, layer, PROJECT, title, content, importance, confidence, status, source),
    )

def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        "INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?) ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note",
        (source, target, edge_type, note),
    )

DECISION_CONTENT = """【K10 动机六问——KV 搬运量定量账本（2026-08-16）】

1. 问题：B 类核心主张"KV 搬运量大幅减少"目前只有机制性定性和容量账面比（2.7x），缺 bytes-on-wire 定量台账，三期 vLLM PD 对比无法用同口径字节数裁决。主线 decode 已是 SD（stationary_decode），但它的 perf event 只有 sends/recvs 计数、无任何字节字段——这是最大缺口。

2. 现状：perf event 字节字段碎片化且口径不一——ring_attention（prefill KV ring）有 kv_sent_bytes/kv_recv_bytes（tensor payload 估算）、ring_decode（legacy Q-ring）有 packet_sent_bytes/packet_recv_bytes（估算）、stationary_continuation 与主线 stationary_decode 都只有 sends/recvs 无 bytes。且这些是 payload 估算（numel×elem_bytes），不是真实 wire frame 字节（含 meta JSON + 4 字节 length prefix）。transport trait 无统一计量接口，未来 NIXL 无法复用同口径。

3. 终态：(a) KvTransport trait 增加 wire_bytes_sent()/wire_bytes_recv() 累计计数器（默认 0），TCP/QUIC 报告真实 serialized frame 字节；(b) 主线 stationary_decode 与 stationary_continuation 的 perf event 补齐 sent_bytes/recv_bytes（从 per-layer transport 计数器差值累计）；(c) scripts/kv_transport_ledger.py 聚合 HCP_PERF_LOG 为 per-request/per-token 的 KV 搬运量台账，并给出 HCP ring vs vLLM PD（NIXL 整段 KV 一次搬移）vs TP（每层 all-reduce activation）的同口径对比公式。

4. 他者：vLLM PD 的 NIXL 用 getXferTelemetry 报告真实 transfer bytes；TP 的 all-reduce 字节 = 2×(N-1)/N × activation 字节 × layers。对比口径都是"实际传输的数据字节数"，因此 HCP 也必须用真实 wire bytes 而非 payload 估算，否则低估 meta 开销、口径不对齐。

5. 本方案：trait 加 wire_bytes_sent/recv 增量计数器（不改变现有 send/recv 返回值签名，默认实现 0，mock 保持 0）；TCP 在 flush_send 累加写出字节、在 frame decode 累加读入字节；QUIC 在主线程 submit_send 累加 frame.len()、recv task 经 Arc<AtomicU64> 累加（recv_frame_from_stream 改为返回 frame wire 长度）；stationary_decode/continuation 逐层用 transport 计数器差值累计 sd_sent_bytes/sd_recv_bytes 写入 perf event；聚合脚本读 JSONL 输出台账。

6. 为什么：真实 wire bytes 是唯一能同时对齐 NIXL telemetry 与 TP all-reduce 的口径；trait 层累计让 NIXL transport 接入时自然复用（NIXL 的 getXferTelemetry 直接填同字段）；逐层差值累计避免在 worker 层重复实现 serialize 逻辑，单一事实源在 transport。

VERDICT: IMPLEMENT（先 K10 字节台账，再接 NIXL transport，两任务共用同一计量接口）。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    # Task: mark in-progress
    conn.execute("UPDATE nodes SET status='active' WHERE id=?", (TASK,))

    upsert_node(conn, DECISION, "decision", "active",
        "K10 决策：transport trait 加 wire-bytes 计量 + 主线 SD decode 补字节字段 + 聚合台账（对齐 NIXL telemetry / TP all-reduce 口径）",
        DECISION_CONTENT, "held", 1.0, 1.0, SOURCE)

    edges = [
        (DECISION, TASK, "PART_OF", "K10 byte-ledger decision"),
        (DECISION, "decision-hcp-first-principles-value-20260815", "FOLLOWS", "quantifies the KV-transport-reduction selling point (B-class)"),
        (DECISION, "inquiry-nixl-as-hcp-transport-20260815", "LEADS_TO", "the wire-bytes interface is the NIXL transport's byte-reporting surface"),
        (DECISION, "decision-decode-route-merge-sd-mainline-20260816", "FOLLOWS", "byte fields target the mainline stationary_decode route"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("decision=k10-kv-byte-ledger-20260816")

if __name__ == "__main__":
    main()
