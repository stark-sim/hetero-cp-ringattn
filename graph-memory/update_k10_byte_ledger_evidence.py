import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@4375ded"
EVIDENCE = "evidence-k10-kv-byte-ledger-20260816"
DECISION = "decision-k10-kv-byte-ledger-20260816"
TASK = "inquiry-kv-transport-quantitative-ledger-20260815"

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

EVIDENCE_CONTENT = """K10 KV 搬运量定量账本实现完成（commit 4375ded）。

实现：
1. KvTransport trait 新增 wire_bytes_sent()/wire_bytes_recv()（默认 0）；TCP 在 flush_send 累加写出字节、frame decode 累加读入字节；QUIC 在主线程 submit 累加 frame.len()、recv task 经 Arc<AtomicU64> 累加（recv_frame_from_stream 改为返回 (RingMessage, u64) frame wire 长度）。
2. 四个 perf event 统一补 wire_sent_bytes/wire_recv_bytes：ring_attention（prefill KV ring）、ring_decode（legacy Q-ring）、stationary_continuation、主线 stationary_decode（逐层 transport 计数器差值累计）。
3. scripts/kv_transport_ledger.py：聚合 HCP_PERF_LOG 为 per-request/per-token KV 搬运量台账，并给出同口径 vLLM PD（整段 KV 一次搬移 = layers*2*kv_heads*head_dim*seq*elem）与 TP（每层 all-reduce = seq*hidden*elem）参考字节。

验证：DYLD_LIBRARY_PATH=/Users/stark_sim/libtorch/lib cargo test --features tch-backend --lib = 160 passed / 0 failed / 5 ignored（新增 tcp_wire_bytes_account_for_self_driving_packet 与 quic_wire_bytes_account_for_self_driving_packet，断言 sent==recv==serialize frame len）；rustfmt --edition 2021 五个文件 + git diff --check 均 exit 0；ledger 脚本 AST parse + 合成 JSONL smoke 通过（HCP total / PD full-KV 比值输出正确）。

证据边界：这是 Mac 本地 CPU/mock/TCP/QUIC 的字节计量正确性证据，不是真实三机 MPS/CUDA/HIP 运行、不是真实 NIXL telemetry 对照、不是 TP all-reduce 实测；reference 字节是模型几何推导公式而非实测；wire bytes 是序列化 frame 字节（含 JSON meta + length prefix），与 NIXL transfer telemetry / TP all-reduce activation 字节口径对齐但未经跨栈实测核对。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "progress",
        "K10 完成：transport wire-byte 计量 + 四类 perf event 统一 bytes-on-wire + 聚合台账（160 tests 绿）",
        EVIDENCE_CONTENT, "verified", 1.0, 1.0, SOURCE)

    conn.execute("UPDATE nodes SET status='completed' WHERE id=?", (TASK,))

    edges = [
        (EVIDENCE, DECISION, "SUPPORTS", "wire-byte interface + ledger implemented and tested"),
        (EVIDENCE, TASK, "CONFIRMS", "K10 byte ledger delivered"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence=k10-kv-byte-ledger-20260816")
    print("task=completed")

if __name__ == "__main__":
    main()
