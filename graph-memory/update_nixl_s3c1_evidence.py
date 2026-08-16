import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@aedba92"

EVIDENCE = "evidence-nixl-s3c1-real-kv-transfer-20260816"
S3_TASK = "task-nixl-s3-cross-machine-transfer-20260816"

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

CONTENT = """NIXL block-direct S3c-1 真实 KV 几何跨机 transfer 验证通过（commit aedba92）。

【验证结果】nixl-xfer-probe 扩展为真实 KV 几何（bf16 [batch=1, num_kv_heads=2, seq, head_dim=64]，Qwen2-0.5B 的 KV 形状），white(CUDA)↔pearl(ROCm) 双向 register→transfer→poll→dump：
- seq=64：block len=16384（16KB = 2*64*64*2 bf16 bytes），双向 max|diff|=0.0（字节级一致），telemetry_bytes=16384，wire_bytes_sent=recv=16384。
- seq=1024：block len=262144（256KB），双向 max|diff|=0.0，telemetry_bytes=262144，wire_bytes_sent=recv=262144。
- 均 CROSS-MACHINE TRANSFER: PASS。

【价值】把 S3a 的 96-byte f32 冒烟扩展到真实 KV 规模（16KB~256KB bf16），证明 block-direct 的 register/transfer/poll 生命周期能处理真实 prefill KV block 的大小与精度，且字节级无损（memcpy 语义，不改变数值）。telemetry 字节对齐 K10 wire-byte 口径（block-direct 无序列化 meta 开销，真实字节 = block len）。

【边界】这是合成 KV 几何 tensor 的 transfer（arange pattern），不是真实模型 prefill 输出的 KV；不接 ring_attention 的 exchange 数据面；不覆盖 ring 的 double-buffering/轮次同步/动态 desc 交换。接进 ring_attention（S3c-2）仍为后续节点：那是 block-direct 数据面的生产接入，需在 KvTransport 字节流之外加 block-direct 分支 + 固定 buffer + 环上同步。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "active",
        "NIXL S3c-1 真实 KV 几何跨机 transfer 验证通过（16KB~256KB bf16，字节级一致）",
        CONTENT, "verified", 0.95, 1.0, SOURCE)

    edges = [
        (EVIDENCE, S3_TASK, "PART_OF", "S3c-1 real-KV transfer is the data-plane scale check before wiring into ring_attention"),
        (EVIDENCE, "evidence-nixl-s3a-cross-machine-transfer-20260816", "BUILDS_ON", "scales S3a's 96-byte probe to real KV geometry (bf16 [1,2,seq,64])"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence-nixl-s3c1-real-kv-transfer-20260816")

if __name__ == "__main__":
    main()
