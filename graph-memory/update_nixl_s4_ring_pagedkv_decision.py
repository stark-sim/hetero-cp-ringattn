import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@nixl-s4-ring-pagedkv-20260816"

DECISION = "decision-nixl-s4-ring-pagedkv-20260816"

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

CONTENT = """【NIXL block 数据面接 prefill KV ring（N=3）+ paged-KV 化（S4-2/3）的动机六问（2026-08-16）】

1. 问题：prefill KV ring 仍走 KvTransport 字节流（序列化 KvBlock），block 数据面（KvBlockTransport register/transfer）只验证了独立 transfer（S3a/S3c-1/S4-1），未接进 ring_attention 生产数据面；KV 也未 paged 化（block_size=16 + block_table），不能对接 vLLM paged KV。这是 S4 的核心目标。

2. 现状：HcpRingAttentionBackend 有 kv_transport: Option<Box<dyn KvTransport>>，ring_attention exchange 用 submit_send(KvBlock)+recv_kv_block。KvBlockTransport + NixlBlockTransport 已验证 register/transfer/poll 生命周期 + 控制面 side-channel（S3b）+ 真实 KV 规模（S3c-1）+ N=3 三机两两（S4-1，同构 CUDA VRAM 可行、异构 host DRAM）。三机（white CUDA + pearl ROCm + laptop CUDA）同一 WiFi（192.168.8.x）均有 NIXL。但 block-direct 未接 ring，KV 未 paged 化。

3. 终态：prefill KV ring 在"有 block transport 时"走 register/transfer（不序列化），N=3 跨机数值对照 single-node reference；KV 切成 paged block（block_size=16 token + block_table），对齐 vLLM 物理 block 形状；QUIC 字节流回退保留。

4. 他者：vLLM 用 paged KV（block_size=16 + block_table）+ NixlConnector（register → side channel 交换 desc → transfer 整段 KV）。HCP 复用其 paged block 形状（BlockDesc 与 vLLM 物理 block 同构，S1 设计），但数据流是 ring 逐 hop + capacity-weighted 分片，非整段搬移；desc 交换复用 coordinator 控制面（S3b），非独立 TCP side channel。

5. 本方案：分两步。S4-2 先做 N=3 通用 block-direct ring（double-buffering + 每轮同步 + desc 交换，不特化 N=2）；S4-3 再做 paged-KV 化（KV 切 block_size=16 物理 block + block_table，每个 paged block 独立 register/transfer）。两步紧耦合（paged block 定义与 block 数据面 transfer 一起设计），但先 S4-2（数据面机制，用合成 KV tensor 验证 ring 逐 hop 正确性）再 S4-3（paged 结构 + 接 vLLM 形状）。

6. 为什么：block 数据面的价值是"不序列化"（真实 KV 直接 register/transfer），且是 S4-4（对接 vLLM paged KV）的必经之路——vLLM 物理 block 与 BlockDesc 同构，paged-KV 化后 block 数据面就是 vLLM paged KV 的传输层。代价/边界：跨异构 host staging（消费级 WiFi 无 RDMA）让 block-direct 相对字节流的收益有限（省 JSON 序列化，不省 host staging 拷贝），本节点的价值主要是"block 数据面闭环 + vLLM paged KV 前置"，非 NIXL 零拷贝的量化收益。

VERDICT: IMPLEMENT（S4-2 先做 N=3 通用 block-direct ring 数据面验证，S4-3 再做 paged-KV；不特化 N=2，避免沉没成本）。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, DECISION, "decision", "active",
        "NIXL block 数据面接 prefill KV ring（N=3）+ paged-KV 化（S4-2/3）动机六问",
        CONTENT, "held", 0.95, 1.0, SOURCE)

    edges = [
        (DECISION, "evidence-nixl-s4-1-n3-transfer-20260816", "BUILDS_ON", "N=3 edges validated, wires block-direct into the N=3 ring"),
        (DECISION, "evidence-nixl-s3b-sidechannel-control-plane-20260816", "BUILDS_ON", "reuses the coordinator control-plane side channel for desc exchange"),
        (DECISION, "decision-nixl-s3c2-ring-wiring-20260816", "SUPERSEDES", "S3c-2's N=2 specialization is dropped for a generic N=3 S4"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("decision-nixl-s4-ring-pagedkv-20260816")

if __name__ == "__main__":
    main()
