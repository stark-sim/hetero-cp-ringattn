import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@nixl-s3c2-ring-wiring-20260816"

DECISION = "decision-nixl-s3c2-ring-wiring-20260816"
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

CONTENT = """【NIXL block-direct 接进 prefill KV ring（S3c-2）的动机六问（2026-08-16）】

1. 问题：prefill KV ring 的数据面现在走 KvTransport（QUIC 字节流，序列化 KvBlock 为 JSON meta + length prefix + tensor bytes）。block-direct（KvBlockTransport register/transfer）只在独立探针验证过（S3a/S3c-1），没接进 ring_attention 的生产数据面。"prefill KV ring 走 block 路径"是 S3 的收官目标，未达成。

2. 现状：HcpRingAttentionBackend 有 kv_transport: Option<Box<dyn KvTransport>>，ring_attention 的 exchange（串行/pipeline 两模式）用 submit_send(KvBlock)+recv_kv_block 交换 micro block。KvBlockTransport trait + NixlBlockTransport（register/transfer/poll）已实现并通过 S3a（生命周期）+S3b（控制面 side-channel）+S3c-1（真实 KV 规模 16KB~256KB 字节级一致）。两者接口不同构：字节流（submit_send/recv）vs block-direct（register/transfer/poll）。

3. 终态：prefill KV ring 在"有 block transport 时"走 register/transfer（不序列化），N=2 跨机数值对照 single-node reference（attention 输出一致）；QUIC 字节流回退保留（无 block transport 时）。N=2 是"一轮双向交换"（num_rounds=1），无需 double-buffering/多轮转发，是最小接入面。

4. 他者：vLLM PD 用 NixlConnector 做 prefill→decode 整段 KV 一次 block-direct 搬移（register → 独立 TCP side channel 交换 desc → transfer）。HCP ring 是逐 hop + micro block + capacity-weighted 分片，数据流不同构；且 HCP 的 desc 交换复用 coordinator 控制面（S3b 已建），不新增端口。可复用"register → side channel 交换 desc → transfer"生命周期，但 ring 的 micro block 是运行时切分、desc 动态，需预注册固定 buffer 或每轮交换 desc。

5. 本方案：N=2 特化优先。HcpRingAttentionBackend 加 block_transport 字段；ring_attention 在 num_domains==2 且 block_transport 存在时走 block-direct 分支——register 本地 KV micro block + 接收 buffer，经控制面/预留通道交换 desc，submit_transfer + poll，读接收 buffer 得 peer KV，再走既有 process_kv_block 计算。数值对照 single-node reference。

6. 为什么：block-direct 的价值是不序列化（真实 KV 直接 register/transfer），跨异构虽 host staging（S3a 发现：跨厂商 GPU-direct 无协议），但仍省 JSON meta + length prefix + 反序列化开销；且这是 S4（paged-KV 化，block_size=16 + block_table，对接 vLLM paged KV）的前置——block 数据面的 ring 接入是 S4 的必经之路。代价/边界：跨异构 host staging 让 block-direct 相对字节流的收益有限（用户已判定"NIXL 收益不多"），本节点的价值主要是"block 数据面闭环"与"S4 前置"，而非 NIXL 零拷贝的量化收益。

VERDICT: IMPLEMENT（N=2 特化，作为 S3 收官 + S4 前置；不追求通用 N 节点 double-buffering，那是 S4 的生产化）。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, DECISION, "decision", "active",
        "NIXL block-direct 接进 prefill KV ring（S3c-2）动机六问：N=2 特化，S3 收官 + S4 前置",
        CONTENT, "held", 0.95, 1.0, SOURCE)

    edges = [
        (DECISION, S3_TASK, "PART_OF", "S3c-2 ring wiring is the final checkpoint of the S3 task"),
        (DECISION, "evidence-nixl-s3c1-real-kv-transfer-20260816", "BUILDS_ON", "wires the real-KV block-direct transfer validated in S3c-1 into ring_attention"),
        (DECISION, "evidence-nixl-s3b-sidechannel-control-plane-20260816", "BUILDS_ON", "reuses the coordinator control-plane side channel for desc exchange"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("decision-nixl-s3c2-ring-wiring-20260816")

if __name__ == "__main__":
    main()
