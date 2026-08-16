import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@0112e75"

EVIDENCE = "evidence-nixl-s3b-sidechannel-control-plane-20260816"
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

CONTENT = """NIXL block-direct S3b 跨机 side-channel 接 coordinator 控制面验证通过（commit 0112e75）。

【验证结果】white(CUDA)+pearl(ROCm) 双 worker + coordinator，NIXL metadata 经 coordinator QUIC 控制面交换，load_remote_metadata 成功（UCX 连接建立）：
- coordinator 打印 worker 0/1 各 reported NIXL metadata 405/409 bytes、descs 45 bytes，exchanged NIXL metadata across 2 workers。
- worker 0(white) loaded NIXL metadata from domain 1 (agent hcp-worker-1)；worker 1(pearl) loaded NIXL metadata from domain 0 (agent hcp-worker-0)。
- 脚本 scripts/nixl_sidechannel_probe.sh 判定 SIDE-CHANNEL EXCHANGE: PASS。

【实现】
1. protocol.rs 加三个变体（纯 bytes，coordinator 透明转发不反序列化，不依赖 nixl-backend）：WorkerCommand::NixlExchange（请求上报）、WorkerCommand::NixlPeers{peers: Vec<(u64, Vec<u8>, Vec<u8>)>}（广播 peer 的 domain_id+metadata+descs）、WorkerResponse::NixlMetadata{metadata, block_descs}（worker 上报）。
2. coordinator.rs 加 exchange_nixl_metadata：handshake 后广播 NixlExchange → 收集各 worker NixlMetadata → 广播 NixlPeers；--nixl-exchange 标志触发后 shutdown。coordinator 是唯一拓扑知识源，NIXL metadata 走既有控制面，不新增 side-channel 端口。
3. worker_sdk/runtime.rs：nixl-backend 下 WorkerRuntime::new 创建 NixlBlockTransport（agent hcp-worker-{domain_id}）+ 注册 probe block；NixlExchange 上报 local_metadata+block desc，NixlPeers 对每个 peer load_remote_metadata（跳过 self，agent 不能 load 自己的 metadata）。

【关键设计】side channel 复用 HCP 控制面（coordinator），不新增端口/依赖——与 vLLM PD 的独立 TCP side channel（VLLM_NIXL_SIDE_CHANNEL_HOST/PORT）不同。HCP 的 coordinator 本就掌握全拓扑，是 metadata 交换的天然枢纽；worker 与 coordinator 的 QUIC 双向流同时承载推理命令（Prefill/Decode/StationaryDecode）和 NIXL metadata（NixlExchange/NixlPeers），协议层只扩变体、不加连接。

【证据边界】N=2 white↔pearl、NIXL metadata 交换 + load_remote_metadata（UCX 连接建立）；probe block 是 1x2x3x4 host 内存（DRAM），不涉及真实 KV block 的 transfer（那是 S3c 的数据面）。不覆盖 prefill KV ring 走 block 路径、真实 KV tensor 注册、多请求并发、性能。S3c 是本系列最后一步。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "active",
        "NIXL S3b 跨机 side-channel 接 coordinator 控制面验证通过（metadata 走既有 QUIC 控制面，不新增端口）",
        CONTENT, "verified", 0.95, 1.0, SOURCE)

    edges = [
        (EVIDENCE, S3_TASK, "PART_OF", "S3b side-channel is the second checkpoint of the S3 task"),
        (EVIDENCE, "evidence-nixl-s3a-cross-machine-transfer-20260816", "BUILDS_ON", "S3b replaces S3a's file-exchange probe channel with the coordinator control plane"),
        (EVIDENCE, "decision-nixl-s3-cross-machine-20260816", "PART_OF", "S3b evidence for the S3 cross-machine verification decision"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence-nixl-s3b-sidechannel-control-plane-20260816")

if __name__ == "__main__":
    main()
