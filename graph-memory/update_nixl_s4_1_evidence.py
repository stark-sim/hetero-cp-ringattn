import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@a13009a"

EVIDENCE = "evidence-nixl-s4-1-n3-transfer-20260816"

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

CONTENT = """NIXL S4-1：N=3 三机两两跨机 transfer 验证通过（commit a13009a）。

【验证结果】scripts/nixl_transfer_pair.sh（参数化 host 对，复用 nixl-xfer-probe）：
- white↔laptop（CUDA↔CUDA 同构，host DRAM + UCX_TLS=tcp）：双向 max|diff|=0.0，PASS。
- pearl↔laptop（ROCm↔CUDA 异构，host DRAM + UCX_TLS=tcp）：双向 max|diff|=0.0，PASS。
- white↔laptop（同构，--device cuda VRAM + --no-tcp 允许 GPU-direct）：双向 max|diff|=0.0，PASS。
- white↔pearl（异构）已由 S3a 验证。至此 N=3 三机（white CUDA + pearl ROCm + laptop CUDA）任意两两 transfer 均字节级一致。

【关键发现】同构 CUDA↔CUDA 的 VRAM transfer 可行（--device cuda + 默认 UCX_TLS），而异构 CUDA↔ROCm 的 VRAM put 无 UCX 协议（S3a 已证）。印证：NIXL 的 VRAM 传输「同构可行、异构退化 host DRAM staging」。但本环境三机走 WiFi（192.168.8.x），无 InfiniBand/RoCE，所以同构 VRAM 的 transport 应是 cuda_copy（host staging）而非 gdr_copy（真 GPU-direct RDMA）——消费级网络下 NIXL 零拷贝无法兑现，强化 CXL/类 RDMA 论据（网络自由=手段，上限白吃 CXL 带宽）。

【网络修复】laptop 无法访问 white 有线 192.168.100.1（laptop 走 WiFi），UCX 默认选 white 的 enp10s0（有线）导致 laptop load white md 报 "No route to host"。修复：脚本按 host 设 UCX_NET_DEVICES 到 WiFi 接口（white=wlp11s0, laptop=wlp3s0, pearl=wlo1），三机 192.168.8.x 互通。代价：white↔pearl 的 0.15ms 有线直连被 WiFi 取代（一个 agent 只能绑定一个 UCX 网络设备，无法对 pearl 用有线、对 laptop 用 WiFi）。

【边界】三机两两 transfer（合成 KV 几何 bf16 [1,2,seq,64]），非 N=3 环上逐 hop，非 ring_attention 数据面，非 paged-KV。S4-2（block 数据面接 N=3 ring）与 S4-3（paged-KV）为后续。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "active",
        "NIXL S4-1：N=3 三机两两 transfer 验证通过（同构 CUDA VRAM 可行，异构退化 host staging）",
        CONTENT, "verified", 0.95, 1.0, SOURCE)

    edges = [
        (EVIDENCE, "evidence-nixl-s4-0-laptop-onboard-20260816", "BUILDS_ON", "N=3 edges validated after laptop onboard"),
        (EVIDENCE, "evidence-nixl-s3c1-real-kv-transfer-20260816", "BUILDS_ON", "reuses the real-KV geometry probe across the third node"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence-nixl-s4-1-n3-transfer-20260816")

if __name__ == "__main__":
    main()
