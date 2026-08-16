import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@a032829"

EVIDENCE = "evidence-nixl-s4-0-laptop-onboard-20260816"

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

CONTENT = """NIXL S4-0：laptop（RTX 4060 Laptop，CUDA）接入 NIXL，单机 register/metadata 冒烟通过（commit a032829）。

【环境】laptop = RTX 4060 Laptop GPU（8188 MiB，sm_89 Ada，driver 610.43.02），libtorch CUDA 13（libcudart-96c42e41.so.13），Ubuntu 24.04。与 white（RTX 4090 CUDA）+ pearl（RX9060XT ROCm）同一 WiFi（laptop wlp3s0 192.168.8.109，white wlp11s0 192.168.8.173），Tailscale 100.96.154.1。至此 N=3 三机（white CUDA + pearl ROCm + laptop CUDA）均有 NIXL。

【接入步骤】
1. laptop 的 vllm-v1 conda 环境 pip install nixl==1.4.0（装上 nixl-cu12 + nixl-cu13，匹配 CUDA 13 libtorch，与 white 同布局）。
2. 缺 libclang-18-dev（clang 内置头 stdbool.h）导致 bindgen 报 fatal error: 'stdbool.h' file not found → apt install clang-18 libclang-18-dev 解决（laptop sudo 密码来自 inventory）。
3. scripts/nixl_transport_probe.sh 加 laptop case（conda 轮 nixl_cu13 + clang 18 + 无 LD_PRELOAD，复用 white 布局但 LIBCLANG_PATH=/usr/lib/llvm-18/lib）。
4. cargo build --features tch-backend,nixl-backend --bin nixl-probe 编译通过；run 输出 agent created + registered block id=0 len=96 addr=1099683594240（VRAM 地址）+ local metadata bytes=778 + OK。

【意义】N=3 的硬件前提成立：三台 GPU（CUDA×2 + ROCm×1）都在同一 WiFi 且都有 NIXL。S3c-2 的 N=2 特化因此被放弃，改为 S4 直接做 N=3 通用 block 数据面接 ring，避免 N=2 特化的沉没成本。laptop 是 CUDA（与 white 同构），为 N=3 带来一对「同构 CUDA↔CUDA」邻居（white↔laptop），可首次验证同构 GPU-direct VRAM 是否可行（此前 S3a 只验证了异构 host DRAM staging）。

【边界】单机 register/metadata 冒烟；未做跨机 transfer、N=3 ring、paged-KV。跨机 N=3 验证（S4-1）与 block 数据面接 ring（S4-2）为后续。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "active",
        "NIXL S4-0：laptop（RTX 4060 CUDA）接入 NIXL，三机 N=3 硬件前提成立",
        CONTENT, "verified", 0.95, 1.0, SOURCE)

    edges = [
        (EVIDENCE, "decision-nixl-s3c2-ring-wiring-20260816", "FOLLOWS", "laptop onboard enables N=3, so S3c-2's N=2 specialization is dropped for a generic N=3 S4"),
        (EVIDENCE, "evidence-nixl-sys-white-cuda-verified-20260816", "BUILDS_ON", "laptop mirrors white's conda-wheel nixl_cu13 layout"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence-nixl-s4-0-laptop-onboard-20260816")

if __name__ == "__main__":
    main()
