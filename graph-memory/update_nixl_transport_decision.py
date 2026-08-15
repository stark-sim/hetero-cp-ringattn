import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@nixl-transport-decision-20260816"

TASK = "inquiry-nixl-as-hcp-transport-20260815"
DECISION = "decision-nixl-as-transport-20260816"

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

DECISION_CONTENT = """【NIXL 接入 transport trait 的动机六问（2026-08-16）】

1. 问题：HCP 的 KvTransport trait 目前只有 QUIC（生产）+ TCP（测试）+ Mock（单测）三个实现。用户要求把 NIXL 接上 transport trait，作为 QUIC/TCP 之外的第三种通信选择——服务"网络自由=手段"卖点（P2P ring 不绑定厂商 collective 栈，从 2.5GbE 到 CXL 都能跑）。这是 K4（评估 NIXL 作为 HCP ring 传输轮子）从"评估"升级为"实现"。

2. 现状：NIXL 是 C++/CUDA/ROCm 库（ai-dynamo/nixl），语义是 block 级 GPU 内存传输（Agent → register_memory → transfer → notification），有官方 Rust 绑定（nixl-sys FFI crate + 高层 nixl crate，crates.io 1.4.0，bindgen 构建）。KvTransport 是"序列化 tensor → frame bytes → 流式 send/recv"抽象。两者语义不同构：NIXL 是 block 传输 + 通知，不是字节流。NIXL 只能在 white(CUDA)/pearl(ROCm) 构建运行（已探明：pearl 有 /home/stark/build/nixl-1.4.0 源码构建 + libnixl.so + src/api/cpp/nixl.h；white 有 conda 轮内 libnixl.so），Mac 无 UCX/CUDA/ROCm，无法本地构建或运行 NIXL。

3. 终态：cargo feature（如 nixl-backend，默认 off）门控的 NixlKvTransport 实现 KvTransport，作为第三种传输；默认 Mac 构建保持绿（feature off）；feature on 只在 white/pearl 编译 + smoke。NIXL 侧的 wire_bytes_sent/recv 复用 K10 刚建的计量接口（用 getXferTelemetry 或 transfer 字节数填同字段）。

4. 他者：vLLM PD 用 NixlConnector 做 prefill→decode 整段 KV 搬移（block 级、GPU-direct）。Dynamo 的 block_manager/storage/nixl.rs 是同一 FFI 的官方用法。NIXL 的既有价值全在 GPU-direct block 传输（避免序列化+拷贝）；若只是"用 NIXL 搬序列化后的 frame bytes"，等于把 UCX 当字节管道，丢掉了 NIXL 的零拷贝优势。

5. 本方案（两形态，需用户裁定）：
   - 形态 A（frame-carrier，薄适配）：serialize frame → 注册 host/device buffer → NIXL transfer → 对端 deserialize。复用 K10 wire-byte 口径，trait 不变，是"第三种传输"的字面实现（网络自由），但不兑现 NIXL 零拷贝价值。Mac 不可构建，验证在 white/pearl。
   - 形态 B（block-direct，新抽象）：直接注册 K/V tensor device 内存做 block 传输，绕开序列化。兑现 NIXL 价值，但 KvTransport trait 是字节流语义，需要新的 block-transport trait 面或并行抽象，改动远大于"接上 transport trait"。

6. 为什么：用户指令是"接上 transport trait"——字面对应形态 A。形态 A 是网络自由卖点的正确落点（第三种可插拔传输），且 K10 刚建的 wire-byte 接口让 NIXL 与 QUIC/TCP 同口径可对比。形态 B 是更大的架构重构（block 级数据面 + 调度改造），应作为独立后续节点，不混入本节点。硬约束：本节点代码 Mac 上只能 cargo check 门控状态（feature off 绿），真实编译+smoke 必须在 white/pearl 经 git pull + rebuild。

VERDICT: 待用户裁定形态 A（frame-carrier，推荐，字面接 trait + 网络自由）vs 形态 B（block-direct，独立大节点）。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    conn.execute("UPDATE nodes SET status='active' WHERE id=?", (TASK,))

    upsert_node(conn, DECISION, "decision", "active",
        "NIXL 接入 transport trait 决策：frame-carrier（薄适配，推荐）vs block-direct（独立大节点）",
        DECISION_CONTENT, "held", 0.9, 1.0, SOURCE)

    edges = [
        (DECISION, TASK, "PART_OF", "NIXL transport integration decision"),
        (DECISION, "decision-k10-kv-byte-ledger-20260816", "BUILDS_ON", "reuses the wire-byte accounting interface for NIXL telemetry"),
        (DECISION, "decision-hcp-first-principles-value-20260815", "FOLLOWS", "serves network-freedom = P2P ring over any transport"),
        (DECISION, "evidence-phase3-pd-spike-green-20260814", "BASED_ON", "NIXL/UCX CUDA-ROCm heterogeneous transfer already proven in vLLM PD spike"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("decision=nixl-as-transport-20260816")

if __name__ == "__main__":
    main()
