import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@358c4c5"
EVIDENCE = "evidence-nixl-block-transport-s1s2-20260816"
DECISION = "decision-nixl-as-transport-20260816"
TASK = "inquiry-nixl-as-hcp-transport-20260815"

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

EVIDENCE_CONTENT = """NIXL block-direct transport 实现 S1+S2 完成（commit 358c4c5，形态 B）。

实现：
1. S1 — KvBlockTransport trait（block 级数据面：register → side-channel metadata → async transfer → poll），描述符模型 addr+len+dev_id+meta 对齐 NIXL nixlBasicDesc 与 vLLM 物理 block；SerializedBlockTransport in-memory fallback 作为 Mac 可测参考基线（register → metadata 交换 → submit_transfer → poll_transfers round-trip，wire_bytes sent==recv==frame size）。
2. S2 — NixlBlockTransport FFI：手写声明稳定 C API（nixl_capi_*，libnixl_capi.so），实现 KvBlockTransport（register_mem → get_local_md/load_remote_md → create_xfer_req/post_xfer_req → get_xfer_status/get_xfer_telemetry），telemetry.total_bytes 填 K10 wire-byte 口径。feature nixl-backend 门控（默认 off）。
3. 设计文档 docs/NIXL_BLOCK_TRANSPORT.md：数据面分两路径（prefill KV ring=block-direct，decode SD packet=字节流不动）、side channel 复用 HCP 控制面、paged-KV 化（block_size=16 + block_table）后置为 S4。
4. 脚本 scripts/nixl_transport_probe.sh：white/pearl 远程 build/probe 入口。

验证：
- 手写 extern "C"（非 bindgen nixl-sys）使 Mac 无需 libclang 即可 cargo check --features tch-backend,nixl-backend --lib 类型检查通过（check 不链接）。
- Mac 默认路径 cargo build/test --features tch-backend --lib = 161 passed / 0 failed / 5 ignored（新增 serialized_block_transport_roundtrips_registered_block）。
- rustfmt + git diff --check 绿；clippy nixl-backend 无 nixl.rs 诊断。
- NIXL C API 符号面已从 pearl /home/stark/build/nixl-1.4.0/src/bindings/rust/wrapper.h 逐函数核对（create_agent/register_mem/get_local_md/load_remote_md/create_xfer_req/post_xfer_req/get_xfer_status/get_xfer_telemetry/notif map）。

证据边界：S2 的 NixlBlockTransport 仅在 Mac 上做了类型检查（cargo check，feature on 不链接），未在 white/pearl 做真实 link + register→transfer→poll 运行时 smoke——这是 S2 的剩余验证，需远端 git pull + rebuild（scripts/nixl_transport_probe.sh）。本证据不声称 NIXL FFI 运行时正确，不覆盖 prefill KV ring 接线（S3）或 paged-KV 化（S4）。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "progress",
        "NIXL block-direct transport S1+S2：KvBlockTransport trait + fallback（Mac 绿）+ NixlBlockTransport FFI（Mac 类型检查，远端 smoke 待做）",
        EVIDENCE_CONTENT, "verified", 0.9, 0.9, SOURCE)

    # K4 inquiry: S1+S2 delivered, but S3 (ring wiring) + remote smoke remain -> keep active, not completed.
    edges = [
        (EVIDENCE, DECISION, "SUPPORTS", "form-B block transport trait + NIXL FFI implemented"),
        (EVIDENCE, TASK, "SUPPORTS", "S1+S2 done; S3 ring wiring + remote smoke remain"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence=nixl-block-transport-s1s2-20260816")

if __name__ == "__main__":
    main()
