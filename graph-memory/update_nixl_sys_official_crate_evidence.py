import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@367fe04"
EVIDENCE = "evidence-nixl-sys-official-crate-verified-20260816"
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

EVIDENCE_CONTENT = """NIXL block transport 改用官方 nixl-sys crate 并真实验证通过（commit 367fe04）。

裁决（用户 2026-08-16）：S2 应复用官方 nixl-sys crate（bindgen FFI + safe Agent/Backend/XferDescList/XferRequest 封装），而非手写 extern "C"。white/pearl 用 inventory 的 sudo 密码装 clang + libclang-dev。

实现：
1. rust/Cargo.toml：nixl-sys = { version = "1.4", optional = true, features = ["stub-api"] }；nixl-backend = ["dep:nixl-sys"]。stub-api 使 crate 运行时 dlopen libnixl_capi.so（无需构建期链接 libnixl），对 pearl 源码构建树与 white conda 轮都适用。
2. rust/src/distributed/transport/nixl.rs 重写：删手写 extern "C"（~500 行），改用 nixl_sys::{Agent, Backend, OptArgs, MemType, XferDescList, XferOp, XferRequest, NixlDescriptor, MemoryRegion, RegistrationHandle}。自定义 VramRegion 实现 MemoryRegion + NixlDescriptor（MemType::Vram）注册 device tensor；Agent::new → get_available_plugins 确认 UCX → get_plugin_params + create_backend("UCX") → OptArgs::add_backend；register_block 用 register_memory、deregister 靠 RegistrationHandle 的 Drop；submit_transfer 用 create_xfer_req(XferOp::Write) + post_xfer_req；poll_transfers 用 get_xfer_status + XferRequest::get_telemetry().total_bytes 填 K10 wire-byte 口径。unsafe impl Send（OptArgs 内部 NonNull 非 Send，与官方对 Backend/XferRequest 的 unsafe Send 一致）。KvBlockTransport trait 面不变。
3. clang 安装：white(Ubuntu 26.04) clang 21.1.8 + libclang-21-dev；pearl(Ubuntu 24.04) clang 18.1.3 + libclang-18-dev；两机 stdbool.h 均能找到。

验证：
- pearl 上 cargo build --features tch-backend,nixl-backend --bin nixl-probe 真实编译链接通过（nixl-sys bindgen + stub-api dlopen libnixl_capi.so）。
- 探针输出：agent created / registered block id=0 len=96 / local metadata bytes=686 / OK，与手写 FFI 版一致。
- spike：官方 crate is_stub()==false 且完整 register(host block)+get_local_md 生命周期跑通。
- Mac：cargo build/test --features tch-backend --lib = 161 passed / 0 failed / 5 ignored（nixl feature off 不碰 nixl-sys）；rustfmt + git diff --check 绿。

证据边界：这是单机 pearl(ROCm) 的 register + local_metadata 运行时验证（官方 crate 版），不覆盖跨机 CUDA↔ROCm register→transfer→poll 全生命周期、不覆盖双 agent side-channel 交换、不覆盖 prefill KV ring 接线（S3）或 paged-KV 化（S4）。white(CUDA) 端未用 nixl-sys 版跑（其 libnixl_capi.so 在 conda 轮内路径不同，S3 时统一）。Mac 上 nixl-backend 无法编译（无 libclang.dylib），nixl-sys 仅在 white/pearl 编译。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "progress",
        "NIXL 改用官方 nixl-sys crate 并真实验证：pearl bindgen 编译 + register/metadata 运行通过（367fe04）",
        EVIDENCE_CONTENT, "verified", 0.95, 0.95, SOURCE)

    edges = [
        (EVIDENCE, DECISION, "CONFIRMS", "official nixl-sys crate is the reused wheel; hand-written FFI replaced"),
        (EVIDENCE, TASK, "SUPPORTS", "S2 official-crate path verified; S3 cross-node transfer + ring wiring remain"),
        (EVIDENCE, "evidence-nixl-s2-remote-verified-20260816", "REPLACED_BY", "hand-written FFI evidence superseded by official nixl-sys crate evidence"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence=nixl-sys-official-crate-verified-20260816")

if __name__ == "__main__":
    main()
