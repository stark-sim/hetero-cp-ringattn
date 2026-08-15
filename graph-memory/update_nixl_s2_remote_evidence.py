import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@106d2e2"
EVIDENCE = "evidence-nixl-s2-remote-verified-20260816"
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

EVIDENCE_CONTENT = """NIXL block-direct transport S2 在 pearl(ROCm) 真实运行时验证通过（commit 106d2e2）。

验证链路（在 pearl 上）：
1. git pull --ff-only 同步到 106d2e2。
2. cargo build --features tch-backend,nixl-backend --bin nixl-probe 真实链接成功（NIXL_LD 指向 /home/stark/build/nixl-1.4.0/build/src 的 bindings/core/infra/utils/serdes/utils/stream/utils/common/plugins/ucx）。
3. 运行探针（LD_PRELOAD=/home/stark/libtorch/lib/libtorch_hip.so + NIXL_PLUGIN_DIR=.../plugins/ucx + HCP_TCH_DEVICE=cuda:0）输出：
   - agent created: hcp-probe-agent
   - registered block id=0 len=96 addr=125726875451392
   - local metadata bytes=686
   - OK

过程中定位的三个真实障碍（已修复，均 commit）：
a) 手写 extern "C" 而非 bindgen nixl-sys——nixl-sys 的 bindgen 在 white/pearl 都缺 clang-dev（white 无 clang 二进制、pearl 的 ROCm libclang 版本串 23.0git 不被 bindgen 识别），装 clang-dev 需 sudo；手写 FFI 绑定官方 libnixl_capi.so（稳定 C ABI，正是 bindgen 的对象）绕开该依赖。
b) 裸 agent 报 "no available backends for mem type VRAM_SEG"——官方示例 single_process_example.rs 显示必须 create_backend("UCX") + opt_args_add_backend，并把 opt_args 传入 register_mem；已补齐插件发现/get_plugin_params/create_backend/opt_args_add_backend FFI，register/deregister/create_xfer_req/post_xfer_req 全部携带 opt_args，Drop 逆序销毁 backend/opt_args/agent。
c) 运行时需 NIXL_PLUGIN_DIR 指向 plugins/ucx（否则插件发现报目录不存在）；VRAM 注册需 LD_PRELOAD=libtorch_hip.so 否则 tch 返回 host 指针被 UCX 拒为 host。

证据边界：这是单主机、单 block、register+local_metadata 的运行时正确性验证（pearl ROCm 单机）。不覆盖跨机 CUDA↔ROCm 的 register→transfer→poll 全生命周期、不覆盖双 agent metadata 交换（side channel）、不覆盖 prefill KV ring 接线（S3）或 paged-KV 化（S4）。white(CUDA) 端未跑（其 libnixl_capi.so 在 conda 轮内，路径不同，待 S3 时统一）。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "progress",
        "NIXL S2 真实验证：pearl(ROCm) 手写 FFI 链接 + agent 创建 + UCX backend 实例化 + VRAM block 注册 + metadata 拉取全通过",
        EVIDENCE_CONTENT, "verified", 0.95, 0.95, SOURCE)

    edges = [
        (EVIDENCE, DECISION, "CONFIRMS", "form-B block transport FFI links + runs on real NIXL/UCX"),
        (EVIDENCE, TASK, "SUPPORTS", "S2 done; S3 cross-node transfer + ring wiring remain"),
        (EVIDENCE, "evidence-nixl-block-transport-s1s2-20260816", "REFUTES", "S1S2 evidence claimed NIXL FFI only Mac type-checked, runtime unverified; this confirms runtime now"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence=nixl-s2-remote-verified-20260816")

if __name__ == "__main__":
    main()
