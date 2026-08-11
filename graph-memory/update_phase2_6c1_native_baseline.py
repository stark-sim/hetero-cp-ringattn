import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@6c1"

TASK = "task-phase2-rust-6c1-native-service-baseline-20260812"
EVIDENCE = "evidence-phase2-rust-6c1-native-baseline-20260812"
SESSION = "session-next-phase2-rust-6c1-native-baseline-20260812"
SESSION_NEXT = "session-next-phase2-rust-6d-n3-service-20260812"
TASK_6D = "task-phase2-rust-6d-n3-service-readiness-20260812"
TASK_6C0 = "task-phase2-benchmark-6c0-observability-20260812"
EV_6C0 = "evidence-phase2-rust-6c0-observability-20260812"
SCRIPT = "scripts/test_phase2_6c1_native_baseline.sh"


def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence, source):
    conn.execute(
        """
        INSERT INTO nodes
        (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type,layer=excluded.layer,project=excluded.project,
          title=excluded.title,content=excluded.content,importance=excluded.importance,
          confidence=excluded.confidence,status=excluded.status,source=excluded.source,
          updated_at=datetime('now')
        """,
        (node_id, node_type, layer, PROJECT, title, content, importance, confidence, status, source),
    )


def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        """
        INSERT INTO edges(source,target,type,weight,note)
        VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note
        """,
        (source, target, edge_type, note),
    )


TASK_RESULT = """[2026-08-12 完成] 6c.1 native 服务稳定性基线。
新增 scripts/test_phase2_6c1_native_baseline.sh：本地 Mac MPS 2-domain 真实 HTTP 服务（HCP_TCH_DEVICE=mps、Qwen2-0.5B），带 --trace-jsonl 运行 concurrency 1（2 串行不等长）、2（2 同时不等长）、4（4 同时不等长），共 8 个请求。
验收：1) 0 错误：8/8 请求完成，metrics failed=0、active=0；2) token/reference：greedy 生成均非 [error:，text 非空；3) queue/active/release：metrics total=completed=8、queued=0；4) reserved bytes 与 release 一致（trace 每请求 reserved==released，字节数随 prompt/decodes 单调增长）；5) ring hops/bytes：trace 断言 prefill_hops=24=L*(N-1)（N=2 L=24）、decode_hops=decode_steps*24。
边界：这是二期内部稳定性基线，不是 vLLM benchmark 性能结论；仅 Mac 本机 MPS 2-domain（N=3 异构见 6d）。"""

EVIDENCE_CONTENT = """实现脚本 scripts/test_phase2_6c1_native_baseline.sh；配套实现 78be1d0（6c.0 trace 平面，本基线复用）。
验证（本地 Mac MPS、N=2、L=24、Qwen2-0.5B、release build）：
1. concurrency 1：2 串行不等长请求（prompt 2/33 tokens，max_tokens 3/8）→ responses=2 errors=0。
2. concurrency 2：2 同时不等长（prompt 8/3，max 6/2）→ responses=2 errors=0。
3. concurrency 4：4 同时不等长（prompt 2/13/26/3，max 4/10/7/1）→ responses=4 errors=0。
4. /metrics 后验：total=completed=8、failed=0、queued=0、active=0。
5. trace 8 条记录：request_id 1..8；每请求 prefill_accepted/completed elapsed > 0、error=null、reserved==released（如 req2 prompt33 max8 reserved=[258048,245760]）；prefill_hops=24=L*(N-1)、decode_hops=steps*24；finish_reason 均 length。
证据边界：Mac 本机 MPS 2-domain 稳定性基线；不证明 N=3 异构（white CUDA/pearl HIP）、真实网络、吞吐或 vLLM 兼容；不含性能结论。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "6c.1：Rust/native client 分级服务稳定性基线",
        TASK_RESULT,
        "completed",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "active",
        "6c.1 native 服务稳定性基线通过（N=2 concurrency 1/2/4）",
        EVIDENCE_CONTENT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )
    conn.execute("UPDATE nodes SET status='closed', updated_at=datetime('now') WHERE id=?", (SESSION,))
    upsert_node(
        conn,
        SESSION_NEXT,
        "session",
        "active",
        "二期下一检查点：6d N=3 异构 Rust 服务 readiness",
        "6c.1 已完成并验证：N=2 本地 MPS concurrency 1/2/4 稳定性基线 0 错误，trace 与 metrics 一致。下一候选节点保持 pending：Mac MPS + white CUDA + pearl HIP N=3 production QUIC 真实 Qwen 服务闭环（neighbor-only ring）。二期 benchmark-readiness 五项（API/资源/调度/观测/稳定性）在 6d 完成后收口。",
        "active",
        0.9,
        1.0,
        SOURCE,
    )

    edges = (
        (TASK, EVIDENCE, "CONFIRMS", "6c.1 task confirmed by evidence"),
        (EVIDENCE, TASK, "SUPPORTS", "evidence supports task completion"),
        (EV_6C0, EVIDENCE, "SUPPORTS", "trace plane enables 6c.1 correlation"),
        (TASK_6C0, TASK, "DEPENDS_ON", "6c.1 builds on 6c.0 observability"),
        (SCRIPT, EVIDENCE, "BASED_ON", "baseline script is the repeatable artifact"),
        (TASK_6D, SESSION_NEXT, "PART_OF", "next checkpoint targets 6d N=3 service"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("6c1_task=completed")
    print("6c1_evidence=verified")
    print("next_checkpoint=6d")


if __name__ == "__main__":
    main()
