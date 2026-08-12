import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@6b3"

TASK = "task-phase2-benchmark-6b3-fifo-runtime-contract-20260812"
EVIDENCE = "evidence-phase2-rust-6b3-fifo-runtime-20260812"
SESSION = "session-next-phase2-rust-6b3-fifo-runtime-20260812"
SESSION_NEXT = "session-next-phase2-rust-6c0-observability-20260812"
TASK_6C0 = "task-phase2-benchmark-6c0-observability-20260812"
TASK_6B2B = "task-phase2-benchmark-6b2b-active-reservation-20260812"
EV_6A2 = "evidence-route-b-6a2-qring-unequal-isolation-20260812"


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


TASK_RESULT = """[2026-08-12 完成] 6b.3 真实 runtime 固化跨 worker DecodeBatch FIFO 合同。
coordinator decode_iteration 新增 batch_request_tokens helper：每个 iteration 只构造一次 request_tokens 向量并按 request_id 排序，原样广播给所有 worker（WorkerCommand::DecodeBatch 同一向量）；worker runtime 原样转发到 backend.decode_batch 按序逐请求 decode。这是 multi-request Q-ring 依赖的 FIFO 合同——RingPacket 无 request_id，所有 worker 必须按同一 per-layer 顺序 decode。
验收：1) 所有 worker 每轮观测相同 request_id 序列（coordinator 广播同一排序向量 + worker 默认按序 decode）；2) 无错包/死锁（6a.2 真实 TCP Q-ring 交错 decode oracle 已覆盖）；3) 两请求 token 与独立 reference 一致（6a.2 oracle 每步 argmax/diff 断言）；4) 完成后 cache 释放（新增 release_request focused 测试：release 后 context 移除、幂等、再 decode 报错）。
边界：不增加 RingPacket request_id/decode_step；真实 runtime 未证明会重排，故无需协议修订；不含真实 HTTP 并发 E2E（6d）与吞吐。"""

EVIDENCE_CONTENT = """实现提交 abddbf1（rust: lock DecodeBatch FIFO contract and request release）。
TDD 与验证：
1. batch_request_tokens 新增 2 个单测：(a) 乱序插入 active requests(30/10/20) 后 batch 严格按 request_id 排序 [(10,5),(20,6),(30,7)]；(b) 无 active 时返回空向量。
2. release_request 新增 1 个 focused 测试：prefill + decode 后 release 移除 context，重复 release 幂等，再 decode 报错（不复用 stale state）。
3. 完整回归：cargo test --features tch-backend --lib = 139 passed、0 failed、5 ignored（129 基线 + 3 service_layer_capacities + 4 kv_ledger + 2 batch + 1 release）。
4. rustfmt --edition 2021 与 git diff --check 均 exit 0；改动限 coordinator.rs + tch_backend.rs 两个文件。
证据边界：FIFO 合同是 backend 层 + coordinator 层确定性单测；多 worker 交错数值由 6a.2 真实 TCP Q-ring oracle（证据 evidence-route-b-6a2-qring-unequal-isolation-20260812）覆盖；未在真实 HTTP 并发服务上复跑（6d 完成）。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "6b.3：真实 runtime 固化跨 worker DecodeBatch FIFO 合同",
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
        "6b.3 DecodeBatch FIFO runtime 合同通过",
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
        "二期下一检查点：6c.0 benchmark 双平面观测",
        "6b.3 已完成并验证：DecodeBatch FIFO 合同（按 request_id 排序 + 原样广播）与 request release 均以单测锁定。下一候选节点保持 pending：建立 benchmark 最小双平面观测——request queue/prefill/first-token/decode/release 与 ring hops/bytes/reserved bytes 可关联。",
        "active",
        0.9,
        1.0,
        SOURCE,
    )

    edges = (
        (TASK, EVIDENCE, "CONFIRMS", "6b.3 task confirmed by evidence"),
        (EVIDENCE, TASK, "SUPPORTS", "evidence supports task completion"),
        (EV_6A2, EVIDENCE, "SUPPORTS", "multi-worker interleave numerics covered by 6a.2 oracle"),
        (TASK_6B2B, TASK, "DEPENDS_ON", "6b.3 builds on 6b.2b active reservation"),
        (TASK_6C0, SESSION_NEXT, "PART_OF", "next checkpoint targets 6c.0 observability"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("6b3_task=completed")
    print("6b3_evidence=verified")
    print("next_checkpoint=6c0")


if __name__ == "__main__":
    main()
