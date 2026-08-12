import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@6b2b"

TASK = "task-phase2-benchmark-6b2b-active-reservation-20260812"
EVIDENCE = "evidence-phase2-rust-6b2b-active-reservation-20260812"
SESSION = "session-next-phase2-rust-6b2b-active-reservation-20260812"
SESSION_NEXT = "session-next-phase2-rust-6b3-fifo-runtime-20260812"
TASK_6B3 = "task-phase2-benchmark-6b3-fifo-runtime-contract-20260812"
TASK_6B2A = "task-phase2-benchmark-6b2a-service-admission-20260812"


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


TASK_RESULT = """[2026-08-12 完成] 6b.2b 最小 active-request KV byte reserve/release 计数。
coordinator 新增 ActiveKvReservation ledger：request_id -> per-domain reserved bytes 映射 + per-domain used sum；prefill 在 dispatch 前原子检查 active sum + new request 是否都 <= 各 worker byte budget（try_reserve），成功才下 Prefill；完成、decode 失败、prefill 失败三种路径各自恰好释放一次（RAII ReservationGuard 保证 prefill 中间失败也释放，成功时 committed 保留）。
验收：1) 两个 individually-fit 但 jointly-over-budget 的请求中第二个在 dispatch 前被拒绝（try_reserve 返回错误并经 job.tx 返回）；2) 完成第一个后预算恢复，后续请求可进入（release 后 used 归零）；3) 重复 release 不产生负数或双重返还（release 幂等、saturating_sub）。
边界：correctness 确定性占用计数，无 paged allocator、preemption、priority、eviction、repair planner、无限队列治理；不含真实 HTTP 并发 E2E（6d）；batch mode 不接入。"""

EVIDENCE_CONTENT = """实现提交 9ec8f96（rust: track active-request KV byte reservations in HTTP service）。
TDD 与验证：
1. 新增 4 个 ActiveKvReservation 单测：(a) 两个 individually-fit 请求同时容纳；(b) 第二个请求 joint 超预算时原子拒绝、账本不变、释放后恢复；(c) duplicate reserve 报错、重复/未知 release 幂等且不产生负数；(d) checked_add overflow 与 domain 数不匹配拒绝且不留账本。
2. RED/GREEN：首轮运行 1 个测试因 overflow 断言触发条件错误而失败（used=0 时 u64::MAX 不溢出），修正测试为已有 live reservation 后 4 passed。
3. 完整回归：cargo test --features tch-backend --lib = 136 passed、0 failed、5 ignored（129 基线 + 3 service_layer_capacities + 4 kv_ledger）。
4. rustfmt --edition 2021 <file> 与 git diff --check 均 exit 0；改动仅限 coordinator.rs。
证据边界：ledger 单测证明确定性计数语义（原子拒绝、一次释放、幂等）；未在真实 HTTP 并发 + 多 worker 上复跑（6d 完成）；不证明吞吐、eviction 或 queue 治理。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "6b.2b：最小 active-request KV byte reserve/release 计数",
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
        "6b.2b active-request KV byte reserve/release 计数通过",
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
        "二期下一检查点：6b.3 DecodeBatch FIFO runtime 合同",
        "6b.2b 已完成并验证：coordinator 维护 request_id -> per-domain reserved bytes 映射，active sum 原子 admission，三种完成/失败路径恰好释放一次。下一候选节点保持 pending：在 coordinator/runtime 命令路径验证 request_tokens 只生成一次并原样广播；两个不等长请求跨 worker 交错 decode，记录 command 顺序、request horizon、token/reference 和 release。不得增加 RingPacket request_id/decode_step；若真实 runtime 证明会重排，再单独做协议修订动机剖析。",
        "active",
        0.9,
        1.0,
        SOURCE,
    )

    edges = (
        (TASK, EVIDENCE, "CONFIRMS", "6b.2b task confirmed by evidence"),
        (EVIDENCE, TASK, "SUPPORTS", "evidence supports task completion"),
        (TASK_6B2A, TASK, "DEPENDS_ON", "6b.2b builds on 6b.2a admission"),
        (TASK_6B3, SESSION_NEXT, "PART_OF", "next checkpoint targets 6b.3"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("6b2b_task=completed")
    print("6b2b_evidence=verified")
    print("next_checkpoint=6b3")


if __name__ == "__main__":
    main()
