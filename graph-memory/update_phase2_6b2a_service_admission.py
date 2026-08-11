import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@6b2a"

TASK = "task-phase2-benchmark-6b2a-service-admission-20260812"
EVIDENCE = "evidence-phase2-rust-6b2a-service-admission-20260812"
SESSION = "session-next-phase2-rust-6b2a-admission-20260812"
SESSION_NEXT = "session-next-phase2-rust-6b2b-active-reservation-20260812"
BENCH_TASK = "task-phase2-rust-benchmark-readiness-20260812"
TASK_6B2B = "task-phase2-benchmark-6b2b-active-reservation-20260812"


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


TASK_RESULT = """[2026-08-12 完成] 6b.2a 普通 HTTP service prefill 接入 frozen reservation 与 byte admission。
接入点 rust/src/distributed/coordinator.rs prefill_single_request：任何 Prefill command 发出前，按 prompt/max_tokens、capacity tickets(chunk_sizes)、模型 KV geometry 冻结本请求 per-domain per-layer reservation（service_layer_capacities：prefix split + 全量 decode horizon [prompt_len, prompt_len+max_tokens)，每个 decode position 由 position%domains owner 持有，与 ring.rs keep 规则一致），随后 capacity_mb_to_bytes + admit_reserved_kv_bytes 做 exact KV payload byte admission；通过后把 capacities 随 Prefill 命令下发（layer_kv_capacities: Some(...)）。
验收：1) 合法请求在 prefill 前打印 required<=budget（新增 coordinator 日志）；2) unknown/overflow/one-byte-short 在任意 worker prefill 前拒绝（fail-closed，错误经 job.tx 返回）；3) 既有 token correctness 不变（132 passed，含 6a decode_batch/request isolation oracle）。新增 3 个 service_layer_capacities 单测（full horizon/zero max_tokens/N=3）。边界：单请求 post-plan admission；batch mode(process_single_request) 保持 layer_kv_capacities: None 未接入；不涉并发总量(6b.2b)、eviction、repair、迁移。worker 端 ReservedPositioned decode 数值语义已由 6a 证明与独立参考一致。"""

EVIDENCE_CONTENT = """实现提交 <COMMIT>（rust: gate HTTP service prefill with KV byte admission）。
TDD 与验证：
1. 纯 helper 单测：service_layer_capacities 新增 3 个测试覆盖 (a) 2-domain prompt4 max_tokens3 -> capacities [3,4]（domain0 前缀1+{4,6}，domain1 前缀3+{5}）；(b) max_tokens=0 只留前缀 [1,3]；(c) N=3 prompt6 [2,2,2] max_tokens4 -> [4,3,3]。
2. focused test：service_layer_capacities 3 passed。
3. 完整回归：cargo test --features tch-backend --lib = 132 passed、0 failed、5 ignored（基线 129 + 3 新增）；含 6a 的 decode_batch isolation 与 decode_qring request isolation oracle。
4. rustfmt --edition 2021 <file> 与 git diff --check 均 exit 0；改动仅限 coordinator.rs 一个文件。
证据边界：coordinator 本地单元 + 既有 in-process worker oracle；未在真实 HTTP 服务 + 多 worker 上复跑（跨机 HTTP E2E 属 6d）；不证明并发 admission(6b.2b)、吞吐、eviction 或 batch mode 路径。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    evidence_content = EVIDENCE_CONTENT.replace("<COMMIT>", "d57b9ca")
    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "6b.2a：普通 service prefill 接入冻结 reservation 与 byte admission",
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
        "6b.2a service prefill frozen reservation + byte admission 通过",
        evidence_content,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )
    # Close the current checkpoint and open the next one.
    conn.execute("UPDATE nodes SET status='closed', updated_at=datetime('now') WHERE id=?", (SESSION,))
    upsert_node(
        conn,
        SESSION_NEXT,
        "session",
        "active",
        "二期下一检查点：6b.2b active-request reserve/release 计数",
        "6b.2a 已完成并验证：service prefill 在任何 Prefill 前做 exact byte admission，capacities 随命令下发。下一候选节点保持 pending：coordinator 维护 request_id -> per-domain reserved bytes 映射；admission 原子检查 active sum + new request；完成/拒绝/失败路径只释放一次。不得扩成 paged allocator、preemption、priority、eviction、repair planner 或无限队列治理。",
        "active",
        0.9,
        1.0,
        SOURCE,
    )

    edges = (
        (TASK, EVIDENCE, "CONFIRMS", "6b.2a task confirmed by evidence"),
        (EVIDENCE, TASK, "SUPPORTS", "evidence supports task completion"),
        (TASK_6B2B, SESSION_NEXT, "PART_OF", "next checkpoint targets 6b.2b"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("6b2a_task=completed")
    print("6b2a_evidence=verified")
    print("next_checkpoint=6b2b")


if __name__ == "__main__":
    main()
