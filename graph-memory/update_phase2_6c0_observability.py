import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@6c0"

TASK = "task-phase2-benchmark-6c0-observability-20260812"
EVIDENCE = "evidence-phase2-rust-6c0-observability-20260812"
SESSION = "session-next-phase2-rust-6c0-observability-20260812"
SESSION_NEXT = "session-next-phase2-rust-6c1-native-baseline-20260812"
TASK_6C1 = "task-phase2-rust-6c1-native-service-baseline-20260812"
TASK_6B3 = "task-phase2-benchmark-6b3-fifo-runtime-contract-20260812"


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


TASK_RESULT = """[2026-08-12 完成] 6c.0 benchmark 最小双平面观测。
coordinator 新增 --trace-jsonl <path>：每个完成的请求输出一条 JSONL 记录（keyed by request_id），包含 enqueue/prefill-accepted/first-token/completed elapsed ms、reserved/released bytes、prompt/max tokens、finish_reason 或 error、prefill/decode hops。hop 数按已知 N/L 公式派生（prefill = L*(N-1)，每个 decode step 同样），无需逐 hop 埋点。默认关闭；开启仅追加 JSONL，不改推理结果。
验收：1) 一个请求可通过 request_id 关联 client 结果与 HCP 记录（request_id 从 1 递增，trace 含同 id）；2) 计数与已知 N/L 公式一致（测试断言 prefill_hops=L*(N-1)、decode_hops=steps*L*(N-1)）；3) disabled 不改变推理结果（TraceSink::new(None) writer=None，lifecycle 调用 no-op）。
边界：JSONL/现有 /metrics 计数，不引入 Prometheus、trace backend、dashboard、生产告警。"""

EVIDENCE_CONTENT = """实现提交 78be1d0（rust: add per-request JSONL trace plane to HTTP service）。
TDD 与验证：
1. 新增 2 个 trace_sink 单测：(a) disabled（None 路径）writer 为 None、lifecycle 调用 no-op 且不 panic；(b) N=3 L=24 trace 记录含 enqueue->accepted->2 decode->complete，断言 prefill_hops=48、decode_hops=96、reserved/released 字节、finish_reason 与时间戳字段。
2. RED/GREEN：首轮 2 个编译错误（elapsed_ms 与 in_flight 的 borrow 冲突；InferenceJob 在 prefill 失败路径被 move 后引用）已修复，随后 2 passed。
3. 完整回归：cargo test --features tch-backend --lib = 141 passed、0 failed、5 ignored（129 基线 + 3 service_layer_capacities + 4 kv_ledger + 2 batch_request_tokens + 1 release_request + 2 trace_sink）。
4. rustfmt --edition 2021 与 git diff --check 均 exit 0；改动仅限 coordinator.rs。
证据边界：trace 平面单测证明字段与 hop 公式；真实 HTTP 服务端到端 trace 由 6c.1 native baseline 完成（证据 evidence-phase2-rust-6c1-native-baseline-20260812）。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "6c.0：建立 benchmark 最小双平面观测",
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
        "6c.0 per-request JSONL trace 双平面通过",
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
        "二期下一检查点：6c.1 native 服务稳定性基线",
        "6c.0 已完成并验证：--trace-jsonl 输出 per-request 双平面记录。下一候选节点保持 pending：用仓库原生 HTTP smoke/测试客户端运行 concurrency 1、2、4 与不等长请求，验证 0 错误、token/reference、queue/active/release、reserved bytes 与 ring hops/bytes。结果是二期内部稳定性基线，不是 vLLM benchmark 性能结论。",
        "active",
        0.9,
        1.0,
        SOURCE,
    )

    edges = (
        (TASK, EVIDENCE, "CONFIRMS", "6c.0 task confirmed by evidence"),
        (EVIDENCE, TASK, "SUPPORTS", "evidence supports task completion"),
        (TASK_6B3, TASK, "DEPENDS_ON", "6c.0 builds on 6b.3 FIFO contract"),
        (TASK_6C1, SESSION_NEXT, "PART_OF", "next checkpoint targets 6c.1 native baseline"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("6c0_task=completed")
    print("6c0_evidence=verified")
    print("next_checkpoint=6c1")


if __name__ == "__main__":
    main()
