import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
TASK_ID = "task-continuation-prefill-inference-research-20260807"
DECISION_ID = "decision-continuation-prefill-research-before-packet-20260807"
PREFERENCE_ID = "preference-motivation-analysis-20260721"

TASK_CONTENT = """只研究推理场景中的 initial prefill、continuation/extend prefill 与 decode。以已有历史长度 T、本轮新增 segment 长度 m 为主变量，核验 Ring Attention 原论文、官方 inference Context Parallel 实现与 serving extend attention 的数据流；比较完整 committed KV ring、历史 KV 原地且 Q/O/LSE 环传、query-shard partial merge 与阶段混合路线。交付物是中文、带数学推导、权威来源与证据强度的多路线实验预期；研究完成前不扩展 Rust activation packet wire。"""

DECISION_CONTENT = """【动机六问】
1. 问题：当前 continuation KV-ring baseline 会在每层环传完整 committed KV，包括长度 T 的历史；batched positioned accumulator 只证明 m>1 attention 数学，尚未证明 continuation 整层数据流。尚不能回答历史 KV 是否必须移动，也不能决定 m 长度 activation packet 是否是正确实验。
2. 现状：永久不变量是每个 (request, layer, position) KV 只驻留在 capacity-weighted owner。initial/continuation baseline 移动 KV；单 token decode 移动 Q/O/LSE；m>1 packet 尚未实现。直接扩 wire 会把未决通信对象固化。
3. 目标：仅针对推理，在统一 T、m、N、H、H_kv、D 和元素字节数符号下，还原原论文与业界方案，推导每条候选路线的通信、临时显存、永久显存、计算与 hop 边界，并给出最小可证伪实验。完成标准是严格回答“历史 KV 是否数学上必须传输”以及各路线何时占优。
4. 他者：优先查 Ring Attention 论文、PyTorch/Megatron/TensorRT-LLM/vLLM/SGLang/FlashInfer 官方文档与官方源码；关注 KV ring/all-gather、QKV all-to-all、query/partial-result 路线和 request-owned cache。业界机制只对照，未经单独决策不引入 HCP。
5. 本方案：四主题各两轮研究：原 Ring Attention；inference CP；serving continuation/extend；HCP 数学综合与现有 Rust 映射。来源发现后回查原文，处理矛盾并标注证据强度。
6. 为什么：HCP 只允许 neighbor-only P2P ring，异构 capacity-weighted KV 永久驻留，不能直接照搬同构 collective 或中心 runtime。先研究可以把“数学必要性”“某实现习惯”和“HCP 设计选择”分离。
【范围】不研究训练性能，不修改 Rust runtime/wire，不引入生产级调度器。m 长度 activation packet 与 continuation 路线直接交合，因此其整层实验依赖本研究结论。
VERDICT: IMPLEMENT RESEARCH FIRST。用户于 2026-08-07 批准。"""


def upsert_node(conn, node_id, node_type, title, content, status):
    conn.execute(
        """
        INSERT INTO nodes
        (id, type, layer, project, title, content, importance, confidence,
         status, source, created_at, updated_at)
        VALUES (?, ?, 'active', ?, ?, ?, 1.0, 1.0, ?,
                'user-approved-2026-08-07', datetime('now'), datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type = excluded.type,
          layer = excluded.layer,
          project = excluded.project,
          title = excluded.title,
          content = excluded.content,
          importance = excluded.importance,
          confidence = excluded.confidence,
          status = excluded.status,
          source = excluded.source,
          updated_at = datetime('now')
        """,
        (node_id, node_type, PROJECT, title, content, status),
    )


def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        """
        INSERT INTO edges (source, target, type, weight, note)
        VALUES (?, ?, ?, 1.0, ?)
        ON CONFLICT(source, target, type) DO UPDATE SET
          weight = excluded.weight,
          note = excluded.note
        """,
        (source, target, edge_type, note),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        TASK_ID,
        "task",
        "研究 continuation prefill 是否需要移动历史 KV",
        TASK_CONTENT,
        "active",
    )
    upsert_node(
        conn,
        DECISION_ID,
        "decision",
        "先研究 continuation prefill 数据移动，再实现整段 packet",
        DECISION_CONTENT,
        "held",
    )

    edges = (
        (DECISION_ID, TASK_ID, "PART_OF", "approved motivation analysis"),
        (
            PREFERENCE_ID,
            DECISION_ID,
            "GOVERNS",
            "pre-action motivation analysis",
        ),
        (
            TASK_ID,
            "task-multiround-stage-dataflow-analysis-20260805",
            "PART_OF",
            "research resolves the multi-round continuation branch",
        ),
        (
            TASK_ID,
            "task-continuation-request-reuse-20260803",
            "DEPENDS_ON",
            "correct reusable mixed-history baseline is the comparison point",
        ),
        (
            TASK_ID,
            "task-continuation-batched-accumulator-contract-20260805",
            "DEPENDS_ON",
            "attention-level m>1 oracle supplies one candidate mechanism",
        ),
        (
            "task-continuation-batched-accumulator-ring-20260803",
            TASK_ID,
            "DEPENDS_ON",
            "do not freeze the full-layer packet before choosing the communication object",
        ),
        (
            "task-continuation-route-comparison-20260803",
            TASK_ID,
            "DEPENDS_ON",
            "research supplies the analytical routes and cost model",
        ),
    )
    for edge in edges:
        upsert_edge(conn, *edge)

    conn.commit()

    task_content = conn.execute(
        "SELECT content FROM nodes WHERE id = ?", (TASK_ID,)
    ).fetchone()[0]
    decision_content = conn.execute(
        "SELECT content FROM nodes WHERE id = ?", (DECISION_ID,)
    ).fetchone()[0]
    relevant_edges = conn.execute(
        """
        SELECT COUNT(*) FROM edges
        WHERE source IN (?, ?) OR target IN (?, ?)
        """,
        (TASK_ID, DECISION_ID, TASK_ID, DECISION_ID),
    ).fetchone()[0]
    assert task_content == TASK_CONTENT
    assert decision_content == DECISION_CONTENT
    assert relevant_edges >= len(edges)
    conn.close()

    print("task_exact=1")
    print("decision_exact=1")
    print(f"decision_newlines={DECISION_CONTENT.count(chr(10))}")
    print(f"relevant_edges={relevant_edges}")


if __name__ == "__main__":
    main()
