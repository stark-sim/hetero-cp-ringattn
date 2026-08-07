import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@5777d51"

TASK = "task-continuation-packet-layer-contract-20260807"
DECISION = "decision-continuation-packet-layer-contract-motivation-20260807"
ROUTE = "hypothesis-continuation-route-full-activation-packet-20260807"
PORTFOLIO = "task-continuation-route-experiment-portfolio-20260807"
EVIDENCE = "evidence-continuation-packet-layer-contract-20260807"
NEXT_TASK = "task-continuation-position-owner-local-kv-20260807"

TASK_RESULT = """[2026-08-07 完成]
m>1 单层完整 LayerPacket 合同已由 Rust synthetic oracle 验证。LayerPacket 接受 [1,m,H] hidden 与 [1,m] absolute position_ids，携带 residual、normalized、Q、O/LSE，依次合并各节点 ReservedPositionedKvShard 的 causal partial；finisher 唯一执行 W_o、residual、post-attention Norm 与 MLP。历史 KV 不进入 packet。当前实现把整段 m 个新 K/V 暂时 append 到单一 assignee；它证明整层数学与 shape 合同，不证明 capacity-weighted per-position placement、wire/runtime 或 24 层 continuation 闭环。"""

EVIDENCE_CONTENT = """实现提交 5777d51（rust: validate multi-token stationary layer packet）。
新鲜验证：
1. cargo test ... multi_token_layer_packet_completes_positioned_causal_layer_without_history_payload：1 passed。m=3、T=6、两个非连续 positioned shards 的 attention 与 dense reference max diff <1e-4，整层 hidden max diff <2e-4；新 KV 仅 append 到指定 shard，预分配 storage pointer 不变。
2. cargo test ... layer_packet_payload_does_not_grow_with_history_context：1 passed。m=3 时 T=2 与 T=47 的 packet tensor element count 相等；payload 公式为 m*(4H+h_q+1)，不含历史 T。
3. cargo test ... model::self_driving::tests：21 passed、0 failed、1 ignored（需要本地 Qwen 权重的既有测试）。既有 m=1 decode、任意 N、wrap-around、24 层 cache reuse 等回归保持通过。
4. cargo clippy --features tch-backend --lib --tests：exit 0，只有仓内既存 warnings；本节点两文件 rustfmt --check 与 git diff --check：exit 0。
调试记录：首次编译发现 validate_route 使用 position_ids 却遗漏函数参数；确认唯一调用点后以显式借用参数修复，同一测试随后通过。
证据边界：CPU synthetic correctness，不是 MPS/CUDA/HIP 性能或真实网络结果；整段新 KV 仍由一个 assignee 接收。"""

NEXT_TASK_CONTENT = """下一候选小节点：让 m-segment 内不同 absolute positions 按既有 frozen capacity-weighted calendar 分配到不同 worker；每个 worker 只投影并 append 自己负责的 position subset，同时 packet 的每个 query 仍对所有 local positioned KV 做 causal partial。验收应覆盖 m 内 assignees 不均等且跨 worker、每个 (layer,position) K/V exact-once、各 shard 不越 reservation、合并结果与 dense reference 一致。范围只到单层 synthetic；不接 wire/runtime、不做 24 层、多请求或动态 planner。开始前需完成独立动机剖析并确认 LayerPacket 如何表达 position assignment。"""


def upsert_node(
    conn,
    node_id,
    node_type,
    layer,
    title,
    content,
    status,
    importance,
    confidence,
    source,
):
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
        (
            node_id,
            node_type,
            layer,
            PROJECT,
            title,
            content,
            importance,
            confidence,
            status,
            source,
        ),
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


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    old_task = conn.execute(
        "SELECT title,content FROM nodes WHERE id=?", (TASK,)
    ).fetchone()
    assert old_task is not None
    task_content = old_task[1]
    if TASK_RESULT not in task_content:
        task_content += "\n\n" + TASK_RESULT
    upsert_node(
        conn,
        TASK,
        "task",
        "progress",
        old_task[0],
        task_content,
        "closed",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "progress",
        "m>1 stationary LayerPacket 单层合同验证完成",
        EVIDENCE_CONTENT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        NEXT_TASK,
        "task",
        "active",
        "候选：m-segment 新 KV 按 position owner-local 生成",
        NEXT_TASK_CONTENT,
        "planning",
        1.0,
        0.95,
        "proposed-after-5777d51",
    )

    edges = (
        (EVIDENCE, TASK, "CONFIRMS", "single-layer packet contract and regression verified"),
        (EVIDENCE, DECISION, "SUPPORTS", "accepted smallest experiment succeeded within its boundary"),
        (EVIDENCE, ROUTE, "SUPPORTS", "first active-experiment checkpoint for full activation packet"),
        (NEXT_TASK, TASK, "DEPENDS_ON", "position-subset generation builds on the m>1 layer contract"),
        (NEXT_TASK, ROUTE, "PART_OF", "second checkpoint in the active stationary-packet route"),
        (NEXT_TASK, PORTFOLIO, "PART_OF", "keeps the experiment inside the retained route portfolio"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)

    conn.commit()

    task_state = conn.execute(
        "SELECT layer,status,source,content FROM nodes WHERE id=?", (TASK,)
    ).fetchone()
    evidence_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (EVIDENCE,)
    ).fetchone()
    next_state = conn.execute(
        "SELECT layer,status,content FROM nodes WHERE id=?", (NEXT_TASK,)
    ).fetchone()
    assert task_state[:3] == ("progress", "closed", SOURCE)
    assert TASK_RESULT in task_state[3]
    assert evidence_state[:2] == ("verified", SOURCE)
    assert evidence_state[2] == EVIDENCE_CONTENT
    assert next_state[:2] == ("active", "planning")
    assert next_state[2] == NEXT_TASK_CONTENT
    edge_count = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source=?", (EVIDENCE,)
    ).fetchone()[0]
    assert edge_count == 3
    conn.close()

    print("packet_task=closed")
    print("evidence=verified")
    print("next_task=planning")
    print("evidence_edges=3")


if __name__ == "__main__":
    main()
