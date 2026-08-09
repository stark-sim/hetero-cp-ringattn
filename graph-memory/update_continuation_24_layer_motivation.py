import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-confirmed-2026-08-08"

TASK = "task-continuation-24-layer-mixed-history-20260808"
DECISION = "decision-continuation-24-layer-mixed-history-20260808"
METHOD = "preference-motivation-analysis-20260721"
PREVIOUS = "task-continuation-position-owner-local-kv-20260807"
ROUTE = "hypothesis-continuation-route-full-activation-packet-20260807"

TASK_CONTENT = """当前小节点只验证路线 B 的 24 层 mixed-history stationary continuation。N=3、L=24、tickets=[1,3,2]；先用 positioned shards 建立 initial prefill 历史，再追加一轮 m=1 decode，使 continuation 起点同时含 prefix KV 与 decode KV；随后仅运行一个 m=6 continuation segment。每层用 position owner-local KV 和 self-driving LayerPacket 遍历三节点，历史 KV 不进入 packet。验收包括：每层新增 position union 完整无重复且 owner counts=[1,3,2]；每层各 worker 增量匹配冻结 schedule；reservation 不越界且 storage pointer 不变；starter/finisher 逐层轮转；总 hops=24*(3-1)=48；最终 hidden/logits 对齐 contiguous dense reference。范围限本机 CPU synthetic correctness，不追加 continuation 后 decode，不接 wire/runtime、多请求或性能测量。"""

DECISION_CONTENT = """动机剖析六问：
1. 问题：单层 position owner-local 实验证明了 m>1 的局部数学和显存归属，但尚未证明同一 activation packet 机制能在 24 层递推中持续成立。尤其是层间 finisher-to-starter handoff、mixed history 的 causal 可见性、每层 capacity-weighted append 和累计 48 hops 尚未由同一测试共同约束。
2. 现状：已有 24 层 prefill-decode-prefill-decode 回归能证明 ReservedPositionedKvShard 与 contiguous reference 兼容，但其中 m>1 prefill 使用直接 QKV 投影和按连续 token_splits append，不经过 LayerPacket、O/LSE accumulator 或 position-local processing API；已有 route B 测试只覆盖单层。因此两者分别成立仍不足以证明 route B 的完整 24 层 continuation 数据流。
3. 终态：在 N=3、L=24、m=6、tickets=[1,3,2] 的 deterministic CPU synthetic case 中，从 initial prefill 加一轮 decode 的 positioned mixed history 出发，逐层运行 stationary-history activation packet。每层六个新 position 精确按 [1,3,2] owner-local 生成并只 append 一次，shard 不越预留容量且 storage pointer 稳定；starter/finisher 轮转和 48 hops 明确；最终 hidden/logits 与 contiguous dense reference 在既有容差内一致。
4. 业界做法：vLLM/SGLang 等 serving runtime 以 request block table 或 paged cache metadata 维持历史 KV placement，forward-extend 只生成新增位置 KV；多层递推由模型执行器串联。可复用的是“历史 cache 原地、只写新增 KV、层间传 activation”的边界，但其中心 block manager、collective/full-connect 通信和生产调度不适合直接作为 HCP neighbor-only P2P ring 的本节点证明。
5. 本方案：复用现有 LayerPacket、process_layer_packet_with_reserved_history_for_positions、FrozenKvAssigneeSchedule 和 ReservedPositionedKvShard，只增加一个最小 24 层实验 runner/oracle。初始 prefix 仍由既有 positioned prefill helper 建立，一轮 decode 建立 mixed history；continuation 对每层复用同一个 position-level frozen owner plan，每层按 ring 顺序调用现有单节点处理 API，finisher 成为下一层 starter，并记录 owner/route/hop 观测值。
6. 为什么：直接扩 wire/runtime 会把尚未验证的 24 层合同放大成部署复杂度；只扩现有 24 层 prefill helper又绕过 route B 的核心数据流。最小 runner 同时复用业界的 stationary cache 原则和 HCP 已验证的 neighbor-only packet 原语，能用一个可证伪节点填补真正缺口，而不引入 planner、动态 assignee 或新的永久协议。
边界：本节点不比较性能，不证明真实网络或异构设备，也不证明 continuation 后能恢复 m=1 decode；后者登记为下一个独立节点。
VERDICT: IMPLEMENT。"""


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
            SOURCE,
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

    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "24 层 mixed-history stationary continuation",
        TASK_CONTENT,
        "active",
        1.0,
        1.0,
    )
    upsert_node(
        conn,
        DECISION,
        "decision",
        "active",
        "用最小 runner 验证 24 层 route B continuation",
        DECISION_CONTENT,
        "held",
        1.0,
        1.0,
    )

    edges = (
        (DECISION, TASK, "PART_OF", "approved pre-action motivation analysis"),
        (METHOD, DECISION, "GOVERNS", "required six-question analysis"),
        (TASK, PREVIOUS, "DEPENDS_ON", "requires the verified single-layer position-local contract"),
        (TASK, ROUTE, "PART_OF", "third route B correctness checkpoint"),
        (DECISION, ROUTE, "SUPPORTS", "tests stationary history across the full 24-layer recurrence"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()

    for node_id, expected_content in ((TASK, TASK_CONTENT), (DECISION, DECISION_CONTENT)):
        row = conn.execute(
            "SELECT layer,status,source,content FROM nodes WHERE id=?", (node_id,)
        ).fetchone()
        assert row is not None
        assert row[2] == SOURCE
        assert row[3] == expected_content
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source=? AND target=? AND type='PART_OF'",
        (DECISION, TASK),
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source=? AND target=? AND type='GOVERNS'",
        (METHOD, DECISION),
    ).fetchone()[0] == 1
    conn.close()

    print("continuation_24_layer_task=active")
    print("motivation=exact")
    print("required_edges=present")


if __name__ == "__main__":
    main()
