import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-confirmed-2026-08-08"

TASK = "task-continuation-after-stationary-decode-20260808"
DECISION = "decision-continuation-after-stationary-decode-20260808"
METHOD = "preference-motivation-analysis-20260721"
PREVIOUS = "task-continuation-24-layer-mixed-history-20260808"
ROUTE = "hypothesis-continuation-route-full-activation-packet-20260807"
FORMAT_LESSON = "lesson-file-scoped-rustfmt-in-route-worktree-20260808"

TASK_CONTENT = """当前小节点只验证同一 request 从 stationary continuation 返回 m=1 decode。N=3、L=24、tickets=[1,3,2]；数据流为 initial prefill positions 0..5、第一轮 decode position 6、m=6 route B continuation positions 7..12、第二轮 decode position 13。两轮 decode 的 48 个 layer-KV append 必须在执行前共同进入 frozen schedule 与 reservation。验收包括 continuation 末位置 token 与 dense reference 一致；第二轮 decode 由 continuation finisher 启动；每层只在计划 assignee append position 13；两轮 decode 总 counts=[8,24,16]；每层 position union=0..14；storage pointer 稳定；第二轮 decode 仍为 48 hops；hidden/logits/token 对齐 contiguous dense reference。范围限本机 CPU synthetic correctness，不接 wire/runtime、多请求或性能测量。"""

DECISION_CONTENT = """动机剖析六问：
1. 问题：上一节点已经证明 route B 的 m=6 continuation 可从 prefix+decode mixed history 完成 24 层，但流程终止在 continuation logits；尚未证明同一 positioned request state 能立刻切回常规 m=1 self-driving decode。若这一边界不成立，continuation 只能作为一次性末端实验，不能表达真实多轮会话。
2. 现状：run_model_ring_with_reserved_history_for_positions 已返回 continuation hidden/logits 和末层 producer domain；run_reserved_positioned_decode 已能从任意 starter 对同一 ReservedPositionedKvShard 追加单位置 KV。旧的 24 层 prefill-decode-prefill-decode 回归证明 direct prefill 后可恢复 decode，但第二段 prefill 没有经过 route B LayerPacket，因此不能替代当前阶段组合证据。
3. 终态：在 N=3、L=24、tickets=[1,3,2] 的 deterministic CPU case 中执行 prefill(0..5)->decode(6)->stationary continuation(7..12)->decode(13)。两轮 decode 的 48 个 layer append 在运行前共同按 [8,24,16] 预留；第二轮每层只 append 一次，position union 完整为 0..14，storage pointer 不变；continuation token、最终 hidden/logits/token 与 contiguous dense reference 一致，第二轮 decode route 为 48 hops。
4. 业界做法：vLLM/SGLang 等 serving engine 让 prefill/extend/decode 共享 request-scoped paged KV 与 position/block metadata，阶段改变只切换 forward mode，不重建历史 cache；scheduler 在 forward 前预留将写入的 slots。可复用的是同一 request cache 与预分配生命周期，不直接复用其中心 block manager、continuous batching 或 collective 通信。
5. 本方案：先只扩展现有 24 层 route B 行为 oracle。把 decode schedule horizon 从 24 扩为 48 个 layer units，并把两轮 assignee 都纳入 reservation；continuation 完成后对其最后位置 logits 采样，复用现有 run_reserved_positioned_decode 从 finisher 在 position 13 继续。若现有 primitives 直接通过，则不新增 production 或实验 runner。
6. 为什么：阶段边界的未知点是已有组件能否组合，不是缺少新的调度或协议。直接接 runtime/wire 会混入 request 生命周期和序列化问题；新增 wrapper 只会复制现有调用关系。test-only composition 是最小可证伪方案，也最符合当前“核心必要能力、小步推进”的范围。
边界：本节点不证明真实网络、异构硬件、并发请求或性能；不把 test helper 提升为服务 API。
VERDICT: IMPLEMENT TEST-ONLY COMPOSITION FIRST。"""


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
        "stationary continuation 后恢复 decode",
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
        "先用 test-only composition 验证 continuation 到 decode",
        DECISION_CONTENT,
        "held",
        1.0,
        1.0,
    )

    edges = (
        (DECISION, TASK, "PART_OF", "approved pre-action motivation analysis"),
        (METHOD, DECISION, "GOVERNS", "required six-question analysis"),
        (TASK, PREVIOUS, "DEPENDS_ON", "requires the verified 24-layer stationary continuation state"),
        (TASK, ROUTE, "PART_OF", "fourth route B correctness checkpoint"),
        (DECISION, ROUTE, "SUPPORTS", "tests the stage transition without changing stationary-history mechanics"),
        (FORMAT_LESSON, TASK, "GOVERNS", "format only the modified Rust file"),
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

    print("post_continuation_decode_task=active")
    print("motivation=exact")
    print("required_edges=present")


if __name__ == "__main__":
    main()
