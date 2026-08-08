import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@b523bc7"

TASK = "task-continuation-after-stationary-decode-20260808"
DECISION = "decision-continuation-after-stationary-decode-20260808"
ROUTE = "hypothesis-continuation-route-full-activation-packet-20260807"
PORTFOLIO = "task-continuation-route-experiment-portfolio-20260807"
EVIDENCE = "evidence-continuation-after-stationary-decode-20260808"
NEXT_TASK = "task-continuation-multi-token-wire-contract-20260808"

TASK_RESULT = """[2026-08-08 完成]
路线 B 的同一 request 已验证 prefill(0..5)->decode(6)->stationary continuation(7..12)->decode(13) 的 24 层闭环。第二轮 decode 直接复用 continuation 的 logits、finisher domain 与原地 ReservedPositionedKvShard；无需新 runner、planner、cache 转换或 owner metadata。两轮 decode 的 48 个 layer-KV append 在执行前共同按 [8,24,16] 预留；完成后每层 position union=0..14，24 层总 domain KV=[56,168,112]，仍严格 1:3:2。结论限本机 CPU synthetic correctness。"""

EVIDENCE_CONTENT = """路线分支 codex/route-b-continuation-stationary-packet；动机与计划提交 100847a；test-only checkpoint b523bc7（test: validate decode after stationary continuation）。
测试先行结果：扩展 twenty_four_layer_stationary_continuation_returns_to_decode 后第一次 focused run 即 1 passed、0 failed，没有出现 RED。这是有效的组合证据：run_model_ring_with_reserved_history_for_positions 已返回下一阶段需要的 logits 与 finisher，run_reserved_positioned_decode 已能消费相同 positioned shards，因此没有实现缺口；未为制造 RED 新增 wrapper。
数据与调度：decode frozen horizon=2*24，counts=[8,24,16]；两轮各自 layer counts=[4,12,8]。reservation 在 initial prefill 前同时包含 prefix、两轮 decode 和 continuation。第二轮 decode 只在每层计划 assignee append position 13，所有其他 shard 增量为 0；storage pointer 全程不变。
正确性：continuation 末位置 token 与 dense reference 一致；第二轮 decode hidden/logits/greedy token 对齐 contiguous reference；每层实际 position union=0..14；最终 24 层 domain KV totals=[56,168,112]；第二轮 decode 总 hops=24*(3-1)=48，末层 producer 与 continuation producer 一致。
完整验证：focused test 1 passed、0 failed；model::self_driving::tests 24 passed、0 failed、1 ignored；cargo test --features tch-backend 106 passed、0 failed、3 ignored；cargo clippy --features tch-backend --lib --tests exit 0，仅既存 warnings；file-scoped rustfmt --check、git diff --check 均 exit 0。
证据边界：mac-local-shell + libtorch CPU synthetic；不证明 MPS/CUDA/HIP、TCP/QUIC、runtime、多请求或性能。"""

NEXT_TASK_CONTENT = """下一候选小节点：在继续真实网络计算前，先验证 route B 的 m>1 SelfDrivingPacket wire 合同。构造带 m=6 residual、normalized、position_ids、Q、O/LSE accumulator 和 route metadata 的 packet，在现有共享 codec/transport 边界完成 roundtrip，断言 tensor shape/value 与 metadata 不变、payload 不包含历史 KV且不随历史 T 增长。开始前通过独立动机剖析决定最小边界是共享 codec、TCP 还是 TCP+QUIC；不得顺带接 24 层网络循环、runtime、多请求或性能测量。"""


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

    old_task = conn.execute("SELECT title,content FROM nodes WHERE id=?", (TASK,)).fetchone()
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
        "stationary continuation 后恢复 decode 验证完成",
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
        "候选：m>1 SelfDrivingPacket wire 合同",
        NEXT_TASK_CONTENT,
        "planning",
        1.0,
        0.9,
        "proposed-after-b523bc7",
    )

    edges = (
        (EVIDENCE, TASK, "CONFIRMS", "post-continuation decode task verified on route B"),
        (EVIDENCE, DECISION, "CONFIRMS", "test-only composition was sufficient without a new runner"),
        (EVIDENCE, ROUTE, "SUPPORTS", "route B can return to normal decode on the same request cache"),
        (NEXT_TASK, TASK, "DEPENDS_ON", "wire proof follows the completed in-process phase cycle"),
        (NEXT_TASK, ROUTE, "PART_OF", "next route B correctness checkpoint"),
        (NEXT_TASK, PORTFOLIO, "PART_OF", "retains route comparison boundaries"),
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
        "SELECT layer,status,source,content FROM nodes WHERE id=?", (NEXT_TASK,)
    ).fetchone()
    assert task_state[:3] == ("progress", "closed", SOURCE)
    assert TASK_RESULT in task_state[3]
    assert evidence_state == ("verified", SOURCE, EVIDENCE_CONTENT)
    assert next_state == ("active", "planning", "proposed-after-b523bc7", NEXT_TASK_CONTENT)
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source=?", (EVIDENCE,)
    ).fetchone()[0] == 3
    conn.close()

    print("post_continuation_decode_task=closed")
    print("evidence=verified")
    print("next_task=planning")


if __name__ == "__main__":
    main()
