import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@f9d90c7"

TASK = "task-remove-post-continuation-oracle-magic-numbers-20260808"
DECISION = "decision-derive-oracle-horizon-from-phase-scenario-20260808"
EVIDENCE = "evidence-oracle-derived-horizon-20260808"

TASK_CONTENT = """修订 24 层 post-continuation decode oracle 中把场景语义写成裸数字的部分。保持有意的实验参数 N=3、L=24、tickets=[1,3,2]、一个 m=6 continuation，以及 initial decode 与 post-continuation decode 两个验收事件；先列出两个具名 checkpoint position，再从 checkpoint 集合长度派生 schedule horizon。验收：positions、history ranges、reservation 总长、capacity-weighted counts 和最终 KV totals 都从具名场景参数派生；测试中不存在 2*layers、0..2、decode_assignees[0]/[1] 等隐藏组合关系的裸表达；测试语义、数值与完整回归保持。"""

TASK_RESULT = """[2026-08-08 完成]
24 层 oracle 仍验证 initial prefill -> decode -> stationary continuation prefill -> decode，因此场景中客观存在两个 decode checkpoint；但 scheduler 不再接收手写的 2*layers。测试先列出 initial_decode_position 与 post_continuation_decode_position，再以 decode_checkpoint_positions.len()*layers 派生有限 horizon。positions、history bounds、reservation 长度、capacity-weighted counts 和最终 domain KV totals 也全部从具名场景参数派生。该 horizon 只属于有限 correctness oracle，不是 runtime decode 上限。"""

DECISION_CONTENT = """动机剖析六问：
1. 问题：oracle 中的 2*layers、0..2、decode_assignees[0]/[1]、positions 6/13 和 totals 常量虽然与当前场景数学一致，但没有说明单位和来源，容易被误读为系统把 decode 长度写死为 2，也让后续调整 prefix/continuation 长度时产生联动遗漏。
2. 现状：FrozenKvAssigneeSchedule 的 total_kv_units 实际单位是 (decode checkpoint, layer) append 事件；本场景为了证明 continuation 后仍可 decode，客观包含 initial decode 与 post-continuation decode 两个 checkpoint。生产调度器没有两 token 上限，问题只在 test fixture 表达不清。N=3、L=24、tickets=[1,3,2] 本身是有意选择的验收参数，不应隐藏。
3. 终态：先用 initial_decode_position 与 post_continuation_decode_position 表达两个验收事件，再从 decode_checkpoint_positions.len() 派生 checkpoint 数；horizon、history ranges、reservation 总长和最终 KV totals 均从具名场景参数派生。focused/full tests 保持通过。
4. 业界做法：可维护的 table-driven 测试通常把场景输入作为命名 fixture，把期望值从独立的不变量公式派生；关键固定参数仍显式断言，避免测试完全复制被测实现而失去检错能力。
5. 本方案：只重构现有 oracle 的局部变量和期望计算，不改 FrozenKvAssigneeSchedule、runner、packet、cache 或 runtime。capacity-weighted 期望由 tickets 与总 unit 数计算，最终 KV totals 由 prefix split、continuation counts 和 decode counts 组合。
6. 为什么：引入通用 scenario struct 会为单个测试增加不必要抽象；保留裸数字则继续掩盖单位。具名 checkpoint 集合加独立公式是最小、可审查且不会改变核心方案的修订。
VERDICT: IMPLEMENT。"""

EVIDENCE_CONTENT = """提交 f9d90c7（test: derive continuation decode horizon from scenario）仅修改 rust/src/model/self_driving.rs 的 24 层 stationary continuation oracle，没有修改 scheduler、runner、packet、cache 或 runtime API。
场景语义：两个 decode checkpoint 分别位于 initial prefill 后与 continuation prefill 后；decode_steps=decode_checkpoint_positions.len()，decode_horizon=decode_steps*layers。固定验收参数 N=3、L=24、tickets=[1,3,2] 保持显式；positions、history bounds、reservation 总长、每轮/全 horizon capacity counts 与最终 KV totals 改为派生值。同一请求的 decode 与 continuation schedule 统一使用 request_id=41。
验证：rustfmt --edition 2021 --check rust/src/model/self_driving.rs 与 git diff --check 均 exit 0；focused oracle 1 passed、0 failed；model::self_driving::tests 24 passed、0 failed、1 ignored；cargo test --manifest-path rust/Cargo.toml --features tch-backend 106 passed、0 failed、3 ignored；cargo clippy --manifest-path rust/Cargo.toml --features tch-backend --lib --tests exit 0，仅仓库既有 warnings。
证据边界：这是 mac-local libtorch CPU synthetic correctness oracle；不声称 runtime 支持被限制为两个 decode step，也不新增性能、真实模型或跨节点结论。"""


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

    old_task = conn.execute("SELECT title FROM nodes WHERE id=?", (TASK,)).fetchone()
    assert old_task is not None
    task_content = TASK_CONTENT + "\n\n" + TASK_RESULT
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

    old_decision = conn.execute(
        "SELECT title FROM nodes WHERE id=?", (DECISION,)
    ).fetchone()
    assert old_decision is not None
    upsert_node(
        conn,
        DECISION,
        "decision",
        "active",
        old_decision[0],
        DECISION_CONTENT,
        "held",
        1.0,
        1.0,
        SOURCE,
    )

    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "progress",
        "continuation oracle horizon 已从 checkpoint 场景派生",
        EVIDENCE_CONTENT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )

    upsert_edge(
        conn,
        EVIDENCE,
        TASK,
        "CONFIRMS",
        "magic-number cleanup verified at f9d90c7",
    )
    upsert_edge(
        conn,
        EVIDENCE,
        DECISION,
        "CONFIRMS",
        "finite oracle horizon is derived without changing runtime semantics",
    )
    conn.commit()

    task_state = conn.execute(
        "SELECT layer,status,source,content FROM nodes WHERE id=?", (TASK,)
    ).fetchone()
    decision_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (DECISION,)
    ).fetchone()
    evidence_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (EVIDENCE,)
    ).fetchone()
    assert task_state[:3] == ("progress", "closed", SOURCE)
    assert TASK_RESULT in task_state[3]
    assert decision_state == ("held", SOURCE, DECISION_CONTENT)
    assert evidence_state == ("verified", SOURCE, EVIDENCE_CONTENT)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source=? AND type='CONFIRMS'",
            (EVIDENCE,),
        ).fetchone()[0]
        == 2
    )
    conn.close()

    print("magic_number_task=closed")
    print("decision=updated")
    print("evidence=verified")


if __name__ == "__main__":
    main()
