import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-review-2026-08-08"

TASK = "task-remove-post-continuation-oracle-magic-numbers-20260808"
DECISION = "decision-derive-oracle-horizon-from-phase-scenario-20260808"
PREFERENCE = "preference-avoid-magic-numbers-20260808"
METHOD = "preference-motivation-analysis-20260721"
COMPLETED_TASK = "task-continuation-after-stationary-decode-20260808"
EVIDENCE = "evidence-continuation-after-stationary-decode-20260808"

TASK_CONTENT = """修订 24 层 post-continuation decode oracle 中把场景语义写成裸数字的部分。保持实验参数 N=3、L=24、tickets=[1,3,2]、一个 m=6 continuation 和两次 decode 不变，但用命名的 decode_steps、阶段索引、prefix_len、position 边界与派生 expected totals 表达。验收：schedule horizon 明确等于 decode_steps*layers；第一/第二 decode position 与 history ranges 从阶段长度派生；不再使用 decode_assignees[0]/[1] 或 2*layers；测试语义、数值与完整回归保持。"""

DECISION_CONTENT = """动机剖析六问：
1. 问题：oracle 中的 2*layers、0..2、decode_assignees[0]/[1]、positions 6/13 和 totals 常量虽然与当前场景数学一致，但没有说明单位和来源，容易被误读为系统把 decode 长度写死为 2，也让后续调整 prefix/continuation 长度时产生联动遗漏。
2. 现状：FrozenKvAssigneeSchedule 的 total_kv_units 实际单位是 (decode_step, layer) append 事件；本场景有两次 decode、每次 24 层，所以 horizon=48。生产调度器没有两 token 上限，问题只在 test fixture 表达不清。N=3、L=24、tickets=[1,3,2] 本身是有意选择的验收参数，不应隐藏。
3. 终态：用 decode_steps、initial_decode_step、post_continuation_decode_step、prefix_len、initial_decode_position、continuation_start_position 和 post_continuation_decode_position 表达阶段；horizon、history ranges、reservation 总长和最终 KV totals 从这些参数派生。focused/full tests 保持通过。
4. 业界做法：可维护的 table-driven 测试通常把场景输入作为命名 fixture，把期望值从独立的不变量公式派生；关键固定参数仍显式断言，避免测试完全复制被测实现而失去检错能力。
5. 本方案：只重构现有 oracle 的局部变量和期望计算，不改 FrozenKvAssigneeSchedule、runner、packet、cache 或 runtime。capacity-weighted 期望由 tickets 与总 unit 数计算，最终 KV totals 由 prefix split、continuation counts 和 decode counts 组合。
6. 为什么：引入通用 scenario struct 会为单个测试增加不必要抽象；保留裸数字则继续掩盖单位。局部命名参数加独立公式是最小、可审查且不会改变核心方案的修订。
VERDICT: IMPLEMENT。"""

PREFERENCE_CONTENT = """测试与实验代码不得用裸数字隐藏阶段数、位置边界、schedule horizon 或容量期望。固定研究参数可以显式保留，但必须有领域名称；由多个参数组合得到的值应从命名参数派生。尤其要区分“有限 oracle 场景长度”与“系统运行时上限”，不得让测试 fixture 看起来像架构硬限制。"""


def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence):
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
        (node_id,node_type,layer,PROJECT,title,content,importance,confidence,status,SOURCE),
    )


def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        """
        INSERT INTO edges(source,target,type,weight,note)
        VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note
        """,
        (source,target,edge_type,note),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn,TASK,"task","active","移除 continuation oracle 的魔法数字",TASK_CONTENT,"active",1.0,1.0)
    upsert_node(conn,DECISION,"decision","active","从阶段场景派生 decode horizon 与 positions",DECISION_CONTENT,"held",1.0,1.0)
    upsert_node(conn,PREFERENCE,"preference","active","实验代码避免隐藏领域语义的魔法数字",PREFERENCE_CONTENT,"held",1.0,1.0)

    edges = (
        (DECISION,TASK,"PART_OF","accepted code-review correction"),
        (METHOD,DECISION,"GOVERNS","required six-question analysis"),
        (PREFERENCE,DECISION,"GOVERNS","user requires named and derived scenario values"),
        (TASK,COMPLETED_TASK,"DEPENDS_ON","refines the verified oracle without changing its result"),
        (DECISION,EVIDENCE,"CLARIFIES","2*24 was a finite test horizon, not a runtime decode limit"),
    )
    for edge in edges:
        upsert_edge(conn,*edge)
    conn.commit()

    for node_id, expected in ((TASK,TASK_CONTENT),(DECISION,DECISION_CONTENT),(PREFERENCE,PREFERENCE_CONTENT)):
        row = conn.execute("SELECT source,content FROM nodes WHERE id=?",(node_id,)).fetchone()
        assert row == (SOURCE,expected)
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source=? AND target=? AND type='CLARIFIES'",
        (DECISION,EVIDENCE),
    ).fetchone()[0] == 1
    conn.close()

    print("magic_number_task=active")
    print("decision=exact")
    print("preference=exact")


if __name__ == "__main__":
    main()
