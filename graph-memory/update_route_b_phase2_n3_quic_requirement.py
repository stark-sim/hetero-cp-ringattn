import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-confirmed-2026-08-09"

PHASE2 = "task-route-b-phase2-engineering-20260809"
TASK = "task-route-b-phase2-n3-production-quic-e2e-20260809"
DECISION = "decision-route-b-phase2-require-n3-production-quic-20260809"
OLD_BELIEF = "belief-route-b-n2-quic-satisfies-phase2-item5-20260809"
METHOD = "preference-motivation-analysis-20260721"
N2_EVIDENCE = "evidence-route-b-p2-command-plan-e2e-20260809"
N3_TCP_EVIDENCE = "evidence-route-b-three-node-ring-mac-white-pearl-20260809"

PHASE2_CONTENT = """出口标准（真实跨节点 E2E 通过才考虑工程层合 main）：
1. [done] m>1 stationary packet 的 TCP/QUIC wire codec（129de9b；codec shape-generic，m>1 回环测试与 N=3 QUIC loopback 已固化）
2. [done] WorkerRuntime/coordinator 一等命令（StationaryContinuation 主路径；2592795/a4ab7f4/1aee8e6）
3. [done] frozen request plan 的分发/重建机制（plan 随 StationaryContinuation 命令广播，worker 无状态推导；1aee8e6）
4. [pending] capacity/placement byte-level admission（重启 decision-defer-placement-ledger-wip-20260809 的必要子集，不引入三期多请求能力）
5. [partial] production-path QUIC E2E：Mac MPS + white CUDA 的 N=2 已通过（generated=[198,15,15]），但 N=2 是 predecessor==successor 的退化环，不能证明 middle relay。正式出口要求 Mac MPS + white CUDA + pearl HIP 三个 worker 经 coordinator/WorkerRuntime/StationaryContinuation 在 N=3 neighbor-only QUIC ring 上通过。
依赖一期完成；当前剩余第 4 项与第 5 项 N=3 部分，每项独立小节点验证。"""

TASK_CONTENT = """把二期 production path 的跨机验证从 N=2 扩展为 N=3：Mac 运行 coordinator + domain0 worker(MPS)，white 运行 domain1 worker(CUDA)，pearl 运行 domain2 worker(HIP/ROCm)。三个 worker 的数据面只能连接 predecessor/successor，StationaryContinuation packet 必须经过 middle worker 逐跳到 finisher；coordinator 仅广播命令与收集唯一 logits，不参与模型计算或 worker 间数据转发。
验收：真实 Qwen2-0.5B 请求完成 prefill -> decode -> stationary continuation -> 后续 decode；三端正常退出；generated tokens 与同场景 golden/tie-aware 数值判据一致；日志证明每个 worker 只有 neighbor data-plane 连接且 packet 为 N-1=2 hops；记录 capacity tickets、prefix splits、offsets、finisher 与 KV totals。边界：单请求 correctness，不含性能、多请求、故障恢复或 byte-level admission。"""

OLD_BELIEF_CONTENT = """二期第 5 项曾因 Mac MPS + white CUDA 的 N=2 production-path QUIC E2E 通过而被标记完成。该证据确实证明跨机 QUIC、runtime/coordinator 命令和异构双 worker 数值正确，但 N=2 时 predecessor 与 successor 是同一 peer，无法区分 neighbor-only ring 与普通点对点直连，因此不足以完成拓扑出口。"""

DECISION_CONTENT = """动机剖析六问：
1. 问题：N=2 production QUIC E2E 被记作二期第 5 项完成，但两节点环中 predecessor==successor，不存在可观察的中间节点转发，不能证明 HCP 的 neighbor-only P2P ring 拓扑。
2. 现状：一期的 Mac MPS + white CUDA + pearl HIP 三机 TCP smoke 已证明 N=3 数学、异构 kernel 与逐跳 ring；二期的 N=3 QUIC 只在 loopback smoke 通过；production WorkerRuntime/coordinator/StationaryContinuation 路径目前只有 Mac+white N=2 跨机证据。这三份证据尚未在同一次实验中取交集。
3. 终态：Mac coordinator+MPS worker、white CUDA worker、pearl HIP worker 经 production QUIC 数据面完成同一真实 Qwen continuation 请求；每个 worker 只连接 predecessor/successor，packet 每层走 N-1=2 hops，输出与 golden 对齐，三端正常退出。
4. 他者：Ring Attention 和常见 P2P ring 都以相邻 send/recv 定义；N=2 是合法但退化的功能场景，只有 N>=3 才能用 middle relay 证伪全连接或 coordinator relay。其 collective runtime 不能替代 HCP 的 QUIC neighbor-only 验证。
5. 本方案：复用 1aee8e6 的 production command path、129de9b 的 QUIC packet wire 与既有三机 inventory；先做 N=3 production-path 配置/代码审计，只修实际阻塞任意 N 的最小缺口，再按 Mac->white->pearl->Mac 接线运行真实 Qwen E2E。
6. 为什么：单独重复三机 TCP smoke不能证明 production QUIC 接线；重复 N=2 production run 不能证明 middle relay。三机 production QUIC 是最小且直接覆盖项目拓扑主张的交叉证据，不引入性能、多请求或动态 planner。
VERDICT: IMPLEMENT。用户明确要求所有跨机验证从 Mac+white 扩展到 Mac+white+pearl，只有 N=3 才计为 neighbor-only P2P ring 出口。"""


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

    phase2_title = conn.execute(
        "SELECT title FROM nodes WHERE id=?", (PHASE2,)
    ).fetchone()
    assert phase2_title is not None
    upsert_node(
        conn,
        PHASE2,
        "task",
        "active",
        phase2_title[0],
        PHASE2_CONTENT,
        "active",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        OLD_BELIEF,
        "belief",
        "progress",
        "N=2 production QUIC 足以完成二期拓扑出口",
        OLD_BELIEF_CONTENT,
        "superseded",
        1.0,
        1.0,
        "hetero-cp-ringattn@45f2f54",
    )
    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "二期第 5 项：三机 production QUIC neighbor-only E2E",
        TASK_CONTENT,
        "active",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        DECISION,
        "decision",
        "active",
        "N=3 才能完成 production neighbor-only ring 出口",
        DECISION_CONTENT,
        "held",
        1.0,
        1.0,
        SOURCE,
    )

    edges = (
        (DECISION, TASK, "PART_OF", "N=3 production QUIC task motivation"),
        (METHOD, DECISION, "GOVERNS", "required six-question analysis"),
        (DECISION, OLD_BELIEF, "SUPERSEDES", "N=2 is a degenerate ring without middle relay"),
        (PHASE2, TASK, "DEPENDS_ON", "revised phase-2 item 5 topology exit"),
        (N2_EVIDENCE, DECISION, "SUPPORTS", "proves production QUIC and N=2 correctness, not N=3 topology"),
        (N3_TCP_EVIDENCE, DECISION, "SUPPORTS", "proves N=3 neighbor-only topology and three-platform numerical feasibility on the smoke path"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()

    phase2_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (PHASE2,)
    ).fetchone()
    decision_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (DECISION,)
    ).fetchone()
    old_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (OLD_BELIEF,)
    ).fetchone()
    assert phase2_state == ("active", SOURCE, PHASE2_CONTENT)
    assert decision_state == ("held", SOURCE, DECISION_CONTENT)
    assert old_state == (
        "superseded",
        "hetero-cp-ringattn@45f2f54",
        OLD_BELIEF_CONTENT,
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source=? AND target=? AND type='SUPERSEDES'",
            (DECISION, OLD_BELIEF),
        ).fetchone()[0]
        == 1
    )
    conn.close()

    print("phase2_item5=partial")
    print("n3_production_quic_task=active")
    print("n2_completion_belief=superseded")


if __name__ == "__main__":
    main()
