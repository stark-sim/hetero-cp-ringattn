import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@phase2-merge"

DECISION = "decision-route-b-phase2-merge-main-20260812"
EVIDENCE = "evidence-route-b-phase2-merge-verification-20260812"
PHASE2_TASK = "task-route-b-phase2-engineering-20260809"
BENCH_TASK = "task-phase2-rust-benchmark-readiness-20260812"
EV_6D = "evidence-phase2-rust-6d-n3-service-20260812"
EV_PHASE1_MERGE = "evidence-route-b-phase1-merge-verification-20260809"
DECISION_PHASE1 = "decision-route-b-phase1-merge-main-20260809"


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


DECISION_CONTENT = """【动机六问】
1. 问题：二期工程层（benchmark-readiness 五项出口）已在 codex 分支验证完成，成果仍未在 main；后续路线组合/公平比较需要共同锚点，且 placement/ledger WIP 等待三期重启前需有稳定 main 基线。
2. 现状：分支领先 main 40 个 checkpoint 提交（6b.0/6b.2a/6b.2b/6b.3/6c.0/6c.1/6d + graph 记录）；一期已确立合并模式（5f7a3e2 纯增量 + 边界文档 + graph.db 冲突取超集版）；二期全部证据为 correctness/服务稳定性，不含性能声明。
3. 终态：merge commit 547e970 把二期服务化资产纯增量合入 main；边界文档 docs/CONTINUATION_ROUTE_BOUNDARIES.md 补充 G 节（二期资产/语义边界/验证矩阵）与证据索引；main 行为零变化；分支保留。
4. 他者：一期合 main 的决策（decision-route-b-phase1-merge-main-20260809）确立六类边界与 graph 冲突处理；本决策沿用同一纪律。
5. 本方案：--no-ff merge 保留 40 个 checkpoint 的 RED/GREEN 历史；代码全部自动合并无冲突；graph.db/active.md 冲突取 codex 超集版（483 节点）后重新 export；边界文档 G 节定义二期服务化语义边界（admission/active ledger/FIFO decode/trace，均不含性能）。
6. 为什么：二期以三期里程碑门禁第二阶段（工程性能力）过关，使 Rust HCP 服务具备接受外部 benchmark 的基础；合 main 使二期资产成为可组合的共同锚点，同时以文档明确"不含性能结论、placement WIP 留三期"。
VERDICT: MERGED(547e970)。"""

EVIDENCE_CONTENT = """合并后 main(547e970) 完整回归验证通过：
1. cargo test --features tch-backend --lib = 141 passed、0 failed、5 ignored（与 codex 分支二期收口时一致，main 行为零回归）。
2. graph.db 冲突取 codex 超集版（483 节点）后重新 export，active/progress/systemPatterns/productContext/techContext 与 graph.db 一致。
3. 边界文档 docs/CONTINUATION_ROUTE_BOUNDARIES.md 补充 G 节：二期服务化资产表（4784acf/d57b9ca/9ec8f96/abddbf1/78be1d0/f249d90/9a42934）、五条语义边界（admission/active ledger/FIFO/trace/不含性能）、二期验证矩阵（N=2 concurrency 1/2/4 + N=3 异构真实 Qwen）；证据索引追加二期 7 条。
4. 二期全部证据为 correctness/服务稳定性；不含性能结论；placement/ledger WIP 保持 main 工作区 stash DEFER 状态（三期素材）。
5. 代码冲突为零（40 个 checkpoint 全部纯增量），仅 graph-memory 两文件冲突按一期约定处理。
边界：二期 benchmark-readiness 五项出口达成；三期（生态：多请求 batching、placement/ledger WIP 重启、外部 benchmark）待用户规划。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        DECISION,
        "decision",
        "active",
        "二期工程层以纯增量合入 main",
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
        "active",
        "二期合 main 后回归验证通过",
        EVIDENCE_CONTENT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )
    conn.execute("UPDATE nodes SET status='completed', updated_at=datetime('now') WHERE id=?", (PHASE2_TASK,))
    conn.execute("UPDATE nodes SET status='completed', updated_at=datetime('now') WHERE id=?", (BENCH_TASK,))

    edges = (
        (EVIDENCE, DECISION, "CONFIRMS", "merge verification confirms decision"),
        (DECISION, EV_6D, "BASED_ON", "6d evidence is the final engineering gate"),
        (DECISION, DECISION_PHASE1, "FOLLOWS", "reuses phase-1 merge discipline"),
        (DECISION, EV_PHASE1_MERGE, "BASED_ON", "phase-1 merge verification precedent"),
        (DECISION, PHASE2_TASK, "PART_OF", "phase-2 engineering graduates via this merge"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("phase2_merge=547e970")
    print("phase2_task=completed")
    print("benchmark_readiness=completed")


if __name__ == "__main__":
    main()
