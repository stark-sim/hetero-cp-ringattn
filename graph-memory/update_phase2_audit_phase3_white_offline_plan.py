import sqlite3
import subprocess
import sys
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-direction-2026-08-14"

TASK = "task-phase2-audit-phase3-white-offline-plan-20260814"
DECISION = "decision-phase3-plan-around-white-offline-20260814"
METHOD = "preference-motivation-analysis-20260721"
PHASE2_MERGE = "evidence-route-b-phase2-merge-verification-20260812"
PHASE3 = "task-phase3-vllm-bench-and-ecosystem-20260812"
WHITE_RECOVERY = "task-white-single-network-mode-20260814"

TASK_CONTENT = """审计路线 B 二期 benchmark-readiness 的剩余缺口，并把三期从宽泛生态目标细化为可独立验证的任务图。当前 white 因网络迁移事故暂时离线，因此计划必须区分：Mac 本地立即可做、pearl/laptop 经即时 inventory 门禁后可做、white 恢复后才能完成的真实异构门禁。产出应包含二期已充分完成项、按严重度排序的不足、每个三期节点的目标/输入/产出/验证门槛/依赖/失败判据/commit 边界，以及需要 owner 确认的 material trade-off。此节点只做审计与规划，不启动远程实验或实现。"""

DECISION_CONTENT = """动机剖析六问：
1. 问题：二期已以 benchmark-readiness 名义完成并合入 main，但其明确不含性能结论；三期已完成 vllm bench 黑盒与 continuation 服务路径 E2E，却仍有生态能力、placement/ledger、故障恢复和可比性能等开放面。white 临时离线使原先依赖 N=2/N=3 真实异构节点的顺序需要重排。
2. 现状：二期提供 admission、active reservation、FIFO decode、observability、native baseline 与 N=3 服务正确性；三期已有 7a vllm bench、8 continuation E2E 和 5-rep N=2 基线，但最新证据判定该基线受约 44 Mbit/s 双端 WiFi 链路约束，网络元数据不等价时不可比较。white 的单 networkd 迁移待物理恢复验证。
3. 终态：形成一份证据化缺口清单与依赖有向的三期执行计划；不依赖 white 的节点可以立即启动，依赖真实异构链路的节点有明确恢复门禁，所有性能结论都先通过网络等价性门禁。
4. 他者：vLLM 等 serving 生态通常把协议兼容、请求调度/batching、KV placement、故障处理和 benchmark 方法拆成独立层，并通过标准客户端与可复现环境分别验收；不能用一次端到端成功替代每层门禁。
5. 本方案：先独立审计二期边界和现有三期证据，再按本地、当前在线节点、white 恢复后三类重排任务；每个节点坚持最小可证伪产出与单独 commit 边界，远程执行前重新读取 infrastructure inventory 并做只读可达性门禁。
6. 为什么：继续按原硬件顺序会让 white 离线把全部三期串行阻塞；直接跳到实现又会重复二期已完成能力或把 WiFi 环境噪声误当算法结论。按依赖和证据类型拆分，既保持推进，也避免把环境性事实混入方案判断。
VERDICT: PLAN_AND_AUDIT；实现与远程实验等待具体节点确认。"""


def upsert_node(conn, node_id, node_type, layer, title, content, status):
    conn.execute(
        """
        INSERT INTO nodes
        (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,1.0,1.0,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type,layer=excluded.layer,project=excluded.project,
          title=excluded.title,content=excluded.content,status=excluded.status,
          source=excluded.source,updated_at=datetime('now')
        """,
        (node_id, node_type, layer, PROJECT, title, content, status, SOURCE),
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
        "审计二期缺口并重排 white 离线期间的三期计划",
        TASK_CONTENT,
        "active",
    )
    upsert_node(
        conn,
        DECISION,
        "decision",
        "active",
        "三期按证据依赖而非 white 可用性串行推进",
        DECISION_CONTENT,
        "held",
    )
    for edge in (
        (METHOD, DECISION, "GOVERNS", "required six-question pre-action analysis"),
        (DECISION, TASK, "PART_OF", "approved scope for audit and planning"),
        (TASK, PHASE2_MERGE, "DEPENDS_ON", "audit starts from the verified phase-2 merge boundary"),
        (TASK, PHASE3, "PART_OF", "refines the active phase-3 ecosystem task"),
        (TASK, WHITE_RECOVERY, "RELATES_TO", "hardware-dependent nodes wait for white recovery; planning does not"),
    ):
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("phase2_audit_phase3_plan=active")
    print("motivation=recorded")


if __name__ == "__main__":
    main()
