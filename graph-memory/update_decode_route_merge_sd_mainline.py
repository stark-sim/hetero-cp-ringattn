import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@decode-route-merge-sd-20260816"

DECISION = "decision-decode-route-merge-sd-mainline-20260816"
INQUIRY = "inquiry-decode-applicability-ab-20260815"


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


DECISION_CONTENT = """【用户裁决 2026-08-16】接受自驱动环为 decode 升级版：省通信量、不增跳数；decode 路线分支线关闭；合并入主线作为默认 decode；回到主线 HCP 视角；同一节点不再并行研究多个方案。

【动机六问】
1. 问题：decode 路线（Q-ring vs 自驱动环 StationaryContinuation）对比已穷尽当前可测空间（N=2/3/4 + 低 RTT 理论），但代码主线 serving loop 的 decode 仍默认 Q-ring（decode_iteration -> DecodeBatch -> ring_decode_attention），SD 仅存在于实验 --continuation-segment 命令与 session append 路径；两个方案在主线同一节点并存，违背用户"单一方案"纪律。
2. 现状：N=2 打平（通信 99.7%）；N=3 SD 省带宽 33.8% 但 tailscale RTT 下延迟 +9-15%（recv_wait 占 98-100%）；N=4 SD 省带宽 50.3%（线程版/进程版/理论三方一致，延迟受 2-GPU 争用污染不可裁决）；低 RTT 理论 SD 延迟省 ~2.5x（15.6 vs 39ms）+ 带宽省 33-50%。两条路线同为 N-1 hop/layer（ring 拓扑下界），hop 维度无法区分；通信量是唯一可实测区分的维度，SD 随 N 单调更省（0.7%/33.8%/50.3%/75.2%@N=8）。
3. 终态：SD = 主线 decode 默认路线；Q-ring 降级为 legacy 回退（保留 HCP_RING_DECODE_RING=0 语义回退路径与对照测试）；decode 路线对比分支正式关闭；主线回到 HCP 视角（route B 主线弧：prefill KV ring + decode 自驱动环）。
4. 他者：无现成轮子可复用——SD 是 HCP 特有数据流（单包 N-1 跳 + 角色轮转 + 零冗余 forward）；vLLM PD/TP 是同构集群形态，已被第一性原理（heterogeneous selling point）排除为 HCP 主线。
5. 本方案：决策层接受 SD 并关闭分支；代码层把 per-token 自驱动 decode 接入主线 serving loop（worker per-token SD decode + coordinator decode_iteration 驱动 + reservation 按 frozen schedule 对齐），Q-ring 保留为 legacy 回退。
6. 为什么：跳数同为 N-1（ring 下界）→ hop 无法区分；通信量 SD 随 N 单调更省且低 RTT 下计算主导 SD 延迟更优；单一方案纪律要求每个节点只保留一条 decode 路线。

VERDICT: SD merged into mainline as default decode; Q-ring demoted to legacy fallback; decode-route comparison branch CLOSED."""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        DECISION,
        "decision",
        "active",
        "decode 路线合并主线：自驱动环=升级版（省通信、不增跳数），Q-ring 降级 legacy，分支关闭",
        DECISION_CONTENT,
        "held",
        1.0,
        1.0,
        SOURCE,
    )

    edges = (
        # closes the branch line: supersedes the Q-ring-default stances
        (DECISION, "decision-decode-route-verdict-qring-20260815", "SUPERSEDES", "Q-ring default stance superseded: SD is the mainline decode default"),
        (DECISION, "decision-decode-route-n3-verdict-20260815", "SUPERSEDES", "N=3 'Q-ring keeps default' point superseded by SD-as-default verdict"),
        # builds on the comparison evidence
        (DECISION, "decision-decode-route-n2-verdict-20260815", "BASED_ON", "N=2 tie evidence input"),
        (DECISION, "decision-decode-route-n3-verdict-20260815", "BASED_ON", "N=3 bandwidth evidence input"),
        (DECISION, "decision-decode-route-low-rtt-n3n4-20260815", "BASED_ON", "low-RTT theory input"),
        (DECISION, "decision-decode-route-n4-emulation-verdict-20260815", "BASED_ON", "N=4 emulation evidence input"),
        (DECISION, "decision-decode-route-n4-process-verdict-20260815", "BASED_ON", "N=4 process evidence input"),
        (DECISION, "evidence-decode-route-comparison-20260815", "BASED_ON", "code-level route comparison test input"),
        # resolves the decode-applicability inquiry at the route level
        (DECISION, INQUIRY, "PARTIALLY_RESOLVES", "decode route settled (SD); decode-suitability conclusion now follows the SD mainline"),
        # mainline HCP anchor
        (DECISION, "decision-hcp-first-principles-value-20260815", "FOLLOWS", "serves the heterogeneous-selling-point mainline"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("decision=decode-route-merge-sd-mainline-20260816")
    print("branch=CLOSED")
    print("qring=legacy-fallback")


if __name__ == "__main__":
    main()

