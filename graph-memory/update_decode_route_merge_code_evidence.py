import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@decode-route-merge-code-20260816"

DECISION = "decision-decode-route-merge-sd-mainline-20260816"
EVIDENCE = "evidence-decode-route-merge-code-verification-20260816"


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


EVIDENCE_CONTENT = """自驱动环已合并入主线 decode（commit 474e9cb，2026-08-16）：
1. 协议：新增 WorkerCommand::StationaryDecode（per-token SD decode step：token/position/capacity_tickets/starter_domain/token_offset/decode_horizon）+ validate_stationary_decode。
2. Worker：新增 decode_stationary_step（单 token 逐层单包 N-1 跳，per-(token,layer) frozen assignee 来自 decode_horizon*layers 单元计划；仅 assignee 追加增长 KV；finisher 就地完成 W_o+MLP；HCP_PERF_LOG stationary_decode 事件与 ring_decode/stationary_continuation 同形）。
3. Coordinator：ActiveRequest 增加 next_position/next_starter（prefill 与 session append 均初始化）；decode_iteration 改为主线 SD 驱动（每请求一轮 StationaryDecode，next_starter 按 finisher 轮转）；service_layer_capacities_sd 按 frozen schedule per-(token,layer) 计数做 per-domain per-layer 预留；Q-ring DecodeBatch 降级为 legacy fallback（HCP_RING_DECODE_SD=0）。
4. 验证：158 passed / 0 failed / 5 ignored（新增 decode_stationary_step_driver_matches_reference_on_mock_ring 与 service_layer_capacities_sd_covers_frozen_decode_assignees，均对照单节点 reference / golden 派生）。
5. 行为语义：SD 与 Q-ring 同为 N-1 hop/layer，通信量 SD 省 33-50%（N=3/4 实测）；合并后主线 decode 默认走自驱动环，Q-ring 仅测试/回退。
边界：多请求并发（SD 串行逐请求）与真 4 节点干净 N=4 延迟验证仍为后续工作项。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "active",
        "自驱动环合并入主线 decode 的代码与验证",
        EVIDENCE_CONTENT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_edge(conn, EVIDENCE, DECISION, "CONFIRMS", "code merge confirms the SD-as-mainline-decode decision")
    upsert_edge(conn, DECISION, EVIDENCE, "BASED_ON", "merge verified by mock-ring + reservation tests")
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence=decode-route-merge-code-verification-20260816")


if __name__ == "__main__":
    main()

