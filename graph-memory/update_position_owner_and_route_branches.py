import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-confirmed-2026-08-07"

TASK = "task-continuation-position-owner-local-kv-20260807"
MOTIVATION = "decision-continuation-position-owner-local-kv-20260807"
BRANCH_PREFERENCE = "preference-route-experiments-use-isolated-branches-20260807"
BRANCH_DECISION = "decision-continuation-route-branch-discipline-20260807"
METHOD = "preference-motivation-analysis-20260721"
ROUTE = "hypothesis-continuation-route-full-activation-packet-20260807"
PREVIOUS = "task-continuation-packet-layer-contract-20260807"
PORTFOLIO = "task-continuation-route-experiment-portfolio-20260807"

TASK_CONTENT = """当前小节点只证明 m-segment 的新 K/V 能按 position 分散到多个 ReservedPositionedKvShard。使用 tickets=[1,3,2]、N=3、m=6 的单层 synthetic case：请求级 frozen schedule 产生 1:3:2 owner counts；每个 domain 只投影并 append 自己负责的 normalized/position subset；全部 m 个 Q 仍逐节点合并 local causal partial。验收包括 owner 完备且无重复、各 shard 不越预留容量、storage pointer 稳定、attention 与整层 hidden 对 dense reference 数值一致、packet payload 不携带历史 KV 或 per-layer owner vector。范围不含 wire/runtime、24 层、多请求、动态 planner 或性能结论。"""

MOTIVATION_CONTENT = """动机剖析六问：
1. 问题：上一节点虽证明 m>1 stationary packet 的整层数学正确，但把整段新 KV 都交给一个 assignee；长 continuation segment 会让单节点承担整段增量，尚未满足 HCP 的 capacity-weighted 永久 KV 压力目标。
2. 现状：LayerPacket 只有 decode 沿用的单 assignee；ReservedPositionedKvShard 已支持任意 absolute positions 与预分配 append；FrozenKvAssigneeSchedule 已能按 tickets 产生确定性完整日历，但尚未连接到 m-segment 内的 position subset projection。
3. 终态：在 N=3、m=6、tickets=[1,3,2] 单层实验中，新 positions 精确按 1:3:2 分散；每个 (layer,position) 只生成和 append 一次；所有 query 对 history+causally-visible new KV 的输出与 dense reference 一致；无 shard 越界或 storage 重分配。
4. 业界做法：vLLM/PagedAttention 类 serving 系统把 block/page placement 放在请求级 block table 或 scheduler metadata，而不是塞进每层 attention activation。可复用的是“placement 为控制状态、kernel 只消费本地映射”的边界；其 collective/full-connect 通信机制不适用于 HCP 的 neighbor-only P2P ring。
5. 本方案：position ownership 由共享 frozen request plan 本地推导，不进入 LayerPacket。新增最小 positioned subset processing API：输入当前 domain 的 position offsets，index-select normalized 与 absolute position_ids，仅投影/append 本地新 K/V，然后照旧以全量 Q 对本地完整 shard 计算 partial。旧单-assignee wrapper 保留并映射为 all-or-empty offsets。
6. 为什么：相比在 packet 中传 m 个 owner id，本方案不增加逐层链路 payload，且 ownership 与 reservation 使用同一个冻结计划，避免两份真相；相比拆成多个 query packet，它保持单 packet、N-1 hops 和现有 finisher 数据流。
边界与风险：packet 层不独立证明全局 owner 集合完备，正确性由冻结计划生成/校验和本节点 oracle 共同保证；真实 runtime 如何分发或重建该计划以后单独处理。没有删除旧单-assignee能力，也不引入动态 planner。
VERDICT: IMPLEMENT。"""

BRANCH_PREFERENCE_CONTENT = """不同 continuation 理论路线必须使用不同 Git 分支实现和测量，以便比较真实性能且避免优化相互污染。共同 correctness/benchmark 修复应从明确的公共锚点单独提交，再 cherry-pick 或等价同步到各路线；报告必须记录 branch、commit、hardware inventory、模型、T、m、N、dtype、warmup/repetition 与 payload/hop 定义。main 用于共同结论和 Graph 记忆，不把某条路线的胜出预设为主架构。"""

BRANCH_DECISION_CONTENT = """Continuation 路线实验采用 branch-per-route。公共路线对比代码锚点暂定为 93e83cb（七路线 portfolio 已记录、m>1 路线 B 实现尚未进入）；路线 B 当前分支命名 codex/route-b-continuation-stationary-packet，并包含 5777d51 之后的 stationary packet checkpoints。路线 A/C/D/E 只在真正开始实验时从公共锚点或后续明确 supersede 的 benchmark anchor 建分支，不提前创建空分支。路线 F 是可组合 kernel 技术，应从它所组合的主路线分叉；路线 G 是显式阶段组合，只有 A/B 都有同口径数据后再建。比较时禁止直接用含不同 correctness 修复或 benchmark harness 的提交做结论，必须先对齐公共测量提交。VERDICT: IMPLEMENT。"""


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

    upsert_node(conn,TASK,"task","active","m-segment 新 KV 按 position owner-local 生成",TASK_CONTENT,"active",1.0,1.0)
    upsert_node(conn,MOTIVATION,"decision","active","position ownership 留在 frozen request plan",MOTIVATION_CONTENT,"held",1.0,1.0)
    upsert_node(conn,BRANCH_PREFERENCE,"preference","active","理论路线使用隔离 Git 分支测量",BRANCH_PREFERENCE_CONTENT,"held",1.0,1.0)
    upsert_node(conn,BRANCH_DECISION,"decision","active","Continuation 性能实验采用 branch-per-route",BRANCH_DECISION_CONTENT,"held",1.0,1.0)

    edges = (
        (MOTIVATION,TASK,"PART_OF","approved pre-action motivation analysis"),
        (METHOD,MOTIVATION,"GOVERNS","required six-question analysis"),
        (TASK,PREVIOUS,"DEPENDS_ON","requires the verified m>1 whole-layer packet contract"),
        (TASK,ROUTE,"PART_OF","second checkpoint on route B"),
        (MOTIVATION,ROUTE,"SUPPORTS","preserves stationary-history route without owner metadata payload"),
        (BRANCH_PREFERENCE,BRANCH_DECISION,"GOVERNS","user-required route isolation for comparison"),
        (BRANCH_DECISION,PORTFOLIO,"PART_OF","governs all retained continuation experiments"),
        (BRANCH_DECISION,TASK,"GOVERNS","current node runs on route B branch"),
    )
    for edge in edges:
        upsert_edge(conn,*edge)
    conn.commit()

    expected = {
        TASK: TASK_CONTENT,
        MOTIVATION: MOTIVATION_CONTENT,
        BRANCH_PREFERENCE: BRANCH_PREFERENCE_CONTENT,
        BRANCH_DECISION: BRANCH_DECISION_CONTENT,
    }
    for node_id, content in expected.items():
        got = conn.execute("SELECT content FROM nodes WHERE id=?",(node_id,)).fetchone()
        assert got is not None and got[0] == content
    assert conn.execute("SELECT status FROM nodes WHERE id=?",(TASK,)).fetchone()[0] == "active"
    assert conn.execute("SELECT COUNT(*) FROM edges WHERE source=?",(MOTIVATION,)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM edges WHERE source=?",(BRANCH_DECISION,)).fetchone()[0] == 2
    conn.close()

    print("position_owner_task=active")
    print("motivation=exact")
    print("branch_preference=exact")
    print("branch_decision=exact")


if __name__ == "__main__":
    main()
